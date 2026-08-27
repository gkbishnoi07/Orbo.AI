"""Composition root: artifacts in, explained recommendations out.

Deliberately knows nothing about Streamlit. The UI is one caller; the scripted
test cases in `scripts/06_test_cases.py` are another, and neither should be able
to drift from the other on what "the hybrid" actually means.

Everything is loaded from `artifacts/` — no Kaggle credentials, no raw CSV, no
sentence-transformers. Fitting happens once at construction (about five seconds,
dominated by the cohort similarity matrices) and the UI caches the instance for
the life of the container.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import artifacts
from .collaborative import CohortCFRecommender
from .explain import Explainer
from .hybrid import CFOnlyRecommender, ContentOnlyRecommender, HybridRecommender
from .recommender import PopularityRecommender, RandomRecommender, Recommender
from .schema import CONCERNS, ProductCols, Query, ReviewCols, Scored
from .tone import ToneAffinity

DEFAULT_METHOD = "tfidf"
"""TF-IDF, not MiniLM.

Measured, not assumed: content-tfidf beat content-minilm on warm-start NDCG@10
in the recorded run, because the product text is highlight tokens and INCI
ingredient lists rather than prose. The live figures are in
reports/evaluation.json; no ratio is repeated here, so nothing can go stale."""


@dataclass(frozen=True)
class StrategyInfo:
    """One selectable approach, described twice.

    `blurb` is what a shopper reads and `technical` is what an engineer reads.
    Keeping both means the UI can lead with "products people like you rated
    highly" without losing "item-item cosine similarity per skin-type cohort" —
    a shopper should never have to learn the word cohort to use this, and an
    evaluator should never have to guess what is underneath the friendly wording.
    """

    key: str
    label: str
    blurb: str
    technical: str


STRATEGIES: tuple[StrategyInfo, ...] = (
    StrategyInfo(
        key="hybrid",
        label="Best overall",
        blurb="Weighs everything: how well a product fits your profile, what "
        "shoppers like you rated highly, and how well loved it is generally.",
        technical="Weighted blend of cohort CF, TF-IDF content similarity and a "
        "log-damped popularity prior, reranked with MMR for intra-list diversity. "
        "Switches to a popularity-led blend when there is no history.",
    ),
    StrategyInfo(
        key="cohort-cf",
        label="Similar shoppers",
        blurb="Finds products that were rated highly by people with a skin "
        "profile like yours who liked what you like.",
        technical="Item-item cosine similarity rebuilt per skin-type cohort, with "
        "co-occurrence shrinkage. Blind to the 72% of the catalogue with no reviews.",
    ),
    StrategyInfo(
        key="content",
        label="Product match",
        blurb="Reads each product's own description, ingredients and labels, and "
        "matches them against what you asked for.",
        technical="Cosine similarity over precomputed TF-IDF vectors of the product "
        "text blob, plus exact concern and skin-type tag matching. The only layer "
        "that can rank a product with no reviews.",
    ),
    StrategyInfo(
        key="popularity",
        label="Most popular",
        blurb="Simply what the most people rated highly — the same list for "
        "everybody, with no personalisation at all.",
        technical="Log-damped count of ratings at 4+. A plain bestselling page "
        "gives this away for free, so it is the bar personalisation must clear.",
    ),
    StrategyInfo(
        key="random",
        label="Random",
        blurb="Picks at random from whatever passes your filters. Included so you "
        "can see what no useful signal looks like.",
        technical="The floor. Any approach that cannot beat this is broken.",
    ),
)

STRATEGIES_BY_KEY = {s.key: s for s in STRATEGIES}

EVAL_ROW_FOR_STRATEGY: dict[str, str] = {
    "hybrid": "hybrid-tfidf",
    "cohort-cf": "cf-only",
    "content": "content-tfidf",
    "popularity": "popularity",
    "random": "random",
}
"""Which row of reports/evaluation.json corresponds to each UI strategy.

The UI used to carry its own copy of these numbers as literals. They were
correct when typed and would have silently gone stale on the next re-tune, while
still being rendered next to live-computed catalogue facts — indistinguishable
to a reader from something measured."""


def load_benchmark(root: Path = Path("reports")) -> dict | None:
    """The recorded evaluation run, or None when it has not been generated.

    Never computed at request time: a full leave-one-out sweep is minutes of
    work, so the UI reports the recorded run and says which commit produced it.
    """
    path = Path(root) / "evaluation.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

LAYER_LABELS: dict[str, str] = {
    "content": "Product match",
    "collaborative": "Similar shoppers",
    "popularity": "Overall popularity",
    "concern": "Addresses your concerns",
    "skin": "Suits your skin type",
    "embedding": "Description similarity",
    "tone": "Rated well by your skin tone",
}
"""Shopper-facing names for the score components. The internal keys stay as they
are — renaming them would ripple through the evaluation reports."""


@dataclass(frozen=True)
class CatalogueFacts:
    """Numbers the UI shows so a reviewer can see the shape of the data."""

    n_products: int
    n_brands: int
    n_categories: int
    n_interactions: int
    n_reviewed_products: int
    price_min: float
    price_max: float
    price_median: float

    @property
    def reviewed_share(self) -> float:
        return self.n_reviewed_products / self.n_products if self.n_products else 0.0


class RecommendationService:
    """Holds every fitted strategy and the explainer, ready to answer queries."""

    def __init__(
        self,
        products: pd.DataFrame,
        interactions: pd.DataFrame,
        cohort_stats: pd.DataFrame,
        embeddings: np.ndarray,
        embedding_ids: list[str],
        tone_stats: pd.DataFrame | None = None,
    ) -> None:
        self.products = products
        self._indexed = products.set_index(ProductCols.ID, drop=False)

        # The hybrid is fitted first and its sub-models are then reused as the
        # standalone strategies. Constructing separate instances would rebuild
        # the same cohort similarity matrices a second time for no benefit.
        self.tone = ToneAffinity(tone_stats) if tone_stats is not None else None
        hybrid = HybridRecommender(
            content=ContentOnlyRecommender(embeddings, embedding_ids),
            collaborative=CFOnlyRecommender(),
            popularity=PopularityRecommender(),
            tone=self.tone,
        )
        hybrid.fit(products, interactions)

        self._models: dict[str, Recommender] = {
            "hybrid": hybrid,
            "cohort-cf": hybrid.collaborative,
            "content": hybrid.content,
            "popularity": hybrid.popularity,
            "random": RandomRecommender(seed=0).fit(products, interactions),
        }
        self.explainer = Explainer(products, cohort_stats)
        self._interaction_counts = (
            interactions.groupby(ReviewCols.ITEM).size().sort_values(ascending=False)
        )
        self._facts = self._compute_facts(interactions)

    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls, root: Path = artifacts.ARTIFACTS, method: str = DEFAULT_METHOD
    ) -> "RecommendationService":
        products = artifacts.load_products(root)
        interactions = artifacts.load_interactions(root)
        cohort_stats = artifacts.load_cohort_stats(root)
        matrix, ids = artifacts.load_embeddings(method, root)
        tone_stats = artifacts.load_tone_stats(root)
        return cls(products, interactions, cohort_stats, matrix, ids, tone_stats)

    def _compute_facts(self, interactions: pd.DataFrame) -> CatalogueFacts:
        price = pd.to_numeric(self.products[ProductCols.PRICE], errors="coerce")
        return CatalogueFacts(
            n_products=len(self.products),
            n_brands=self.products[ProductCols.BRAND].nunique(),
            n_categories=self.products[ProductCols.CATEGORY].nunique(),
            n_interactions=len(interactions),
            n_reviewed_products=interactions[ReviewCols.ITEM].nunique(),
            price_min=float(price.min()),
            price_max=float(price.max()),
            price_median=float(price.median()),
        )

    # ------------------------------------------------------------------
    # What the UI needs to build its controls
    # ------------------------------------------------------------------

    @property
    def facts(self) -> CatalogueFacts:
        return self._facts

    def categories(self) -> list[tuple[str, str]]:
        """(primary, secondary) pairs, so the picker can group sensibly."""
        pairs = (
            self.products[[ProductCols.PRIMARY_CATEGORY, ProductCols.CATEGORY]]
            .dropna()
            .drop_duplicates()
            .sort_values([ProductCols.PRIMARY_CATEGORY, ProductCols.CATEGORY])
        )
        return [(str(a), str(b)) for a, b in pairs.itertuples(index=False)]

    def concerns_for(self, domain: str) -> list[tuple[str, str]]:
        return [(c.slug, c.label) for c in CONCERNS if c.domain == domain]

    def history_pool(self, limit: int = 2000) -> list[tuple[str, str]]:
        """Products a user can declare as owned, most-interacted-with first.

        Ordered by presence in the *interaction* data, not by the catalogue's
        `review_count`. Those two disagree badly and it is a trap: `review_count`
        is Sephora's site-wide figure, so a product can show 21,281 reviews and
        still have zero rows here — the review files only cover 2,343 of 8,494
        products. Ranking the picker by `review_count` offers mostly products
        collaborative filtering has never seen, which leaves it cold while
        looking like it should have fired.
        """
        ids = [
            product_id
            for product_id in self._interaction_counts.index[: limit * 2]
            if product_id in self._indexed.index
        ][:limit]
        return [
            (
                str(product_id),
                f"{self._indexed.loc[product_id, ProductCols.BRAND]} — "
                f"{self._indexed.loc[product_id, ProductCols.NAME]}",
            )
            for product_id in ids
        ]

    def product(self, product_id: str) -> pd.Series | None:
        if product_id not in self._indexed.index:
            return None
        return self._indexed.loc[product_id]

    def search(self, text: str, limit: int = 40) -> pd.DataFrame:
        """Substring match over brand and name, for the 'products I already use' box."""
        if not text or len(text) < 2:
            return self.products.head(0)
        needle = text.strip().casefold()
        haystack = (
            self.products[ProductCols.BRAND].astype(str)
            + " "
            + self.products[ProductCols.NAME].astype(str)
        ).str.casefold()
        return self.products[haystack.str.contains(needle, regex=False)].head(limit)

    # ------------------------------------------------------------------
    # The actual ask
    # ------------------------------------------------------------------

    def recommend(
        self, query: Query, strategy: str = "hybrid", k: int = 10
    ) -> list[Scored]:
        """Rank, then attach the evidence for each result."""
        if strategy not in self._models:
            raise KeyError(f"unknown strategy {strategy!r}")
        results = self._models[strategy].recommend(query, k=k)
        for scored in results:
            scored.evidence = self.explainer.explain(scored, query)
        return results

    def excluded_counts(self, query: Query, strategy: str = "hybrid") -> dict[str, int]:
        """Which rules removed how many products — shown so filtering is visible."""
        return self._models[strategy].excluded_counts(query)

    def eligible_count(self, query: Query, strategy: str = "hybrid") -> int:
        """Products surviving the hard filters, before ranking."""
        model = self._models[strategy]
        return len(model._apply_filters(query))

    def cohort_size(self, skin_type: str | None) -> int:
        """Interactions behind the cohort this query will be scored against."""
        cf = self._models["cohort-cf"]
        assert isinstance(cf, CohortCFRecommender)
        sizes = cf.cohort_sizes()
        if skin_type and skin_type in sizes:
            return sizes[skin_type]
        from .collaborative import GLOBAL_COHORT

        return sizes.get(GLOBAL_COHORT, 0)
