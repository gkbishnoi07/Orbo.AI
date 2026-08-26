"""Recommender interface and non-personalised baselines.

Every strategy implements the same two methods, so `evaluate.py` can score all
of them through one loop and the UI can swap strategies without changing.

The baselines here exist to make the hybrid's numbers mean something. "NDCG@10
of 0.21" is not a result; "NDCG@10 of 0.21 against 0.09 for popularity" is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import rules
from .schema import POSITIVE_RATING, ProductCols, Query, ReviewCols, Scored


class Recommender(ABC):
    """Base class for every recommendation strategy."""

    name: str = "base"

    def __init__(self) -> None:
        self._products: pd.DataFrame | None = None
        self._irritant_flags: pd.DataFrame | None = None
        self._skin_type_flags: pd.DataFrame | None = None
        self._concern_flags: pd.DataFrame | None = None
        self._components: dict[str, pd.Series] = {}
        self._fitted = False

    def fit(self, products: pd.DataFrame, reviews: pd.DataFrame) -> "Recommender":
        """Prepare the shared state every strategy needs, then mark as fitted.

        Subclasses that learn something of their own override this and call
        `super().fit(...)` first. The rule matrices are built here rather than
        per strategy so a baseline cannot accidentally skip them.
        """
        self._products = products.reset_index(drop=True)
        self._irritant_flags = rules.build_irritant_flags(self._products)
        self._skin_type_flags = rules.build_skin_type_flags(self._products)
        self._concern_flags = rules.build_concern_flags(self._products)
        self._fitted = True
        return self

    @abstractmethod
    def _score(self, query: Query) -> pd.Series:
        """Score every product for this query. Index is product_id.

        Implementations that blend several signals should record them in
        `self._components` so the UI can show what drove each result. The
        contract is deliberately write-only scratch space for one call: it is
        read immediately by `recommend` and never relied on afterwards.
        """

    def _select(self, scores: pd.Series, query: Query, k: int) -> pd.Series:
        """Choose the final k from the ranked candidates.

        A plain head(k) here; `HybridRecommender` overrides it to trade a little
        relevance for intra-list variety.
        """
        return scores.head(k)

    def recommend(self, query: Query, k: int = 10) -> list[Scored]:
        """Apply hard filters, score, and return the top k."""
        if not self._fitted:
            raise RuntimeError(f"{self.name}: call fit() before recommend()")

        eligible = self._apply_filters(query)
        if eligible.empty:
            return []

        self._components = {}
        scores = self._score(query).reindex(eligible[ProductCols.ID])
        scores = scores.dropna().sort_values(ascending=False)

        # Never recommend something the user told us they already have.
        scores = scores.drop(labels=list(query.liked_product_ids), errors="ignore")
        if scores.empty:
            return []

        chosen = self._select(scores, query, k)

        return [
            Scored(
                product_id=str(pid),
                score=float(value),
                components={
                    name: float(series.get(pid, 0.0))
                    for name, series in self._components.items()
                },
            )
            for pid, value in chosen.items()
        ]

    def _apply_filters(self, query: Query) -> pd.DataFrame:
        """Hard constraints. These are non-negotiable regardless of score.

        Kept in the base class so every strategy honours budget, category and
        the suitability rules identically — a baseline that quietly ignored the
        budget cap would make the comparison table dishonest.
        """
        assert self._products is not None
        frame = self._products

        if query.category and ProductCols.CATEGORY in frame.columns:
            match = frame[ProductCols.CATEGORY].astype(str).str.casefold()
            frame = frame[match == query.category.casefold()]

        if query.budget_max is not None and ProductCols.PRICE in frame.columns:
            price = pd.to_numeric(frame[ProductCols.PRICE], errors="coerce")
            frame = frame[price.notna() & (price <= query.budget_max)]

        if self._irritant_flags is not None and self._skin_type_flags is not None:
            drop = rules.excluded(
                self._irritant_flags,
                self._skin_type_flags,
                skin_type=query.skin_type,
                concerns=query.concerns,
            )
            keep = ~frame[ProductCols.ID].map(drop).fillna(False).to_numpy()
            frame = frame[keep]

        return frame

    def excluded_counts(self, query: Query) -> dict[str, int]:
        """Which rules removed how many products, for the UI and the write-up."""
        if self._irritant_flags is None or self._skin_type_flags is None:
            return {}
        return rules.exclusion_breakdown(
            self._irritant_flags,
            self._skin_type_flags,
            skin_type=query.skin_type,
            concerns=query.concerns,
        )


class RandomRecommender(Recommender):
    """Floor. Anything that cannot beat this is broken."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def fit(self, products: pd.DataFrame, reviews: pd.DataFrame) -> "RandomRecommender":
        super().fit(products, reviews)
        return self

    def _score(self, query: Query) -> pd.Series:
        assert self._products is not None
        ids = self._products[ProductCols.ID]
        return pd.Series(self._rng.random(len(ids)), index=ids)


class PopularityRecommender(Recommender):
    """The bar that actually matters.

    Popularity is a genuinely strong baseline in sparse retail data and is what
    a plain "sort by bestselling" page already gives users for free. A
    personalised model earns its complexity only by beating this.
    """

    name = "popularity"

    def __init__(self, min_reviews: int = 5) -> None:
        super().__init__()
        self.min_reviews = min_reviews
        self._scores: pd.Series | None = None

    def fit(
        self, products: pd.DataFrame, reviews: pd.DataFrame
    ) -> "PopularityRecommender":
        super().fit(products, reviews)

        positives = reviews[reviews[ReviewCols.RATING] >= POSITIVE_RATING]
        counts = positives.groupby(ReviewCols.ITEM).size()
        counts = counts[counts >= self.min_reviews]

        # Log-damped so a handful of megasellers do not flatten everything else.
        self._scores = np.log1p(counts).reindex(
            products[ProductCols.ID], fill_value=0.0
        )
        self._fitted = True
        return self

    def _score(self, query: Query) -> pd.Series:
        assert self._scores is not None
        return self._scores
