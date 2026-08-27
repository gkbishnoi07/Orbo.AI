"""Hybrid reranker: blend the layers, then diversify.

Content and collaborative filtering fail in opposite directions here, which is
the whole reason for combining them. CF knows things product copy never says,
but it can only speak about the 27.7% of the catalogue that has reviews.
Content can rank anything with text, but it cannot tell a beloved product from a
merely well-described one. Blending covers both; a small popularity prior breaks
ties in the direction of things people actually buy.

Then MMR. Blending alone tends to return ten near-identical products — the same
hyaluronic serum from ten brands — because whatever the profile matches, it
matches repeatedly. Maximal Marginal Relevance gives up a little relevance per
slot to buy variety, which is why `intra_list_diversity` is reported next to
NDCG rather than instead of it.

A blend needs comparable inputs, so each layer's raw score is min-max scaled
across the catalogue before weighting. That matters more than it sounds: CF
scores are zero for the ~72% of products it has never seen, and rescaling makes
that abstention explicit instead of letting an arbitrary scale decide the
ranking.

The weights below come from `scripts/05_sweep.py`, and the sweep produced an
awkward answer worth stating plainly rather than burying: **measured on
leave-one-out NDCG, the optimum is degenerate.** Warm-start NDCG rises
monotonically as the content weight falls to zero (0.3748 -> 0.3812), and
cold-start NDCG rises monotonically as popularity rises to one (0.0289 ->
0.0665). On the metric alone, the right hybrid is no hybrid.

The reason is a limitation of the protocol, not a verdict on the layers.
Held-out items are drawn from the review table, so every correct answer in
leave-one-out is by construction a product that *has* reviews — one of the 2,343
CF can rank. The 6,151 products with no reviews can never be a hit, so the one
thing content uniquely provides, reach across the other 72% of the catalogue,
is invisible to this measurement. An offline metric built on interaction data
cannot reward covering items that have no interaction data.

So the shipped defaults keep small content and CF terms against the metric's
advice, and the cost is stated rather than hidden: -0.4% warm NDCG and -9% cold
NDCG versus the degenerate optima. What that buys is a cold-start list that
responds to the profile someone actually typed in. Pure popularity scored best
on cold start while returning roughly twelve distinct products across 400 users
(coverage 0.0014) — the same list for everyone, which is not a product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .collaborative import CohortCFRecommender
from .content import ContentRecommender
from .recommender import PopularityRecommender, Recommender
from .schema import ProductCols, Query
from .tone import ToneAffinity


def _scale(series: pd.Series) -> pd.Series:
    low, high = float(series.min()), float(series.max())
    if high - low < 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


class HybridRecommender(Recommender):
    """Weighted blend of content, cohort CF and popularity, reranked by MMR."""

    name = "hybrid"

    def __init__(
        self,
        content: ContentRecommender,
        collaborative: CohortCFRecommender,
        popularity: PopularityRecommender | None = None,
        tone: ToneAffinity | None = None,
        *,
        weight_content: float = 0.15,
        weight_cf: float = 0.80,
        weight_popularity: float = 0.05,
        cold_weight_content: float = 0.10,
        cold_weight_cf: float = 0.30,
        cold_weight_popularity: float = 0.60,
        weight_tone: float = 0.10,
        mmr_lambda: float = 0.85,
        candidate_pool: int = 60,
    ) -> None:
        super().__init__()
        self.content = content
        self.collaborative = collaborative
        self.popularity = popularity or PopularityRecommender()
        self.tone = tone
        # Added on top of the three existing weights rather than folded into
        # them, so the blend those weights were swept for is untouched. The tone
        # term is centred on zero, so a query with no tone — or a product with no
        # tone history — contributes exactly nothing and the result is identical
        # to the previous behaviour.
        self.weight_tone = weight_tone
        self.weight_content = weight_content
        self.weight_cf = weight_cf
        self.weight_popularity = weight_popularity
        # Cold start is a different problem, so it gets different weights.
        # Measured: with no history the content layer's embedding term is close
        # to worthless (NDCG@10 0.002 at 0.4% coverage — it hands nearly every
        # new user the same ten products) because the profile prototype is a
        # centroid shared by everyone with that skin type. Leaning on cohort
        # popularity instead is an honest account of what is actually known
        # about someone who has told us their skin type and nothing else.
        self.cold_weight_content = cold_weight_content
        self.cold_weight_cf = cold_weight_cf
        self.cold_weight_popularity = cold_weight_popularity
        self.mmr_lambda = mmr_lambda
        self.candidate_pool = candidate_pool

    def fit(self, products: pd.DataFrame, reviews: pd.DataFrame, **kwargs):
        super().fit(products, reviews)
        self.content.fit(products, reviews)
        self.collaborative.fit(products, reviews, **kwargs)
        self.popularity.fit(products, reviews)
        return self

    # ------------------------------------------------------------------

    def _weights(self, query: Query) -> tuple[float, float, float]:
        if query.is_cold_start:
            return (
                self.cold_weight_content,
                self.cold_weight_cf,
                self.cold_weight_popularity,
            )
        return self.weight_content, self.weight_cf, self.weight_popularity

    def _score(self, query: Query) -> pd.Series:
        w_content, w_cf, w_popularity = self._weights(query)
        content_scores = _scale(self.content._score(query))
        content_parts = dict(self.content._components)

        cf_scores = _scale(self.collaborative._score(query))
        cf_parts = dict(self.collaborative._components)

        popularity_scores = _scale(self.popularity._score(query))

        total = (
            w_content * content_scores
            + w_cf * cf_scores.reindex(content_scores.index, fill_value=0.0)
            + w_popularity
            * popularity_scores.reindex(content_scores.index, fill_value=0.0)
        )

        # Signed, unscaled, and added last. It is not min-max scaled like the
        # others because scaling would move a genuine "no signal" of 0 up to
        # whatever the lowest affinity happens to be, turning an abstention into
        # an opinion.
        if self.tone is not None and query.skin_tone:
            tone_scores = self.tone.scores(query.skin_tone, content_scores.index)
            total = total + self.weight_tone * tone_scores
        else:
            tone_scores = pd.Series(0.0, index=content_scores.index)

        self._components = {
            "content": content_scores,
            "collaborative": cf_scores,
            "popularity": popularity_scores,
            "tone": tone_scores,
            **{k: v for k, v in content_parts.items()},
            **{k: v for k, v in cf_parts.items()},
        }
        return total

    # ------------------------------------------------------------------

    def _select(self, scores: pd.Series, query: Query, k: int) -> pd.Series:
        """Greedy MMR over the top `candidate_pool` products."""
        if self.mmr_lambda >= 1.0 or len(scores) <= k:
            return scores.head(k)

        pool = scores.head(max(self.candidate_pool, k))
        rows = [
            self.content._row_of[pid]
            for pid in pool.index
            if pid in self.content._row_of
        ]
        if len(rows) != len(pool):
            # Cannot measure similarity for part of the pool; relevance only.
            return scores.head(k)

        vectors = self.content.embeddings[rows]
        similarity = vectors @ vectors.T
        relevance = _scale(pool).to_numpy()

        # Greedy MMR, but the penalty is carried forward as a running maximum
        # instead of recomputed against every selected item each round. Same
        # result, and it keeps the reranker off the request's critical path.
        wanted = min(k, len(pool))
        selected: list[int] = []
        available = np.ones(len(pool), dtype=bool)
        worst_case = np.zeros(len(pool), dtype=np.float32)

        for _ in range(wanted):
            value = (
                self.mmr_lambda * relevance
                - (1.0 - self.mmr_lambda) * worst_case
            )
            value[~available] = -np.inf
            best = int(np.argmax(value))
            selected.append(best)
            available[best] = False
            worst_case = np.maximum(worst_case, similarity[best])

        return pool.iloc[selected]


class ContentOnlyRecommender(ContentRecommender):
    """Named separately so the comparison table reads clearly."""

    name = "content-only"


class CFOnlyRecommender(CohortCFRecommender):
    name = "cf-only"
