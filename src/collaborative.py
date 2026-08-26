"""Cohort-scoped item-item collaborative filtering.

`scripts/01_inspect.py` cleared this design: skin_type is populated on 89.8% of
reviews and the median product-by-skin-type cohort holds 32 reviews, so slicing
the interaction matrix by skin type leaves cohorts with something to say. Had the
audit come back negative the fallback was global item-item CF with profile
fields demoted to filters; it did not, so the cohort version stands.

Why bother scoping at all: "people with oily skin who liked this also liked
that" is a genuinely different statement from "people who liked this also liked
that", and it captures preferences no amount of product text can express — which
foundation actually oxidises, which moisturiser pills under sunscreen. Product
copy never says that; reviewers do.

Two things this layer cannot do, both handled elsewhere:

* It can only rank the 2,351 products that have reviews, 27.7% of the
  catalogue. Content covers the rest.
* With no interaction history it has no user-specific signal at all and falls
  back to cohort popularity.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .recommender import Recommender
from .schema import POSITIVE_RATING, SKIN_TYPES, ProductCols, Query, ReviewCols

GLOBAL_COHORT = "__all__"
"""Fallback cohort for users who did not state a skin type, or stated one whose
cohort turned out too thin to trust."""


class CohortCFRecommender(Recommender):
    """Item-item CF, with the similarity matrix rebuilt per skin-type cohort."""

    name = "cohort-cf"

    def __init__(
        self,
        *,
        top_n: int = 50,
        min_cohort_reviews: int = 500,
        shrinkage: float = 10.0,
    ) -> None:
        super().__init__()
        self.top_n = top_n
        self.min_cohort_reviews = min_cohort_reviews
        self.shrinkage = shrinkage
        # Neighbours are stored as (catalogue row indices, similarities) rather
        # than Series. Scoring then accumulates with numpy fancy indexing; the
        # Series version spent most of its time reindexing a 50-element vector
        # up to catalogue width, once per liked product, on every request.
        self._neighbours: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self._cohort_popularity: dict[str, np.ndarray] = {}
        self._cohort_sizes: dict[str, int] = {}
        self._position: dict[str, int] = {}

    # ------------------------------------------------------------------

    def fit(
        self,
        products: pd.DataFrame,
        reviews: pd.DataFrame,
        *,
        exclude_pairs: Iterable[tuple[str, str]] | None = None,
    ) -> "CohortCFRecommender":
        """Build per-cohort neighbour lists from positive interactions.

        `exclude_pairs` removes specific (user, product) interactions before
        anything is learned. The evaluation harness passes every held-out pair,
        because a similarity matrix built over all reviews has already seen the
        interaction it is later asked to predict — a leak that flatters CF
        specifically and would make the comparison table meaningless.
        """
        super().fit(products, reviews)
        catalogue_ids = products[ProductCols.ID].astype(str).tolist()
        self._position = {pid: row for row, pid in enumerate(catalogue_ids)}
        self._catalogue_index = pd.Index(catalogue_ids, name=ProductCols.ID)

        positives = reviews[reviews[ReviewCols.RATING] >= POSITIVE_RATING]
        positives = positives[
            [ReviewCols.USER, ReviewCols.ITEM, ReviewCols.SKIN_TYPE]
        ].dropna(subset=[ReviewCols.USER, ReviewCols.ITEM])

        if exclude_pairs:
            held = pd.MultiIndex.from_tuples(list(exclude_pairs))
            keys = pd.MultiIndex.from_arrays(
                [positives[ReviewCols.USER], positives[ReviewCols.ITEM]]
            )
            positives = positives[~keys.isin(held)]

        catalogue = set(products[ProductCols.ID].astype(str))
        positives = positives[positives[ReviewCols.ITEM].astype(str).isin(catalogue)]

        cohorts: dict[str, pd.DataFrame] = {GLOBAL_COHORT: positives}
        for skin_type in SKIN_TYPES:
            subset = positives[positives[ReviewCols.SKIN_TYPE] == skin_type]
            if len(subset) >= self.min_cohort_reviews:
                cohorts[skin_type] = subset

        for name, frame in cohorts.items():
            self._cohort_sizes[name] = len(frame)
            self._neighbours[name] = self._build_neighbours(frame)
            counts = np.log1p(frame.groupby(ReviewCols.ITEM).size())
            self._cohort_popularity[name] = (
                counts.reindex(self._catalogue_index, fill_value=0.0)
                .to_numpy(dtype=np.float32)
            )

        return self

    def _build_neighbours(
        self, frame: pd.DataFrame
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Top-N cosine neighbours per item, with co-occurrence shrinkage."""
        if frame.empty:
            return {}

        users = pd.Categorical(frame[ReviewCols.USER])
        items = pd.Categorical(frame[ReviewCols.ITEM])
        item_labels = list(items.categories)
        n_items = len(item_labels)
        if n_items < 2:
            return {}

        matrix = sparse.csr_matrix(
            (
                np.ones(len(frame), dtype=np.float32),
                (users.codes.astype(np.int64), items.codes.astype(np.int64)),
            ),
            shape=(len(users.categories), n_items),
        )
        matrix.data[:] = 1.0  # binary: one user liking an item once is enough

        # Co-occurrence counts. The diagonal is each item's own like count.
        co = (matrix.T @ matrix).toarray().astype(np.float32)
        own = np.diag(co).copy()
        np.fill_diagonal(co, 0.0)

        denominator = np.sqrt(np.outer(own, own))
        denominator[denominator == 0] = 1.0
        cosine = co / denominator

        # Shrinkage: two items sharing one reviewer out of a thousand are not
        # similar, however flattering the cosine looks. Damping by the raw
        # co-count is the standard fix and matters a lot at 0.09% density.
        #
        # Guarded because co == 0 with shrinkage == 0 is 0/0, and a NaN here
        # would propagate into every score built from this matrix without ever
        # raising. Zero co-occurrence means zero similarity, so that is what
        # the damping factor becomes.
        damping = np.zeros_like(co)
        np.divide(co, co + self.shrinkage, out=damping, where=co > 0)
        cosine *= damping

        keep = min(self.top_n, n_items - 1)
        neighbours: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for row in range(n_items):
            similarities = cosine[row]
            if not similarities.any():
                continue
            top = np.argpartition(-similarities, keep - 1)[:keep]
            top = top[similarities[top] > 0]
            if top.size == 0:
                continue
            # Translate to catalogue positions now, so scoring never has to.
            positions, values = [], []
            for i in top:
                position = self._position.get(item_labels[i])
                if position is not None:
                    positions.append(position)
                    values.append(similarities[i])
            if not positions:
                continue
            neighbours[item_labels[row]] = (
                np.asarray(positions, dtype=np.int32),
                np.asarray(values, dtype=np.float32),
            )
        return neighbours

    # ------------------------------------------------------------------

    def _cohort_for(self, query: Query) -> str:
        if query.skin_type and query.skin_type in self._neighbours:
            return query.skin_type
        return GLOBAL_COHORT

    def _score(self, query: Query) -> pd.Series:
        assert self._products is not None
        index = self._catalogue_index
        cohort = self._cohort_for(query)

        neighbours = self._neighbours.get(cohort, {})
        popularity = self._cohort_popularity.get(
            cohort, np.zeros(len(index), dtype=np.float32)
        )

        totals = np.zeros(len(index), dtype=np.float32)
        matched = 0
        for liked in query.liked_product_ids:
            entry = neighbours.get(str(liked))
            if entry is None:
                continue
            matched += 1
            positions, values = entry
            totals[positions] += values

        if matched:
            # Mean rather than sum, so a user with twenty likes is not scored on
            # a different scale from one with two.
            totals /= matched
            self._components = {
                "cf_neighbours": pd.Series(totals, index=index),
                "cf_cohort_popularity": pd.Series(popularity, index=index),
                "cf_liked_matched": pd.Series(float(matched), index=index),
            }
            return pd.Series(totals, index=index)

        # Cold start: no usable history, so all this layer knows is what the
        # cohort as a whole likes.
        maximum = float(popularity.max()) if popularity.size else 0.0
        scaled = popularity / maximum if maximum else popularity
        self._components = {
            "cf_neighbours": pd.Series(0.0, index=index),
            "cf_cohort_popularity": pd.Series(scaled, index=index),
            "cf_liked_matched": pd.Series(0.0, index=index),
        }
        return pd.Series(scaled, index=index)

    # ------------------------------------------------------------------

    def coverage(self) -> dict[str, int]:
        """Items each cohort can actually rank. Reported in the evaluation."""
        return {
            name: len(neighbours) for name, neighbours in self._neighbours.items()
        }

    def cohort_sizes(self) -> dict[str, int]:
        return dict(self._cohort_sizes)
