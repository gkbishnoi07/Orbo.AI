"""Content-based scoring over precomputed product vectors.

Handles the cold-start half of the problem: 72% of this catalogue has never been
reviewed, so collaborative filtering cannot rank it at all. Content can, because
it only needs the product to have text.

The design is shaped by one hard constraint. The deployed app has no text
encoder — sentence-transformers is a development dependency only — so there is
no way to embed a user's profile at request time. Instead the profile is turned
into a **prototype vector assembled from the catalogue itself**: the centroid of
products already tagged for that skin type and those concerns, blended with the
centroid of anything the user has liked. That needs nothing but the precomputed
matrix and a boolean tag lookup, both of which are cheap.

Scores blend three signals, kept separate so the UI can show which one carried a
result:

* `embedding` — cosine to the prototype. Generalises past the tag vocabulary.
* `concern`   — fraction of the user's stated concerns the product is tagged for.
  Sparse but exact, and the only signal an explanation can quote verbatim.
* `skin`      — stated suitability. Untagged scores a neutral 0.5 rather than 0,
  since 88% of the catalogue is untagged and zeroing it would hand the entire
  ranking to a 12% minority.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .recommender import Recommender
from .schema import ProductCols, Query

NEUTRAL_SKIN_SCORE = 0.5
"""What an untagged product scores on skin-type fit. Deliberately not 0."""


def _unit_scale(values: np.ndarray) -> np.ndarray:
    """Squash to [0, 1] so components with different natural ranges can be added.

    Cosine similarity over SVD-reduced TF-IDF is signed and rarely uses the full
    [-1, 1] range, so a fixed rescale would leave the embedding term compressed
    against whatever the tag terms do. Scaling to the observed spread keeps the
    configured weights meaning what they say.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


class ContentRecommender(Recommender):
    """Ranks by similarity to a profile prototype plus explicit tag matches."""

    name = "content"

    def __init__(
        self,
        embeddings: np.ndarray,
        ids: Sequence[str],
        *,
        weight_embedding: float = 0.50,
        weight_concern: float = 0.35,
        weight_skin: float = 0.15,
        history_weight: float = 0.6,
    ) -> None:
        super().__init__()
        if embeddings.shape[0] != len(ids):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but {len(ids)} ids"
            )
        # Normalising once here turns every later cosine into a single matmul.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.ids = [str(i) for i in ids]
        self._row_of = {pid: row for row, pid in enumerate(self.ids)}

        self.weight_embedding = weight_embedding
        self.weight_concern = weight_concern
        self.weight_skin = weight_skin
        self.history_weight = history_weight

    def fit(self, products: pd.DataFrame, reviews: pd.DataFrame) -> "ContentRecommender":
        super().fit(products, reviews)
        missing = set(products[ProductCols.ID]) - set(self._row_of)
        if missing:
            raise ValueError(
                f"{len(missing)} products have no embedding row; rebuild artifacts"
            )
        # Align the matrix to catalogue order once, so scoring never has to.
        order = [self._row_of[pid] for pid in products[ProductCols.ID]]
        self.embeddings = self.embeddings[order]
        self.ids = list(products[ProductCols.ID].astype(str))
        self._row_of = {pid: row for row, pid in enumerate(self.ids)}
        return self

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def _centroid(self, mask: np.ndarray) -> np.ndarray | None:
        if not mask.any():
            return None
        return self.embeddings[mask].mean(axis=0)

    def _tag_mask(self, query: Query) -> np.ndarray:
        """Products tagged for this profile's concerns or skin type."""
        assert self._concern_flags is not None and self._skin_type_flags is not None
        mask = np.zeros(len(self.ids), dtype=bool)
        for concern in query.concerns:
            if concern in self._concern_flags.columns:
                mask |= self._concern_flags[concern].to_numpy()
        if not mask.any() and query.skin_type in self._skin_type_flags.columns:
            mask |= self._skin_type_flags[query.skin_type].to_numpy()
        return mask

    def _prototype(self, query: Query) -> np.ndarray | None:
        """Blend a history centroid with a profile centroid.

        History dominates when it exists — what someone actually bought beats
        what a tag says they should want — but the profile still contributes, so
        a user with three moisturisers in their history can still be steered by
        newly declared concerns.
        """
        rows = [self._row_of[p] for p in query.liked_product_ids if p in self._row_of]
        history = self.embeddings[rows].mean(axis=0) if rows else None
        profile = self._centroid(self._tag_mask(query))

        if history is not None and profile is not None:
            blended = (
                self.history_weight * history + (1.0 - self.history_weight) * profile
            )
        else:
            blended = history if history is not None else profile

        if blended is None:
            return None
        norm = np.linalg.norm(blended)
        return blended / norm if norm else None

    # ------------------------------------------------------------------

    def _score(self, query: Query) -> pd.Series:
        assert self._products is not None
        assert self._concern_flags is not None and self._skin_type_flags is not None
        index = pd.Index(self.ids, name=ProductCols.ID)

        prototype = self._prototype(query)
        if prototype is None:
            # No history and no recognised profile: content has nothing to say.
            # Returning a flat score is honest — the hybrid's popularity prior
            # is what should carry this case, not a fabricated ranking.
            embedding = np.zeros(len(self.ids), dtype=np.float32)
        else:
            embedding = _unit_scale(self.embeddings @ prototype)

        if query.concerns:
            matched = np.zeros(len(self.ids), dtype=np.float32)
            known = 0
            for concern in query.concerns:
                if concern in self._concern_flags.columns:
                    matched += self._concern_flags[concern].to_numpy(dtype=np.float32)
                    known += 1
            concern_score = matched / known if known else matched
        else:
            concern_score = np.zeros(len(self.ids), dtype=np.float32)

        if query.skin_type and query.skin_type in self._skin_type_flags.columns:
            tagged_for = self._skin_type_flags[query.skin_type].to_numpy()
            untagged = self._skin_type_flags["untagged"].to_numpy()
            skin_score = np.where(
                tagged_for, 1.0, np.where(untagged, NEUTRAL_SKIN_SCORE, 0.0)
            ).astype(np.float32)
        else:
            skin_score = np.full(len(self.ids), NEUTRAL_SKIN_SCORE, dtype=np.float32)

        total = (
            self.weight_embedding * embedding
            + self.weight_concern * concern_score
            + self.weight_skin * skin_score
        )

        self._components = {
            "embedding": pd.Series(embedding, index=index),
            "concern": pd.Series(concern_score, index=index),
            "skin": pd.Series(skin_score, index=index),
        }
        return pd.Series(total, index=index)
