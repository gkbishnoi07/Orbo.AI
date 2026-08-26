"""Offline evaluation metrics.

Two families live here:

* **Accuracy** (precision, recall, NDCG, MAP) — did we retrieve the items the
  user actually went on to like?
* **List quality** (coverage, diversity, novelty) — is the list any good as a
  *list*? A recommender can score well on accuracy while returning ten near
  duplicates drawn from 2% of the catalogue, which is a bad product.

All functions are pure and take plain sequences so they can be unit tested
without touching the dataset.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item made the top k. Equals recall under leave-one-out."""
    return float(any(item in relevant for item in ranked[:k]))


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain NDCG. Rewards putting relevant items near the top."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """AP for one user. Mean of this across users is MAP@k."""
    if not relevant:
        return 0.0
    hits = 0
    running = 0.0
    for rank, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            running += hits / (rank + 1)
    return running / min(len(relevant), k)


def catalog_coverage(all_ranked: Sequence[Sequence[str]], catalog_size: int) -> float:
    """Fraction of the catalogue that appears in at least one recommendation list.

    Low coverage means the system funnels every user toward the same bestsellers,
    which is a business problem even when accuracy looks fine.
    """
    if catalog_size <= 0:
        return 0.0
    seen = {item for ranked in all_ranked for item in ranked}
    return len(seen) / catalog_size


def intra_list_diversity(
    ranked: Sequence[str],
    embeddings: dict[str, np.ndarray],
) -> float:
    """Mean pairwise cosine *distance* within one list. Higher is more varied.

    Items missing from `embeddings` are skipped rather than treated as zero
    vectors, which would inflate the score.
    """
    vectors = [embeddings[i] for i in ranked if i in embeddings]
    if len(vectors) < 2:
        return 0.0
    matrix = np.vstack(vectors).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(len(vectors), k=1)]
    return float(1.0 - upper.mean())


def novelty(ranked: Sequence[str], popularity: dict[str, int], n_users: int) -> float:
    """Mean self-information of the list: -log2(P(item interacted with)).

    A list of blockbusters scores near zero; a list of long-tail items scores
    high. Reported alongside accuracy because the two trade off against each
    other and the tradeoff is the interesting part.
    """
    if n_users <= 0 or not ranked:
        return 0.0
    scores = []
    for item in ranked:
        count = popularity.get(item, 0)
        probability = (count + 1) / (n_users + 1)  # Laplace, avoids log2(0)
        scores.append(-math.log2(probability))
    return float(np.mean(scores))


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0
