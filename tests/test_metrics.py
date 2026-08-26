"""Metric functions checked against values computed by hand.

These are the ruler. If they are wrong, every number in the report is wrong and
no amount of model work will reveal it, so they are pinned to arithmetic done
on paper rather than to whatever the code currently returns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import metrics

RANKED = ["a", "b", "c"]
RELEVANT = {"a", "c"}


def test_precision_counts_hits_over_k():
    assert metrics.precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(2 / 3)


def test_precision_is_penalised_by_a_short_list():
    # Only one hit in the top 3 even though the list has two overall.
    assert metrics.precision_at_k(["a", "x", "y", "c"], RELEVANT, 3) == pytest.approx(
        1 / 3
    )


def test_recall_is_over_the_relevant_set_not_k():
    assert metrics.recall_at_k(RANKED, RELEVANT, 3) == pytest.approx(1.0)
    assert metrics.recall_at_k(["a"], RELEVANT, 3) == pytest.approx(0.5)


def test_ndcg_discounts_by_position():
    # dcg  = 1/log2(2) + 1/log2(4)  (ranks 0 and 2)
    # idcg = 1/log2(2) + 1/log2(3)  (ranks 0 and 1)
    expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
    assert metrics.ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx(expected)


def test_ndcg_is_one_when_all_relevant_items_lead():
    assert metrics.ndcg_at_k(["a", "c", "b"], RELEVANT, 3) == pytest.approx(1.0)


def test_average_precision_rewards_early_hits():
    # hit at rank 1 -> 1/1, hit at rank 3 -> 2/3, averaged over min(|rel|, k) = 2
    assert metrics.average_precision_at_k(RANKED, RELEVANT, 3) == pytest.approx(
        (1.0 + 2 / 3) / 2
    )
    front_loaded = metrics.average_precision_at_k(["a", "c", "b"], RELEVANT, 3)
    assert front_loaded > metrics.average_precision_at_k(RANKED, RELEVANT, 3)


def test_hit_rate_is_binary():
    assert metrics.hit_rate_at_k(RANKED, RELEVANT, 3) == 1.0
    assert metrics.hit_rate_at_k(["x", "y"], RELEVANT, 3) == 0.0


@pytest.mark.parametrize(
    "fn",
    [
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.ndcg_at_k,
        metrics.average_precision_at_k,
    ],
)
def test_no_relevant_items_or_empty_list_scores_zero(fn):
    assert fn(RANKED, set(), 3) == 0.0
    assert fn([], RELEVANT, 3) == 0.0


def test_precision_with_nonpositive_k_is_zero_not_a_crash():
    assert metrics.precision_at_k(RANKED, RELEVANT, 0) == 0.0


def test_catalog_coverage_counts_distinct_items_across_all_lists():
    assert metrics.catalog_coverage([["a", "b"], ["b", "c"]], 4) == pytest.approx(0.75)
    assert metrics.catalog_coverage([], 4) == 0.0
    assert metrics.catalog_coverage([["a"]], 0) == 0.0


def test_diversity_is_one_for_orthogonal_items_and_zero_for_duplicates():
    orthogonal = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    assert metrics.intra_list_diversity(["a", "b"], orthogonal) == pytest.approx(1.0)

    identical = {"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0])}
    assert metrics.intra_list_diversity(["a", "b"], identical) == pytest.approx(0.0)


def test_diversity_needs_two_known_vectors():
    embeddings = {"a": np.array([1.0, 0.0])}
    assert metrics.intra_list_diversity(["a"], embeddings) == 0.0
    # 'b' is unknown and must be skipped, not treated as a zero vector.
    assert metrics.intra_list_diversity(["a", "b"], embeddings) == 0.0


def test_novelty_rises_as_items_get_rarer():
    popular = metrics.novelty(["a"], {"a": 99}, n_users=100)
    obscure = metrics.novelty(["a"], {"a": 0}, n_users=100)
    assert obscure > popular
    # Laplace smoothing: an unseen item is 1/(n+1), never log2(0).
    assert obscure == pytest.approx(-math.log2(1 / 101))


def test_novelty_of_an_empty_list_is_zero():
    assert metrics.novelty([], {"a": 1}, n_users=10) == 0.0


def test_percentile_handles_the_empty_case():
    assert metrics.percentile([], 95) == 0.0
    assert metrics.percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
