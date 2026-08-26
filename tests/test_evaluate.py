"""End-to-end checks on the leave-one-out harness and the baselines.

The synthetic catalogue has a Zipf popularity curve deliberately built into it,
so two things *must* be true and both are asserted here:

* popularity beats random by a wide margin on NDCG — the signal is learnable,
  and the accuracy metrics can see it;
* popularity's catalogue coverage collapses while it does so — the list-quality
  metrics can see the cost.

A harness that only showed the first half would let a recommender that serves
every user the same ten bestsellers look like a success.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import build_eval_cases, compare, evaluate
from src.recommender import PopularityRecommender, RandomRecommender
from src.schema import POSITIVE_RATING, ProductCols, Query, ReviewCols


@pytest.fixture(scope="module")
def cases(synthetic_reviews):
    return build_eval_cases(synthetic_reviews, min_positives=2, max_users=None)


def test_cases_are_built_and_hold_out_exactly_one_item(cases):
    assert len(cases) > 100, "synthetic data should yield plenty of eval cases"
    for case in cases:
        assert case.held_out not in case.query.liked_product_ids
        assert len(case.query.liked_product_ids) >= 1


def test_held_out_item_is_the_users_most_recent_like(synthetic_reviews, cases):
    """With timestamps present the harness must predict forward, not interpolate."""
    positives = synthetic_reviews[synthetic_reviews[ReviewCols.RATING] >= POSITIVE_RATING]
    latest = (
        positives.sort_values(ReviewCols.TIME)
        .groupby(ReviewCols.USER)[ReviewCols.ITEM]
        .last()
    )
    for case in cases[:50]:
        assert case.held_out == latest[case.user_id]


def test_users_below_min_positives_are_excluded(synthetic_reviews):
    strict = build_eval_cases(synthetic_reviews, min_positives=4, max_users=None)
    loose = build_eval_cases(synthetic_reviews, min_positives=2, max_users=None)
    assert len(strict) < len(loose)
    for case in strict:
        assert len(case.query.liked_product_ids) >= 3


def test_max_users_caps_the_case_count(synthetic_reviews):
    capped = build_eval_cases(synthetic_reviews, min_positives=2, max_users=25)
    assert len(capped) == 25


def test_profile_fields_are_carried_onto_the_query(cases):
    assert any(case.query.skin_type for case in cases)
    assert any(case.query.skin_tone for case in cases)


# --------------------------------------------------------------------------
# The headline claim: the metrics can see both sides of the tradeoff.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_table(synthetic_products, synthetic_reviews, cases):
    popularity = {
        str(item): int(n)
        for item, n in synthetic_reviews.groupby(ReviewCols.ITEM).size().items()
    }
    models = [
        RandomRecommender(seed=0).fit(synthetic_products, synthetic_reviews),
        PopularityRecommender().fit(synthetic_products, synthetic_reviews),
    ]
    return compare(
        models, cases, synthetic_products, k=10, popularity=popularity
    )


def test_popularity_decisively_beats_random_on_accuracy(baseline_table):
    random_ndcg = baseline_table.loc["random", "ndcg@10"]
    popular_ndcg = baseline_table.loc["popularity", "ndcg@10"]
    assert popular_ndcg > 4 * random_ndcg, baseline_table.to_string()


def test_popularity_pays_for_it_in_coverage(baseline_table):
    assert baseline_table.loc["popularity", "coverage"] < 0.20
    assert baseline_table.loc["random", "coverage"] > 0.70


def test_popularity_is_less_novel_than_random(baseline_table):
    assert (
        baseline_table.loc["popularity", "novelty"]
        < baseline_table.loc["random", "novelty"]
    )


def test_compare_returns_one_row_per_model_with_every_metric(baseline_table):
    assert list(baseline_table.index) == ["random", "popularity"]
    for column in ("precision@10", "recall@10", "ndcg@10", "map@10", "coverage"):
        assert column in baseline_table.columns
    assert baseline_table["latency_p95_ms"].ge(baseline_table["latency_p50_ms"]).all()
    assert (baseline_table["empty_result_rate"] == 0.0).all()


def test_evaluate_refuses_to_score_an_empty_case_list(synthetic_products):
    model = RandomRecommender()
    model.fit(synthetic_products, pd.DataFrame())
    with pytest.raises(ValueError, match="no evaluation cases"):
        evaluate(model, [], synthetic_products)


# --------------------------------------------------------------------------
# Hard filters live in the base class, so they must hold for *every* strategy.
# --------------------------------------------------------------------------


@pytest.fixture(params=["random", "popularity"])
def fitted_model(request, synthetic_products, synthetic_reviews):
    cls = {"random": RandomRecommender, "popularity": PopularityRecommender}[
        request.param
    ]
    return cls().fit(synthetic_products, synthetic_reviews)


def test_budget_is_never_exceeded(fitted_model, synthetic_products):
    prices = synthetic_products.set_index(ProductCols.ID)[ProductCols.PRICE]
    budget = 25.0
    results = fitted_model.recommend(Query(budget_max=budget), k=10)
    assert results, "a 25.0 budget should still match part of the catalogue"
    assert all(prices[r.product_id] <= budget for r in results)


def test_category_is_respected(fitted_model, synthetic_products):
    categories = synthetic_products.set_index(ProductCols.ID)[ProductCols.CATEGORY]
    results = fitted_model.recommend(Query(category="treatments"), k=10)
    assert results
    # Matching is case-insensitive: the UI sends whatever the user typed.
    assert all(categories[r.product_id] == "Treatments" for r in results)


def test_already_liked_products_are_not_recommended_back(
    fitted_model, synthetic_products
):
    everything = tuple(synthetic_products[ProductCols.ID].head(50))
    results = fitted_model.recommend(Query(liked_product_ids=everything), k=10)
    assert all(r.product_id not in everything for r in results)


def test_impossible_filters_return_an_empty_list_not_an_error(fitted_model):
    assert fitted_model.recommend(Query(budget_max=-1.0), k=10) == []
    assert fitted_model.recommend(Query(category="no-such-category"), k=10) == []


def test_recommend_before_fit_is_an_error(synthetic_products):
    with pytest.raises(RuntimeError, match="call fit"):
        RandomRecommender().recommend(Query(), k=10)


def test_results_are_returned_in_descending_score_order(fitted_model):
    results = fitted_model.recommend(Query(), k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
