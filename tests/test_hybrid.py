"""The hybrid blend and the MMR reranker."""

from __future__ import annotations

import numpy as np
import pytest

from src.collaborative import CohortCFRecommender
from src.hybrid import HybridRecommender, _scale
from src.metrics import intra_list_diversity
from src.recommender import PopularityRecommender
from src.content import ContentRecommender
from src.schema import ProductCols, Query


def build(synthetic_embeddings, synthetic_products, synthetic_reviews, **kwargs):
    matrix, ids = synthetic_embeddings
    model = HybridRecommender(
        content=ContentRecommender(matrix, ids),
        collaborative=CohortCFRecommender(min_cohort_reviews=50),
        popularity=PopularityRecommender(),
        **kwargs,
    )
    return model.fit(synthetic_products, synthetic_reviews)


@pytest.fixture
def model(synthetic_embeddings, synthetic_products, synthetic_reviews):
    return build(synthetic_embeddings, synthetic_products, synthetic_reviews)


def test_scale_maps_onto_the_unit_interval():
    import pandas as pd

    scaled = _scale(pd.Series([2.0, 4.0, 6.0]))
    assert scaled.tolist() == [0.0, 0.5, 1.0]
    flat = _scale(pd.Series([3.0, 3.0]))
    assert flat.tolist() == [0.0, 0.0]  # no spread means no signal


def test_every_layer_contributes_a_component(model):
    results = model.recommend(
        Query(skin_type="dry", concerns=("dryness",), liked_product_ids=("P0003",)), k=5
    )
    assert results
    for scored in results:
        assert {"content", "collaborative", "popularity"} <= set(scored.components)


def test_cold_start_uses_different_weights_from_warm_start(model):
    warm = model._weights(Query(liked_product_ids=("P0003",)))
    cold = model._weights(Query())
    assert warm != cold
    assert cold[2] > warm[2], "cold start should lean harder on popularity"


def test_weights_are_actually_applied(synthetic_embeddings, synthetic_products, synthetic_reviews):
    """Turning a layer's weight to zero must remove its influence entirely."""
    content_only = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews,
        weight_content=1.0, weight_cf=0.0, weight_popularity=0.0,
    )
    popularity_only = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews,
        weight_content=0.0, weight_cf=0.0, weight_popularity=1.0,
    )
    query = Query(liked_product_ids=("P0003",))
    a = [r.product_id for r in content_only.recommend(query, k=10)]
    b = [r.product_id for r in popularity_only.recommend(query, k=10)]
    assert a != b


def test_mmr_buys_diversity(synthetic_embeddings, synthetic_products, synthetic_reviews):
    """The trade MMR exists to make, measured rather than assumed."""
    matrix, ids = synthetic_embeddings
    lookup = {pid: matrix[i] for i, pid in enumerate(ids)}
    query = Query(skin_type="dry", liked_product_ids=("P0003",))

    relevance_only = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews, mmr_lambda=1.0
    )
    diversified = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews, mmr_lambda=0.3
    )

    plain = [r.product_id for r in relevance_only.recommend(query, k=10)]
    varied = [r.product_id for r in diversified.recommend(query, k=10)]
    assert intra_list_diversity(varied, lookup) > intra_list_diversity(plain, lookup)


def test_mmr_at_lambda_one_is_plain_relevance_ranking(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    model = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews, mmr_lambda=1.0
    )
    query = Query(skin_type="dry", liked_product_ids=("P0003",))
    scores = model._score(query).reindex(
        model._apply_filters(query)[ProductCols.ID]
    ).dropna().sort_values(ascending=False)
    scores = scores.drop(labels=["P0003"], errors="ignore")
    expected = list(scores.head(10).index)
    assert [r.product_id for r in model.recommend(query, k=10)] == expected


def test_mmr_keeps_the_top_result(synthetic_embeddings, synthetic_products, synthetic_reviews):
    """Diversity should reorder the tail, not throw away the best match."""
    model = build(
        synthetic_embeddings, synthetic_products, synthetic_reviews, mmr_lambda=0.5
    )
    query = Query(skin_type="dry", liked_product_ids=("P0003",))
    scores = model._score(query).reindex(
        model._apply_filters(query)[ProductCols.ID]
    ).dropna().sort_values(ascending=False).drop(labels=["P0003"], errors="ignore")
    assert model.recommend(query, k=10)[0].product_id == scores.index[0]


def test_mmr_returns_the_requested_number_without_duplicates(model):
    results = model.recommend(Query(skin_type="dry", liked_product_ids=("P0003",)), k=10)
    ids = [r.product_id for r in results]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_hybrid_still_honours_the_hard_filters(model, synthetic_products):
    prices = synthetic_products.set_index(ProductCols.ID)[ProductCols.PRICE]
    results = model.recommend(Query(budget_max=25.0, concerns=("sensitivity",)), k=10)
    assert results
    assert all(prices[r.product_id] <= 25.0 for r in results)


def test_hybrid_beats_random_ordering_on_a_liked_category(model, synthetic_products):
    indexed = synthetic_products.set_index(ProductCols.ID)
    liked = "P0003"
    target = indexed.loc[liked, ProductCols.CATEGORY]
    results = model.recommend(Query(liked_product_ids=(liked,)), k=10)
    share = sum(
        indexed.loc[r.product_id, ProductCols.CATEGORY] == target for r in results
    ) / len(results)
    baseline = (indexed[ProductCols.CATEGORY] == target).mean()
    assert share > baseline


# --------------------------------------------------------------------------
# Defaults that encode a measured conclusion, not a guess
# --------------------------------------------------------------------------


def test_warm_defaults_are_cf_dominant(model):
    """scripts/05_sweep.py: warm NDCG rises monotonically as content falls."""
    content, cf, popularity = model._weights(Query(liked_product_ids=("P0003",)))
    assert cf > content + popularity
    assert content > 0, "a small content term is kept deliberately; see hybrid docstring"


def test_cold_defaults_are_popularity_dominant(model):
    """Same sweep: cold NDCG rises monotonically as popularity rises."""
    content, cf, popularity = model._weights(Query())
    assert popularity > cf > content
    assert content > 0, "kept so a cold list still responds to declared concerns"


def test_weights_sum_to_one_in_both_regimes(model):
    for query in (Query(liked_product_ids=("P0003",)), Query()):
        assert sum(model._weights(query)) == pytest.approx(1.0)


def test_mmr_default_trades_almost_no_relevance(model):
    """0.85 was chosen because it cost 0.08% NDCG for +7.9% diversity."""
    assert 0.8 <= model.mmr_lambda < 1.0
