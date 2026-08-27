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


# --------------------------------------------------------------------------
# Skin tone: a real ranking signal, not a decorative control
# --------------------------------------------------------------------------


def tone_stats_frame():
    """Two products a band disagrees about, one it has no opinion on.

    P0001: the deep band likes it far more than everyone else.
    P0002: the deep band likes it far less.
    P0003: no tone rows at all -> must contribute exactly nothing.
    """
    import pandas as pd

    rows = []
    for pid, deep_pos, other_pos in [("P0001", 195, 100), ("P0002", 5, 100)]:
        rows.append(
            {"product_id": pid, "skin_tone": "deep", "n_reviews": 200,
             "n_positive": deep_pos}
        )
        for band in ("fair", "light", "medium", "tan"):
            rows.append(
                {"product_id": pid, "skin_tone": band, "n_reviews": 200,
                 "n_positive": other_pos}
            )
    return pd.DataFrame(rows)


def tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews, **kwargs):
    from src.tone import ToneAffinity

    return build(
        synthetic_embeddings, synthetic_products, synthetic_reviews,
        tone=ToneAffinity(tone_stats_frame()), **kwargs,
    )


def test_tone_affinity_is_signed_and_centred_on_zero():
    """The property that makes it safe to add to an existing blend."""
    from src.tone import ToneAffinity
    import pandas as pd

    affinity = ToneAffinity(tone_stats_frame())
    index = pd.Index(["P0001", "P0002", "P0003"])
    deep = affinity.scores("deep", index)

    assert deep["P0001"] > 0, "a band that likes a product should score it up"
    assert deep["P0002"] < 0, "a band that dislikes it should score it down"
    assert deep["P0003"] == 0.0, "no tone data must mean no opinion, not a penalty"


def test_an_unknown_or_absent_band_contributes_nothing():
    from src.tone import ToneAffinity
    import pandas as pd

    affinity = ToneAffinity(tone_stats_frame())
    index = pd.Index(["P0001", "P0002"])
    assert (affinity.scores(None, index) == 0).all()
    assert (affinity.scores("chartreuse", index) == 0).all()


def test_shrinkage_damps_a_thin_cohort_far_more_than_a_thick_one():
    """The deep and tan bands are an order of magnitude smaller than light, so an
    unshrunk rate from a handful of reviewers would swamp a well-evidenced one."""
    from src.tone import ToneAffinity
    import pandas as pd

    def frame(n_deep):
        rows = [{"product_id": "P0001", "skin_tone": "deep",
                 "n_reviews": n_deep, "n_positive": n_deep}]
        rows += [{"product_id": "P0001", "skin_tone": b,
                  "n_reviews": 400, "n_positive": 200} for b in ("fair", "light")]
        return pd.DataFrame(rows)

    index = pd.Index(["P0001"])
    thin = ToneAffinity(frame(4)).scores("deep", index)["P0001"]
    thick = ToneAffinity(frame(400)).scores("deep", index)["P0001"]
    assert 0 < thin < thick, f"thin={thin} should be damped below thick={thick}"


def test_changing_skin_tone_changes_the_ranking(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    """The regression this whole change exists for.

    Skin tone was collected by the UI, banded in the data, displayed as a chip —
    and used by no scoring layer at all, so switching it produced a byte-identical
    list. Asserted over the full ranking rather than a top-k slice: the tone term
    is deliberately a small nudge, so on a catalogue where nothing else differs it
    moves products by tens of places, not into the first page.
    """
    model = tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews)

    def ranking(tone: str) -> list[str]:
        return list(
            model._score(Query(skin_type="dry", skin_tone=tone))
            .sort_values(ascending=False)
            .index
        )

    deep, fair = ranking("deep"), ranking("fair")
    assert deep != fair, "skin tone is still cosmetic"
    assert deep.index("P0001") < fair.index("P0001"), (
        "the band that rates P0001 highly should rank it higher: "
        f"deep={deep.index('P0001')} fair={fair.index('P0001')}"
    )
    assert deep.index("P0002") > fair.index("P0002"), (
        "the band that rates P0002 poorly should rank it lower"
    )


def test_tone_moves_a_product_by_a_meaningful_number_of_places(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    """Small does not mean invisible. If the nudge cannot shift a strongly
    disagreed-about product at all, the weight is effectively zero."""
    model = tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews)

    def rank_of(product: str, tone: str) -> int:
        order = (
            model._score(Query(skin_type="dry", skin_tone=tone))
            .sort_values(ascending=False)
        )
        return list(order.index).index(product)

    moved = rank_of("P0001", "fair") - rank_of("P0001", "deep")
    assert moved >= 10, f"tone shifted P0001 by only {moved} places"


def test_a_query_without_a_tone_is_unchanged_by_the_tone_layer(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    """Guarantees the addition did not perturb the existing blend."""
    with_tone = tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews)
    without = build(synthetic_embeddings, synthetic_products, synthetic_reviews)
    query = Query(skin_type="dry", concerns=("dryness",))
    assert [r.product_id for r in with_tone.recommend(query, k=20)] == [
        r.product_id for r in without.recommend(query, k=20)
    ]


def test_tone_never_removes_a_product_from_the_eligible_pool(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    """A nudge, not a filter. Excluding products a band has not reviewed would
    hit the least-represented bands hardest."""
    model = tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews)
    plain = len(model._apply_filters(Query(skin_type="dry")))
    toned = len(model._apply_filters(Query(skin_type="dry", skin_tone="deep")))
    assert plain == toned


def test_the_tone_term_is_exposed_as_a_score_component(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    model = tone_model(synthetic_embeddings, synthetic_products, synthetic_reviews)
    results = model.recommend(Query(skin_type="dry", skin_tone="deep"), k=5)
    assert results and all("tone" in r.components for r in results)
