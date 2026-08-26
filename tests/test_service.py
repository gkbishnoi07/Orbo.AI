"""The composition root, wired against the real committed artifacts.

These are integration tests on purpose. The unit tests elsewhere use synthetic
fixtures with known structure, which is the right way to check that scoring maths
is correct — but it cannot catch a mismatch between two real files. The bug this
module exists to prevent was exactly that: the history picker was ranked by the
catalogue's `review_count` (Sephora's site-wide figure) while collaborative
filtering reads the interaction table, and those two disagree so badly that most
of the picker's options could not switch CF on at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.schema import ProductCols, Query, ReviewCols
from src.service import STRATEGIES, RecommendationService

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "products.parquet").exists(),
    reason="artifacts not built; run scripts/03_build_artifacts.py",
)


@pytest.fixture(scope="module")
def service() -> RecommendationService:
    return RecommendationService.from_artifacts(ARTIFACTS)


@pytest.fixture(scope="module")
def interaction_ids(service) -> set[str]:
    from src import artifacts

    frame = artifacts.load_interactions(ARTIFACTS)
    return set(frame[ReviewCols.ITEM].astype(str))


# --------------------------------------------------------------------------
# The history-picker bug
# --------------------------------------------------------------------------


def test_every_history_option_can_actually_switch_cf_on(service, interaction_ids):
    """The regression. Each option must exist in the interaction data."""
    pool = service.history_pool(200)
    assert pool
    missing = [pid for pid, _ in pool if pid not in interaction_ids]
    assert not missing, f"{len(missing)} options have no interactions: {missing[:5]}"


def test_a_history_option_produces_a_real_cf_match(service):
    """Not just present in the data — actually reaches the neighbour lists."""
    product_id, _ = service.history_pool(50)[0]
    cf = service._models["cohort-cf"]
    cf._score(Query(liked_product_ids=(product_id,)))
    assert cf._components["cf_liked_matched"].iloc[0] == 1.0


def test_history_pool_is_ordered_by_interaction_count_not_review_count(service):
    """`review_count` is the trap; ordering must not follow it."""
    pool = service.history_pool(300)
    counts = service._interaction_counts
    ordered = [counts[pid] for pid, _ in pool]
    assert ordered == sorted(ordered, reverse=True)

    by_review_count = list(
        service.products.nlargest(300, ProductCols.N_REVIEWS)[ProductCols.ID]
    )
    assert [pid for pid, _ in pool] != by_review_count, (
        "history pool matches the review_count ordering, which is the bug"
    )


def test_history_pool_respects_its_limit(service):
    assert len(service.history_pool(25)) == 25


def test_history_labels_are_brand_and_name(service):
    for product_id, label in service.history_pool(10):
        product = service.product(product_id)
        assert str(product[ProductCols.BRAND]) in label
        assert str(product[ProductCols.NAME]) in label


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def test_every_advertised_strategy_is_actually_wired(service):
    for info in STRATEGIES:
        results = service.recommend(Query(skin_type="dry"), strategy=info.key, k=5)
        assert len(results) == 5, f"{info.key} returned {len(results)}"


def test_an_unknown_strategy_is_rejected(service):
    with pytest.raises(KeyError, match="unknown strategy"):
        service.recommend(Query(), strategy="telepathy")


def test_strategies_disagree(service):
    """If two strategies return the same list, one of them is not wired up."""
    query = Query(skin_type="oily", concerns=("acne",))
    hybrid = [r.product_id for r in service.recommend(query, "hybrid", 10)]
    popularity = [r.product_id for r in service.recommend(query, "popularity", 10)]
    random_list = [r.product_id for r in service.recommend(query, "random", 10)]
    assert hybrid != popularity
    assert hybrid != random_list
    assert popularity != random_list


def test_history_changes_the_hybrid_result(service):
    query_cold = Query(skin_type="dry")
    owned = tuple(pid for pid, _ in service.history_pool(3))
    query_warm = Query(skin_type="dry", liked_product_ids=owned)

    cold = {r.product_id for r in service.recommend(query_cold, "hybrid", 10)}
    warm = {r.product_id for r in service.recommend(query_warm, "hybrid", 10)}
    assert cold != warm, "declaring owned products changed nothing"
    assert not (owned[0] in warm), "a product the user owns was recommended back"


# --------------------------------------------------------------------------
# Explanations and filtering
# --------------------------------------------------------------------------


def test_every_result_carries_evidence(service):
    results = service.recommend(
        Query(skin_type="dry", concerns=("dryness",), budget_max=80.0), "hybrid", 10
    )
    for scored in results:
        assert scored.evidence, f"{scored.product_id} came back with no explanation"
        assert any(e.supported for e in scored.evidence)


def test_cohort_claims_always_carry_a_sample_size(service):
    """A percentage without its n is the thing this project refuses to print."""
    results = service.recommend(Query(skin_type="combination"), "hybrid", 20)
    claims = [e for r in results for e in r.evidence if "%" in e.label]
    assert claims, "expected at least one cohort claim on a large cohort"
    for claim in claims:
        assert "reviews" in claim.detail

def test_filters_narrow_the_pool_and_the_reasons_are_reported(service):
    wide = Query()
    narrow = Query(skin_type="dry", concerns=("sensitivity",), budget_max=40.0)
    assert service.eligible_count(narrow) < service.eligible_count(wide)

    assert service.excluded_counts(wide) == {}
    breakdown = service.excluded_counts(narrow)
    assert breakdown and all(v > 0 for v in breakdown.values())


def test_impossible_constraints_return_nothing_rather_than_nonsense(service):
    results = service.recommend(
        Query(category="Beauty Supplements", budget_max=5.0), "hybrid", 10
    )
    assert results == []


def test_budget_and_category_are_honoured_by_every_strategy(service):
    for info in STRATEGIES:
        results = service.recommend(
            Query(category="Moisturizers", budget_max=30.0), info.key, 10
        )
        for scored in results:
            product = service.product(scored.product_id)
            assert product[ProductCols.CATEGORY] == "Moisturizers"
            assert float(product[ProductCols.PRICE]) <= 30.0


# --------------------------------------------------------------------------
# Facts shown in the UI header
# --------------------------------------------------------------------------


def test_catalogue_facts_match_the_data(service):
    facts = service.facts
    assert facts.n_products == len(service.products)
    assert facts.n_brands > 100
    assert 0 < facts.reviewed_share < 1, "the coverage gap is the whole design premise"
    assert facts.price_min > 0 < facts.price_median < facts.price_max


def test_cohort_size_is_reported_per_skin_type(service):
    dry = service.cohort_size("dry")
    everyone = service.cohort_size(None)
    assert 0 < dry < everyone
    assert service.cohort_size("reptilian") == everyone  # unknown -> global
