"""Suitability rules.

The rule layer is the one place where a plausible-looking bug is catastrophic
rather than merely wrong: only 12% of the real catalogue carries a skin-type
tag, so a rule that treats "untagged" as "unsuitable" silently deletes seven
eighths of the products while still returning a full, confident-looking list of
ten. Several tests below exist purely to pin that down.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import rules
from src.recommender import PopularityRecommender, RandomRecommender
from src.schema import ProductCols, Query

from .conftest import BENIGN_INGREDIENTS, IRRITANT_INGREDIENTS


@pytest.fixture(scope="module")
def irritant_flags(synthetic_products):
    return rules.build_irritant_flags(synthetic_products)


@pytest.fixture(scope="module")
def skin_type_flags(synthetic_products):
    return rules.build_skin_type_flags(synthetic_products)


# --------------------------------------------------------------------------
# Irritant matching
# --------------------------------------------------------------------------


def test_parfum_and_denatured_alcohol_are_flagged():
    frame = pd.DataFrame(
        {
            ProductCols.ID: ["X1"],
            ProductCols.INGREDIENTS: [IRRITANT_INGREDIENTS],
            ProductCols.HIGHLIGHTS: [[]],
        }
    )
    flags = rules.build_irritant_flags(frame)
    assert flags.loc["X1", "fragrance"]
    assert flags.loc["X1", "drying_alcohol"]
    assert flags.loc["X1", "essential_oils"]  # limonene, linalool


def test_cetyl_alcohol_is_not_a_drying_alcohol():
    """The whole reason the alcohol patterns are specific rather than "alcohol"."""
    frame = pd.DataFrame(
        {
            ProductCols.ID: ["X2"],
            ProductCols.INGREDIENTS: [BENIGN_INGREDIENTS],
            ProductCols.HIGHLIGHTS: [[]],
        }
    )
    flags = rules.build_irritant_flags(frame)
    assert not flags.loc["X2", "drying_alcohol"]
    assert not flags.loc["X2", "fragrance"]
    assert not flags.loc["X2", "essential_oils"]


def test_a_fragrance_free_label_waives_the_fragrance_match():
    frame = pd.DataFrame(
        {
            ProductCols.ID: ["X3", "X4"],
            ProductCols.INGREDIENTS: [IRRITANT_INGREDIENTS] * 2,
            ProductCols.HIGHLIGHTS: [["Fragrance Free"], []],
        }
    )
    flags = rules.build_irritant_flags(frame)
    assert not flags.loc["X3", "fragrance"]
    assert flags.loc["X4", "fragrance"]
    # The waiver is scoped to fragrance and must not spill onto other rules.
    assert flags.loc["X3", "drying_alcohol"]


def test_an_empty_ingredient_list_is_unverifiable_not_irritant():
    frame = pd.DataFrame(
        {
            ProductCols.ID: ["X5"],
            ProductCols.INGREDIENTS: [""],
            ProductCols.HIGHLIGHTS: [[]],
        }
    )
    flags = rules.build_irritant_flags(frame)
    assert flags.loc["X5", rules.UNVERIFIABLE]
    assert not flags.loc["X5", [r.slug for r in rules.IRRITANT_RULES]].any()


def test_the_fixture_catalogue_has_all_three_kinds_of_product(irritant_flags):
    """Guards the fixture itself: if it drifts to one kind, the tests go blind."""
    assert irritant_flags["fragrance"].any()
    assert (~irritant_flags["fragrance"]).any()
    assert irritant_flags[rules.UNVERIFIABLE].any()


# --------------------------------------------------------------------------
# Skin-type matching — absence of a tag is not a mismatch
# --------------------------------------------------------------------------


def test_untagged_products_are_marked_untagged(skin_type_flags, synthetic_products):
    untagged = skin_type_flags["untagged"]
    assert untagged.any() and not untagged.all()
    expected = synthetic_products[ProductCols.SKIN_TYPES].map(len).eq(0).sum()
    assert int(untagged.sum()) == int(expected)


def test_an_untagged_product_is_never_excluded(irritant_flags, skin_type_flags):
    drop = rules.excluded(
        irritant_flags, skin_type_flags, skin_type="dry", concerns=()
    )
    assert not drop[skin_type_flags["untagged"]].any()


def test_a_stated_mismatch_is_excluded(irritant_flags, skin_type_flags):
    drop = rules.excluded(
        irritant_flags, skin_type_flags, skin_type="dry", concerns=()
    )
    oily_only = skin_type_flags["oily"] & ~skin_type_flags["dry"]
    assert oily_only.any(), "fixture needs at least one oily-only product"
    assert drop[oily_only].all()


def test_a_matching_tag_is_kept(irritant_flags, skin_type_flags):
    drop = rules.excluded(
        irritant_flags, skin_type_flags, skin_type="dry", concerns=()
    )
    assert not drop[skin_type_flags["dry"]].any()


def test_no_skin_type_means_no_skin_type_exclusions(irritant_flags, skin_type_flags):
    drop = rules.excluded(
        irritant_flags, skin_type_flags, skin_type=None, concerns=()
    )
    assert not drop.any()


# --------------------------------------------------------------------------
# Sensitivity is opt-in
# --------------------------------------------------------------------------


def test_irritants_are_only_excluded_when_sensitivity_is_declared(
    irritant_flags, skin_type_flags
):
    without = rules.excluded(
        irritant_flags, skin_type_flags, skin_type=None, concerns=()
    )
    with_sensitivity = rules.excluded(
        irritant_flags, skin_type_flags, skin_type=None, concerns=("sensitivity",)
    )
    assert not without.any()
    assert with_sensitivity.any()


def test_other_concerns_do_not_trigger_irritant_exclusion(
    irritant_flags, skin_type_flags
):
    drop = rules.excluded(
        irritant_flags, skin_type_flags, skin_type=None, concerns=("dryness", "pores")
    )
    assert not drop.any()


def test_breakdown_reports_a_count_per_active_rule(irritant_flags, skin_type_flags):
    breakdown = rules.exclusion_breakdown(
        irritant_flags, skin_type_flags, skin_type="dry", concerns=("sensitivity",)
    )
    assert breakdown, "some rule must have fired on the fixture"
    assert all(isinstance(v, int) and v > 0 for v in breakdown.values())
    assert "added fragrance" in breakdown

    quiet = rules.exclusion_breakdown(
        irritant_flags, skin_type_flags, skin_type=None, concerns=()
    )
    assert quiet == {}


def test_concern_flags_cover_the_tagged_vocabulary(synthetic_products):
    flags = rules.build_concern_flags(synthetic_products)
    assert "dryness" in flags.columns
    assert "sensitivity" in flags.columns  # from the Fragrance Free highlight
    assert flags.to_numpy().any()


# --------------------------------------------------------------------------
# Rules are enforced through the base class, so they bind every strategy
# --------------------------------------------------------------------------


@pytest.fixture(params=["random", "popularity"])
def fitted_model(request, synthetic_products, synthetic_reviews):
    cls = {"random": RandomRecommender, "popularity": PopularityRecommender}[
        request.param
    ]
    return cls().fit(synthetic_products, synthetic_reviews)


def test_sensitive_profiles_never_receive_a_flagged_product(
    fitted_model, synthetic_products
):
    ingredients = synthetic_products.set_index(ProductCols.ID)[ProductCols.INGREDIENTS]
    highlights = synthetic_products.set_index(ProductCols.ID)[ProductCols.HIGHLIGHTS]
    results = fitted_model.recommend(Query(concerns=("sensitivity",)), k=10)
    assert results
    for scored in results:
        text = ingredients[scored.product_id].casefold()
        waived = "Fragrance Free" in highlights[scored.product_id]
        assert "alcohol denat" not in text
        assert "limonene" not in text
        if not waived:
            assert "parfum" not in text


def test_a_dry_profile_never_receives_an_oily_only_product(
    fitted_model, synthetic_products
):
    tags = synthetic_products.set_index(ProductCols.ID)[ProductCols.SKIN_TYPES]
    results = fitted_model.recommend(Query(skin_type="dry"), k=10)
    assert results
    for scored in results:
        product_types = tags[scored.product_id]
        assert not product_types or "dry" in product_types


def test_rules_shrink_the_pool_without_emptying_it(fitted_model):
    """The failure mode to catch: a rule so aggressive nothing survives it."""
    unfiltered = fitted_model.recommend(Query(), k=10)
    filtered = fitted_model.recommend(
        Query(skin_type="dry", concerns=("sensitivity",)), k=10
    )
    assert len(unfiltered) == 10
    assert len(filtered) == 10, "rules should still leave a full page of results"


def test_excluded_counts_is_reported_for_the_ui(fitted_model):
    counts = fitted_model.excluded_counts(
        Query(skin_type="dry", concerns=("sensitivity",))
    )
    assert counts and all(v > 0 for v in counts.values())
    assert fitted_model.excluded_counts(Query()) == {}
