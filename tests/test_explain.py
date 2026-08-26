"""The explanation layer.

Two properties are worth more than the rest and both are pinned here: a cohort
percentage never appears without the sample size behind it, and it never appears
at all when the sample is too small to mean anything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.explain import Explainer, build_cohort_stats
from src.schema import MIN_COHORT_FOR_CLAIM, Query, ReviewCols, Scored


def cohort_row(product_id: str, skin_type: str, n_reviews: int, n_positive: int):
    return {
        ReviewCols.ITEM: product_id,
        ReviewCols.SKIN_TYPE: skin_type,
        "n_reviews": n_reviews,
        "n_positive": n_positive,
    }


@pytest.fixture
def stats():
    return pd.DataFrame(
        [
            cohort_row("P0001", "dry", 400, 340),      # plenty, mostly positive
            cohort_row("P0002", "dry", 400, 80),        # plenty, mostly negative
            cohort_row("P0003", "dry", MIN_COHORT_FOR_CLAIM - 1, 20),  # too thin
        ]
    )


@pytest.fixture
def explainer(synthetic_products, stats):
    return Explainer(synthetic_products, stats)


def labels(evidence):
    return [e.label for e in evidence]


# --------------------------------------------------------------------------
# Cohort statistics
# --------------------------------------------------------------------------


def test_a_large_cohort_yields_a_quoted_percentage_with_its_sample_size(explainer):
    evidence = explainer.explain(Scored("P0001", 1.0), Query(skin_type="dry"))
    claim = next(e for e in evidence if "%" in e.label)
    assert "85%" in claim.label
    assert "dry-skin reviewers" in claim.label
    assert claim.supported
    assert "400 reviews" in claim.detail, "a percentage must carry its n"


def test_a_mostly_negative_cohort_is_reported_as_unsupported(explainer):
    evidence = explainer.explain(Scored("P0002", 1.0), Query(skin_type="dry"))
    claim = next(e for e in evidence if "%" in e.label)
    assert "20%" in claim.label
    assert not claim.supported


def test_a_thin_cohort_is_suppressed_rather_than_hedged(explainer):
    """A percentage from 29 reviewers reads as authoritative whatever caveat is
    attached, so the honest move is not to print it."""
    evidence = explainer.explain(Scored("P0003", 1.0), Query(skin_type="dry"))
    assert not any("%" in e.label for e in evidence)
    assert evidence, "the checklist must still be produced"


def test_no_skin_type_means_no_cohort_claim(explainer):
    evidence = explainer.explain(Scored("P0001", 1.0), Query())
    assert not any("%" in e.label for e in evidence)


def test_a_product_absent_from_the_stats_table_gets_no_claim(explainer):
    evidence = explainer.explain(Scored("P0050", 1.0), Query(skin_type="dry"))
    assert not any("reviewers" in e.label for e in evidence)


def test_explanations_work_with_no_cohort_table_at_all(synthetic_products):
    explainer = Explainer(synthetic_products, None)
    evidence = explainer.explain(Scored("P0001", 1.0), Query(skin_type="dry"))
    assert evidence
    assert not any("%" in e.label for e in evidence)


# --------------------------------------------------------------------------
# Checklist lines
# --------------------------------------------------------------------------


def test_an_unmatched_concern_is_shown_as_unmet_not_omitted(explainer):
    """A checklist of nothing but ticks is marketing."""
    evidence = explainer.explain(
        Scored("P0001", 1.0), Query(skin_type="dry", concerns=("frizz",))
    )
    line = next(e for e in evidence if "frizz" in e.label.lower())
    assert not line.supported
    assert "not listed" in line.detail


def test_a_matched_concern_is_shown_as_met(synthetic_products, stats):
    explainer = Explainer(synthetic_products, stats)
    tagged = synthetic_products[
        synthetic_products["addresses_concerns"].map(lambda c: "dryness" in c)
    ].iloc[0]
    evidence = explainer.explain(
        Scored(tagged["product_id"], 1.0), Query(concerns=("dryness",))
    )
    line = next(e for e in evidence if "dryness" in e.label.lower())
    assert line.supported


def test_untagged_skin_type_is_explained_as_unknown_not_as_a_mismatch(
    synthetic_products, stats
):
    untagged = synthetic_products[
        synthetic_products["suits_skin_types"].map(len) == 0
    ].iloc[0]
    explainer = Explainer(synthetic_products, stats)
    evidence = explainer.explain(
        Scored(untagged["product_id"], 1.0), Query(skin_type="dry")
    )
    line = next(e for e in evidence if "skin-type" in e.label)
    assert not line.supported
    assert "not a mismatch" in line.detail


def test_a_matching_skin_type_tag_is_reported_as_suitable(synthetic_products, stats):
    tagged = synthetic_products[
        synthetic_products["suits_skin_types"].map(lambda t: "dry" in t)
    ].iloc[0]
    explainer = Explainer(synthetic_products, stats)
    evidence = explainer.explain(
        Scored(tagged["product_id"], 1.0), Query(skin_type="dry")
    )
    line = next(e for e in evidence if "Suitable for" in e.label)
    assert line.supported


def test_sensitivity_reports_an_unverifiable_ingredient_list_honestly(
    synthetic_products, stats
):
    """~14% of the catalogue publishes no ingredients. Silence would imply a pass."""
    blank = synthetic_products[synthetic_products["ingredients"] == ""].iloc[0]
    explainer = Explainer(synthetic_products, stats)
    evidence = explainer.explain(
        Scored(blank["product_id"], 1.0), Query(concerns=("sensitivity",))
    )
    line = next(e for e in evidence if "irritants" in e.label)
    assert not line.supported
    assert "no ingredient list" in line.detail


def test_sensitivity_is_silent_when_it_was_not_asked_about(explainer):
    evidence = explainer.explain(Scored("P0001", 1.0), Query(skin_type="dry"))
    assert not any("irritant" in e.label.lower() for e in evidence)


def test_rating_and_budget_lines_carry_the_numbers(explainer, synthetic_products):
    product = synthetic_products.set_index("product_id").loc["P0001"]
    evidence = explainer.explain(
        Scored("P0001", 1.0), Query(skin_type="dry", budget_max=500.0)
    )
    rating_line = next(e for e in evidence if "Rated" in e.label)
    assert f"{float(product['rating']):.1f}" in rating_line.label

    budget_line = next(e for e in evidence if "budget" in e.label)
    assert budget_line.supported
    assert "$500" in budget_line.label


def test_no_budget_means_no_budget_line(explainer):
    evidence = explainer.explain(Scored("P0001", 1.0), Query(skin_type="dry"))
    assert not any("budget" in e.label for e in evidence)


def test_an_unknown_product_produces_nothing_rather_than_raising(explainer):
    assert explainer.explain(Scored("NOT-A-PRODUCT", 1.0), Query()) == []


# --------------------------------------------------------------------------
# Cohort stats builder
# --------------------------------------------------------------------------


def test_build_cohort_stats_counts_reviews_and_positives(synthetic_reviews):
    stats = build_cohort_stats(synthetic_reviews)
    assert {"n_reviews", "n_positive"} <= set(stats.columns)
    assert (stats["n_positive"] <= stats["n_reviews"]).all()
    assert stats["n_reviews"].sum() == synthetic_reviews["skin_type"].notna().sum()


def test_build_cohort_stats_ignores_reviews_with_no_skin_type():
    reviews = pd.DataFrame(
        {
            ReviewCols.USER: ["U1", "U2"],
            ReviewCols.ITEM: ["P1", "P1"],
            ReviewCols.RATING: [5.0, 5.0],
            ReviewCols.SKIN_TYPE: ["dry", None],
        }
    )
    stats = build_cohort_stats(reviews)
    assert len(stats) == 1
    assert stats.iloc[0]["n_reviews"] == 1
