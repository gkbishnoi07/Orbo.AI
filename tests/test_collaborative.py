"""Cohort-scoped collaborative filtering.

The leak test near the bottom is the one that matters most. A similarity matrix
built over all reviews has already seen the interaction it is later asked to
predict, and the resulting NDCG looks excellent for entirely the wrong reason.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.collaborative import GLOBAL_COHORT, CohortCFRecommender
from src.schema import ProductCols, Query, ReviewCols


def make_reviews(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(user, item, skin_type) triples, all rated 5."""
    return pd.DataFrame(
        [
            {
                ReviewCols.USER: user,
                ReviewCols.ITEM: item,
                ReviewCols.RATING: 5.0,
                ReviewCols.SKIN_TYPE: skin,
                ReviewCols.SKIN_TONE: "light",
                ReviewCols.TIME: i,
            }
            for i, (user, item, skin) in enumerate(rows)
        ]
    )


@pytest.fixture
def model(synthetic_products, synthetic_reviews):
    return CohortCFRecommender(min_cohort_reviews=50).fit(
        synthetic_products, synthetic_reviews
    )


def test_cohorts_are_built_per_skin_type_plus_a_global_fallback(model):
    sizes = model.cohort_sizes()
    assert GLOBAL_COHORT in sizes
    assert {"dry", "oily", "combination", "normal"} <= set(sizes)
    assert sizes[GLOBAL_COHORT] > max(
        v for k, v in sizes.items() if k != GLOBAL_COHORT
    )


def test_a_cohort_too_small_to_trust_is_not_built(synthetic_products, synthetic_reviews):
    strict = CohortCFRecommender(min_cohort_reviews=10**9).fit(
        synthetic_products, synthetic_reviews
    )
    assert list(strict.cohort_sizes()) == [GLOBAL_COHORT]


def test_an_unknown_skin_type_falls_back_to_the_global_cohort(model):
    assert model._cohort_for(Query(skin_type="reptilian")) == GLOBAL_COHORT
    assert model._cohort_for(Query()) == GLOBAL_COHORT
    assert model._cohort_for(Query(skin_type="dry")) == "dry"


def test_co_liked_products_become_neighbours(synthetic_products):
    """Ten users who all liked P0001 and P0002 make them neighbours."""
    reviews = make_reviews(
        [(f"U{i}", item, "dry") for i in range(10) for item in ("P0001", "P0002")]
    )
    model = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products, reviews
    )
    scores = model._score(Query(skin_type="dry", liked_product_ids=("P0001",)))
    assert scores["P0002"] > 0
    assert scores.drop(["P0001", "P0002"]).max() == 0


def test_recommendations_come_from_the_neighbourhood(synthetic_products):
    reviews = make_reviews(
        [(f"U{i}", item, "dry") for i in range(20) for item in ("P0001", "P0002", "P0003")]
    )
    model = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products, reviews
    )
    results = model.recommend(Query(skin_type="dry", liked_product_ids=("P0001",)), k=5)
    top = [r.product_id for r in results[:2]]
    assert set(top) == {"P0002", "P0003"}


def test_cold_start_falls_back_to_cohort_popularity(synthetic_products):
    reviews = make_reviews(
        [(f"U{i}", "P0005", "oily") for i in range(30)]
        + [(f"V{i}", "P0006", "oily") for i in range(5)]
    )
    model = CohortCFRecommender(min_cohort_reviews=1).fit(synthetic_products, reviews)

    scores = model._score(Query(skin_type="oily"))
    assert scores["P0005"] > scores["P0006"] > 0
    assert scores.max() == pytest.approx(1.0)  # scaled
    assert model._components["cf_liked_matched"].iloc[0] == 0.0


def test_a_user_whose_likes_are_all_unknown_degrades_to_the_cohort_prior(
    synthetic_products,
):
    reviews = make_reviews([(f"U{i}", "P0005", "dry") for i in range(30)])
    model = CohortCFRecommender(min_cohort_reviews=1).fit(synthetic_products, reviews)
    # P0100 has no reviews at all, so it has no neighbours to contribute.
    scores = model._score(Query(skin_type="dry", liked_product_ids=("P0100",)))
    assert model._components["cf_liked_matched"].iloc[0] == 0.0
    assert scores["P0005"] > 0


def test_shrinkage_damps_similarity_built_on_one_shared_reviewer(synthetic_products):
    """0.09% density means thin co-occurrence is the common case, not the edge."""
    reviews = make_reviews([("U1", "P0001", "dry"), ("U1", "P0002", "dry")])
    loose = CohortCFRecommender(min_cohort_reviews=1, shrinkage=0.0).fit(
        synthetic_products, reviews
    )
    tight = CohortCFRecommender(min_cohort_reviews=1, shrinkage=50.0).fit(
        synthetic_products, reviews
    )
    loose_score = loose._score(Query(skin_type="dry", liked_product_ids=("P0001",)))
    tight_score = tight._score(Query(skin_type="dry", liked_product_ids=("P0001",)))
    assert loose_score["P0002"] > tight_score["P0002"]


def test_only_catalogue_products_are_scored(synthetic_products):
    reviews = make_reviews(
        [(f"U{i}", item, "dry") for i in range(10) for item in ("P0001", "GHOST")]
    )
    model = CohortCFRecommender(min_cohort_reviews=1).fit(synthetic_products, reviews)
    scores = model._score(Query(skin_type="dry", liked_product_ids=("P0001",)))
    assert "GHOST" not in scores.index
    assert len(scores) == len(synthetic_products)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_excluded_pairs_are_never_learned_from(synthetic_products):
    """The whole basis of an honest evaluation.

    Ten users liked both products, so without exclusion they are strong
    neighbours. Removing every interaction with P0002 must leave P0001 with
    nothing to recommend — if a score survives, the held-out interaction leaked
    into the similarity matrix.
    """
    pairs = [(f"U{i}", item, "dry") for i in range(10) for item in ("P0001", "P0002")]
    reviews = make_reviews(pairs)

    leaky = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products, reviews
    )
    assert leaky._score(Query(skin_type="dry", liked_product_ids=("P0001",)))["P0002"] > 0

    clean = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products,
        reviews,
        exclude_pairs=[(f"U{i}", "P0002") for i in range(10)],
    )
    scores = clean._score(Query(skin_type="dry", liked_product_ids=("P0001",)))
    assert scores["P0002"] == 0.0


def test_partial_exclusion_weakens_but_does_not_erase_similarity(synthetic_products):
    pairs = [(f"U{i}", item, "dry") for i in range(20) for item in ("P0001", "P0002")]
    reviews = make_reviews(pairs)
    full = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products, reviews
    )
    partial = CohortCFRecommender(min_cohort_reviews=1, shrinkage=1.0).fit(
        synthetic_products,
        reviews,
        exclude_pairs=[(f"U{i}", "P0002") for i in range(15)],
    )
    query = Query(skin_type="dry", liked_product_ids=("P0001",))
    assert 0 < partial._score(query)["P0002"] < full._score(query)["P0002"]


def test_coverage_reports_how_much_of_the_catalogue_cf_can_rank(model, synthetic_products):
    """CF is blind to unreviewed products, and the report should say so."""
    coverage = model.coverage()
    assert coverage[GLOBAL_COHORT] > 0
    assert coverage[GLOBAL_COHORT] <= len(synthetic_products)


def test_cf_scores_zero_for_products_nobody_has_reviewed(synthetic_products):
    """The 72% gap that content exists to fill.

    Worth stating explicitly because it is also why leave-one-out cannot reward
    the content layer: held-out items always come from the review table, so an
    unreviewed product can never be the right answer.
    """
    reviews = make_reviews(
        [(f"U{i}", item, "dry") for i in range(10) for item in ("P0001", "P0002")]
    )
    model = CohortCFRecommender(min_cohort_reviews=1).fit(synthetic_products, reviews)
    scores = model._score(Query(skin_type="dry", liked_product_ids=("P0001",)))

    reviewed = {"P0001", "P0002"}
    unreviewed = [p for p in synthetic_products[ProductCols.ID] if p not in reviewed]
    assert (scores[unreviewed] == 0).all()
    assert len(unreviewed) > len(reviewed)
