"""Content-based scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.content import NEUTRAL_SKIN_SCORE, ContentRecommender
from src.schema import ProductCols, Query


@pytest.fixture
def model(synthetic_embeddings, synthetic_products, synthetic_reviews):
    matrix, ids = synthetic_embeddings
    return ContentRecommender(matrix, ids).fit(synthetic_products, synthetic_reviews)


def test_a_mismatched_matrix_is_rejected_at_construction(synthetic_embeddings):
    matrix, ids = synthetic_embeddings
    with pytest.raises(ValueError, match="rows but"):
        ContentRecommender(matrix, ids[:-5])


def test_missing_embedding_rows_are_rejected_at_fit(
    synthetic_embeddings, synthetic_products, synthetic_reviews
):
    """A stale matrix must fail loudly, not silently score part of the catalogue."""
    matrix, ids = synthetic_embeddings
    trimmed = ContentRecommender(matrix[:-10], ids[:-10])
    with pytest.raises(ValueError, match="no embedding row"):
        trimmed.fit(synthetic_products, synthetic_reviews)


def test_liking_a_product_pulls_in_its_category(model, synthetic_products):
    """The core content behaviour, made checkable by category-clustered vectors."""
    indexed = synthetic_products.set_index(ProductCols.ID)
    liked = "P0003"
    target = indexed.loc[liked, ProductCols.CATEGORY]

    results = model.recommend(Query(liked_product_ids=(liked,)), k=10)
    assert results
    share = sum(
        indexed.loc[r.product_id, ProductCols.CATEGORY] == target for r in results
    ) / len(results)
    assert share >= 0.7, f"only {share:.0%} of results shared the liked category"


def test_a_requested_concern_scores_higher_when_the_product_is_tagged(model):
    model._score(Query(concerns=("dryness",)))
    concern = model._components["concern"]
    tagged = model._concern_flags["dryness"].to_numpy()
    assert concern[tagged].mean() > concern[~tagged].mean()
    assert set(np.unique(concern.to_numpy())) <= {0.0, 1.0}


def test_untagged_products_get_a_neutral_skin_score_not_zero(model):
    """Zeroing untagged products would hand the ranking to a 12% minority."""
    model._score(Query(skin_type="dry"))
    skin = model._components["skin"]
    untagged = model._skin_type_flags["untagged"].to_numpy()
    matching = model._skin_type_flags["dry"].to_numpy()

    assert (skin[untagged] == NEUTRAL_SKIN_SCORE).all()
    assert (skin[matching] == 1.0).all()


def test_with_no_profile_and_no_history_the_embedding_term_is_flat(model):
    """Honest abstention: content has nothing to say, so it says nothing."""
    model._score(Query())
    assert (model._components["embedding"] == 0.0).all()


def test_components_reach_the_scored_results(model):
    results = model.recommend(Query(skin_type="dry", concerns=("dryness",)), k=5)
    assert results
    for scored in results:
        assert set(scored.components) == {"embedding", "concern", "skin"}
        assert all(0.0 <= v <= 1.0 for v in scored.components.values())


def test_scores_stay_inside_the_configured_weight_budget(model):
    scores = model._score(Query(skin_type="oily", concerns=("acne",)))
    ceiling = model.weight_embedding + model.weight_concern + model.weight_skin
    assert scores.max() <= ceiling + 1e-6
    assert scores.min() >= 0.0


def test_history_outweighs_the_profile_prototype(model, synthetic_products):
    """history_weight should mean what it says."""
    liked = "P0003"
    with_history = model._prototype(Query(liked_product_ids=(liked,), concerns=("acne",)))
    row = model._row_of[liked]
    profile_only = model._prototype(Query(concerns=("acne",)))

    to_liked = float(model.embeddings[row] @ with_history)
    profile_to_liked = float(model.embeddings[row] @ profile_only)
    assert to_liked > profile_to_liked
