"""Synthetic fixtures.

The real dataset is large and not redistributable, so the harness is verified
against generated data with *known* structure. That is the point: with a
power-law popularity distribution baked in, we know in advance that a
popularity recommender must beat random by a wide margin and must do it by
funnelling everyone toward the same head of the catalogue. If the metrics do
not show that, the metrics are wrong — not the model.

The product fixture deliberately mirrors the real catalogue's awkward shape
rather than an idealised one: only ~12% of products carry a skin-type tag, ~14%
have no usable ingredient list, and some benign ingredients (cetyl alcohol) look
superficially like irritants. A fixture without those properties would let real
bugs through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    _concerns_from_highlights,
    _skin_types_from_highlights,
)
from src.schema import ProductCols, ReviewCols  # noqa: E402

N_PRODUCTS = 200
N_USERS = 400
CATEGORIES = ["Moisturizers", "Cleansers", "Treatments", "Sunscreen"]
SKIN_TYPES = ["oily", "dry", "combination", "normal"]

SKIN_TYPE_TAGS = [
    "Best for Dry, Combo, Normal Skin",
    "Best for Oily, Combo, Normal Skin",
    "Best for Dry Skin",
    "Best for Oily Skin",
]
CONCERN_TAGS = [
    "Good for: Dryness",
    "Good for: Acne/Blemishes",
    "Good for: Anti-Aging",
    "Good for: Pores",
    "Good for: Redness",
]

# Contains parfum and denatured alcohol: must trip the sensitivity rules.
IRRITANT_INGREDIENTS = (
    "Alcohol Denat. (SD Alcohol 39C), Parfum (Fragrance), Limonene, Linalool"
)
# Cetyl alcohol is a fatty alcohol and an emollient. A rule matching a bare
# "alcohol" would wrongly exclude this, so it belongs in the fixture.
BENIGN_INGREDIENTS = (
    "Water, Glycerin, Cetyl Alcohol, Sodium Hyaluronate, Niacinamide, Squalane"
)


@pytest.fixture(scope="session")
def synthetic_products() -> pd.DataFrame:
    rng = np.random.default_rng(7)

    highlights: list[list[str]] = []
    ingredients: list[str] = []
    for i in range(N_PRODUCTS):
        tags: list[str] = []
        if i % 8 == 0:  # ~12% skin-type tagged, as in the real catalogue
            tags.append(SKIN_TYPE_TAGS[(i // 8) % len(SKIN_TYPE_TAGS)])
        if i % 5 in (0, 1):  # ~40% concern tagged
            tags.append(CONCERN_TAGS[i % len(CONCERN_TAGS)])
        if i % 11 == 0:
            tags.append("Fragrance Free")  # waives the fragrance rule
        highlights.append(tags)

        if i % 7 == 0:
            ingredients.append("")  # ~14% unverifiable
        elif i % 3 == 0:
            ingredients.append(IRRITANT_INGREDIENTS)
        else:
            ingredients.append(BENIGN_INGREDIENTS)

    frame = pd.DataFrame(
        {
            ProductCols.ID: [f"P{i:04d}" for i in range(N_PRODUCTS)],
            ProductCols.NAME: [f"Product {i}" for i in range(N_PRODUCTS)],
            ProductCols.BRAND: [f"Brand{i % 20}" for i in range(N_PRODUCTS)],
            ProductCols.CATEGORY: rng.choice(CATEGORIES, N_PRODUCTS),
            ProductCols.PRIMARY_CATEGORY: "Skincare",
            # Log-normal: a realistic retail price curve, cheap items dominating.
            ProductCols.PRICE: np.round(rng.lognormal(3.2, 0.6, N_PRODUCTS), 2),
            ProductCols.RATING: np.round(rng.uniform(3.0, 5.0, N_PRODUCTS), 2),
            ProductCols.N_REVIEWS: rng.integers(0, 500, N_PRODUCTS),
            ProductCols.LOVES: rng.integers(0, 20000, N_PRODUCTS),
            ProductCols.HIGHLIGHTS: highlights,
            ProductCols.INGREDIENTS: ingredients,
        }
    )
    frame[ProductCols.SKIN_TYPES] = frame[ProductCols.HIGHLIGHTS].map(
        _skin_types_from_highlights
    )
    frame[ProductCols.CONCERNS] = frame[ProductCols.HIGHLIGHTS].map(
        _concerns_from_highlights
    )
    frame[ProductCols.TEXT] = (
        frame[ProductCols.BRAND]
        + ". "
        + frame[ProductCols.NAME]
        + ". "
        + frame[ProductCols.CATEGORY]
        + ". "
        + frame[ProductCols.HIGHLIGHTS].map(". ".join)
        + ". "
        + frame[ProductCols.INGREDIENTS]
    )
    return frame


@pytest.fixture(scope="session")
def synthetic_reviews(synthetic_products: pd.DataFrame) -> pd.DataFrame:
    """Reviews drawn with Zipf-weighted item choice, so popularity is learnable."""
    rng = np.random.default_rng(11)
    ids = synthetic_products[ProductCols.ID].to_numpy()

    weights = 1.0 / np.arange(1, N_PRODUCTS + 1)  # Zipf over a shuffled catalogue
    weights = weights / weights.sum()
    order = rng.permutation(N_PRODUCTS)
    weights = weights[order]

    rows = []
    clock = 0
    for user in range(N_USERS):
        n = int(rng.integers(2, 7))
        chosen = rng.choice(ids, size=n, replace=False, p=weights)
        skin = SKIN_TYPES[user % len(SKIN_TYPES)]
        for item in chosen:
            clock += 1
            rows.append(
                {
                    ReviewCols.USER: f"U{user:04d}",
                    ReviewCols.ITEM: item,
                    # Skewed positive, as review data always is.
                    ReviewCols.RATING: float(rng.choice([3, 4, 5], p=[0.2, 0.3, 0.5])),
                    ReviewCols.SKIN_TYPE: skin,
                    ReviewCols.SKIN_TONE: ["fair", "light", "medium", "tan", "deep"][
                        user % 5
                    ],
                    ReviewCols.TIME: clock,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_embeddings(synthetic_products: pd.DataFrame):
    """Vectors clustered by category, so similarity is predictable.

    Real embeddings are opaque, which makes them useless for asserting
    behaviour. Here products in the same category sit near a shared centre, so
    "a user who liked a cleanser should be shown cleansers" becomes a statement
    a test can actually check.
    """
    rng = np.random.default_rng(3)
    dims = 16
    centres = {category: rng.normal(size=dims) for category in CATEGORIES}
    rows = [
        centres[category] + 0.12 * rng.normal(size=dims)
        for category in synthetic_products[ProductCols.CATEGORY]
    ]
    matrix = np.vstack(rows).astype(np.float32)
    return matrix, list(synthetic_products[ProductCols.ID].astype(str))
