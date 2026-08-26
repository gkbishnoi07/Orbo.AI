"""Raw Kaggle CSVs -> canonical frames.

This is the *only* module permitted to know what the source columns are called.
Everything downstream speaks `src.schema`. If the dataset is swapped, this file
is the one that changes.

Source: nadyinky/sephora-products-and-skincare-reviews
  product_info.csv   8,494 products, 27 columns
  reviews_*.csv      1,094,411 reviews across 5 files, 19 columns

Three properties of the raw data drive most of the code below:

1. `highlights` and `ingredients` are *stringified Python lists*, not delimited
   text, so they need literal_eval rather than split.
2. There is **no product description field**. The text used for embeddings has
   to be assembled from name, brand, categories, highlights and ingredients.
3. Explicit skin-type tags ("Best for Dry, Combo, Normal Skin") exist on only
   1,118 of 8,494 products. An untagged product is therefore *unknown*, never
   *unsuitable* — a distinction the scoring layer has to preserve or it will
   silently bury 87% of the catalogue.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from .schema import ProductCols, ReviewCols

# --------------------------------------------------------------------------
# Raw column names. Nothing outside this module should reference these.
# --------------------------------------------------------------------------

RAW_PRODUCT_MAP = {
    "product_id": ProductCols.ID,
    "product_name": ProductCols.NAME,
    "brand_name": ProductCols.BRAND,
    "secondary_category": ProductCols.CATEGORY,
    "primary_category": ProductCols.PRIMARY_CATEGORY,
    "price_usd": ProductCols.PRICE,
    "rating": ProductCols.RATING,
    "reviews": ProductCols.N_REVIEWS,
    "loves_count": ProductCols.LOVES,
}

RAW_REVIEW_COLUMNS = [
    "author_id",
    "product_id",
    "rating",
    "skin_type",
    "skin_tone",
    "submission_time",
    "is_recommended",
]
"""Deliberately excludes review_text. It is ~90% of the 520MB on disk and no
model reads it, so loading it would cost gigabytes of RAM for nothing."""

RAW_REVIEW_MAP = {
    "author_id": ReviewCols.USER,
    "product_id": ReviewCols.ITEM,
    "rating": ReviewCols.RATING,
    "skin_type": ReviewCols.SKIN_TYPE,
    "skin_tone": ReviewCols.SKIN_TONE,
    "submission_time": ReviewCols.TIME,
}

# --------------------------------------------------------------------------
# Vocabulary mappings, read off the real data rather than guessed.
# --------------------------------------------------------------------------

SKIN_TYPE_HIGHLIGHTS: dict[str, tuple[str, ...]] = {
    "Best for Dry Skin": ("dry",),
    "Best for Oily Skin": ("oily",),
    "Best for Normal Skin": ("normal",),
    "Best for Combination Skin": ("combination",),
    "Best for Dry, Combo, Normal Skin": ("dry", "combination", "normal"),
    "Best for Oily, Combo, Normal Skin": ("oily", "combination", "normal"),
}

CONCERN_HIGHLIGHTS: dict[str, str] = {
    "Good for: Dryness": "dryness",
    "Good for: Dullness/Uneven Texture": "dullness",
    "Good for: Anti-Aging": "anti_aging",
    "Good for: Loss of firmness": "firmness",
    "Good for: Pores": "pores",
    "Good for: Acne/Blemishes": "acne",
    "Good for: Dark spots": "dark_spots",
    "Good for: Dark Circles": "dark_circles",
    "Good for: Redness": "redness",
    "Good for: Frizz": "frizz",
    "Good for: Damage": "damage",
    "Good for: Volume": "volume",
    "Good for: Hair Thinning": "hair_thinning",
    "Good for: Color Care": "color_care",
    "Good for: Flaky/Dry Scalp": "flaky_scalp",
    "Good for: Oily Scalp": "oily_scalp",
}

SENSITIVITY_HIGHLIGHTS = frozenset(
    {"Fragrance Free", "Alcohol Free", "Hypoallergenic", "Good for: Redness"}
)
"""There is no 'sensitive' label anywhere in the dataset, so the sensitivity
concern is inferred from formulation flags. Any product carrying one of these
counts as addressing it; `src.rules` additionally *excludes* products whose
ingredient list contains a known irritant."""

SKIN_TONE_BANDS: dict[str, str] = {
    "porcelain": "fair",
    "fair": "fair",
    "fairLight": "fair",
    "light": "light",
    "lightMedium": "light",
    "medium": "medium",
    "olive": "medium",
    "mediumTan": "medium",
    "tan": "tan",
    "deep": "deep",
    "rich": "deep",
    "dark": "deep",
    "ebony": "deep",
    # "notSureST" (70 rows) is the reviewer declining to answer -> null.
}


def _parse_list(value: object) -> list[str]:
    """Parse one of the stringified-list columns.

    Falls back to a comma split, then to a single-element list, then to empty.
    Malformed rows are common enough in this dataset that raising would mean
    losing usable products over a stray bracket.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    return [part.strip() for part in text.split(",") if part.strip()]


def _clean_ingredient_text(items: list[str]) -> str:
    """Flatten a parsed ingredient list into one searchable string.

    Multi-variant products repeat the list once per shade with a heading like
    "Capri Eau de Parfum:", so headings are dropped and the remainder joined.
    """
    kept = [i for i in items if not i.endswith(":") and len(i) > 2]
    return ", ".join(kept)


def load_products(raw_dir: Path) -> pd.DataFrame:
    """Load and canonicalise product_info.csv."""
    path = Path(raw_dir) / "product_info.csv"
    raw = pd.read_csv(path, low_memory=False)

    missing = set(RAW_PRODUCT_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)}")

    products = raw[list(RAW_PRODUCT_MAP)].rename(columns=RAW_PRODUCT_MAP).copy()
    products[ProductCols.ID] = products[ProductCols.ID].astype(str)

    highlights = raw["highlights"].map(_parse_list)
    ingredient_lists = raw["ingredients"].map(_parse_list)

    products[ProductCols.HIGHLIGHTS] = highlights
    products[ProductCols.INGREDIENTS] = ingredient_lists.map(_clean_ingredient_text)

    products[ProductCols.SKIN_TYPES] = highlights.map(_skin_types_from_highlights)
    products[ProductCols.CONCERNS] = highlights.map(_concerns_from_highlights)

    products[ProductCols.PRICE] = pd.to_numeric(
        products[ProductCols.PRICE], errors="coerce"
    )
    products[ProductCols.CATEGORY] = products[ProductCols.CATEGORY].fillna("Other")

    products[ProductCols.TEXT] = _build_text_blob(products)

    # A product with no price cannot be budget-filtered honestly, and a product
    # with no name cannot be displayed. Both are unusable rather than degraded.
    before = len(products)
    products = products[
        products[ProductCols.PRICE].notna() & products[ProductCols.NAME].notna()
    ].reset_index(drop=True)
    if len(products) < before:
        print(f"  dropped {before - len(products)} products with no price or name")

    return products


def _skin_types_from_highlights(highlights: list[str]) -> list[str]:
    types: set[str] = set()
    for token in highlights:
        types.update(SKIN_TYPE_HIGHLIGHTS.get(token, ()))
    return sorted(types)


def _concerns_from_highlights(highlights: list[str]) -> list[str]:
    concerns = {CONCERN_HIGHLIGHTS[t] for t in highlights if t in CONCERN_HIGHLIGHTS}
    if any(t in SENSITIVITY_HIGHLIGHTS for t in highlights):
        concerns.add("sensitivity")
    return sorted(concerns)


def _build_text_blob(products: pd.DataFrame) -> pd.Series:
    """Assemble the string the embedding model sees.

    The dataset has no description field, so this is everything descriptive
    that does exist. Ingredients are truncated: past a few hundred characters
    an INCI list is mostly shared filler (water, glycerin, phenoxyethanol) that
    pushes every product's vector toward the same place.
    """
    parts = [
        products[ProductCols.BRAND].fillna(""),
        products[ProductCols.NAME].fillna(""),
        products[ProductCols.PRIMARY_CATEGORY].fillna(""),
        products[ProductCols.CATEGORY].fillna(""),
        products[ProductCols.HIGHLIGHTS].map(lambda hs: ". ".join(hs)),
        products[ProductCols.INGREDIENTS].str.slice(0, 400).fillna(""),
    ]
    blob = parts[0]
    for part in parts[1:]:
        blob = blob.str.cat(part, sep=". ", na_rep="")
    return blob.str.replace(r"\.\s*\.", ".", regex=True).str.strip(". ")


def load_reviews(raw_dir: Path) -> pd.DataFrame:
    """Load, concatenate and canonicalise the five review files."""
    paths = sorted(Path(raw_dir).glob("reviews_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no reviews_*.csv in {raw_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=RAW_REVIEW_COLUMNS, low_memory=False)
        frames.append(frame)
    reviews = pd.concat(frames, ignore_index=True)
    del frames

    reviews = reviews.rename(columns=RAW_REVIEW_MAP)
    reviews[ReviewCols.USER] = reviews[ReviewCols.USER].astype(str)
    reviews[ReviewCols.ITEM] = reviews[ReviewCols.ITEM].astype(str)
    reviews[ReviewCols.RATING] = pd.to_numeric(
        reviews[ReviewCols.RATING], errors="coerce"
    )
    reviews[ReviewCols.TIME] = pd.to_datetime(
        reviews[ReviewCols.TIME], errors="coerce", format="%Y-%m-%d"
    )

    reviews[ReviewCols.SKIN_TYPE] = _normalise_skin_type(reviews[ReviewCols.SKIN_TYPE])
    reviews[ReviewCols.SKIN_TONE] = _normalise_skin_tone(reviews[ReviewCols.SKIN_TONE])

    reviews = reviews[reviews[ReviewCols.RATING].notna()]

    # 5,525 (user, product) pairs appear twice — edited or re-submitted reviews.
    # Keeping both would let one user's opinion count twice in every cohort
    # statistic, so the most recent survives.
    before = len(reviews)
    reviews = (
        reviews.sort_values(ReviewCols.TIME)
        .drop_duplicates([ReviewCols.USER, ReviewCols.ITEM], keep="last")
        .reset_index(drop=True)
    )
    if len(reviews) < before:
        print(f"  dropped {before - len(reviews):,} duplicate (user, product) pairs")

    return reviews


def _normalise_skin_type(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().replace({"": pd.NA})


def _normalise_skin_tone(series: pd.Series) -> pd.Series:
    """Collapse the raw 14-value tone vocabulary into 5 ordered bands."""
    cleaned = series.astype("string").str.strip()
    return cleaned.map(SKIN_TONE_BANDS).astype("string")


# --------------------------------------------------------------------------
# Cached entry point
# --------------------------------------------------------------------------

PROCESSED_PRODUCTS = "products.parquet"
PROCESSED_REVIEWS = "reviews.parquet"

LIST_COLUMNS = (
    ProductCols.HIGHLIGHTS,
    ProductCols.SKIN_TYPES,
    ProductCols.CONCERNS,
)


def _coerce_list_columns(products: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the list-valued columns really are `list[str]`.

    Parquet stores them faithfully but hands them back as numpy arrays, so a
    cached load and a fresh parse would otherwise return different types for
    the same column — and `if row[SKIN_TYPES]` raises on an empty ndarray
    instead of being falsy. Normalising here keeps the contract in one place
    rather than making every caller defensive.
    """
    for column in LIST_COLUMNS:
        products[column] = products[column].map(
            lambda v: [str(x) for x in v] if v is not None else []
        )
    return products


def load(
    raw_dir: Path = Path("data/raw"),
    cache_dir: Path = Path("data/processed"),
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (products, reviews), reading a parquet cache when one exists.

    Parsing the CSVs takes about a minute and is pure overhead on every rerun,
    so the canonical frames are cached. Parquet keeps the list-valued columns
    as lists, which CSV would flatten back into strings.
    """
    cache_dir = Path(cache_dir)
    product_cache = cache_dir / PROCESSED_PRODUCTS
    review_cache = cache_dir / PROCESSED_REVIEWS

    if not refresh and product_cache.exists() and review_cache.exists():
        return (
            _coerce_list_columns(pd.read_parquet(product_cache)),
            pd.read_parquet(review_cache),
        )

    print("Parsing raw CSVs (cached afterwards)")
    products = load_products(raw_dir)
    reviews = load_reviews(raw_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    products.to_parquet(product_cache, index=False)
    reviews.to_parquet(review_cache, index=False)
    return products, reviews


def summarise(products: pd.DataFrame, reviews: pd.DataFrame) -> str:
    """One-screen description of what came out, for the audit trail."""
    tagged = products[ProductCols.SKIN_TYPES].map(len).gt(0).mean()
    with_concerns = products[ProductCols.CONCERNS].map(len).gt(0).mean()
    reviewed = reviews[ReviewCols.ITEM].nunique()
    return "\n".join(
        [
            f"products:            {len(products):,}",
            f"  skin-type tagged:  {tagged:.1%}",
            f"  concern tagged:    {with_concerns:.1%}",
            f"  categories:        {products[ProductCols.CATEGORY].nunique()}",
            f"reviews:             {len(reviews):,}",
            f"  users:             {reviews[ReviewCols.USER].nunique():,}",
            f"  products reviewed: {reviewed:,} "
            f"({reviewed / len(products):.1%} of catalogue)",
            f"  skin_type filled:  {reviews[ReviewCols.SKIN_TYPE].notna().mean():.1%}",
            f"  skin_tone filled:  {reviews[ReviewCols.SKIN_TONE].notna().mean():.1%}",
        ]
    )
