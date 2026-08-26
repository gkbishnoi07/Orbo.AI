"""Audit the raw dataset before committing to an architecture.

Answers the questions the design depends on:

1. How complete are the reviewer profile fields? Cohort collaborative filtering
   is only viable if skin type and tone are populated for most reviews.
2. How dense is the interaction matrix? Reviews per user and users per product
   determine whether item-item similarity has anything to work with.
3. How large is the average cohort? A "87% of oily-skin reviewers" style claim
   needs a real sample behind it.
4. Is the ingredient field parseable, or free text of varying format?

Run:  python scripts/01_inspect.py data/raw
Writes a markdown report to reports/data_audit.md and prints a verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Substrings we look for when guessing which raw column maps to which concept.
COLUMN_HINTS = {
    "user": ["author_id", "user_id", "reviewer"],
    "item": ["product_id", "item_id", "asin"],
    "rating": ["rating", "stars", "score"],
    "skin_type": ["skin_type", "skintype"],
    "skin_tone": ["skin_tone", "skintone"],
    "price": ["price"],
    "ingredients": ["ingredient"],
    "category": ["category", "secondary_category", "tertiary_category"],
}

MIN_COHORT_FOR_CLAIM = 30
"""Below this many reviews, a cohort percentage is noise and must not be shown."""


def guess_column(frame: pd.DataFrame, concept: str) -> str | None:
    """Find the raw column matching a concept, preferring exact hint matches."""
    lowered = {c.lower(): c for c in frame.columns}
    for hint in COLUMN_HINTS.get(concept, []):
        if hint in lowered:
            return lowered[hint]
    for hint in COLUMN_HINTS.get(concept, []):
        for low, original in lowered.items():
            if hint in low:
                return original
    return None


def load_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every CSV in the directory, tolerating messy quoting."""
    frames = {}
    for path in sorted(raw_dir.glob("*.csv")):
        try:
            frames[path.name] = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  ! could not read {path.name}: {exc}")
    return frames


def classify(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Split loaded files into the product table and the concatenated reviews."""
    products, reviews = None, []
    for name, frame in frames.items():
        has_user = guess_column(frame, "user") is not None
        has_rating = guess_column(frame, "rating") is not None
        if has_user and has_rating:
            reviews.append(frame)
        elif guess_column(frame, "price") is not None or "product" in name.lower():
            products = frame if products is None else products
    combined = pd.concat(reviews, ignore_index=True) if reviews else None
    return products, combined


def audit(raw_dir: Path) -> str:
    lines: list[str] = ["# Data audit", ""]
    frames = load_all(raw_dir)
    if not frames:
        return "No CSV files found in " + str(raw_dir)

    lines.append("## Files")
    for name, frame in frames.items():
        lines.append(f"- `{name}` — {len(frame):,} rows x {len(frame.columns)} cols")
    lines.append("")

    products, reviews = classify(frames)

    if products is not None:
        lines += ["## Products", f"- {len(products):,} products", ""]
        ing_col = guess_column(products, "ingredients")
        if ing_col:
            filled = products[ing_col].notna().mean()
            sample = products[ing_col].dropna()
            comma_ratio = (
                sample.astype(str).str.count(",").gt(2).mean() if len(sample) else 0.0
            )
            lines.append(f"- ingredients column `{ing_col}`: {filled:.1%} populated")
            lines.append(
                f"- comma-separated and parseable in {comma_ratio:.1%} of populated rows"
            )
            lines.append("")

    if reviews is None:
        lines.append("**No review table detected — collaborative filtering is off.**")
        return "\n".join(lines)

    user_col = guess_column(reviews, "user")
    item_col = guess_column(reviews, "item")
    lines += ["## Reviews", f"- {len(reviews):,} reviews", ""]

    lines.append("### Profile field completeness")
    verdict_ok = True
    for concept in ("skin_type", "skin_tone"):
        col = guess_column(reviews, concept)
        if col is None:
            lines.append(f"- **{concept}: column absent**")
            verdict_ok = False
            continue
        filled = reviews[col].notna().mean()
        distinct = reviews[col].nunique()
        lines.append(
            f"- `{col}`: {filled:.1%} populated, {distinct} distinct values"
        )
        if filled < 0.5:
            verdict_ok = False
    lines.append("")

    if user_col and item_col:
        per_user = reviews.groupby(user_col).size()
        per_item = reviews.groupby(item_col).size()
        density = len(reviews) / (per_user.size * per_item.size)
        lines += [
            "### Interaction density",
            f"- {per_user.size:,} distinct users, {per_item.size:,} distinct products",
            f"- reviews per user: median {per_user.median():.0f}, "
            f"share with 2+ {(per_user >= 2).mean():.1%}",
            f"- reviews per product: median {per_item.median():.0f}, "
            f"share with 5+ {(per_item >= 5).mean():.1%}",
            f"- matrix density: {density:.2%}",
            "",
        ]
        if (per_user >= 2).mean() < 0.15:
            verdict_ok = False

    st_col = guess_column(reviews, "skin_type")
    if st_col and item_col:
        cohort = reviews.groupby([item_col, st_col]).size()
        lines += [
            "### Cohort sizes (product x skin type)",
            f"- median cohort: {cohort.median():.0f} reviews",
            f"- share of cohorts with {MIN_COHORT_FOR_CLAIM}+ reviews: "
            f"{(cohort >= MIN_COHORT_FOR_CLAIM).mean():.1%}",
            "",
            f"Cohort statistics may only be shown when n >= {MIN_COHORT_FOR_CLAIM}; "
            "everything below that falls back to the checklist explanation.",
            "",
        ]

    lines.append("## Verdict")
    lines.append(
        "Cohort collaborative filtering is **viable as designed**."
        if verdict_ok
        else "Cohort CF is **not safe as designed** — profile fields are too sparse "
        "or users too thin. Fall back to item-item CF over the full review matrix, "
        "with profile fields used only as a filter."
    )
    return "\n".join(lines)


def main() -> None:
    raw_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    report = audit(raw_dir)
    out = Path("reports/data_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
