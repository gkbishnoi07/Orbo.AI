"""Build the small committed files the deployed app reads.

The deployed app has no Kaggle credentials, no 530MB of CSV and no time to parse
it on a cold start, so everything it needs is precomputed here and committed.
The constraint that shapes this file: keep the total small enough to live in git
comfortably and load in under a second.

What gets written, and why each is the shape it is:

* `products.parquet` — the display and filter table. Small already (3MB).
* `interactions.parquet` — positive interactions only, three columns. This is
  what cohort CF is refitted from at app startup. Serialising the fitted
  similarity matrices instead would be smaller still, but it would mean the app
  and the evaluation ran different code paths, and a serialised matrix that
  silently drifts from the model that produced it is a much worse bug than a
  two-second startup cost.
* `cohort_stats.parquet` — per (product, skin type) review and positive counts.
  This is what lets an explanation say "72% of dry-skin reviewers rated this 4+
  (n=318)" without the app ever seeing a review row.

Run:  python scripts/03_build_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load  # noqa: E402
from src.explain import build_cohort_stats  # noqa: E402
from src.tone import build_tone_stats  # noqa: E402
from src.schema import POSITIVE_RATING, MIN_COHORT_FOR_CLAIM, ProductCols, ReviewCols  # noqa: E402

ARTIFACTS = Path("artifacts")


def main() -> int:
    products, reviews = load()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    products.to_parquet(ARTIFACTS / "products.parquet", index=False)

    # Only positives, only the columns cohort CF actually reads. Ratings and
    # timestamps are dropped because nothing at request time consults them.
    interactions = reviews.loc[
        reviews[ReviewCols.RATING] >= POSITIVE_RATING,
        [ReviewCols.USER, ReviewCols.ITEM, ReviewCols.SKIN_TYPE, ReviewCols.RATING],
    ].reset_index(drop=True)
    interactions.to_parquet(ARTIFACTS / "interactions.parquet", index=False)

    cohort_stats = build_cohort_stats(reviews)
    cohort_stats.to_parquet(ARTIFACTS / "cohort_stats.parquet", index=False)

    # Per (product, tone band). Feeds the tone affinity nudge in the hybrid;
    # kept separate from cohort_stats so neither artifact changes shape.
    tone_stats = build_tone_stats(reviews)
    tone_stats.to_parquet(ARTIFACTS / "tone_stats.parquet", index=False)

    usable = (cohort_stats["n_reviews"] >= MIN_COHORT_FOR_CLAIM).mean()
    print("written to artifacts/:")
    total = 0
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file() and path.suffix in {".parquet", ".npy", ".json"}:
            size = path.stat().st_size
            total += size
            print(f"  {path.name:<26} {size / 1e6:7.1f} MB")
    print(f"  {'TOTAL':<26} {total / 1e6:7.1f} MB")

    print()
    print(f"products:      {len(products):,}")
    print(f"interactions:  {len(interactions):,} positives")
    print(f"cohort rows:   {len(cohort_stats):,}")
    print(f"tone rows:     {len(tone_stats):,}")
    print(
        f"  of which n >= {MIN_COHORT_FOR_CLAIM}: {usable:.1%} "
        "(these can carry a quoted percentage; the rest fall back to the checklist)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
