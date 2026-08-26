"""Sweep the hybrid's blend weights and report the whole curve.

The first full evaluation produced an uncomfortable result: the hybrid lost to
plain cohort CF on warm-start NDCG (0.401 vs 0.413) *and* on catalogue coverage,
and lost to plain popularity on cold start (0.030 vs 0.069). Reporting only the
tuned winner would hide that, so the entire grid is written out.

The likely mechanism, which the sweep tests directly: min-max scaling makes the
two layers look comparable when they are not. Content is dense — every product
scores something, averaging around 0.5 — while CF is zero for the ~72% of the
catalogue it has never seen. So a flat `0.40 * content` term adds a baseline to
everything and dilutes the sharp, genuinely informative CF signal. If that is
right, pushing weight toward CF should recover the loss.

Sub-models are fitted once and the weights mutated between runs; refitting per
combination would spend minutes rebuilding identical similarity matrices.

Run:  python scripts/05_sweep.py [--cases 400]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import artifacts  # noqa: E402
from src.collaborative import CohortCFRecommender  # noqa: E402
from src.data import load  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_cold_start_cases,
    build_eval_cases,
    evaluate,
)
from src.hybrid import ContentOnlyRecommender, HybridRecommender  # noqa: E402
from src.recommender import PopularityRecommender  # noqa: E402
from src.schema import ReviewCols  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from importlib import import_module  # noqa: E402

_evaluate_module = import_module("04_evaluate")
drop_held_out = _evaluate_module.drop_held_out
as_markdown = _evaluate_module.as_markdown

REPORTS = Path("reports")

# (content, cf, popularity). Deliberately spans "content-led" through to
# "CF-led" so the shape of the tradeoff is visible, not just its peak.
WARM_GRID = [
    (0.40, 0.45, 0.15),
    (0.30, 0.60, 0.10),
    (0.20, 0.70, 0.10),
    (0.10, 0.85, 0.05),
    (0.05, 0.95, 0.00),
    (0.00, 1.00, 0.00),
]
COLD_GRID = [
    (0.30, 0.40, 0.30),
    (0.20, 0.40, 0.40),
    (0.10, 0.30, 0.60),
    (0.05, 0.15, 0.80),
    (0.00, 0.00, 1.00),
]
MMR_VALUES = [1.0, 0.85, 0.75, 0.5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--method", default="tfidf")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    products, reviews = load()
    matrix, ids = artifacts.load_embeddings(args.method)
    lookup = artifacts.as_lookup(matrix, ids)

    warm = build_eval_cases(reviews, min_positives=2, max_users=args.cases)
    cold = build_cold_start_cases(reviews, max_users=args.cases)
    train = drop_held_out(reviews, warm + cold)
    popularity_counts = (
        train[train[ReviewCols.RATING] >= 4].groupby(ReviewCols.ITEM).size().to_dict()
    )
    shared = dict(k=args.k, embeddings=lookup, popularity=popularity_counts)

    print(f"warm {len(warm):,}  cold {len(cold):,}  method {args.method}")
    print("fitting sub-models once")
    model = HybridRecommender(
        content=ContentOnlyRecommender(matrix, ids),
        collaborative=CohortCFRecommender(),
        popularity=PopularityRecommender(),
    ).fit(products, train)

    def run(label: str, cases) -> dict:
        row = evaluate(model, cases, products, **shared)
        print(
            f"  {label:<28} ndcg={row[f'ndcg@{args.k}']:.4f}  "
            f"cov={row['coverage']:.4f}  div={row.get('diversity', 0):.4f}"
        )
        return row

    print("\nwarm-start blend weights (content / cf / popularity)")
    warm_rows = {}
    for w_content, w_cf, w_pop in WARM_GRID:
        model.weight_content, model.weight_cf, model.weight_popularity = (
            w_content,
            w_cf,
            w_pop,
        )
        warm_rows[f"{w_content:.2f}/{w_cf:.2f}/{w_pop:.2f}"] = run(
            f"{w_content:.2f}/{w_cf:.2f}/{w_pop:.2f}", warm
        )
    warm_table = pd.DataFrame(warm_rows).T

    best_warm = warm_table[f"ndcg@{args.k}"].idxmax()
    model.weight_content, model.weight_cf, model.weight_popularity = (
        float(x) for x in best_warm.split("/")
    )
    print(f"  best warm blend: {best_warm}")

    print("\ncold-start blend weights (content / cf / popularity)")
    cold_rows = {}
    for w_content, w_cf, w_pop in COLD_GRID:
        (
            model.cold_weight_content,
            model.cold_weight_cf,
            model.cold_weight_popularity,
        ) = (w_content, w_cf, w_pop)
        cold_rows[f"{w_content:.2f}/{w_cf:.2f}/{w_pop:.2f}"] = run(
            f"{w_content:.2f}/{w_cf:.2f}/{w_pop:.2f}", cold
        )
    cold_table = pd.DataFrame(cold_rows).T
    best_cold = cold_table[f"ndcg@{args.k}"].idxmax()
    print(f"  best cold blend: {best_cold}")

    print("\nMMR lambda (1.0 = pure relevance, lower = more variety)")
    mmr_rows = {}
    for value in MMR_VALUES:
        model.mmr_lambda = value
        mmr_rows[f"lambda={value}"] = run(f"lambda={value}", warm)
    mmr_table = pd.DataFrame(mmr_rows).T

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            "# Hybrid weight sweep",
            "",
            f"{len(warm):,} warm cases, {len(cold):,} cold cases, k={args.k}, "
            f"embeddings `{args.method}`. Sub-models fitted once on the training "
            "split; only the blend weights change between rows.",
            "",
            "## Warm-start blend",
            "",
            "Rows are `content / cf / popularity`.",
            "",
            as_markdown(warm_table),
            "",
            f"Best NDCG@{args.k}: **{best_warm}**.",
            "",
            "## Cold-start blend",
            "",
            as_markdown(cold_table),
            "",
            f"Best NDCG@{args.k}: **{best_cold}**.",
            "",
            "## MMR lambda",
            "",
            "The relevance-for-variety trade, measured. Diversity is the column "
            "this is bought with; NDCG is the column it is paid for from.",
            "",
            as_markdown(mmr_table),
            "",
        ]
    )
    out = REPORTS / "weight_sweep.md"
    out.write_text(report)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
