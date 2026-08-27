"""Score every model through one harness and write the comparison table.

Protocol: leave-one-out on positive interactions (rating >= 4), holding out each
user's most recent like so the model predicts forward in time.

The part that matters more than any individual number: **every model is fitted
on the same training set, with the held-out interactions removed.** Fitting a
popularity or CF model on all reviews and then evaluating leave-one-out lets each
model count the very interaction it is being asked to predict. The leak is easy
to miss, it flatters interaction-based models specifically, and it would make the
whole table meaningless. The size of that leak is measured and reported at the
bottom rather than just asserted.

Two case sets are reported because they measure different products:

* **warm start** — users with 2+ positives. Tests the ranking model.
* **cold start** — users with exactly one positive, hidden. Tests the path a new
  visitor actually takes through the UI, where CF has nothing to say.

Run:  python scripts/04_evaluate.py [--cases 1000] [--method tfidf]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import artifacts  # noqa: E402
from src.collaborative import CohortCFRecommender  # noqa: E402
from src.data import load  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvalCase,
    build_cold_start_cases,
    build_eval_cases,
    compare,
    held_out_pairs,
)
from src.hybrid import CFOnlyRecommender, ContentOnlyRecommender, HybridRecommender  # noqa: E402
from src.tone import ToneAffinity, build_tone_stats  # noqa: E402
from src.recommender import PopularityRecommender, RandomRecommender  # noqa: E402
from src.schema import ProductCols, ReviewCols  # noqa: E402
from src.service import DEFAULT_METHOD  # noqa: E402

REPORTS = Path("reports")

LATENCY_BUDGET_MS = 100.0
"""Budget the harness checks p95 model-inference latency against."""


def drop_held_out(reviews: pd.DataFrame, cases: list[EvalCase]) -> pd.DataFrame:
    """Remove every evaluated interaction from the training data."""
    pairs = set(held_out_pairs(cases))
    if not pairs:
        return reviews
    keys = pd.MultiIndex.from_arrays(
        [reviews[ReviewCols.USER].astype(str), reviews[ReviewCols.ITEM].astype(str)]
    )
    mask = ~keys.isin(pd.MultiIndex.from_tuples(sorted(pairs)))
    return reviews[mask]


def build_models(products, train_reviews, matrices: dict[str, tuple]):
    """One model per row of the comparison table.

    Both embedding methods appear as separate content and hybrid rows, scored
    over identical cases in the same pass. Which vectoriser suits this text is
    an empirical question — the blob is highlight tokens and INCI lists, not
    prose — and putting them side by side is the only way to answer it.
    """
    models = [RandomRecommender(seed=0), PopularityRecommender(), CFOnlyRecommender()]

    for method, (matrix, ids) in matrices.items():
        content = ContentOnlyRecommender(matrix, ids)
        content.name = f"content-{method}"
        models.append(content)

    # Tone affinity is rebuilt from the TRAINING split, never from the committed
    # artifact. The artifact is derived from every review including the held-out
    # ones, so using it here would leak exactly the interactions being predicted
    # — the same mistake the train/test split exists to prevent.
    tone = ToneAffinity(build_tone_stats(train_reviews))

    for method, (matrix, ids) in matrices.items():
        hybrid = HybridRecommender(
            content=ContentOnlyRecommender(matrix, ids),
            collaborative=CohortCFRecommender(),
            popularity=PopularityRecommender(),
            tone=tone,
        )
        hybrid.name = f"hybrid-{method}"
        models.append(hybrid)

    for model in models:
        started = time.perf_counter()
        model.fit(products, train_reviews)
        print(f"  fitted {model.name:<18} in {time.perf_counter() - started:5.1f}s")
    return models


def as_markdown(table: pd.DataFrame) -> str:
    """Render a results table as markdown.

    Hand-rolled rather than `DataFrame.to_markdown`, which needs `tabulate`.
    Adding a dependency so a report can draw pipe characters is not a good
    trade, and this keeps the runtime requirements file honest.
    """
    def render(value, column: str) -> str:
        if column == "n_cases":
            return f"{int(value):,}"
        if "latency" in column:
            return f"{value:.1f}"
        return f"{value:.4f}"

    columns = list(table.columns)
    header = ["model", *columns]
    rows = [
        [str(name), *(render(table.loc[name, c], c) for c in columns)]
        for name in table.index
    ]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows))
        for i in range(len(header))
    ]
    def line(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    return "\n".join(
        [line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        + [line(row) for row in rows]
    )


def summarise_findings(warm, cold, leak, k: int) -> str:
    """Derive the conclusions from the tables rather than asserting them.

    Written out even when it is unflattering. The interesting result on this
    dataset is that the hybrid does *not* clear cohort CF, and a report that
    quietly omitted that would be worth less than one that says so.
    """
    ndcg = f"ndcg@{k}"
    lines = []

    best = warm[ndcg].idxmax()
    popularity = warm.loc["popularity", ndcg]
    lines.append(
        f"- **Best warm-start model: `{best}`** at {ndcg} "
        f"{warm.loc[best, ndcg]:.4f}, "
        f"{warm.loc[best, ndcg] / popularity:.1f}x the popularity baseline "
        f"({popularity:.4f}), with catalogue coverage "
        f"{warm.loc[best, 'coverage']:.1%} against popularity's "
        f"{warm.loc['popularity', 'coverage']:.1%}."
    )

    hybrids = [i for i in warm.index if i.startswith("hybrid")]
    if hybrids and "cf-only" in warm.index:
        best_hybrid = warm.loc[hybrids, ndcg].idxmax()
        gap = warm.loc[best_hybrid, ndcg] / warm.loc["cf-only", ndcg] - 1
        verdict = "does not beat" if gap < 0 else "beats"
        lines.append(
            f"- **The hybrid {verdict} cohort CF alone** on warm start: "
            f"`{best_hybrid}` {warm.loc[best_hybrid, ndcg]:.4f} vs `cf-only` "
            f"{warm.loc['cf-only', ndcg]:.4f} ({gap:+.1%}). See "
            "'What this protocol cannot measure' below before reading that as a "
            "verdict on the content layer."
        )

    contents = [i for i in warm.index if i.startswith("content-")]
    if len(contents) >= 2:
        ranked = warm.loc[contents, ndcg].sort_values(ascending=False)
        winner, runner = ranked.index[0], ranked.index[1]
        lines.append(
            f"- **{winner.removeprefix('content-')} beats "
            f"{runner.removeprefix('content-')} for content scoring** "
            f"({ranked.iloc[0]:.4f} vs {ranked.iloc[1]:.4f}, "
            f"{ranked.iloc[0] / ranked.iloc[1]:.1f}x). The product text is "
            "highlight tokens and INCI ingredient lists rather than prose, so "
            "exact term overlap does more work here than semantic similarity."
        )

    cold_best = cold[ndcg].idxmax()
    lines.append(
        f"- **Cold start is hard and everything collapses toward popularity.** "
        f"Best is `{cold_best}` at {cold.loc[cold_best, ndcg]:.4f}; the content "
        f"layers score near zero because a profile with no history gives them "
        f"almost nothing to work with."
    )

    for model in leak.index:
        clean_name = model.removesuffix("-LEAKY")
        if clean_name in warm.index:
            inflation = leak.loc[model, ndcg] / warm.loc[clean_name, ndcg] - 1
            lines.append(
                f"- **Leak check `{clean_name}`:** training on all reviews "
                f"inflates {ndcg} by {inflation:+.1%} "
                f"({warm.loc[clean_name, ndcg]:.4f} -> {leak.loc[model, ndcg]:.4f})."
            )

    # Compare against the budget rather than asserting it. The previous version
    # printed "inside the 100ms budget" unconditionally, and on a loaded machine
    # emitted "p95 110ms ... inside the 100ms budget".
    slowest = warm["latency_p95_ms"].max()
    within = slowest <= LATENCY_BUDGET_MS
    verdict = (
        f"inside the {LATENCY_BUDGET_MS:.0f}ms budget"
        if within
        else f"**over the {LATENCY_BUDGET_MS:.0f}ms budget**"
    )
    lines.append(
        f"- **Model inference latency:** p95 {slowest:.0f}ms across all models, "
        f"{verdict}. This is `Recommender.recommend()` only — it excludes "
        "explanation generation and all Streamlit rendering, so it is not "
        "end-to-end user latency."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--method", default=None, help="embedding method to use")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    products, reviews = load()
    methods = artifacts.available_methods()
    if not methods:
        print("No embeddings found. Run scripts/02_embed.py first.", file=sys.stderr)
        return 1
    # Default to the method the app actually ships (src.service.DEFAULT_METHOD)
    # rather than whichever name sorts first — the benchmark caption in the UI
    # names this, and "minilm" there would describe a model nobody runs.
    default = DEFAULT_METHOD if DEFAULT_METHOD in methods else methods[0]
    method = args.method or default
    print(f"diversity measured in: {method}  (available: {', '.join(methods)})")

    matrices = {m: artifacts.load_embeddings(m) for m in methods}
    # Diversity is measured in one fixed space so the numbers stay comparable
    # across rows; otherwise each model would be scored with its own ruler.
    reference_matrix, reference_ids = matrices[method]
    lookup = artifacts.as_lookup(reference_matrix, reference_ids)

    print("\nbuilding cases")
    warm = build_eval_cases(reviews, min_positives=2, max_users=args.cases)
    cold = build_cold_start_cases(reviews, max_users=args.cases)
    print(f"  warm: {len(warm):,}   cold: {len(cold):,}")

    all_cases = warm + cold
    train = drop_held_out(reviews, all_cases)
    print(f"  training interactions: {len(train):,} of {len(reviews):,} "
          f"({len(reviews) - len(train):,} held out)")

    popularity_counts = (
        train[train[ReviewCols.RATING] >= 4].groupby(ReviewCols.ITEM).size().to_dict()
    )

    print("\nfitting models on the training split")
    models = build_models(products, train, matrices)

    shared = dict(k=args.k, embeddings=lookup, popularity=popularity_counts)

    print("\nevaluating warm start")
    warm_table = compare(models, warm, products, **shared)
    print(warm_table.to_string())

    print("\nevaluating cold start")
    cold_table = compare(models, cold, products, **shared)
    print(cold_table.to_string())

    # ---- how big is the leak we avoided? ----
    print("\nquantifying the leak (models refitted on ALL reviews)")
    leaky = [PopularityRecommender(), CFOnlyRecommender()]
    for model in leaky:
        model.fit(products, reviews)
        model.name = f"{model.name}-LEAKY"
    leak_table = compare(leaky, warm, products, **shared)
    print(leak_table.to_string())

    headline = summarise_findings(warm_table, cold_table, leak_table, args.k)
    print("\n" + headline)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            "# Evaluation",
            "",
            f"Leave-one-out on positive interactions (rating >= 4), k={args.k}. "
            f"Embedding methods compared: {', '.join(f'`{m}`' for m in methods)}. "
            f"Intra-list diversity is measured in the `{method}` space for every "
            "row, so the column stays comparable across models.",
            "",
            f"Catalogue: {len(products):,} products, of which "
            f"{reviews[ReviewCols.ITEM].nunique():,} "
            f"({reviews[ReviewCols.ITEM].nunique() / len(products):.1%}) have any review.",
            "",
            "Every model below is fitted on the same training split, with all "
            f"{len(reviews) - len(train):,} evaluated interactions removed.",
            "",
            "## Headline",
            "",
            headline,
            "",
            "## Warm start",
            "",
            f"Users with 2+ positive interactions ({len(warm):,} cases). Tests ranking.",
            "",
            as_markdown(warm_table),
            "",
            "## Cold start",
            "",
            f"Users with exactly one positive, hidden ({len(cold):,} cases). This is "
            "the path a new visitor takes through the UI: profile only, no history.",
            "",
            as_markdown(cold_table),
            "",
            "## What this protocol cannot measure",
            "",
            f"Held-out items are drawn from the review table, so every correct "
            f"answer here is by construction a product that already has reviews "
            f"— one of the {reviews[ReviewCols.ITEM].nunique():,} that cohort CF can rank. "
            f"The other {len(products) - reviews[ReviewCols.ITEM].nunique():,} products "
            f"({1 - reviews[ReviewCols.ITEM].nunique() / len(products):.1%} of the catalogue) "
            "have no interactions and can never register as a hit.",
            "",
            "That matters for how the table above should be read. The content "
            "layer exists to cover exactly those products, and an offline metric "
            "built on interaction data structurally cannot reward covering items "
            "that have no interaction data. `reports/weight_sweep.md` shows the "
            "consequence: on NDCG alone the optimal blend is degenerate — pure CF "
            "warm, pure popularity cold. The shipped weights keep small content "
            "terms anyway, at a measured cost of -0.4% warm and -9% cold NDCG, "
            "because the alternative returns roughly a dozen distinct products "
            "across a thousand cold-start users.",
            "",
            "## The leak, measured",
            "",
            "The same two interaction-based models refitted on *all* reviews, so each "
            "has seen the interaction it is asked to predict. Compare against the "
            "warm-start table above: the gap is the size of the mistake avoided by "
            "splitting properly.",
            "",
            as_markdown(leak_table),
            "",
        ]
    )
    out = REPORTS / "evaluation.md"
    out.write_text(report)

    # Machine-readable twin of the table above. The UI reads this instead of
    # carrying its own copy of the numbers: a hardcoded metric in the interface
    # is indistinguishable from a measured one to a reader, and silently goes
    # stale the moment a weight changes.
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"

    benchmark = {
        "commit": commit,
        "k": args.k,
        "warm_cases": len(warm),
        "cold_cases": len(cold),
        "embedding_method": method,
        "note": (
            "Latency columns are model inference only: the harness times "
            "Recommender.recommend() and excludes explanation generation and all "
            "Streamlit rendering."
        ),
        "models": {
            str(name): {
                "warm": {m: float(warm_table.loc[name, m]) for m in warm_table.columns},
                "cold": (
                    {m: float(cold_table.loc[name, m]) for m in cold_table.columns}
                    if name in cold_table.index
                    else None
                ),
            }
            for name in warm_table.index
        },
    }
    json_out = REPORTS / "evaluation.json"
    json_out.write_text(json.dumps(benchmark, indent=2))
    print(f"\nwritten to {out} and {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
