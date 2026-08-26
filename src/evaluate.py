"""Offline evaluation harness.

Protocol: leave-one-out on positive interactions.

For each eligible user we hide exactly one product they rated highly, hand the
recommender their profile plus their *remaining* likes, and check whether the
hidden product comes back in the top k. When timestamps exist we hide the most
recent like, so the model is always predicting forward in time rather than
interpolating — an easier, and misleading, task.

The harness is written before any model so the metrics cannot be chosen after
seeing which ones happen to look good.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .recommender import Recommender
from .schema import POSITIVE_RATING, ProductCols, Query, ReviewCols


@dataclass(frozen=True)
class EvalCase:
    """One held-out prediction task."""

    user_id: str
    query: Query
    held_out: str


def build_eval_cases(
    reviews: pd.DataFrame,
    *,
    min_positives: int = 2,
    max_users: int | None = 2000,
    seed: int = 0,
) -> list[EvalCase]:
    """Construct leave-one-out cases from the review table.

    Users with fewer than `min_positives` likes are skipped: hiding their only
    interaction leaves nothing to personalise from, which measures the
    cold-start path rather than the ranking model. Cold start is tested
    separately in the scripted test cases.
    """
    rng = np.random.default_rng(seed)
    positives = reviews[reviews[ReviewCols.RATING] >= POSITIVE_RATING].copy()

    has_time = ReviewCols.TIME in positives.columns and positives[
        ReviewCols.TIME
    ].notna().any()
    if has_time:
        positives = positives.sort_values(ReviewCols.TIME)

    # Pick the users *before* iterating. The obvious implementation — build a
    # case for everyone, then sample — walks all 503,216 user groups in Python
    # to keep a couple of thousand, which takes minutes and dominates the whole
    # evaluation. Filtering and sampling first makes the loop proportional to
    # the sample instead of the dataset.
    counts = positives.groupby(ReviewCols.USER, sort=False)[ReviewCols.ITEM].size()
    eligible = counts.index[counts >= min_positives].to_numpy()
    if max_users is not None and len(eligible) > max_users:
        picked = rng.choice(len(eligible), size=max_users, replace=False)
        eligible = eligible[np.sort(picked)]
    positives = positives[positives[ReviewCols.USER].isin(set(eligible))]

    cases: list[EvalCase] = []
    for user_id, group in positives.groupby(ReviewCols.USER, sort=False):
        if len(group) < min_positives:
            continue

        if has_time:
            held_row = group.iloc[-1]
            history = group.iloc[:-1]
        else:
            index = rng.integers(len(group))
            held_row = group.iloc[index]
            history = group.drop(group.index[index])

        cases.append(
            EvalCase(
                user_id=str(user_id),
                query=Query(
                    skin_type=_first_valid(history, ReviewCols.SKIN_TYPE),
                    skin_tone=_first_valid(history, ReviewCols.SKIN_TONE),
                    liked_product_ids=tuple(
                        str(x) for x in history[ReviewCols.ITEM].tolist()
                    ),
                ),
                held_out=str(held_row[ReviewCols.ITEM]),
            )
        )

    return cases


def build_cold_start_cases(
    reviews: pd.DataFrame,
    *,
    max_users: int | None = 2000,
    seed: int = 0,
) -> list[EvalCase]:
    """Cases for users with exactly one positive interaction.

    These are excluded from `build_eval_cases` on purpose — hiding a user's only
    like leaves nothing to personalise from — but they are not a footnote. A
    brand-new visitor entering a skin profile is the *primary* path through the
    UI, and it is the path where CF has nothing to say and content has to carry
    the result. Measuring it separately is the only way to see that, since
    averaging it into the warm-start table would hide it either way.
    """
    rng = np.random.default_rng(seed)
    positives = reviews[reviews[ReviewCols.RATING] >= POSITIVE_RATING]

    counts = positives.groupby(ReviewCols.USER)[ReviewCols.ITEM].size()
    singles = counts[counts == 1].index
    subset = positives[positives[ReviewCols.USER].isin(singles)]

    if max_users is not None and len(subset) > max_users:
        subset = subset.sample(n=max_users, random_state=int(rng.integers(1 << 31)))

    cases = [
        EvalCase(
            user_id=str(row[ReviewCols.USER]),
            query=Query(
                skin_type=_as_optional_str(row.get(ReviewCols.SKIN_TYPE)),
                skin_tone=_as_optional_str(row.get(ReviewCols.SKIN_TONE)),
                liked_product_ids=(),
            ),
            held_out=str(row[ReviewCols.ITEM]),
        )
        for _, row in subset.iterrows()
    ]
    return cases


def held_out_pairs(cases: list[EvalCase]) -> list[tuple[str, str]]:
    """The (user, item) interactions a model must not be allowed to learn from."""
    return [(case.user_id, case.held_out) for case in cases]


def _as_optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _first_valid(frame: pd.DataFrame, column: str) -> str | None:
    """Most recent non-null value for a profile field, or None."""
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return str(values.iloc[-1]) if len(values) else None


def evaluate(
    model: Recommender,
    cases: list[EvalCase],
    products: pd.DataFrame,
    *,
    k: int = 10,
    embeddings: dict[str, np.ndarray] | None = None,
    popularity: dict[str, int] | None = None,
) -> dict[str, float]:
    """Score one recommender across every metric in one pass."""
    if not cases:
        raise ValueError("no evaluation cases; check min_positives and the data")

    per_case: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "map": [],
        "diversity": [],
        "novelty": [],
    }
    latencies_ms: list[float] = []
    all_lists: list[list[str]] = []
    empty_results = 0

    catalog_size = products[ProductCols.ID].nunique()
    n_users = len(cases)

    for case in cases:
        start = time.perf_counter()
        results = model.recommend(case.query, k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        ranked = [r.product_id for r in results]
        if not ranked:
            empty_results += 1
        all_lists.append(ranked)

        relevant = {case.held_out}
        per_case["precision"].append(metrics.precision_at_k(ranked, relevant, k))
        per_case["recall"].append(metrics.recall_at_k(ranked, relevant, k))
        per_case["ndcg"].append(metrics.ndcg_at_k(ranked, relevant, k))
        per_case["map"].append(metrics.average_precision_at_k(ranked, relevant, k))

        if embeddings:
            per_case["diversity"].append(
                metrics.intra_list_diversity(ranked, embeddings)
            )
        if popularity is not None:
            per_case["novelty"].append(metrics.novelty(ranked, popularity, n_users))

    summary = {
        f"precision@{k}": float(np.mean(per_case["precision"])),
        f"recall@{k}": float(np.mean(per_case["recall"])),
        f"ndcg@{k}": float(np.mean(per_case["ndcg"])),
        f"map@{k}": float(np.mean(per_case["map"])),
        "coverage": metrics.catalog_coverage(all_lists, catalog_size),
        "latency_p50_ms": metrics.percentile(latencies_ms, 50),
        "latency_p95_ms": metrics.percentile(latencies_ms, 95),
        "empty_result_rate": empty_results / n_users,
        "n_cases": float(n_users),
    }
    if per_case["diversity"]:
        summary["diversity"] = float(np.mean(per_case["diversity"]))
    if per_case["novelty"]:
        summary["novelty"] = float(np.mean(per_case["novelty"]))
    return summary


def compare(
    models: list[Recommender],
    cases: list[EvalCase],
    products: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Run every model over the same cases and return the comparison table."""
    rows = {}
    for model in models:
        rows[model.name] = evaluate(model, cases, products, **kwargs)
    return pd.DataFrame(rows).T
