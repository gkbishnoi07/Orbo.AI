"""Run named scenarios through the real service and write reports/test_cases.md.

Generated, never transcribed. A hand-written table of "expected" outputs drifts
from the code the first time a weight changes and nobody notices; this file runs
the shipped `RecommendationService` and prints what actually came back.

Failure cases get equal billing with successes, because the interesting question
about a recommender is not whether it works on the happy path — it is what it
does when the data cannot support an answer. Each failure below is a real
limitation with a number attached, not a hypothetical.

Run:  python scripts/06_test_cases.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schema import CONCERNS_BY_SLUG, ProductCols, Query  # noqa: E402
from src.service import RecommendationService  # noqa: E402

REPORTS = Path("reports")


@dataclass(frozen=True)
class Case:
    name: str
    why: str
    query: Query
    expect: str
    strategy: str = "hybrid"


def describe(query: Query) -> str:
    bits = []
    if query.skin_type:
        bits.append(f"{query.skin_type} skin")
    if query.skin_tone:
        bits.append(f"{query.skin_tone} tone")
    for slug in query.concerns:
        bits.append(CONCERNS_BY_SLUG[slug].label.lower() if slug in CONCERNS_BY_SLUG else slug)
    if query.category:
        bits.append(query.category)
    if query.budget_max is not None:
        bits.append(f"under ${query.budget_max:.0f}")
    if query.liked_product_ids:
        bits.append(f"{len(query.liked_product_ids)} owned product(s)")
    return ", ".join(bits) or "nothing specified"


SUCCESS: list[Case] = [
    Case(
        "Dry, sensitive skin wanting a moisturiser",
        "The core path: a profile with a hard rule attached. Sensitivity is not a "
        "label in this dataset, so it has to be enforced through ingredients.",
        Query(skin_type="dry", concerns=("dryness", "sensitivity"),
              category="Moisturizers", budget_max=60.0),
        "Ten moisturisers under $60, none containing fragrance or drying alcohol, "
        "each carrying a cohort statistic with its sample size.",
    ),
    Case(
        "Oily skin, acne concern",
        "Checks that the concern tag actually steers the list rather than being "
        "decorative, and that salicylic acid surfaces without being hard-coded.",
        Query(skin_type="oily", concerns=("acne",), budget_max=50.0),
        "Products tagged for acne, skewing to treatments and cleansers.",
    ),
    Case(
        "Returning shopper with three products they love",
        "The warm-start path. Collaborative filtering only contributes when the "
        "declared products exist in the interaction data.",
        Query(skin_type="combination", concerns=("pores",),
              liked_product_ids=("P420652", "P7880", "P269122")),
        "A list shifted noticeably by the declared history, with none of the three "
        "recommended back.",
    ),
    Case(
        "Hair concern rather than skin",
        "The catalogue is not only skincare. A frizz concern should reach haircare "
        "without the user having to pick the category.",
        Query(concerns=("frizz",), budget_max=40.0),
        "Haircare products tagged for frizz.",
    ),
    Case(
        "No profile at all",
        "What a first-time visitor sees before saying anything. Should be honest "
        "popularity rather than fake personalisation.",
        Query(budget_max=100.0),
        "The catalogue's most-rated products, identical for every visitor.",
        strategy="popularity",
    ),
]

FAILURE: list[Case] = [
    Case(
        "Budget below the category's cheapest product",
        "Over-constrained filters. The system must say so rather than quietly "
        "relaxing a constraint the user set.",
        Query(category="Beauty Supplements", budget_max=5.0),
        "Zero results, with the active constraints named. The cheapest Beauty "
        "Supplement is $40, so no ranking can help.",
    ),
    Case(
        "Cold start: skin type and nothing else",
        "The weakest measured path. Reported because averaging it into the "
        "headline numbers would hide it.",
        Query(skin_type="normal"),
        "A plausible list, but measured NDCG@10 of 0.065 against 0.069 for plain "
        "popularity — personalisation adds nothing detectable here.",
    ),
    Case(
        "Every concern at once",
        "A user who ticks everything is asking for a product that does not exist. "
        "The concern score becomes an average over ten unrelated tags.",
        Query(skin_type="dry", concerns=tuple(CONCERNS_BY_SLUG)[:10], budget_max=80.0),
        "Results still return, but the concern signal is diluted: no product "
        "satisfies ten concerns, so ranking falls back to the other layers.",
    ),
    Case(
        "Sensitivity plus a tight budget",
        "Two rules compounding. Roughly half the catalogue carries an irritant, so "
        "the eligible pool collapses before scoring begins.",
        Query(skin_type="sensitive" if False else "dry",
              concerns=("sensitivity",), category="Cleansers", budget_max=20.0),
        "A short list, with the exclusion breakdown showing how much the rules "
        "removed and why.",
    ),
    Case(
        "Skin tone stated",
        "Skin tone is collected and used for cohort context but is not a scoring "
        "signal. Included so the gap is documented rather than implied.",
        Query(skin_type="dry", skin_tone="deep", concerns=("dark_spots",)),
        "Identical ranking to the same query without a tone, because shade "
        "matching would need product-level shade data this dataset lacks.",
    ),
]


def render(service: RecommendationService, case: Case, index: str) -> list[str]:
    started = time.perf_counter()
    results = service.recommend(case.query, strategy=case.strategy, k=10)
    elapsed = (time.perf_counter() - started) * 1000
    eligible = service.eligible_count(case.query, case.strategy)
    removed = service.excluded_counts(case.query, case.strategy)

    lines = [
        f"### {index}. {case.name}",
        "",
        f"**Why this case.** {case.why}",
        "",
        f"| | |",
        f"| --- | --- |",
        f"| Input | {describe(case.query)} |",
        f"| Strategy | `{case.strategy}` |",
        f"| Eligible after filters | {eligible:,} of {service.facts.n_products:,} |",
        f"| Returned | {len(results)} |",
        f"| Latency | {elapsed:.0f} ms |",
        "",
        f"**Expected.** {case.expect}",
        "",
    ]

    if removed:
        lines.append("**Rules fired.**")
        lines.append("")
        for label, count in sorted(removed.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {count:,} products {label}")
        lines.append("")

    if not results:
        lines += [
            "**Actual output.** No products. This is the honest empty state: every "
            "candidate was removed by a hard filter rather than scored badly, so "
            "the UI names the active constraints instead of showing a spinner or "
            "silently dropping one.",
            "",
        ]
        return lines

    lines += ["**Actual output.**", "", "| # | Product | Price | Rating | Top reason |",
              "| --- | --- | --- | --- | --- |"]
    for rank, scored in enumerate(results[:5], start=1):
        product = service.product(scored.product_id)
        top = next((e for e in scored.evidence if e.supported), None)
        reason = top.label if top else "—"
        rating = product.get(ProductCols.RATING)
        rating_text = "—" if rating != rating else f"{float(rating):.1f}"
        lines.append(
            f"| {rank} | {product[ProductCols.BRAND]} — {product[ProductCols.NAME]} "
            f"| ${float(product[ProductCols.PRICE]):.0f} | {rating_text} | {reason} |"
        )
    if len(results) > 5:
        lines.append(f"| … | _{len(results) - 5} more_ | | | |")
    lines.append("")

    sample = results[0]
    lines += [
        f"**Full reasoning for #1** (`{sample.product_id}`), as the UI renders it:",
        "",
    ]
    for kind, title in (("match", "Why it matches"), ("evidence", "Evidence"),
                        ("caveat", "Worth knowing")):
        items = [e for e in sample.evidence if e.kind == kind]
        if not items:
            continue
        lines.append(f"*{title}*")
        lines.append("")
        for e in items:
            mark = "✓" if e.supported else "○"
            detail = f" — _{e.detail}_" if e.detail else ""
            lines.append(f"- {mark} {e.label}{detail}")
        lines.append("")
    return lines


def main() -> int:
    service = RecommendationService.from_artifacts()
    facts = service.facts

    out = [
        "# Test cases",
        "",
        "Generated by `scripts/06_test_cases.py` against the shipped",
        "`RecommendationService` — every table below is real output, not an",
        "illustration. Re-run the script to regenerate.",
        "",
        f"Catalogue: {facts.n_products:,} products, {facts.n_brands} brands, "
        f"{facts.n_interactions:,} positive ratings covering "
        f"{facts.reviewed_share:.0%} of the catalogue.",
        "",
        "## Successful scenarios",
        "",
    ]
    for i, case in enumerate(SUCCESS, start=1):
        out += render(service, case, f"S{i}")

    out += [
        "## Failure scenarios",
        "",
        "These are the cases where the system is weak, wrong, or unable to answer.",
        "They are documented with numbers rather than described in the abstract,",
        "because a recommender's limitations are the part a reader cannot verify",
        "for themselves.",
        "",
    ]
    for i, case in enumerate(FAILURE, start=1):
        out += render(service, case, f"F{i}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "test_cases.md"
    path.write_text("\n".join(out))
    print(f"written to {path}  ({len(SUCCESS)} success, {len(FAILURE)} failure cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
