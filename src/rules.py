"""Hard suitability rules, precomputed once per catalogue.

Two rules, both derived from the profile rather than from a score:

* **Irritant exclusion.** Someone reporting sensitivity should not be handed a
  product whose ingredient list opens with denatured alcohol, however well it
  matches on every other axis. No amount of embedding similarity should be able
  to outvote that.
* **Tagged skin-type mismatch.** If a product explicitly says "Best for Oily,
  Combo, Normal Skin" and the user has dry skin, that is a stated conflict.

The second rule carries a trap worth being explicit about: only 12.4% of the
catalogue has any skin-type tag at all. So the rule fires on *stated
disagreement*, never on absence of a tag. Treating untagged as unsuitable would
silently delete seven eighths of the catalogue — the single most destructive
thing this file could get wrong.

Matching runs once in `fit()` and collapses to a boolean matrix, because
scanning 8,494 ingredient strings against a dozen regexes on every request
would dominate the latency budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import SKIN_TYPES, ProductCols


@dataclass(frozen=True)
class IrritantRule:
    """One reason a product might be unsuitable for sensitive skin."""

    slug: str
    label: str
    patterns: tuple[str, ...]
    waived_by: tuple[str, ...] = field(default=())
    """Highlights that override an ingredient match. A product labelled
    "Fragrance Free" that still lists 'Parfum' somewhere in a multi-variant
    ingredient dump is a data artefact, not a fragranced product."""


IRRITANT_RULES: tuple[IrritantRule, ...] = (
    IrritantRule(
        "fragrance",
        "they contain added fragrance",
        ("parfum", "fragrance)"),
        waived_by=("Fragrance Free",),
    ),
    IrritantRule(
        "drying_alcohol",
        "they contain drying alcohol",
        # Deliberately specific. A bare "alcohol" match would also catch cetyl
        # and stearyl alcohol, which are fatty alcohols and are emollients —
        # the opposite of an irritant.
        ("alcohol denat", "sd alcohol", "denatured alcohol"),
        waived_by=("Alcohol Free",),
    ),
    IrritantRule(
        "essential_oils",
        "they contain fragrant essential oils",
        (
            "essential oil",
            "limonene",
            "linalool",
            "citronellol",
            "geraniol",
            "eugenol",
            "menthol",
        ),
    ),
    IrritantRule(
        "harsh_surfactant",
        "they contain a harsh sulfate surfactant",
        ("sodium lauryl sulfate",),
        waived_by=("Without Sulfates SLS & SLES",),
    ),
)

UNVERIFIABLE = "unverifiable"
"""Column marking products with no usable ingredient list (about 11% of the
catalogue). These are *not* excluded — absence of data is not evidence of an
irritant — but the explanation layer has to say the check could not be run
rather than implying it passed."""


def build_irritant_flags(products: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: one row per product, one column per irritant rule.

    Indexed by product_id so lookups downstream are alignment, not merges.
    """
    ingredients = (
        products[ProductCols.INGREDIENTS].fillna("").astype(str).str.casefold()
    )
    highlights = products[ProductCols.HIGHLIGHTS]

    flags = pd.DataFrame(index=pd.Index(products[ProductCols.ID], name=ProductCols.ID))

    for rule in IRRITANT_RULES:
        matched = pd.Series(False, index=ingredients.index)
        for pattern in rule.patterns:
            matched |= ingredients.str.contains(pattern, regex=False, na=False)
        if rule.waived_by:
            waived = highlights.map(
                lambda hs, allowed=rule.waived_by: any(h in allowed for h in hs)
            )
            matched &= ~waived
        flags[rule.slug] = matched.to_numpy()

    flags[UNVERIFIABLE] = (ingredients.str.len() < 20).to_numpy()
    return flags


def build_skin_type_flags(products: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix of stated skin-type suitability, plus an `untagged` column.

    `untagged` is the important one: it lets callers distinguish "this product
    is wrong for you" from "nobody recorded who this product is for".
    """
    tags = products[ProductCols.SKIN_TYPES]
    flags = pd.DataFrame(index=pd.Index(products[ProductCols.ID], name=ProductCols.ID))
    for skin_type in SKIN_TYPES:
        flags[skin_type] = tags.map(
            lambda ts, target=skin_type: target in ts
        ).to_numpy()
    flags["untagged"] = tags.map(len).eq(0).to_numpy()
    return flags


def build_concern_flags(products: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix of which concerns each product is tagged as addressing."""
    tags = products[ProductCols.CONCERNS]
    slugs = sorted({slug for row in tags for slug in row})
    flags = pd.DataFrame(index=pd.Index(products[ProductCols.ID], name=ProductCols.ID))
    for slug in slugs:
        flags[slug] = tags.map(lambda cs, target=slug: target in cs).to_numpy()
    return flags


def excluded(
    irritant_flags: pd.DataFrame,
    skin_type_flags: pd.DataFrame,
    *,
    skin_type: str | None,
    concerns: tuple[str, ...],
) -> pd.Series:
    """Products this profile must not be shown, as a boolean Series.

    Returned rather than applied so the caller can report *how many* products a
    rule removed, which is the difference between a filter and a black box.
    """
    drop = pd.Series(False, index=irritant_flags.index)

    if "sensitivity" in concerns:
        for rule in IRRITANT_RULES:
            drop |= irritant_flags[rule.slug]

    if skin_type and skin_type in skin_type_flags.columns:
        stated_mismatch = ~skin_type_flags[skin_type] & ~skin_type_flags["untagged"]
        drop |= stated_mismatch

    return drop


def exclusion_breakdown(
    irritant_flags: pd.DataFrame,
    skin_type_flags: pd.DataFrame,
    *,
    skin_type: str | None,
    concerns: tuple[str, ...],
) -> dict[str, int]:
    """How many products each active rule removes. Used by the UI and the docs."""
    breakdown: dict[str, int] = {}
    if "sensitivity" in concerns:
        for rule in IRRITANT_RULES:
            count = int(irritant_flags[rule.slug].sum())
            if count:
                breakdown[rule.label] = count
    if skin_type and skin_type in skin_type_flags.columns:
        mismatch = ~skin_type_flags[skin_type] & ~skin_type_flags["untagged"]
        count = int(mismatch.sum())
        if count:
            breakdown[
                f"they are specifically formulated for other skin "
                f"types — not {skin_type}"
            ] = count
    return breakdown
