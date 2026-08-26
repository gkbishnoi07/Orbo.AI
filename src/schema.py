"""Canonical data contract.

Every module in this project speaks in terms of the names defined here, never
in terms of whatever the raw CSV columns happen to be called. `data.py` is the
only place allowed to know about the source dataset's naming.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ProductCols:
    """Canonical product table columns."""

    ID = "product_id"
    NAME = "name"
    BRAND = "brand"
    CATEGORY = "category"  # secondary_category: the useful UI granularity
    PRIMARY_CATEGORY = "primary_category"  # Skincare / Makeup / Hair / ...
    PRICE = "price"
    RATING = "rating"
    N_REVIEWS = "review_count"
    LOVES = "loves_count"
    INGREDIENTS = "ingredients"
    HIGHLIGHTS = "highlights"  # list[str], parsed from the raw stringified list
    TEXT = "text_blob"  # name + brand + categories + highlights + ingredients
    SKIN_TYPES = "suits_skin_types"  # list[str], derived
    CONCERNS = "addresses_concerns"  # list[str], derived

    REQUIRED = [ID, NAME, BRAND, CATEGORY, PRICE]


class ReviewCols:
    """Canonical review table columns."""

    USER = "author_id"
    ITEM = "product_id"
    RATING = "rating"
    SKIN_TYPE = "skin_type"
    SKIN_TONE = "skin_tone"
    EYE_COLOR = "eye_color"
    HAIR_COLOR = "hair_color"
    TIME = "submitted_at"

    REQUIRED = [USER, ITEM, RATING]


POSITIVE_RATING = 4.0
"""A rating at or above this counts as a positive interaction for evaluation.

Worth knowing when reading the numbers: 82% of Sephora reviews are 4 or 5 stars,
so this threshold is generous and the ranking task is correspondingly easier
than it would be on a balanced set. Kept at 4.0 anyway because it is the
conventional cut and moving it after seeing results would be choosing the
threshold that flatters the model. `reports/evaluation.md` repeats the headline
numbers at >= 5 as a sensitivity check.
"""

SKIN_TYPES: tuple[str, ...] = ("dry", "oily", "combination", "normal")
"""The only four values the review data uses. Note that 'sensitive' is *not*
among them — it is modelled as a concern instead, since the dataset expresses
sensitivity through formulation flags (fragrance-free, alcohol-free,
hypoallergenic) rather than as a skin type."""

SKIN_TONE_BANDS: tuple[str, ...] = ("fair", "light", "medium", "tan", "deep")
"""Reviewer skin tone collapsed from the raw 14 values into 5 ordered bands.

The raw vocabulary (porcelain, fairLight, lightMedium, mediumTan, olive, rich,
ebony, ...) splits 1.09M reviews so finely that a per-product tone cohort is
usually too small to say anything about. Banding trades precision we cannot
support for cohorts that clear the n >= 30 bar.
"""


@dataclass(frozen=True)
class ConcernDef:
    """One selectable concern, and which part of the catalogue it applies to."""

    slug: str
    label: str
    domain: str  # "skin" | "hair"


CONCERNS: tuple[ConcernDef, ...] = (
    # Skin — mirrors the "Good for: X" highlight vocabulary.
    ConcernDef("dryness", "Dryness", "skin"),
    ConcernDef("dullness", "Dullness / uneven texture", "skin"),
    ConcernDef("anti_aging", "Anti-aging", "skin"),
    ConcernDef("firmness", "Loss of firmness", "skin"),
    ConcernDef("pores", "Pores", "skin"),
    ConcernDef("acne", "Acne / blemishes", "skin"),
    ConcernDef("dark_spots", "Dark spots", "skin"),
    ConcernDef("dark_circles", "Dark circles", "skin"),
    ConcernDef("redness", "Redness", "skin"),
    # Not a "Good for:" tag. Derived from formulation flags plus an ingredient
    # exclusion rule, because the dataset has no sensitive-skin label.
    ConcernDef("sensitivity", "Sensitivity", "skin"),
    # Hair
    ConcernDef("frizz", "Frizz", "hair"),
    ConcernDef("damage", "Damage", "hair"),
    ConcernDef("volume", "Volume", "hair"),
    ConcernDef("hair_thinning", "Hair thinning", "hair"),
    ConcernDef("color_care", "Colour care", "hair"),
    ConcernDef("flaky_scalp", "Flaky / dry scalp", "hair"),
    ConcernDef("oily_scalp", "Oily scalp", "hair"),
)

CONCERNS_BY_SLUG: dict[str, ConcernDef] = {c.slug: c for c in CONCERNS}


@dataclass(frozen=True)
class Query:
    """A recommendation request.

    Every field is optional so the same object serves both the cold-start UI
    path (profile only) and the evaluation path (profile plus known history).
    """

    skin_type: str | None = None
    skin_tone: str | None = None
    concerns: tuple[str, ...] = ()
    category: str | None = None
    budget_max: float | None = None
    liked_product_ids: tuple[str, ...] = ()

    @property
    def is_cold_start(self) -> bool:
        return len(self.liked_product_ids) == 0


EVIDENCE_KINDS = ("match", "evidence", "caveat")
"""How an explanation line should be grouped when shown.

* `match`    — a criterion the user asked for that this product satisfies.
* `evidence` — a measurement backing the recommendation up, such as a cohort
               percentage or the product's own rating.
* `caveat`   — something unmet, or something that could not be verified.

Presentation metadata only: nothing here affects scoring or ranking. It exists
so the UI can group lines without pattern-matching on their wording, which
breaks the moment a label is reworded.
"""


@dataclass
class Evidence:
    """One line of a recommendation's explanation card.

    `supported` distinguishes a claim we verified from a claim we could not
    check. `detail` carries the sample size or measured value so the UI can
    show the user what the claim rests on. `kind` is a grouping hint.
    """

    label: str
    supported: bool
    detail: str = ""
    kind: str = "match"


@dataclass
class Scored:
    """A single ranked recommendation."""

    product_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)


MIN_COHORT_FOR_CLAIM = 30
"""Smallest cohort a percentage may be quoted from.

Below this a "% of oily-skin reviewers rated this 4+" claim is noise dressed up
as a statistic. The audit found only 51.8% of product-by-skin-type cohorts clear
this bar, so roughly half of all explanations fall back to the checklist form —
by design, not by accident.
"""
