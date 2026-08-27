"""Skin-tone affinity: a small, shrunk ranking nudge.

Tone was collected by the UI and used by nothing. This turns it into a real but
deliberately modest ranking signal, without touching the three existing layers
or their weights.

The signal is the difference between how well a product is rated *by people in
your tone band* and how well it is rated overall:

    affinity(product, band) = shrunk_positive_rate(product, band)
                              - overall_positive_rate(product)

Two properties make this safe to add to an existing blend:

* It is **centred on zero**. A product with no tone history, or one that its
  band likes exactly as much as everyone else does, contributes exactly 0. So
  queries without a tone are bit-identical to the previous behaviour.
* It is **shrunk toward the product's own overall rate** by an empirical-Bayes
  prior. That matters here more than usual: the deep and tan bands hold 26.5k
  and 33.5k reviews against light's 460.8k, so an unshrunk rate from six deep-tone
  reviewers would swamp a rate from six hundred light-tone ones. With PRIOR_STRENGTH
  reviews of prior mass, a band needs real evidence to move a product at all.

It is a nudge, not a filter. Nothing is excluded for being unrated by a band —
that would repeat the "untagged means unsuitable" mistake the rules layer exists
to avoid, and would hit the darkest bands hardest precisely because they are the
least represented in this dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ReviewCols

PRIOR_STRENGTH = 20.0
"""Reviews of prior mass pulling a band's rate toward the product's overall rate.

Chosen against the observed distribution: the median product-by-tone cohort holds
20 reviews, so at the median the band's own evidence and the prior carry equal
weight. Below that the prior dominates, which is the intended behaviour for the
thin bands.
"""


def build_tone_stats(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per (product, tone band): review count and positive count.

    Mirrors `build_cohort_stats` so the two artifacts stay symmetrical.
    """
    from .schema import POSITIVE_RATING

    frame = reviews.dropna(subset=[ReviewCols.SKIN_TONE])
    grouped = frame.groupby([ReviewCols.ITEM, ReviewCols.SKIN_TONE])
    stats = grouped[ReviewCols.RATING].agg(
        n_reviews="size",
        n_positive=lambda s: int((s >= POSITIVE_RATING).sum()),
    )
    return stats.reset_index()


class ToneAffinity:
    """Precomputed per-(product, band) affinity, looked up in O(1) at request time."""

    def __init__(
        self,
        tone_stats: pd.DataFrame,
        *,
        prior_strength: float = PRIOR_STRENGTH,
    ) -> None:
        self.prior_strength = prior_strength
        self._by_band: dict[str, pd.Series] = {}

        if tone_stats is None or tone_stats.empty:
            return

        # A product's overall positive rate, pooled across every band. This is
        # the prior each band is shrunk toward.
        totals = tone_stats.groupby(ReviewCols.ITEM)[["n_reviews", "n_positive"]].sum()
        overall = totals["n_positive"] / totals["n_reviews"].replace(0, np.nan)

        for band, rows in tone_stats.groupby(ReviewCols.SKIN_TONE):
            rows = rows.set_index(ReviewCols.ITEM)
            base = overall.reindex(rows.index)
            shrunk = (rows["n_positive"] + prior_strength * base) / (
                rows["n_reviews"] + prior_strength
            )
            self._by_band[str(band)] = (shrunk - base).dropna()

    def bands(self) -> list[str]:
        return sorted(self._by_band)

    def scores(self, band: str | None, index: pd.Index) -> pd.Series:
        """Affinity for every product in `index`; all zeros when unknown."""
        if not band or band not in self._by_band:
            return pd.Series(0.0, index=index)
        return self._by_band[band].reindex(index).fillna(0.0)
