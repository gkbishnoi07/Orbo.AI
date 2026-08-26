# Comparison with Sephora and Nykaa

The UI takes its cues from Sephora (whose catalogue this is) and Nykaa. Worth
being precise about what is genuinely comparable and what is not: they run
recommender systems against live behavioural data at scale, and this runs one
against a 2023 review dump. The interesting comparison is not "how close is it"
but **which of their design decisions turn out to be forced by the problem, and
which are choices I could make differently.**

## Similarities

**A filter rail beside a product grid.** Both put constraints on the left and
results on the right, and both keep the rail fixed while results scroll. That is
not decoration — beauty shopping is constraint-driven, and a shopper adjusts
budget or category repeatedly while scanning. Copied deliberately.

**Structured skin profiles.** Nykaa and Sephora both ask for skin type and
concerns rather than inferring everything from behaviour, because a first-time
visitor has no behaviour. Same reason the profile here is the primary input.

**Concerns as a first-class facet.** "Dark spots", "dullness", "pores" are how
beauty shoppers describe what they want, not product categories. Both platforms
expose this; so does the sidebar here.

**Popularity as the fallback.** Sephora's "Bestsellers" and Nykaa's trending
rails exist because popularity is a strong baseline when nothing is known about
the visitor. The measurements here agree emphatically: for a brand-new user,
**nothing beat plain popularity** (0.0693 NDCG@10, against 0.0651 for the
hybrid). The industry default is the industry default for a reason.

**Ratings and review counts on the card.** Both show them because they are the
cheapest available trust signal.

## Differences

**Explanations are shown, and they include failures.** This is the real
divergence. Sephora and Nykaa show ratings and sometimes "recommended for you";
neither tells you *why* a specific product was ranked where it was, and neither
says what a product does **not** satisfy. Every card here lists the criteria it
misses. That is commercially unusual — a retailer has little incentive to
volunteer that a product is not tagged for your skin type — but it is the whole
point of the exercise, and it is why the explanation layer is template-driven
rather than generated.

**Sample sizes are attached to every statistic.** "86% of dry-skin reviewers
rated this 4 or above (based on 1,783 reviews)". A retail site would show the
percentage and drop the *n*. Below 30 reviewers this system shows **no
percentage at all**, rather than a hedged one.

**The unfiltered count is visible.** "7,693 products were excluded from your
results — see why", broken down by rule. Retailers hide the funnel; showing it is
the difference between a filter and a black box.

**No behavioural data whatsoever.** They have clicks, dwell time, add-to-cart,
purchases, returns, and repeat purchases. This has stated profiles and 4★+
ratings. That gap is not a tuning difference, it is a different problem — and it
is why the cold-start numbers here are honest rather than good.

**No commercial ranking pressure.** No sponsored placements, no margin
weighting, no inventory bias. Real retail rankings are a negotiation between
relevance and business objectives; this one is purely relevance, which makes it
cleaner and also less realistic.

**Landing state recommends nothing.** Both retailers fill the homepage with
merchandising. This shows an onboarding panel and no products until a profile
exists, because a list identical for everybody is not a recommendation and
labelling it as one would be dishonest.

## Current limitations against them

| | Sephora / Nykaa | This system |
| --- | --- | --- |
| Signal | Clicks, purchases, returns, repeat rate | Stated profile + 4★ ratings |
| Freshness | Live catalogue and stock | 2023 snapshot, no stock |
| Feedback loop | Continuous online learning | None — fully offline |
| Cold start | Rich context: geography, referrer, session | Profile only |
| Shade matching | Shade-level SKUs, virtual try-on | Not attempted; no shade data |
| Imagery | Real product photography | Typographic placeholder plates |
| Scale | Millions of users, live A/B testing | 503k historical reviewers, offline eval |
| Personalisation depth | Per-user embeddings, sequence models | Cohort-level (four skin types) |

The most consequential of these is the **feedback loop**. Every ranking decision
here was tuned against a static offline metric that, as the evaluation section
documents, structurally cannot reward one of the three layers. Sephora tunes
against what people actually clicked yesterday.

## Areas for improvement

1. **Shade-level data.** The single biggest gap for a beauty recommender, and the
   place where Orbo's computer vision would add something reviews cannot. Needs
   product data this dataset does not have.
2. **Real product imagery.** Placeholder plates are honest but a beauty grid
   without photography is not a beauty grid.
3. **Stock and price freshness.** Recommending an out-of-stock product is worse
   than recommending a mediocre one.
4. **Routine-level recommendations.** Both retailers are moving toward "build
   your routine" rather than single products. A cleanser, serum and moisturiser
   that work together is a different and more useful output than three
   independently ranked products.
5. **Per-user personalisation** instead of four skin-type cohorts.

## What I would build next, with more time

In priority order, judged by expected value rather than by what is interesting:

**1. An online test.** Not a model — a measurement. Offline evaluation here
cannot distinguish the hybrid from CF-only in the way that matters, because
held-out items always come from the review table and the 6,151 unreviewed
products can never count as a hit. A week of click-through data would settle what
no amount of offline tuning can. Everything below is guesswork until this exists.

**2. Structured ingredient parsing.** Ingredients are currently a truncated text
blob. Parsing INCI names into a vocabulary with positional weighting would allow
genuine actives matching — "contains niacinamide in the top third of the list" —
and would sharpen the sensitivity rules from keyword matching into something a
dermatologist would recognise.

**3. Fix cold start properly.** Replace centroid similarity with max-similarity
to any tag-matched product. A centroid over 500 tagged products washes out into
"how average is this item"; nearest-neighbour to the relevant set is far more
discriminative. This is the most promising unexplored lever on the weakest path.

**4. A learned reranker.** Replace the hand-tuned linear blend with a small GBDT
over the component scores, trained on the interaction data. The weight sweep
showed the linear optimum is degenerate, which is usually a sign the combination
function is too simple rather than that a layer is worthless.

**5. Routine construction.** Take the ranked list and solve for a compatible
set — no two products with conflicting actives, covering cleanse/treat/moisturise
inside a budget. This is a constrained selection problem on top of ranking, and
it is what a beauty shopper actually wants.

**6. Calibrated confidence.** The system already knows when it is guessing:
untagged product, cohort under 30, no history. It should say so in words rather
than only omitting a claim. "We are not confident about this one, and here is
why" is more useful than silence, and it is the natural extension of showing
sample sizes.
