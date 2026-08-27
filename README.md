# Beauty Recommender

Enter a skin profile — type, tone, concerns, category, budget — and get ranked
beauty products, each with the evidence behind it. Built on 8,494 real Sephora
products and 1.09 million real reviews from people who stated their skin type.

**Nothing in an explanation is generated.** Every line on a product card is read
from a column, so the system can tell you what a product *fails* as readily as
what it satisfies, and it cannot invent a claim.

| | |
| --- | --- |
| **Live app** | **https://orboai.streamlit.app/** |
| **Repo** | https://github.com/gkbishnoi07/Orbo.AI |
| **Tests** | 187, all passing |
| **Latency** | ~16 ms typical, ~31 ms p95 end-to-end (see [Latency](#latency)) |

---

## Contents

- [Problem statement](#problem-statement) · [Use case and motivation](#use-case-and-motivation) · [Approach](#approach)
- [System architecture](#system-architecture) · [Recommendation methodology](#recommendation-methodology)
- [Dataset](#dataset) · [Technologies](#technologies) · [Assumptions](#assumptions)
- [Key design decisions](#key-design-decisions) · [Evaluation](#evaluation-methodology) · [Results](#results)
- [Test cases](#test-cases) · [Known limitations](#known-limitations) · [Future improvements](#future-improvements)
- [Running it](#running-it) · [Repository layout](#repository-layout)
- [Bonus: comparison with Sephora and Nykaa](docs/comparison.md)

---

## Problem statement

Beauty retail has a discovery problem that ordinary e-commerce search does not.
A shopper does not want "a moisturiser" — they want a moisturiser that suits
*combination skin*, targets *dark spots*, contains no fragrance because their
skin reacts, and costs under $40. The catalogue is 8,494 products across 304
brands; the shopper is one person with four constraints and no way to apply them.

Worse, the information that would answer the question is not in the product
description. Which foundation oxidises after four hours, which moisturiser pills
under sunscreen, which "gentle" cleanser is not — that lives in reviews, written
by people whose skin may or may not resemble yours.

So the task is: **turn a stated skin profile into a short ranked list, and show
the evidence for each pick.** The second half matters as much as the first. A
beauty recommendation that a shopper cannot audit is a guess with a UI.

## Use case and motivation

I chose a beauty and skincare recommender because Orbo.AI works in this domain —
visual skin analysis, shade matching, product recommendation — so the design
decisions here are the same ones the company's own products have to make. The
easier option was a MovieLens or generic e-commerce dataset, which is what most
tutorials use and what I would have reached for if I only wanted to show that I
can build a recommender. I picked beauty because it makes the problem harder in
ways worth demonstrating: hard suitability constraints, cohorts that mean
something specific, and explanations where being wrong actually matters.

Those three things shaped most of the architecture below:

- **Suitability is not preference.** A fragranced serum is not merely a worse
  match for reactive skin; it is a wrong answer. Some constraints must override
  the score entirely, which forces a filter layer distinct from ranking.
- **Cohorts are meaningful.** "People with oily skin who liked this also liked
  that" is a genuinely different statement from the un-scoped version. The
  dataset carries reviewer skin type, so this is testable rather than aspirational.
- **Explanations carry real stakes.** People put these products on their face.
  An invented claim about an ingredient is not a bad recommendation, it is a
  safety problem — which is why the explanation layer is template-driven and an
  LLM is deliberately absent.

## Approach

Four layers, in order. The first is non-negotiable; the rest are scores.

1. **Hard filters.** Budget, category, and suitability rules. Applied before
   anything is ranked, in the `Recommender` base class so that *every* strategy —
   baselines included — honours them identically.
2. **Content scoring.** Similarity over precomputed product-text vectors, plus
   exact concern and skin-type tag matching. The only layer that can rank a
   product nobody has reviewed, which is 72% of the catalogue.
3. **Cohort collaborative filtering.** Item-item similarity rebuilt per
   skin-type cohort. Captures what product copy never says.
4. **Skin-tone affinity.** A small, signed nudge toward products that a tone
   band rates more highly than the catalogue average — never a filter.
5. **Blend, then diversify.** Weighted combination, then MMR reranking so ten
   near-identical serums do not fill the page.

Then an **explanation layer** turns each result into evidence lines grouped as
*why it matches*, *evidence*, and *worth knowing*.

The order of work was deliberate too: the evaluation harness and metrics were
written **before any model**, so the metrics could not be chosen after seeing
which ones flattered the result. And the dataset was audited before the
architecture was committed — see [Key design decisions](#key-design-decisions).

## System architecture

```
OFFLINE (run once, results committed)
  scripts/00_download.py ──► data/raw/  530 MB of Kaggle CSVs (gitignored)
  scripts/01_inspect.py  ──► reports/data_audit.md      decides the CF design
  src/data.py            ──► canonical frames           the only module that
                                                        knows raw column names
  scripts/02_embed.py    ──► TF-IDF + MiniLM vectors
  scripts/03_build_artifacts.py ──► artifacts/  34 MB, committed

RUNTIME (per request, ~12 ms p95)
  Query ──► _apply_filters ──► content ─┐
             (budget,          cohort CF ├─► blend ──► MMR ──► Explainer ──► cards
              category,        popularity┘
              rules)
```

The runtime reads **only** `artifacts/`. No Kaggle credentials, no CSV parsing,
no `torch`. Verified: installing `requirements.txt` into a clean virtualenv
pulls 41 packages and zero of `torch`, `transformers`, or `sentence-transformers`.

## Recommendation methodology

### Hard filters — `src/rules.py`

Two rules, both derived from the profile rather than from a score.

**Irritant exclusion.** Declaring *sensitivity* excludes products whose
ingredient list contains fragrance, drying alcohol, fragrant essential oils, or
a harsh sulfate surfactant. The alcohol patterns are deliberately specific
(`alcohol denat`, `sd alcohol`) because a bare `alcohol` match would also catch
cetyl and stearyl alcohol — fatty alcohols, which are emollients, the opposite
of an irritant. A product labelled "Fragrance Free" waives the fragrance match,
since a stray `Parfum` in a multi-variant ingredient dump is a data artefact.

**Stated skin-type mismatch.** A product saying "Best for Oily, Combo, Normal
Skin" is excluded for dry skin. Crucially the rule fires on *stated
disagreement*, never on a missing tag — see the design decisions below.

About 11% of products publish no ingredient list. These are **not** excluded —
absence of data is not evidence of an irritant — but the explanation says the
check could not be run rather than implying it passed.

### Content — `src/content.py`

Cosine similarity between a product vector and a **profile prototype**. The
deployed app has no text encoder, so the prototype is assembled from the
catalogue itself: the centroid of products already tagged for that skin type and
those concerns, blended with the centroid of anything the user says they own
(history weighted 0.6, because what someone actually bought beats what a tag says
they should want).

Three components, kept separate so the UI can show which one carried a result:
embedding similarity, fraction of stated concerns the product is tagged for, and
stated skin-type fit. **Untagged products score a neutral 0.5 on skin fit, not
zero** — zeroing them would hand the entire ranking to the tagged 12%.

### Cohort collaborative filtering — `src/collaborative.py`

Item-item cosine similarity over positive interactions, rebuilt per skin-type
cohort, with co-occurrence shrinkage — two items sharing one reviewer out of a
thousand are not similar however flattering the cosine, and at 0.09% matrix
density that case is the norm rather than the edge.

Neighbours are stored as `(catalogue row indices, similarities)` numpy pairs
rather than pandas Series. The Series version spent most of its time reindexing a
50-element vector up to catalogue width once per liked product per request, and
was the single largest contributor to latency (248 ms p95 → 26 ms).

With no usable history it degrades to cohort popularity rather than returning
nothing.

### Blend and rerank — `src/hybrid.py`

Each layer is min-max scaled across the catalogue before weighting, which makes
CF's abstention explicit: it scores zero for the ~72% of products it has never
seen, and rescaling stops an arbitrary scale from deciding the ranking.

Weights differ by regime, because cold start is a different problem:

| | Content | Cohort CF | Popularity |
| --- | --- | --- | --- |
| Returning user | 0.15 | 0.80 | 0.05 |
| New user | 0.10 | 0.30 | 0.60 |

Then greedy **MMR** at λ=0.85, which in the shipped blend lifts intra-list
diversity from 0.680 to 0.740 while NDCG@10 rises 1.0%. Worth stating plainly: a
naive one-product-per-brand cap reaches the same variety (0.743) — so the
diversity number alone does not justify MMR. What justifies it is the price: the
brand cap costs 6.4% of NDCG@10 to get there, and it only knows about brands,
not about ten near-identical niacinamide serums from ten different houses.

The sweep's own λ table reports 0.7658 at λ=0.85 rather than 0.740. That section
runs after the warm grid has set the blend to `0.05/0.95/0.00` and uses no tone
layer, so it measures MMR on a near-pure-CF blend, not on the shipped hybrid.

### Skin-tone affinity — `src/tone.py`

For a product and a tone band, the signal is the band's positive rate minus the
product's overall positive rate, shrunk toward that overall rate by an
empirical-Bayes prior of 20 reviews (the median product-by-tone cohort size).

Two properties keep it safe to add to an already-swept blend. It is **centred on
zero**, so a query without a tone — or a product with no tone history —
contributes exactly nothing and the ranking is bit-identical to the
three-layer version. And it is **added on top of** the existing weights rather
than folded into them, so the swept 0.15/0.80/0.05 blend is untouched.

It is a nudge, not a filter: nothing is excluded for lacking tone data, which
would penalise the least-represented bands hardest. In practice it moves the
ranking for the `deep`, `medium` and `tan` bands and barely at all for `fair`
and `light` — those two dominate the data, so their rate *is* close to the
pooled average that everything is shrunk toward.

### Explanations — `src/explain.py`

Templates filled from columns. Never an LLM, so a card cannot invent a claim.

A cohort percentage ("86% of dry-skin reviewers rated this 4 or above") is shown
only when that cohort has **at least 30 reviews**, and always with its sample
size. Below the threshold it is suppressed rather than hedged — a percentage from
eleven reviewers reads as authoritative whatever caveat is attached. Only 51.8%
of product-by-skin-type cohorts clear that bar, so roughly half of all
explanations fall back to the checklist form by design.

Unmet criteria are shown, not hidden. A checklist of nothing but ticks is
marketing.

## Dataset

[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
(Kaggle, 2023 snapshot). Chosen because it is the only readily available beauty
dataset that carries **reviewer skin type and skin tone** — without which
cohort-scoped collaborative filtering is not possible.

| | |
| --- | --- |
| Products | 8,494 across 304 brands, 42 categories |
| Reviews | 1,094,411 raw → 1,088,886 after de-duplication |
| Positive interactions (4★+) | 893,393 |
| Distinct reviewers | 503,216 |
| Products with a positive rating (4★+) | **2,343 (27.6%)** |
| Products with any review at all | 2,351 (27.7%) |
| Matrix density | 0.09% |
| `skin_type` populated | 89.9% (4 values) |
| `skin_tone` populated | 84.5% (14 values → 5 bands) |
| Products with a skin-type tag | **12.4%** |
| Products with a concern tag | 41.7% |

Three properties of the raw data shaped most of the code:

1. **There is no product description field.** The text used for vectors has to be
   assembled from brand, name, categories, highlights and the ingredient list.
2. `highlights` and `ingredients` are *stringified Python lists*, not delimited
   text, so they need `literal_eval` rather than `split`.
3. **82% of reviews are 4★ or 5★.** The positive threshold of 4.0 is therefore
   generous and the ranking task correspondingly easier than on a balanced set.
   Kept anyway, because moving it after seeing results would be choosing the
   threshold that flatters the model.

### Where the data lives in this repo

**If you are looking for a dataset file, it is `artifacts/` — not `data/`.** That
directory is deliberately empty apart from a `.gitkeep`.

| Path | Committed? | What it is |
| --- | --- | --- |
| `data/raw/` | **No** — gitignored | The five original Kaggle CSVs, ~530 MB. Not redistributed: too large for git and covered by Kaggle's licence terms. `scripts/00_download.py` fetches them. |
| `artifacts/` | **Yes** — 34 MB, 7 files | The derived data the app actually reads: the product table, positive interactions, per-cohort and per-tone counts, and the product vectors. **This is the shipped dataset.** |

That split is why a clone runs with no Kaggle account: the app never opens a CSV,
it reads `artifacts/products.parquet` and friends. It is also why the repo is
34 MB rather than 560 MB.

To inspect the shipped data directly:

```python
import pandas as pd
pd.read_parquet("artifacts/products.parquet").head()      # 8,494 products
pd.read_parquet("artifacts/interactions.parquet").head()  # 893,393 ratings
```

## Technologies

| Layer | Choice | Why |
| --- | --- | --- |
| UI | Streamlit | One language end to end; the brief values function over a bespoke frontend |
| Data | pandas, pyarrow | Parquet keeps list-valued columns as lists, which CSV flattens |
| Vectors | scikit-learn TF-IDF + TruncatedSVD | **Measured 2.1× better than MiniLM here** — see Results |
| Vectors (compared) | sentence-transformers MiniLM-L6-v2 | Development only; never imported at runtime |
| Sparse maths | scipy | Cohort item-item similarity |
| Tests | pytest + `streamlit.testing.v1` | Drives the real page headlessly, no browser needed |

## Assumptions

1. **A rating of 4 or 5 is a positive signal.** Not a purchase, not a repeat
   purchase — the dataset has neither.
2. **Stated skin type is accurate and stable.** Self-reported, unverified, and in
   reality skin changes seasonally.
3. **A missing tag means unknown, not unsuitable.** The single most consequential
   assumption in the codebase.
4. **Product metadata is current as of the 2023 snapshot.** Prices and rating
   counts were true when scraped, not today.
5. **Sensitivity can be inferred from formulation flags.** The dataset has no
   sensitive-skin label, so it is derived from fragrance-free / alcohol-free /
   hypoallergenic plus an ingredient scan. A dermatologist would want more.
6. **The 14 raw skin tones can be banded into 5.** Precision we cannot support
   is traded for cohorts large enough to say anything.
7. **Reviewers are broadly representative of shoppers.** Review data
   over-represents people motivated enough to write.

## Key design decisions

**The dataset was audited before the architecture was fixed.** `scripts/01_inspect.py`
existed before any model and answered one question: is cohort CF viable? It found
`skin_type` on 89.9% of reviews and a median product-by-skin-type cohort of 32,
so yes. The documented fallback — global item-item CF with profile fields demoted
to filters — was not needed. Committing to cohort CF without checking would have
been a guess.

**Untagged is not unsuitable.** Only 12.4% of products carry a skin-type tag.
Treating untagged as a mismatch would silently delete seven eighths of the
catalogue while still returning a full, confident-looking list of ten. This is
the most destructive bug the rule layer could have, so it is asserted in three
separate tests.

**Sensitivity is a concern, not a skin type.** There is no sensitive-skin label
anywhere in the data, and inventing one as a fifth skin type would have made the
cohorts fictional.

**Filters live in the base class.** `Recommender._apply_filters` runs for every
strategy including random and popularity. A baseline that quietly ignored the
budget cap would make the entire comparison table dishonest.

**Explanations are computed, never asserted.** An LLM asked to justify a
recommendation will happily produce "clinically proven to reduce redness in two
weeks". A template filled from a column cannot.

**Embeddings are precomputed and committed.** The deployed app never imports
sentence-transformers, which keeps `torch` out of the runtime, the image inside a
free tier, and p95 latency at 26 ms.

**TF-IDF over a sentence transformer.** Not an assumption — a measurement. See
Results.

## Evaluation methodology

Leave-one-out on positive interactions (rating ≥ 4). Where timestamps exist the
**most recent** like is held out, so the model predicts forward in time rather
than interpolating between known likes.

**Every model is fitted on the same training split, with all evaluated
interactions removed.** This matters more than any individual number. Fitting a
popularity or CF model on all reviews and then evaluating leave-one-out lets each
model count the very interaction it is being asked to predict. The leak is easy to
miss and flatters interaction-based models specifically, so its size is measured
and reported rather than asserted away: training on everything inflates CF's
NDCG@10 by **3.7%** and popularity's by **1.9%**.

Two case sets, reported separately because they measure different products:

- **Warm start** — users with 2+ positives. Tests the ranking model.
- **Cold start** — users with exactly one positive, hidden. Tests the path a new
  visitor actually takes. Averaging it into the headline would hide it either way.

Metrics: precision@10, recall@10, NDCG@10, MAP@10, recommendation coverage,
intra-list diversity, novelty, p50/p95 latency, empty-result rate.

**"Coverage" here means one specific thing** and it is easy to confuse with
another: it is the share of the catalogue a model actually *returned* across the
whole evaluation run. It is not the share a layer is *able* to score. The content
layer, for instance, is applicable to the 72% of the catalogue that has no
interactions — that is reach — while returning 16.6% of the catalogue across
1,000 evaluation cases. Both numbers are about the content model and neither
contradicts the other.

Recommendation coverage and diversity are reported *beside* accuracy because a
model that serves every user the same ten bestsellers can score well on accuracy
alone.

Reproduce: `.venv/bin/python scripts/04_evaluate.py` and `.venv/bin/python scripts/05_sweep.py`.

## Latency

Two different numbers get quoted for systems like this and they are not
interchangeable. Both are measured, neither is typed into this file:

| What | Where measured | p50 | p95 |
| --- | --- | --- | --- |
| **Model inference** — `Recommender.recommend()` alone | `src/evaluate.py`, reported in `reports/evaluation.json` | see artifact | see artifact |
| **End-to-end recommendation** — inference + explanation generation, k=20 | measured live, shown in the UI's result strip | ~16 ms | ~31 ms |

Neither includes Streamlit rendering or network time, so neither is what a
browser tab experiences. Cold start is ~2.8 s, almost entirely rebuilding the
cohort similarity matrices, cached thereafter for the life of the container.

Earlier revisions of this README quoted "26 ms p95" without saying which of the
two it was. It was model inference at k=10, excluding explanations.

## Results

1,000 warm and 1,000 cold cases, k=10. Full tables in
[`reports/evaluation.md`](reports/evaluation.md); the same numbers in
machine-readable form in [`reports/evaluation.json`](reports/evaluation.json),
which is what the app's Model performance tab reads.

| Model | NDCG@10 warm | NDCG@10 cold | Recommendation coverage | Diversity (warm) | p95 latency |
| --- | --- | --- | --- | --- | --- |
| random | 0.0004 | 0.0000 | 68.8% | 0.904 | 7.6 ms |
| popularity | 0.0307 | **0.0693** | 0.2% | 0.874 | 6.9 ms |
| content (TF-IDF) | 0.2404 | 0.0019 | 16.6% | 0.521 | 9.8 ms |
| content (MiniLM) | 0.1166 | 0.0024 | 11.0% | 0.646 | 10.4 ms |
| cohort CF | **0.4135** | 0.0693 | 15.5% | 0.748 | 7.2 ms |
| hybrid (TF-IDF) | 0.4112 | 0.0653 | 14.2% | 0.740 | 12.2 ms |

Latency is model inference only, and coverage is the share of the catalogue a
model *returned* across the run — not the share it is able to score. The
diversity column is the shipped configuration; see the note under the reranker
above for why `reports/weight_sweep.md` reports a higher figure for the same λ.

Three findings, including the unflattering one.

**TF-IDF beats MiniLM 2.1× for content scoring.** The dataset has no description
field, so the text is brand, name, category, highlight tokens and an INCI
ingredient list — keyword soup, where exact term overlap does more work than
semantic similarity. Both were built and measured rather than assumed.

**The hybrid does not beat cohort CF alone** (0.4112 vs 0.4135), and the weight
sweep is blunter still: on NDCG the optimum is near-degenerate — 95% CF warm,
pure popularity cold. That is largely a limitation of the protocol. Held-out items are
drawn from the review table, so **every correct answer is a product that already
has reviews**; the 6,151 products with none can never register as a hit, and the
content layer's whole contribution is invisible to the measurement. The shipped
weights keep small content terms anyway, at a stated cost of −0.5% warm and −6%
cold NDCG. `reports/weight_sweep.md` shows the weights that were rejected.

**Cold start is genuinely weak.** Nothing beats plain popularity by much. Said
plainly rather than buried.

## Test cases

Ten scenarios — five success, five failure — generated by
`scripts/06_test_cases.py` running the shipped service, with real output tables:
[`reports/test_cases.md`](reports/test_cases.md).

Generated rather than transcribed, because a hand-written table of "expected"
output drifts from the code the first time a weight changes and nobody notices.

Failure cases get equal billing: over-constrained filters returning an honest
empty state, cold start measurably no better than popularity, ten concerns at
once diluting the signal to nothing, compounding rules collapsing the eligible
pool, and skin tone being collected but not scored.

## Known limitations

1. **Cold start barely beats popularity** (0.0653 vs 0.0693). With a profile and
   no history there is little to personalise on.
2. **The hybrid loses to CF alone on offline accuracy**, and the offline metric
   structurally cannot see what the content layer adds.
3. **CF is blind to 72% of the catalogue.** Only 2,343 products have a positive
   rating to learn from (2,351 have any review at all).
4. **Skin tone is a weak signal, and deliberately so.** It now affects ranking
   (`src/tone.py`) but only as a shrunk nudge, and it visibly moves results only
   for the `deep`, `medium` and `tan` bands — `fair` and `light` dominate the
   data, so their rate is already the average everything is shrunk toward. True
   *shade* matching remains impossible: this dataset has no product-level shade
   data.
5. **The 4★ threshold is generous** — 82% of reviews qualify.
6. **Self-reported, unverified profile fields.**
7. **No feedback loop.** Everything is offline; no clicks, so nothing learns.
8. **A 2023 snapshot.** Prices and counts are historical, not live.
9. **Popularity bias survives in CF.** Item-item similarity on sparse retail data
   returns blockbusters as almost everyone's neighbour; the median popularity rank
   of a top-50 CF neighbour is 96 of 8,494. MMR mitigates, does not solve.
10. **The result list is capped at 100.** Rendering thousands of cards would take
    ~45,000 Streamlit elements and make the page unusable.
11. **English-language, US catalogue, USD pricing.**

## Future improvements

**Fix what the evaluation cannot see.** The honest next step is an online test.
Offline leave-one-out cannot reward catalogue reach, so the content layer's value
is unmeasurable here. A click-through comparison of hybrid against CF-only would
settle in a week what no amount of offline tuning can.

**Score the ingredient list properly.** Ingredients are currently a text blob
truncated at 400 characters. Parsing INCI names into a structured vocabulary
would allow real actives matching ("contains a retinoid at a tolerable position
in the list") instead of highlight-tag matching.

**Replace centroid similarity for cold start** with max-similarity to any
tag-matched product. A centroid over 500 products washes out; nearest-neighbour
to the relevant set is far more discriminative. This is the most promising
unexplored lever on the weakest measured path.

**Two-stage retrieval** — ANN candidate generation then a learned reranker
(LambdaMART or a small GBDT over the component scores) instead of hand-tuned
linear weights.

**Sequence awareness.** Routines are ordered: cleanser then serum then
moisturiser. A session-based model would capture that; leave-one-out on a bag of
likes cannot.

**Shade matching**, which is where Orbo's computer vision would genuinely add
something the review text cannot — but it needs shade-level product data.

**Calibrated confidence.** The system knows when it is guessing (untagged
product, thin cohort, no history). It should say "low confidence" explicitly
rather than only omitting a claim.

## Running it

### Use the deployed app

See the link at the top. No setup required.

### Run locally

Requires **Python 3.10 or newer**. On Windows use `.venv\Scripts\` in place of
`.venv/bin/`.

```bash
git clone https://github.com/gkbishnoi07/Orbo.AI.git
cd Orbo.AI
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open http://localhost:8501.

That is all — the 34 MB of precomputed artifacts is committed, so **no Kaggle
account and no dataset download is needed**. Expect a few seconds on first load
while the cohort similarity matrices are built; they are cached afterwards.

### Run the tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q        # 187 tests
```

### Reproduce everything from the raw data

Needs Kaggle credentials at `~/.kaggle/kaggle.json`.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/00_download.py            # ~150 MB zip -> data/raw/
.venv/bin/python scripts/01_inspect.py data/raw    # audit; decides the CF design
.venv/bin/python scripts/02_embed.py               # TF-IDF + MiniLM vectors
.venv/bin/python scripts/03_build_artifacts.py     # the committed runtime files
.venv/bin/python scripts/04_evaluate.py            # reports/evaluation.md
.venv/bin/python scripts/05_sweep.py               # reports/weight_sweep.md
.venv/bin/python scripts/06_test_cases.py          # reports/test_cases.md
```

## Repository layout

```
app.py                      Streamlit UI (presentation only)
.streamlit/config.toml      committed theme
src/schema.py               the contract: canonical names, Query, Scored, Evidence
src/data.py                 the ONLY module that knows raw Kaggle column names
src/rules.py                hard suitability rules, precomputed per catalogue
src/recommender.py          Recommender ABC + random and popularity baselines
src/content.py              content scoring over precomputed vectors
src/collaborative.py        cohort-scoped item-item CF
src/hybrid.py               weighted blend + MMR reranker
src/explain.py              template-based evidence lines
src/metrics.py              pure metric functions, no dataset dependency
src/evaluate.py             leave-one-out harness, compare()
src/artifacts.py            loads the committed runtime files
src/service.py              composition root shared by the UI and the test cases
src/tone.py                 skin-tone affinity (shrunk, signed ranking nudge)
scripts/00..06              download, audit, embed, build, evaluate, sweep, cases
tests/                      187 tests
reports/                    audit, evaluation, sweep, test cases
artifacts/                  34 MB of committed runtime files
docs/comparison.md          bonus: Sephora / Nykaa comparison
```

`src/schema.py` is the contract. Everything downstream of `data.py` speaks
canonical names only, so swapping the dataset touches exactly one file.

## Deployment

**Live at https://orboai.streamlit.app/** — no setup, no account needed.

Hosted on Streamlit Community Cloud, which redeploys on every push to `main`.

Requires Python **3.10+**. No floor is declared in the repo (no
`.python-version` or `pyproject.toml`), so the version is selected in the
Streamlit Cloud UI.

It needs nothing but this repository: `requirements.txt` resolves to 41 packages
with no `torch`, and the app reads only the committed `artifacts/`. Cold start is
about three seconds, almost all of it rebuilding the cohort similarity matrices,
which are then cached for the life of the container.
