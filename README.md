# Beauty recommender

Personalised skincare and cosmetics recommendations from a skin profile. Enter
skin type, tone, concerns, category and budget; get ten ranked products, each
with a plain-English explanation of why it was chosen.

Built on the Sephora Products and Skincare Reviews dataset: 8,494 products and
1.09M reviews carrying reviewer skin type and tone.

## Status

| Stage | State |
| --- | --- |
| Schema and contracts | done |
| Metrics | done |
| Evaluation harness | done |
| Baselines (random, popularity) | done |
| Data audit | done — `reports/data_audit.md` |
| Data loader | done |
| Content-based model | done |
| Cohort collaborative model | done |
| Hybrid and MMR reranker | done |
| Explanations | done |
| Evaluation + weight sweep | done — `reports/evaluation.md`, `reports/weight_sweep.md` |
| UI | not started |
| Deployment | not started |
| Full documentation | not started |

114 tests passing.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

The runtime needs only `requirements.txt`; `requirements-dev.txt` adds the
Kaggle client, sentence-transformers and pytest. The deployed app deliberately
never imports sentence-transformers — see "Design decisions" below.

### Reproducing from scratch

Needs Kaggle credentials at `~/.kaggle/kaggle.json`.

```bash
python scripts/00_download.py            # ~150MB zip -> data/raw/
python scripts/01_inspect.py data/raw    # audit; decides the CF design
python scripts/02_embed.py               # product vectors -> artifacts/
python scripts/03_build_artifacts.py     # committed runtime files
python scripts/04_evaluate.py            # reports/evaluation.md
python scripts/05_sweep.py               # reports/weight_sweep.md
pytest
```

Steps 2 and 3 are already committed as `artifacts/` (34MB), so the app and the
evaluation run without a Kaggle account.

## Architecture

```
app.py                      Streamlit entrypoint (must stay at repo root)
src/schema.py               canonical column names, Query, Scored, Evidence, vocabulary
src/data.py                 the only module that knows raw Kaggle column names
src/metrics.py              pure metric functions, no dataset dependency
src/rules.py                hard suitability rules, precomputed per catalogue
src/recommender.py          Recommender ABC + random and popularity baselines
src/content.py              content scoring over precomputed vectors
src/collaborative.py        cohort-scoped item-item CF
src/hybrid.py               weighted blend + MMR reranker
src/explain.py              template-based evidence lines
src/artifacts.py            loads the committed runtime files
src/evaluate.py             leave-one-out harness, compare()
scripts/00..05              download, audit, embed, build, evaluate, sweep
```

`src/schema.py` is the contract. Everything downstream of `data.py` speaks
canonical names only.

## Results

Leave-one-out on positives (rating >= 4), most recent like held out, k=10,
1,000 warm and 1,000 cold cases. Every model is fitted on the same training
split with all evaluated interactions removed.

| model | NDCG@10 warm | NDCG@10 cold | coverage | p95 latency |
| --- | --- | --- | --- | --- |
| random | 0.0004 | 0.0000 | 68.8% | 22ms |
| popularity | 0.0307 | **0.0693** | 0.2% | 22ms |
| content (tf-idf) | 0.2404 | 0.0019 | 16.6% | 16ms |
| content (minilm) | 0.1166 | 0.0024 | 11.0% | 17ms |
| cohort CF | **0.4135** | 0.0693 | 15.5% | 20ms |
| hybrid (tf-idf) | 0.4115 | 0.0651 | 14.1% | 19ms |

Three findings worth stating plainly:

**TF-IDF beats MiniLM 2.1x for content scoring.** The dataset has no product
description field, so the text is brand, name, category, highlight tokens and an
INCI ingredient list — keyword soup, where exact term overlap does more work
than semantic similarity. Both are built and compared rather than assumed.

**The hybrid does not beat cohort CF alone** (0.4115 vs 0.4135). The weight
sweep is worse than that: on NDCG the optimum is degenerate, pure CF warm and
pure popularity cold. This is largely a limitation of the protocol. Held-out
items are drawn from the review table, so every correct answer is a product that
already has reviews — one of the 2,343 CF can rank. The 6,151 products with no
reviews (72% of the catalogue) can never register as a hit, so the one thing the
content layer uniquely provides is invisible to the measurement. The shipped
weights keep small content terms anyway, at a stated cost of -0.5% warm and -6%
cold NDCG.

**The leak is real and quantified.** Fitting CF on all reviews instead of the
training split inflates NDCG@10 by 3.7%. Easy to miss, and it flatters
interaction-based models specifically.

## Design decisions

- **Cohort CF was validated before it was built.** `scripts/01_inspect.py` found
  skin_type populated on 89.8% of reviews and a median product-by-skin-type
  cohort of 32, so slicing by skin type leaves cohorts with something to say.
  The documented fallback was global item-item CF; it was not needed.
- **Untagged is not unsuitable.** Only 12.4% of products carry a skin-type tag,
  so the rules fire on stated disagreement, never on a missing tag. Treating
  untagged as unsuitable would silently delete seven eighths of the catalogue.
- **Sensitivity is a concern, not a skin type.** The dataset has no
  sensitive-skin label, so it is derived from formulation flags plus an
  ingredient exclusion rule. The alcohol patterns are specific (`alcohol denat`,
  `sd alcohol`) because a bare `alcohol` match would exclude cetyl and stearyl
  alcohol, which are emollients.
- **Filters live in the base class.** Budget, category and the suitability rules
  are applied in `Recommender._apply_filters`, so no strategy — baselines
  included — can quietly skip them and flatter the comparison table.
- **Explanations are computed, never asserted.** Templates filled from columns,
  never LLM-generated, so an explanation cannot invent a claim about a product.
  A cohort percentage appears only at n >= 30 and always with its sample size;
  below that it is suppressed rather than hedged, because a percentage from
  eleven reviewers reads as authoritative whatever caveat is attached. Only
  51.8% of cohorts clear the bar, so roughly half of explanations use the
  checklist form by design.
- **Unmet criteria are shown, not hidden.** A checklist of nothing but ticks is
  marketing.
- **Embeddings are precomputed and committed.** The deployed app never imports
  sentence-transformers, keeping torch out of the runtime and p95 latency at
  ~20ms.

## Evaluation protocol

Leave-one-out on positive interactions. Where timestamps exist the most recent
like is held out, so the model predicts forward in time rather than
interpolating between known likes. Users with fewer than two positives are
excluded from the warm-start table and measured separately as cold start — that
is the path a new visitor actually takes through the UI, and averaging it in
would hide it either way.

Every model is scored through the same `compare()` call over the same cases,
baselines included. A result is only meaningful relative to those baselines.

Metrics: precision@10, recall@10, NDCG@10, MAP@10, catalogue coverage,
intra-list diversity, novelty, p50/p95 latency, empty-result rate.

## Data

Not redistributed: `data/raw/` is gitignored and `scripts/00_download.py`
reproduces it from Kaggle. `artifacts/` holds derived files only — the product
table, positive interactions, per-cohort review counts and product vectors.
