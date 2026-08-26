# Project context

Internship assignment for Orbo.AI (Mumbai beauty-AI company: skin analysis,
virtual try-on, foundation shade finder, "BeautyGPT" recommender).

**Deadline: Thursday 27 August, end of day.** Roughly two working days. Scope
discipline matters more than sophistication — the brief explicitly values
"thoughtful engineering and clarity of execution over the most complex
solution."

## What we are building

A deployed web app where a user enters a skin profile (type, tone, concerns,
budget, category) and receives ~10 ranked beauty products, each with a
plain-English explanation of why it was chosen.

Beauty was chosen deliberately over the obvious MovieLens-style project because
it mirrors Orbo's actual product.

## Required deliverables (all three are mandatory)

1. Public GitHub repo — code, setup instructions, sample data
2. Live deployment link, usable without local setup
3. Documentation — problem statement, architecture, methodology, dataset,
   assumptions, design decisions, evaluation, test cases (success *and*
   failure), limitations, future work

Bonus: UI inspired by Sephora/Nykaa plus a written comparison
(similarities, differences, limitations, what we would build next).

## Architecture

Three scoring layers, then a reranker:

1. **Content-based** — embeddings over product text (description, ingredients,
   highlights) plus hard rule filters (e.g. drop fragrance for sensitive skin).
   Handles cold-start products.
2. **Collaborative** — cohort-scoped: users sharing a skin profile, what they
   rated highly. Captures behaviour that product text cannot express.
3. **Hybrid reranker** — blends both scores, applies MMR for intra-list
   diversity, adds a small popularity prior, enforces budget/category.
4. **Explanation layer** — template-based, never LLM-generated. Fast, and it
   cannot hallucinate a claim about a product.

## Decisions already made — do not silently reverse these

- **The collaborative design is provisional.** It assumes reviewer skin-type
  and skin-tone fields are densely populated. `scripts/01_inspect.py` decides.
  If the audit verdict is negative, fall back to item-item CF over the full
  review matrix with profile fields used only as filters. Do not force cohort
  CF onto data that cannot support it.
- **Explanations must be computed, never asserted.** A cohort statistic
  ("X% of oily-skin reviewers rated this 4+") may only be shown when the cohort
  has >= 30 reviews, and the sample size must be displayed alongside it. Below
  that threshold, fall back to the checklist form (matches skin type, addresses
  concern, contains ingredient, average rating, within budget). The `Evidence`
  dataclass has a `detail` field for exactly this.
- **Embeddings are precomputed offline.** The transformer runs once locally and
  vectors are committed as `.npy`. The deployed app must not import
  sentence-transformers — it keeps the runtime footprint small enough for a
  free tier and inference under 100ms. Keep heavy deps in a separate
  requirements-dev.txt.
- **Filters live in `Recommender._apply_filters`, in the base class.** Every
  strategy including the baselines must honour budget and category identically,
  or the comparison table is dishonest.
- **Selfie / face analysis is out of scope.** Tempting given Orbo's domain, but
  the brief does not ask for computer vision. Only consider it if everything
  required is finished and deployed.

## Evaluation

Leave-one-out on positive interactions (rating >= 4). Where timestamps exist,
hold out the user's *most recent* like so the model predicts forward in time
rather than interpolating between known likes.

Report every model — random, popularity, content-only, CF-only, hybrid —
through the same `compare()` call over the same cases. Numbers are meaningless
without the baselines beside them. Popularity is a strong baseline in sparse
retail data; the hybrid has to beat it on NDCG *without* catalogue coverage
collapsing.

Metrics: precision@10, recall@10, NDCG@10, MAP@10, catalog coverage, intra-list
diversity, novelty, p50/p95 latency, empty-result rate.

## Layout

```
app.py                 Streamlit entrypoint (must stay at repo root)
src/__init__.py        makes src a package — required, the modules use
                       relative imports and break without it
src/schema.py          canonical column names, Query, Scored, Evidence
src/metrics.py         pure metric functions, no dataset dependency
src/recommender.py     Recommender ABC + random and popularity baselines
src/evaluate.py        leave-one-out harness, compare()
scripts/01_inspect.py  dataset audit — run this before writing any model
data/raw/              Sephora CSVs, gitignored
artifacts/             precomputed embeddings, gitignored
reports/               audit output, evaluation tables
```

`src/schema.py` is the contract. Only the data loader may know what the raw
Kaggle columns are actually named; everything downstream uses canonical names.

## Order of work

Deploy a trivial version first, then: dataset audit -> working recommender ->
evaluation -> UI -> full deployment -> documentation -> polish -> optional
bonus. Do not leave deployment or documentation until Thursday; they carry more
marks than model refinement.

## Current state

Done: schema, metrics, evaluation harness, baselines, audit script, hello-world
Streamlit app. Harness verified against synthetic data — popularity beat random
17x on NDCG while coverage fell to 3.8%, confirming the metrics are wired
correctly.

Blocked on the dataset audit: content model, collaborative model, hybrid,
explanations.
