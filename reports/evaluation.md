# Evaluation

Leave-one-out on positive interactions (rating >= 4), k=10. Embedding methods compared: `minilm`, `tfidf`. Intra-list diversity is measured in the `minilm` space for every row, so the column stays comparable across models.

Catalogue: 8,494 products, of which 2,351 (27.7%) have any review.

Every model below is fitted on the same training split, with all 2,000 evaluated interactions removed.

## Headline

- **Best warm-start model: `cf-only`** at ndcg@10 0.4135, 13.5x the popularity baseline (0.0307), with catalogue coverage 15.5% against popularity's 0.2%.
- **The hybrid does not beat cohort CF alone** on warm start: `hybrid-tfidf` 0.4115 vs `cf-only` 0.4135 (-0.5%). See 'What this protocol cannot measure' below before reading that as a verdict on the content layer.
- **tfidf beats minilm for content scoring** (0.2404 vs 0.1166, 2.1x). The product text is highlight tokens and INCI ingredient lists rather than prose, so exact term overlap does more work here than semantic similarity.
- **Cold start is hard and everything collapses toward popularity.** Best is `popularity` at 0.0693; the content layers score near zero because a profile with no history gives them almost nothing to work with.
- **Leak check `popularity`:** training on all reviews inflates ndcg@10 by +1.9% (0.0307 -> 0.0313).
- **Leak check `cf-only`:** training on all reviews inflates ndcg@10 by +3.7% (0.4135 -> 0.4290).
- **Latency:** p95 26ms across all models, inside the 100ms budget. Embeddings are precomputed, so no model does heavy work per request.

## Warm start

Users with 2+ positive interactions (1,000 cases). Tests ranking.

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| random         | 0.0001       | 0.0010    | 0.0004  | 0.0002 | 0.6877   | 16.0           | 26.1           | 0.0000            | 1,000   | 0.4870    | 8.2332  |
| popularity     | 0.0060       | 0.0600    | 0.0307  | 0.0220 | 0.0021   | 12.5           | 18.8           | 0.0000            | 1,000   | 0.4132    | -2.5979 |
| cf-only        | 0.0466       | 0.4660    | 0.4135  | 0.3974 | 0.1552   | 13.4           | 20.0           | 0.0000            | 1,000   | 0.3150    | 0.0292  |
| content-minilm | 0.0147       | 0.1470    | 0.1166  | 0.1072 | 0.1103   | 13.4           | 17.1           | 0.0000            | 1,000   | 0.1855    | 3.8933  |
| content-tfidf  | 0.0283       | 0.2830    | 0.2404  | 0.2269 | 0.1659   | 13.0           | 16.5           | 0.0000            | 1,000   | 0.2485    | 3.7614  |
| hybrid-minilm  | 0.0460       | 0.4600    | 0.4108  | 0.3957 | 0.1291   | 15.7           | 19.9           | 0.0000            | 1,000   | 0.3002    | -0.4076 |
| hybrid-tfidf   | 0.0459       | 0.4590    | 0.4115  | 0.3967 | 0.1407   | 15.4           | 19.1           | 0.0000            | 1,000   | 0.3112    | -0.1308 |

## Cold start

Users with exactly one positive, hidden (1,000 cases). This is the path a new visitor takes through the UI: profile only, no history.

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| random         | 0.0000       | 0.0000    | 0.0000  | 0.0000 | 0.6858   | 12.0           | 18.9           | 0.0000            | 1,000   | 0.4843    | 8.2190  |
| popularity     | 0.0110       | 0.1100    | 0.0693  | 0.0570 | 0.0014   | 10.8           | 17.5           | 0.0000            | 1,000   | 0.4141    | -2.6111 |
| cf-only        | 0.0103       | 0.1030    | 0.0693  | 0.0587 | 0.0024   | 11.9           | 17.5           | 0.0000            | 1,000   | 0.3389    | -2.5223 |
| content-minilm | 0.0006       | 0.0060    | 0.0024  | 0.0013 | 0.0038   | 12.5           | 15.8           | 0.0000            | 1,000   | 0.1673    | 5.0303  |
| content-tfidf  | 0.0003       | 0.0030    | 0.0019  | 0.0016 | 0.0038   | 12.1           | 15.7           | 0.0000            | 1,000   | 0.2273    | 4.9320  |
| hybrid-minilm  | 0.0097       | 0.0970    | 0.0665  | 0.0570 | 0.0021   | 15.4           | 21.7           | 0.0000            | 1,000   | 0.3267    | -2.5496 |
| hybrid-tfidf   | 0.0096       | 0.0960    | 0.0651  | 0.0555 | 0.0021   | 14.6           | 20.3           | 0.0000            | 1,000   | 0.3447    | -2.5237 |

## What this protocol cannot measure

Held-out items are drawn from the review table, so every correct answer here is by construction a product that already has reviews — one of the 2,351 that cohort CF can rank. The other 6,143 products (72.3% of the catalogue) have no interactions and can never register as a hit.

That matters for how the table above should be read. The content layer exists to cover exactly those products, and an offline metric built on interaction data structurally cannot reward covering items that have no interaction data. `reports/weight_sweep.md` shows the consequence: on NDCG alone the optimal blend is degenerate — pure CF warm, pure popularity cold. The shipped weights keep small content terms anyway, at a measured cost of -0.4% warm and -9% cold NDCG, because the alternative returns roughly a dozen distinct products across a thousand cold-start users.

## The leak, measured

The same two interaction-based models refitted on *all* reviews, so each has seen the interaction it is asked to predict. Compare against the warm-start table above: the gap is the size of the mistake avoided by splitting properly.

| model            | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|------------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| popularity-LEAKY | 0.0062       | 0.0620    | 0.0313  | 0.0222 | 0.0021   | 14.3           | 23.6           | 0.0000            | 1,000   | 0.4128    | -2.5979 |
| cf-only-LEAKY    | 0.0495       | 0.4950    | 0.4290  | 0.4088 | 0.1551   | 11.6           | 21.9           | 0.0000            | 1,000   | 0.3147    | 0.0278  |
