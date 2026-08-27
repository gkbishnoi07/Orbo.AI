# Evaluation

Leave-one-out on positive interactions (rating >= 4), k=10. Embedding methods compared: `minilm`, `tfidf`. Intra-list diversity is measured in the `tfidf` space for every row, so the column stays comparable across models.

Catalogue: 8,494 products, of which 2,351 (27.7%) have any review.

Every model below is fitted on the same training split, with all 2,000 evaluated interactions removed.

## Headline

- **Best warm-start model: `cf-only`** at ndcg@10 0.4135, 13.5x the popularity baseline (0.0307), with catalogue coverage 15.5% against popularity's 0.2%.
- **The hybrid does not beat cohort CF alone** on warm start: `hybrid-tfidf` 0.4112 vs `cf-only` 0.4135 (-0.5%). See 'What this protocol cannot measure' below before reading that as a verdict on the content layer.
- **tfidf beats minilm for content scoring** (0.2404 vs 0.1166, 2.1x). The product text is highlight tokens and INCI ingredient lists rather than prose, so exact term overlap does more work here than semantic similarity.
- **Cold start is hard and everything collapses toward popularity.** Best is `popularity` at 0.0693; the content layers score near zero because a profile with no history gives them almost nothing to work with.
- **Leak check `popularity`:** training on all reviews inflates ndcg@10 by +1.9% (0.0307 -> 0.0313).
- **Leak check `cf-only`:** training on all reviews inflates ndcg@10 by +3.7% (0.4135 -> 0.4290).
- **Model inference latency:** p95 14ms across all models, inside the 100ms budget. This is `Recommender.recommend()` only — it excludes explanation generation and all Streamlit rendering, so it is not end-to-end user latency.

## Warm start

Users with 2+ positive interactions (1,000 cases). Tests ranking.

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| random         | 0.0001       | 0.0010    | 0.0004  | 0.0002 | 0.6877   | 6.7            | 8.0            | 0.0000            | 1,000   | 0.9037    | 8.2332  |
| popularity     | 0.0060       | 0.0600    | 0.0307  | 0.0220 | 0.0021   | 6.1            | 7.3            | 0.0000            | 1,000   | 0.8744    | -2.5979 |
| cf-only        | 0.0466       | 0.4660    | 0.4135  | 0.3974 | 0.1552   | 6.3            | 7.5            | 0.0000            | 1,000   | 0.7482    | 0.0292  |
| content-minilm | 0.0147       | 0.1470    | 0.1166  | 0.1072 | 0.1103   | 8.9            | 10.8           | 0.0000            | 1,000   | 0.6462    | 3.8933  |
| content-tfidf  | 0.0283       | 0.2830    | 0.2404  | 0.2269 | 0.1659   | 8.7            | 10.7           | 0.0000            | 1,000   | 0.5212    | 3.7614  |
| hybrid-minilm  | 0.0460       | 0.4600    | 0.4110  | 0.3959 | 0.1297   | 11.3           | 14.1           | 0.0000            | 1,000   | 0.7558    | -0.3908 |
| hybrid-tfidf   | 0.0458       | 0.4580    | 0.4112  | 0.3967 | 0.1419   | 11.0           | 13.5           | 0.0000            | 1,000   | 0.7396    | -0.1209 |

## Cold start

Users with exactly one positive, hidden (1,000 cases). This is the path a new visitor takes through the UI: profile only, no history.

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| random         | 0.0000       | 0.0000    | 0.0000  | 0.0000 | 0.6858   | 6.0            | 7.2            | 0.0000            | 1,000   | 0.9045    | 8.2190  |
| popularity     | 0.0110       | 0.1100    | 0.0693  | 0.0570 | 0.0014   | 5.4            | 6.5            | 0.0000            | 1,000   | 0.8757    | -2.6111 |
| cf-only        | 0.0103       | 0.1030    | 0.0693  | 0.0587 | 0.0024   | 5.7            | 6.9            | 0.0000            | 1,000   | 0.8377    | -2.5223 |
| content-minilm | 0.0006       | 0.0060    | 0.0024  | 0.0013 | 0.0038   | 8.6            | 10.6           | 0.0000            | 1,000   | 0.6098    | 5.0303  |
| content-tfidf  | 0.0003       | 0.0030    | 0.0019  | 0.0016 | 0.0038   | 8.3            | 10.2           | 0.0000            | 1,000   | 0.4788    | 4.9320  |
| hybrid-minilm  | 0.0097       | 0.0970    | 0.0665  | 0.0570 | 0.0021   | 10.7           | 13.0           | 0.0000            | 1,000   | 0.8141    | -2.5496 |
| hybrid-tfidf   | 0.0096       | 0.0960    | 0.0653  | 0.0557 | 0.0025   | 10.5           | 12.9           | 0.0000            | 1,000   | 0.8346    | -2.5160 |

## What this protocol cannot measure

Held-out items are drawn from the review table, so every correct answer here is by construction a product that already has reviews — one of the 2,351 that cohort CF can rank. The other 6,143 products (72.3% of the catalogue) have no interactions and can never register as a hit.

That matters for how the table above should be read. The content layer exists to cover exactly those products, and an offline metric built on interaction data structurally cannot reward covering items that have no interaction data. `reports/weight_sweep.md` shows the consequence: on NDCG alone the optimal blend is degenerate — pure CF warm, pure popularity cold. The shipped weights keep small content terms anyway, at a measured cost of -0.4% warm and -9% cold NDCG, because the alternative returns roughly a dozen distinct products across a thousand cold-start users.

## The leak, measured

The same two interaction-based models refitted on *all* reviews, so each has seen the interaction it is asked to predict. Compare against the warm-start table above: the gap is the size of the mistake avoided by splitting properly.

| model            | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|------------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| popularity-LEAKY | 0.0062       | 0.0620    | 0.0313  | 0.0222 | 0.0021   | 6.5            | 8.5            | 0.0000            | 1,000   | 0.8756    | -2.5979 |
| cf-only-LEAKY    | 0.0495       | 0.4950    | 0.4290  | 0.4088 | 0.1551   | 6.2            | 7.5            | 0.0000            | 1,000   | 0.7481    | 0.0278  |
