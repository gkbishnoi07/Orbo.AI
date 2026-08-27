# Hybrid weight sweep

400 warm cases, 400 cold cases, k=10, embeddings `tfidf`. Sub-models fitted once on the training split; only the blend weights change between rows. The sweep model carries no skin-tone layer, so every row here is content / CF / popularity only — the shipped hybrid has a fourth term and its recorded numbers live in `reports/evaluation.json`.

## Warm-start blend

Rows are `content / cf / popularity`.

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| 0.40/0.45/0.15 | 0.0415       | 0.4150    | 0.3731  | 0.3601 | 0.1020   | 13.6           | 21.3           | 0.0000            | 400     | 0.6719    | -1.1414 |
| 0.30/0.60/0.10 | 0.0420       | 0.4200    | 0.3728  | 0.3581 | 0.1053   | 13.7           | 20.5           | 0.0000            | 400     | 0.6989    | -1.1995 |
| 0.20/0.70/0.10 | 0.0435       | 0.4350    | 0.3787  | 0.3613 | 0.0969   | 13.6           | 19.2           | 0.0000            | 400     | 0.7343    | -1.6192 |
| 0.10/0.85/0.05 | 0.0442       | 0.4425    | 0.3820  | 0.3633 | 0.0974   | 13.2           | 21.9           | 0.0000            | 400     | 0.7613    | -1.6271 |
| 0.05/0.95/0.00 | 0.0445       | 0.4450    | 0.3838  | 0.3649 | 0.1243   | 14.8           | 35.5           | 0.0000            | 400     | 0.7658    | -0.8991 |
| 0.00/1.00/0.00 | 0.0440       | 0.4400    | 0.3835  | 0.3659 | 0.1141   | 12.4           | 17.1           | 0.0000            | 400     | 0.8123    | -1.0383 |

Best NDCG@10: **0.05/0.95/0.00**.

## Cold-start blend

| model          | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|----------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| 0.30/0.40/0.30 | 0.0055       | 0.0550    | 0.0285  | 0.0207 | 0.0031   | 12.5           | 20.3           | 0.0000            | 400     | 0.7619    | -3.3012 |
| 0.20/0.40/0.40 | 0.0077       | 0.0775    | 0.0446  | 0.0341 | 0.0027   | 11.5           | 17.0           | 0.0000            | 400     | 0.8044    | -3.6064 |
| 0.10/0.30/0.60 | 0.0090       | 0.0900    | 0.0585  | 0.0486 | 0.0021   | 11.8           | 19.2           | 0.0000            | 400     | 0.8335    | -3.8417 |
| 0.05/0.15/0.80 | 0.0097       | 0.0975    | 0.0618  | 0.0506 | 0.0018   | 12.5           | 18.5           | 0.0000            | 400     | 0.8653    | -3.9234 |
| 0.00/0.00/1.00 | 0.0112       | 0.1125    | 0.0663  | 0.0523 | 0.0014   | 11.7           | 19.1           | 0.0000            | 400     | 0.8760    | -3.9334 |

Best NDCG@10: **0.00/0.00/1.00**.

## MMR lambda

The relevance-for-variety trade, measured. Diversity is the column this is bought with; NDCG is the column it is paid for from.

Check the configuration before comparing these rows against numbers reported elsewhere. This section runs on warm cases with the blend left at `0.05/0.95/0.00` — the NDCG winner of the grid above, not the shipped weights — and, like every row in this file, without the skin-tone layer. So it measures MMR on top of *that* blend. The shipped hybrid's own diversity is in `reports/evaluation.json`, and it is a lower number for that reason, not because the metric changed.

| model       | precision@10 | recall@10 | ndcg@10 | map@10 | coverage | latency_p50_ms | latency_p95_ms | empty_result_rate | n_cases | diversity | novelty |
|-------------|--------------|-----------|---------|--------|----------|----------------|----------------|-------------------|---------|-----------|---------|
| lambda=1.0  | 0.0442       | 0.4425    | 0.3833  | 0.3650 | 0.1196   | 12.7           | 17.4           | 0.0000            | 400     | 0.7033    | -0.8498 |
| lambda=0.85 | 0.0445       | 0.4450    | 0.3838  | 0.3649 | 0.1243   | 14.1           | 29.9           | 0.0000            | 400     | 0.7658    | -0.8991 |
| lambda=0.75 | 0.0435       | 0.4350    | 0.3801  | 0.3630 | 0.1227   | 12.4           | 17.3           | 0.0000            | 400     | 0.7801    | -0.9055 |
| lambda=0.5  | 0.0425       | 0.4250    | 0.3756  | 0.3604 | 0.1275   | 13.7           | 19.5           | 0.0000            | 400     | 0.8137    | -0.8457 |
