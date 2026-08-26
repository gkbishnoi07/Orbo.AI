# Data audit

## Files
- `product_info.csv` — 8,494 rows x 27 cols
- `reviews_0-250.csv` — 602,130 rows x 19 cols
- `reviews_1250-end.csv` — 49,977 rows x 19 cols
- `reviews_250-500.csv` — 206,725 rows x 19 cols
- `reviews_500-750.csv` — 116,262 rows x 19 cols
- `reviews_750-1250.csv` — 119,317 rows x 19 cols

## Products
- 8,494 products

- ingredients column `ingredients`: 88.9% populated
- comma-separated and parseable in 97.6% of populated rows

## Reviews
- 1,094,411 reviews

### Profile field completeness
- `skin_type`: 89.8% populated, 4 distinct values
- `skin_tone`: 84.4% populated, 14 distinct values

### Interaction density
- 503,216 distinct users, 2,351 distinct products
- reviews per user: median 1, share with 2+ 41.8%
- reviews per product: median 164, share with 5+ 94.6%
- matrix density: 0.09%

### Cohort sizes (product x skin type)
- median cohort: 32 reviews
- share of cohorts with 30+ reviews: 51.8%

Cohort statistics may only be shown when n >= 30; everything below that falls back to the checklist explanation.

## Verdict
Cohort collaborative filtering is **viable as designed**.