# SR + FP per Mode

Standalone Layer 0a/0b aggregation from paper-grade B0 per-task `summary_v2.json` files.

## Main Table

| site | mode | n | raw SR | adjusted SR | FP count | FP rate | FP breakdown |
|---|---|---:|---:|---:|---:|---:|---|
| reddit | DOM | 210 | 11.43% | 9.52% | 4 | 1.90% | na_fp=4 |
| reddit | P-text | 210 | 13.81% | 12.38% | 3 | 1.43% | na_fp=3 |
| reddit | Phantom-SoM | 210 | 14.29% | 13.81% | 1 | 0.48% | na_fp=1 |
| reddit | SoM | 210 | 11.90% | 10.48% | 3 | 1.43% | na_fp=3 |
| reddit | Vision | 210 | 8.57% | 6.67% | 4 | 1.90% | na_fp=4 |
| classifieds | DOM | 234 | 14.96% | 14.10% | 2 | 0.85% | na_fp=2 |
| classifieds | P-text | 234 | 16.67% | 14.53% | 5 | 2.14% | na_fp=5 |
| classifieds | Phantom-SoM | 234 | 15.81% | 14.53% | 3 | 1.28% | na_fp=3 |
| classifieds | SoM | 234 | 23.08% | 21.37% | 4 | 1.71% | na_fp=4 |
| classifieds | Vision | 234 | 15.81% | 13.68% | 5 | 2.14% | na_fp=5 |

## FP rate ranking per site

- reddit: Phantom-SoM 0.48% < P-text 1.43% < SoM 1.43% < DOM 1.90% < Vision 1.90%
- classifieds: DOM 0.85% < Phantom-SoM 1.28% < SoM 1.71% < P-text 2.14% < Vision 2.14%

## Method

Raw SR counts `success == true`; adjusted SR counts `adjusted_success == true` with fallback to `success` when the adjusted field is absent. FP count is raw success minus adjusted success.
