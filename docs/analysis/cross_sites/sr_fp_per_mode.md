# SR + FP per Mode

Standalone Outcome 0a/0b aggregation from paper-grade per-task `summary_v2.json` files (B0 + B1).

## Main Table

| baseline | site | mode | n | raw SR | adjusted SR | FP count | FP rate | FP breakdown |
|---|---|---|---:|---:|---:|---:|---:|---|
| B0 | reddit | DOM | 210 | 11.43% | 9.52% | 4 | 1.90% | na_fp=4 |
| B0 | reddit | P-text | 210 | 13.81% | 12.38% | 3 | 1.43% | na_fp=3 |
| B0 | reddit | Phantom-prompt | 210 | 10.48% | 9.52% | 2 | 0.95% | na_fp=2 |
| B0 | reddit | Phantom-SoM | 210 | 14.29% | 13.81% | 1 | 0.48% | na_fp=1 |
| B0 | reddit | SoM | 210 | 11.90% | 10.48% | 3 | 1.43% | na_fp=3 |
| B0 | reddit | Vision | 210 | 8.57% | 6.67% | 4 | 1.90% | na_fp=4 |
| B0 | classifieds | DOM | 234 | 14.96% | 14.10% | 2 | 0.85% | na_fp=2 |
| B0 | classifieds | P-text | 234 | 16.67% | 14.53% | 5 | 2.14% | na_fp=5 |
| B0 | classifieds | Phantom-SoM | 234 | 15.81% | 14.53% | 3 | 1.28% | na_fp=3 |
| B0 | classifieds | SoM | 234 | 23.08% | 21.37% | 4 | 1.71% | na_fp=4 |
| B0 | classifieds | Vision | 234 | 15.81% | 13.68% | 5 | 2.14% | na_fp=5 |
| B1 | reddit | DOM | 210 | 10.00% | 7.62% | 5 | 2.38% | na_fp=5 |
| B1 | reddit | SoM | 210 | 8.10% | 5.71% | 5 | 2.38% | na_fp=5 |
| B1 | reddit | Vision | 210 | 4.76% | 2.38% | 5 | 2.38% | na_fp=5 |
| B1 | classifieds | DOM | 234 | 11.11% | 8.55% | 6 | 2.56% | na_fp=6 |
| B1 | classifieds | Phantom-SoM | 234 | 10.26% | 7.69% | 6 | 2.56% | na_fp=6 |
| B1 | classifieds | SoM | 234 | 17.52% | 13.68% | 9 | 3.85% | na_fp=9 |
| B1 | classifieds | Vision | 234 | 11.11% | 7.26% | 9 | 3.85% | na_fp=9 |

## FP rate ranking per (baseline, site)

- B0 reddit: Phantom-SoM 0.48% < Phantom-prompt 0.95% < P-text 1.43% < SoM 1.43% < DOM 1.90% < Vision 1.90%
- B0 classifieds: DOM 0.85% < Phantom-SoM 1.28% < SoM 1.71% < P-text 2.14% < Vision 2.14%
- B1 reddit: DOM 2.38% < SoM 2.38% < Vision 2.38%
- B1 classifieds: DOM 2.56% < Phantom-SoM 2.56% < SoM 3.85% < Vision 3.85%

## Method

Raw SR counts `success == true`; adjusted SR counts `adjusted_success == true` with fallback to `success` when the adjusted field is absent. FP count is raw success minus adjusted success. B1 phantom data is partial: only B1 classifieds Phantom-SoM is available (P-text pending, B1 reddit phantom pending).
