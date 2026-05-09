# SR + FP per Mode

Standalone Outcome 0a/0b aggregation from paper-grade per-task `summary_v2.json` files (B0 + B1).

## Main Table

| baseline | site | mode | n | raw SR | adjusted SR | FP count | FP rate | FP breakdown |
|---|---|---|---:|---:|---:|---:|---:|---|

## FP rate ranking per (baseline, site)


## Method

Raw SR counts `success == true`; adjusted SR counts `adjusted_success == true` with fallback to `success` when the adjusted field is absent. FP count is raw success minus adjusted success. B1 phantom data is partial: only B1 classifieds Phantom-SoM is available (P-text pending, B1 reddit phantom pending).
