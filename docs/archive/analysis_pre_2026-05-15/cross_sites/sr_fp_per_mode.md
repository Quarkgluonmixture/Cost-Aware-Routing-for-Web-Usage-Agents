# SR + FP per Mode

Standalone Outcome 0a/0b aggregation from paper-grade per-task `summary_v2.json` files (B0 + B1).

## Main Table

| baseline | site | mode | n | raw SR | adjusted SR | FP count | FP rate | FP breakdown |
|---|---|---|---:|---:|---:|---:|---:|---|

## FP rate ranking per (baseline, site)


## Method

§139.8: the adjusted_success post-hoc layer is retired — `success` is the canonical paper-grade outcome (na_fp / eval_fp fixed at the source: B-91 evaluator empty-pred guard + N/A task exclusion at load). Raw SR == adjusted SR; FP count is structurally 0. The dual columns are kept for schema stability. B1 phantom data is partial: only B1 classifieds Phantom-SoM is available (P-text pending, B1 reddit phantom pending).
