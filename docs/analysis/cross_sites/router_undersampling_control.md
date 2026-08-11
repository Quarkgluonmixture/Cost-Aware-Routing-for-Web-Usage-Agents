# Is the routing negative result undersampling?

> Regenerate: `.venv/bin/python3 scripts/analysis/router_undersampling_control.py`
> `post_hoc_exploratory=True`, `h10_eligible=False` — this answers an attack on a negative result; it is not a gate.

Triage label, 5-fold stratified CV × 20 repeats. Training rows are thinned; **test folds are never thinned**, so the curve isolates training size. AP is reported beside AUROC because these cells are 57–93% negative and AUROC alone flatters a model there.

## A + C. Learning curve, annotated vs deployment feature set

| cell | n | pos | set | AUROC @25% | @55% | @100% | Δ(25→100) | AP @100% |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `classifieds·B0` | 224 | 97 | 20 (incl. benchmark difficulty) | 0.677 | 0.698 | 0.722 | +0.045 | 0.675 |
| `classifieds·B0` | 224 | 97 | **18 (deployment)** | 0.637 | 0.663 | 0.685 | +0.048 | 0.654 |
| `reddit·B0` | 203 | 53 | 20 (incl. benchmark difficulty) | 0.731 | 0.758 | 0.781 | +0.050 | 0.665 |
| `reddit·B0` | 203 | 53 | **18 (deployment)** | 0.658 | 0.686 | 0.707 | +0.049 | 0.583 |
| `classifieds·B1` | 224 | 55 | 20 (incl. benchmark difficulty) | 0.665 | 0.699 | 0.725 | +0.060 | 0.499 |
| `classifieds·B1` | 224 | 55 | **18 (deployment)** | 0.634 | 0.673 | 0.696 | +0.062 | 0.509 |
| `reddit·B1` | 203 | 24 | 20 (incl. benchmark difficulty) | 0.742 | 0.800 | 0.846 | +0.104 | 0.534 |
| `reddit·B1` | 203 | 24 | **18 (deployment)** | 0.643 | 0.677 | 0.726 | +0.083 | 0.376 |
| `classifieds·B2` | 224 | 16 | 20 (incl. benchmark difficulty) | 0.609 | 0.617 | 0.652 | +0.043 | 0.190 |
| `classifieds·B2` | 224 | 16 | **18 (deployment)** | 0.626 | 0.650 | 0.658 | +0.032 | 0.187 |
| `reddit·B2` | 203 | 15 | 20 (incl. benchmark difficulty) | 0.683 | 0.736 | 0.757 | +0.073 | 0.488 |
| `reddit·B2` | 203 | 15 | **18 (deployment)** | 0.631 | 0.684 | 0.685 | +0.053 | 0.409 |

**Saturation.** The comparison that decides A is not the total rise but whether the increments shrink. The second half of the sweep adds more absolute rows than the first; if it nonetheless buys less AUROC, the curve is saturating and further rows of the same kind are worth progressively less.

| cell | set | Δ AUROC 25→55% | Δ AUROC 55→100% | ratio | verdict |
|---|---|---:|---:|---:|---|
| `classifieds·B0` | 20 | +0.021 | +0.024 | 1.10 | still climbing |
| `classifieds·B0` | **18** | +0.026 | +0.022 | 0.85 | saturating |
| `reddit·B0` | 20 | +0.027 | +0.023 | 0.85 | saturating |
| `reddit·B0` | **18** | +0.028 | +0.021 | 0.75 | saturating |
| `classifieds·B1` | 20 | +0.034 | +0.026 | 0.77 | saturating |
| `classifieds·B1` | **18** | +0.038 | +0.023 | 0.61 | saturating |
| `reddit·B1` | 20 | +0.058 | +0.046 | 0.78 | saturating |
| `reddit·B1` | **18** | +0.034 | +0.049 | 1.43 | still climbing |
| `classifieds·B2` | 20 | +0.008 | +0.035 | 4.37 | still climbing |
| `classifieds·B2` | **18** | +0.024 | +0.008 | 0.32 | saturating |
| `reddit·B2` | 20 | +0.052 | +0.021 | 0.40 | saturating |
| `reddit·B2` | **18** | +0.052 | +0.001 | 0.02 | saturating |

## B. In-sample separability (near-unregularised, C=1e6)

Scored on the rows it was fitted to — a memorisation test, made interpretable only by the permuted-label arm beside it.

| cell | set | train AUROC (real) | train AUROC (permuted labels) | excess | p |
|---|---|---:|---:|---:|---:|
| `classifieds·B0` | 20 | 0.796 | 0.647 (p95 0.699) | +0.148 | 0.0050 |
| `classifieds·B0` | **18** | 0.749 | 0.634 (p95 0.683) | +0.116 | 0.0050 |
| `reddit·B0` | 20 | 0.855 | 0.684 (p95 0.740) | +0.172 | 0.0050 |
| `reddit·B0` | **18** | 0.784 | 0.669 (p95 0.722) | +0.115 | 0.0050 |
| `classifieds·B1` | 20 | 0.807 | 0.668 (p95 0.719) | +0.139 | 0.0050 |
| `classifieds·B1` | **18** | 0.768 | 0.654 (p95 0.709) | +0.114 | 0.0050 |
| `reddit·B1` | 20 | 0.930 | 0.734 (p95 0.804) | +0.195 | 0.0050 |
| `reddit·B1` | **18** | 0.828 | 0.715 (p95 0.788) | +0.113 | 0.0100 |
| `classifieds·B2` | 20 | 0.844 | 0.775 (p95 0.853) | +0.069 | 0.0697 |
| `classifieds·B2` | **18** | 0.829 | 0.755 (p95 0.829) | +0.075 | 0.0547 |
| `reddit·B2` | 20 | 0.917 | 0.788 (p95 0.860) | +0.129 | 0.0050 |
| `reddit·B2` | **18** | 0.876 | 0.764 (p95 0.845) | +0.111 | 0.0050 |

## D. Benchmark size the which-mode half would need

Two classes must each retain ≥10 training rows under a 5-fold split, so each needs ≥12.5 minted labels. Labels are minted at the solvable rate, so the required benchmark size is arithmetic.

| cell | tasks now | minted | solvable % | 2nd-largest class | × needed | tasks needed |
|---|---:|---:|---:|---:|---:|---:|
| `B0_classifieds` | 224 | 97 | 43.3 | 24 | 0.5× | **117** |
| `B0_reddit` | 203 | 53 | 26.1 | 6 | 2.1× | **423** |
| `B1_classifieds` | 224 | 55 | 24.6 | 14 | 0.9× | **200** |
| `B1_reddit` | 203 | 24 | 11.8 | 4 | 3.1× | **634** |
| `B2_classifieds` | 224 | 16 | 7.1 | 5 | 2.5× | **560** |
| `B2_reddit` | 203 | 15 | 7.4 | 3 | 4.2× | **846** |
