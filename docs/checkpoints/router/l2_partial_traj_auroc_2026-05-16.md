# L2 partial-trajectory AUROC — F1 defuse

> v5 §3.1 claims fig0g `ep_mean_verbalized` AUROC ≥ 0.7 anchors L2 runtime trigger. But fig0g uses *full-episode* mean. L2 fires at step ≥ 3 with *partial* trajectory. This script computes prefix-k AUROC for k ∈ {1, 2, 3, 5, 8, full}.

**Hypothesis**: if step-3 AUROC ≥ 0.65 → v5 anchor partially salvaged (with calibrated threshold). If step-3 AUROC < 0.6 → v5 L2 verbose trigger 不能 paper-grade 落地.

## AUROC by prefix-k per cell

| Cell | n_tasks | n_no_verb | k=1 | k=2 | k=3 | k=5 | k=8 | k=full | k=3 vs full Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_classifieds_dom | 234 | 0 | 0.555 | 0.547 | 0.615 | 0.644 | 0.677 | 0.686 | -0.070 |
| B0_classifieds_som | 234 | 0 | 0.480 | 0.554 | 0.642 | 0.681 | 0.687 | 0.688 | -0.047 |
| B0_classifieds_vision | 234 | 0 | 0.527 | 0.631 | 0.625 | 0.648 | 0.668 | 0.682 | -0.057 |
| B0_classifieds_phantom_text | 234 | 0 | 0.597 | 0.504 | 0.586 | 0.644 | 0.670 | 0.670 | -0.085 |
| B0_classifieds_phantom_prompt | 4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| B0_classifieds_phantom_som | 234 | 0 | 0.510 | 0.541 | 0.594 | 0.654 | 0.638 | 0.657 | -0.063 |
| B0_reddit_dom | 209 | 1 | 0.571 | 0.589 | 0.677 | 0.722 | 0.742 | 0.737 | -0.060 |
| B0_reddit_som | 199 | 11 | 0.550 | 0.533 | 0.556 | 0.625 | 0.656 | 0.666 | -0.110 |
| B0_reddit_vision | 210 | 0 | 0.475 | 0.536 | 0.574 | 0.593 | 0.645 | 0.662 | -0.088 |
| B0_reddit_phantom_text | 208 | 2 | 0.518 | 0.564 | 0.680 | 0.716 | 0.726 | 0.733 | -0.054 |
| B0_reddit_phantom_prompt | 209 | 1 | 0.554 | 0.614 | 0.698 | 0.772 | 0.788 | 0.788 | -0.090 |
| B0_reddit_phantom_som | 205 | 5 | 0.518 | 0.577 | 0.629 | 0.690 | 0.713 | 0.706 | -0.077 |

## Summary — verdict on v5 §3.1 L2 trigger anchor

- Cells where k=3 AUROC < 0.65 (anchor not viable): **8**
- Cells where full-episode AUROC ≥ 0.7 (v5 cited threshold): 4

**Verdict**: if `k=3 AUROC < 0.65` in majority of cells → v5 §3.1 L2 verbose AUROC anchor is category error. L2 falls back to cycle-only triggers (max_repeat / url_revisit) which are more directly computable at runtime.