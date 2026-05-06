"""Mechanistic interpretability pipeline for B1 (Qwen3-VL-4B) mirage feature analysis.

Per advisor 5/5 sync: in-house contrastive set (P-text / P-SoM / DOM × site × model)
already supports activation patching + linear probe + (future) SAE feature steering.

Pipeline stages:
- Stage 1 (linear probe): validate "mirage info linearly separable in hidden states"
- Stage 2 (activation patching): identify mirage critical layer via causal patch
- Stage 3 (SAE feature, future): direct steer mirage feature (deferred — public Qwen3-VL SAE 不存在)

Reference: Tool Calling Linear Steerable Circuit (ACL 2026, Qwen3-4B) is the
B1-side method template anchor — per-step hidden state → PCA action direction
→ cosine gap → AUROC.
"""

__all__ = ["HiddenStateExtractor", "linear_probe_per_layer", "plot_auroc_curve"]

from p79.mechanistic.extract_hidden_states import HiddenStateExtractor
from p79.mechanistic.linear_probe import linear_probe_per_layer, plot_auroc_curve
