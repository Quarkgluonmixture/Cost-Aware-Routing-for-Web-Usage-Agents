"""Activation patching via PyTorch forward hooks (Stage 2 — causal mechanism analysis).

Advisor 5/5 instruction: "patch 到哪一层的时候, 它的结果就切换了" — find the layer
where source-into-target hidden-state injection causes output to flip. That's
the mirage critical layer.

Hand-rolled (not nnsight) — nnsight wheel build failed on aarch64/GB10. PyTorch
register_forward_hook gives equivalent control with no extra dependency.

Convention:
- "source" run = ground-truth condition (e.g. SoM with image — no mirage)
- "target" run = perturbed condition (e.g. P-SoM no image — mirage induced)
- Patch source's layer-L last-token hidden state into target run at same layer.
- If patched output flips toward source's behavior → layer L carries source info.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_transformer_layers(model) -> torch.nn.ModuleList:
    """Locate transformer decoder layer ModuleList in Qwen3-VL.

    Tested on Qwen3VLForConditionalGeneration (verified 2026-05-06):
        model.model.language_model.layers — 36 × Qwen3VLTextDecoderLayer
    """
    return model.model.language_model.layers


class ActivationPatcher:
    """Cache + patch interface for transformer layer outputs."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.layers = get_transformer_layers(model)
        self.n_layers = len(self.layers)

    @torch.no_grad()
    def cache_hidden_states(self, **inputs) -> list[torch.Tensor]:
        """Forward inputs and return per-layer post-block hidden states.

        Returns:
            list of (batch, seq_len, hidden_dim) tensors, length = n_layers.
            Tensors are detached + cloned (CPU or device, matches model device).
        """
        cached: list[Optional[torch.Tensor]] = [None] * self.n_layers
        hooks = []
        for i, layer in enumerate(self.layers):
            def hook(module, layer_input, layer_output, idx=i):
                hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                cached[idx] = hs.detach().clone()
            hooks.append(layer.register_forward_hook(hook))
        try:
            self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            for h in hooks:
                h.remove()
        return cached

    @torch.no_grad()
    def patched_generate(
        self,
        layer_idx: int,
        source_hidden: torch.Tensor,
        max_new_tokens: int = 30,
        **inputs,
    ) -> torch.Tensor:
        """Patch last-token hidden state at layer_idx on FIRST forward, then greedy-generate.

        With use_cache=True, the first forward processes full input (seq_len = N input
        tokens). The hook only fires for this first forward — subsequent forwards
        process 1-token-at-a-time and shouldn't be patched (they're new generated content,
        not source's input). Patched first-token hidden state propagates through KV cache
        so subsequent generations attend to it.

        Returns:
            Generated token IDs (1D tensor, only generated portion not input).
        """
        layer = self.layers[layer_idx]
        src = source_hidden.to(self.model.device)
        fire_count = [0]

        def hook(module, layer_input, layer_output):
            fire_count[0] += 1
            if fire_count[0] > 1:
                return None  # subsequent forwards: pass through unchanged
            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            hs_patched = hs.clone()
            hs_patched[:, -1, :] = src[:, -1, :]
            if isinstance(layer_output, tuple):
                return (hs_patched,) + layer_output[1:]
            return hs_patched

        h = layer.register_forward_hook(hook)
        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
            )
        finally:
            h.remove()

        input_len = inputs["input_ids"].shape[1]
        return out.sequences[0, input_len:]

    @torch.no_grad()
    def patched_forward(
        self,
        layer_idx: int,
        source_hidden: torch.Tensor,
        position: str = "last",
        **inputs,
    ):
        """Forward inputs with hook on layer_idx that swaps hidden state.

        Args:
            layer_idx: which transformer block to patch at (0 .. n_layers-1)
            source_hidden: source-run cached hidden state at same layer
                (batch, source_seq_len, hidden_dim)
            position: 'last' = patch only last-token position (works across
                different seq_len between source/target); 'all' = swap entire
                sequence (requires matching seq_len)
            **inputs: target run inputs (model kwargs)

        Returns:
            model output dict (.logits at last position used for downstream metric)
        """
        layer = self.layers[layer_idx]
        src = source_hidden.to(self.model.device)

        def hook(module, layer_input, layer_output):
            hs = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            hs_patched = hs.clone()
            if position == "last":
                hs_patched[:, -1, :] = src[:, -1, :]
            elif position == "all":
                if hs.shape != src.shape:
                    raise ValueError(
                        f"shape mismatch for position='all': target {hs.shape} vs source {src.shape}"
                    )
                hs_patched = src
            else:
                raise ValueError(f"unknown position={position!r}")
            if isinstance(layer_output, tuple):
                return (hs_patched,) + layer_output[1:]
            return hs_patched

        h = layer.register_forward_hook(hook)
        try:
            output = self.model(**inputs, use_cache=False, return_dict=True)
        finally:
            h.remove()
        return output


@torch.no_grad()
def patching_grid(
    patcher: ActivationPatcher,
    source_inputs: dict,
    target_inputs: dict,
    layers: Optional[list[int]] = None,
) -> dict:
    """Per-layer source-into-target patching, return causal-effect metrics.

    Standard activation patching protocol (Meng et al. 2022 ROME-style):
    1. Cache source's per-layer hidden states (source_cache[L])
    2. Run target unperturbed → baseline target distribution
    3. Run source unperturbed → baseline source distribution
    4. For each L: forward target with hook injecting source_cache[L] at last
       token → patched distribution. Compare to source/target baselines.

    Metrics:
        argmax_match_source: 1 if patched_argmax == source_argmax else 0
        logit_shift_to_source: (patched_logit_src - target_logit_src) /
                              (source_logit_src - target_logit_src). 1.0 = full
                              shift to source, 0.0 = no shift.
        kl_patched_to_source: KL(patched || source). Lower = closer to source.
        kl_patched_to_target: KL(patched || target). Higher = further from target.

    Returns dict with above keys (each mapped to list[float] of length n_layers).
    """
    if layers is None:
        layers = list(range(patcher.n_layers))

    # 1. Cache source hidden states
    source_cache = patcher.cache_hidden_states(**source_inputs)

    # 2. Baseline source + target output dist
    source_out = patcher.model(**source_inputs, use_cache=False, return_dict=True)
    source_logits = source_out.logits[0, -1, :].float().cpu()  # (vocab,)
    source_probs = torch.softmax(source_logits, dim=-1)
    source_argmax = int(source_probs.argmax())

    target_out = patcher.model(**target_inputs, use_cache=False, return_dict=True)
    target_logits = target_out.logits[0, -1, :].float().cpu()
    target_probs = torch.softmax(target_logits, dim=-1)
    target_argmax = int(target_probs.argmax())

    eps = 1e-12
    argmax_match_source = []
    logit_shift_to_source = []
    kl_patched_to_source = []
    kl_patched_to_target = []

    denom_src_logit = source_logits[source_argmax].item() - target_logits[source_argmax].item()

    for L in layers:
        patched_out = patcher.patched_forward(
            layer_idx=L,
            source_hidden=source_cache[L],
            position="last",
            **target_inputs,
        )
        patched_logits = patched_out.logits[0, -1, :].float().cpu()
        patched_probs = torch.softmax(patched_logits, dim=-1)
        patched_argmax = int(patched_probs.argmax())

        argmax_match_source.append(1.0 if patched_argmax == source_argmax else 0.0)

        if abs(denom_src_logit) > 1e-6:
            shift = (
                (patched_logits[source_argmax].item() - target_logits[source_argmax].item())
                / denom_src_logit
            )
        else:
            shift = 0.0
        logit_shift_to_source.append(shift)

        kl_ps = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(source_probs + eps))))
        kl_pt = float(torch.sum(patched_probs * (torch.log(patched_probs + eps) - torch.log(target_probs + eps))))
        kl_patched_to_source.append(kl_ps)
        kl_patched_to_target.append(kl_pt)

        if (L + 1) % 6 == 0:
            logger.info(
                f"  patched L{L}: argmax_match={argmax_match_source[-1]}, "
                f"logit_shift={shift:.3f}, KL→src={kl_ps:.3f}, KL→tgt={kl_pt:.3f}"
            )

    return {
        "layers": layers,
        "argmax_match_source": argmax_match_source,
        "logit_shift_to_source": logit_shift_to_source,
        "kl_patched_to_source": kl_patched_to_source,
        "kl_patched_to_target": kl_patched_to_target,
        "source_argmax_token_id": source_argmax,
        "target_argmax_token_id": target_argmax,
        "source_logit_at_argmax": float(source_logits[source_argmax].item()),
        "target_logit_at_argmax": float(target_logits[target_argmax].item()),
    }


def _token_seq_overlap(seq_a, seq_b) -> float:
    """Ratio of positions where seq_a[i] == seq_b[i] (prefix-aligned). 1.0 = identical."""
    n = min(len(seq_a), len(seq_b))
    if n == 0:
        return 0.0
    return sum(int(seq_a[i] == seq_b[i]) for i in range(n)) / n


def _levenshtein_token(a, b) -> int:
    """Token-level edit distance between two integer sequences (DP, no extra dep)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


@torch.no_grad()
def patching_grid_continuation(
    patcher: ActivationPatcher,
    source_inputs: dict,
    target_inputs: dict,
    max_new_tokens: int = 15,
    layers: Optional[list[int]] = None,
    randomize_source_hidden: bool = False,
) -> dict:
    """Multi-token continuation patching.

    Per-layer patch source's last-token hidden into target run, then greedy-generate
    `max_new_tokens` tokens. Compare patched output sequence to source/target baselines.

    This addresses the first-token-trivial-agree problem of patching_grid: by
    generating 10+ tokens, divergence between source/target output sequences emerges
    (e.g. action_type / element_id values vary). Layer L is causal if patching at L
    pulls patched output toward source's full sequence.

    Returns:
        dict with:
        - "source_tokens": list[int] (source's greedy sequence)
        - "target_tokens": list[int]
        - "source_text": decoded
        - "target_text": decoded
        - "per_layer": list of {layer, patched_tokens, patched_text,
                                token_overlap_to_source, token_overlap_to_target,
                                ld_to_source, ld_to_target, exact_match_source}
    """
    if layers is None:
        layers = list(range(patcher.n_layers))

    # 1. Source baseline generation
    source_gen = patcher.model.generate(
        **source_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    src_input_len = source_inputs["input_ids"].shape[1]
    source_tokens = source_gen.sequences[0, src_input_len:].cpu().tolist()
    source_text = patcher.processor.tokenizer.decode(source_tokens, skip_special_tokens=True)

    # 2. Target baseline generation
    target_gen = patcher.model.generate(
        **target_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, use_cache=True,
    )
    tgt_input_len = target_inputs["input_ids"].shape[1]
    target_tokens = target_gen.sequences[0, tgt_input_len:].cpu().tolist()
    target_text = patcher.processor.tokenizer.decode(target_tokens, skip_special_tokens=True)

    logger.info(f"  source generated: {source_text!r}")
    logger.info(f"  target generated: {target_text!r}")

    # 3. Cache source's per-layer hidden states (full forward)
    source_cache = patcher.cache_hidden_states(**source_inputs)

    # Random-injection control (paper §5 reviewer Q "is L17 disruption from
    # specific source content or any non-zero injection?"): replace each
    # layer's cached source hidden with Gaussian noise matched to that
    # layer's mean+std. Preserves activation magnitude while destroying
    # task-specific structure. If L17 disruption persists with random
    # injection → mechanism is non-specific (any patch disrupts). If it
    # vanishes → source-content-specific causal claim valid.
    if randomize_source_hidden:
        import torch as _torch_for_random
        randomized = []
        for L_idx, h in enumerate(source_cache):
            mean = h.mean()
            std = h.std()
            noise = _torch_for_random.randn_like(h) * std + mean
            randomized.append(noise)
        source_cache = randomized
        logger.info(
            "  RANDOMIZED source hidden: replaced cached activations with "
            "Gaussian noise matched to per-layer mean/std"
        )

    # 4. Per-layer patched generate
    per_layer = []
    for L in layers:
        patched_token_tensor = patcher.patched_generate(
            layer_idx=L,
            source_hidden=source_cache[L],
            max_new_tokens=max_new_tokens,
            **target_inputs,
        )
        patched_tokens = patched_token_tensor.cpu().tolist()
        patched_text = patcher.processor.tokenizer.decode(patched_tokens, skip_special_tokens=True)

        per_layer.append({
            "layer": L,
            "patched_tokens": patched_tokens,
            "patched_text": patched_text,
            "token_overlap_to_source": _token_seq_overlap(patched_tokens, source_tokens),
            "token_overlap_to_target": _token_seq_overlap(patched_tokens, target_tokens),
            "ld_to_source": _levenshtein_token(patched_tokens, source_tokens),
            "ld_to_target": _levenshtein_token(patched_tokens, target_tokens),
            "exact_match_source": patched_tokens == source_tokens,
            "exact_match_target": patched_tokens == target_tokens,
        })

        if (L + 1) % 6 == 0:
            r = per_layer[-1]
            logger.info(
                f"  L{L}: overlap→src={r['token_overlap_to_source']:.2f}, "
                f"overlap→tgt={r['token_overlap_to_target']:.2f}, "
                f"LD→src={r['ld_to_source']}, LD→tgt={r['ld_to_target']}"
            )

    return {
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "source_text": source_text,
        "target_text": target_text,
        "per_layer": per_layer,
        "max_new_tokens": max_new_tokens,
    }
