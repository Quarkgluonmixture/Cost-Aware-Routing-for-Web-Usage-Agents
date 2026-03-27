import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _parse_sm_tag(tag: str) -> Optional[Tuple[int, int]]:
    # Examples: sm_80, sm_120
    if not tag.startswith("sm_"):
        return None
    raw = tag[3:]
    if not raw.isdigit() or len(raw) < 2:
        return None
    if len(raw) == 2:
        return int(raw[0]), int(raw[1])
    return int(raw[:-1]), int(raw[-1])


def _max_supported_sm() -> Optional[Tuple[int, int]]:
    try:
        archs = torch.cuda.get_arch_list()
    except Exception:
        return None
    parsed = [_parse_sm_tag(a) for a in archs]
    parsed = [p for p in parsed if p is not None]
    return max(parsed) if parsed else None


def _needs_nvrtc_prod_workaround() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    max_sm = _max_supported_sm()
    if max_sm is None:
        return False
    # Trigger workaround when runtime device capability is newer than
    # architectures baked into this torch CUDA build.
    return cap > max_sm


def _runtime_error_is_nvrtc_arch(err: BaseException) -> bool:
    msg = str(err).lower()
    return "invalid value for --gpu-architecture" in msg or "nvrtc" in msg and "gpu-architecture" in msg


def apply_nvrtc_prod_fallback_if_needed() -> bool:
    """Patch torch.prod/Tensor.prod to CPU-fallback on NVRTC arch errors.

    Returns True when patching is enabled, False otherwise.
    """
    if os.environ.get("P79_DISABLE_NVRTC_PROD_FALLBACK", "0") == "1":
        return False

    if getattr(torch, "_p79_nvrtc_prod_fallback_enabled", False):
        return True

    if not _needs_nvrtc_prod_workaround():
        return False

    original_tensor_prod = torch.Tensor.prod
    original_torch_prod = torch.prod

    def _cpu_prod_tensor(tensor: torch.Tensor, *args, **kwargs):
        if kwargs.get("out", None) is not None:
            # Keep semantics predictable; do not emulate out= across devices.
            raise RuntimeError("CPU prod fallback does not support out= argument")

        cpu_tensor = tensor.detach().to("cpu")
        try:
            out = original_tensor_prod(cpu_tensor, *args, **kwargs)
        except Exception:
            if cpu_tensor.is_floating_point():
                cast = cpu_tensor.float()
                out = original_tensor_prod(cast, *args, **kwargs)
                if isinstance(out, torch.Tensor):
                    out = out.to(dtype=tensor.dtype)
            else:
                raise
        if isinstance(out, torch.Tensor):
            return out.to(device=tensor.device)
        return out

    def tensor_prod_wrapper(self, *args, **kwargs):
        try:
            return original_tensor_prod(self, *args, **kwargs)
        except RuntimeError as err:
            if self.is_cuda and _runtime_error_is_nvrtc_arch(err):
                return _cpu_prod_tensor(self, *args, **kwargs)
            raise

    def torch_prod_wrapper(input_tensor, *args, **kwargs):
        try:
            return original_torch_prod(input_tensor, *args, **kwargs)
        except RuntimeError as err:
            if isinstance(input_tensor, torch.Tensor) and input_tensor.is_cuda and _runtime_error_is_nvrtc_arch(err):
                return _cpu_prod_tensor(input_tensor, *args, **kwargs)
            raise

    torch.Tensor.prod = tensor_prod_wrapper
    torch.prod = torch_prod_wrapper
    torch._p79_nvrtc_prod_fallback_enabled = True
    logger.warning(
        "Enabled NVRTC prod fallback: CUDA capability exceeds torch-supported SM list. "
        "set P79_DISABLE_NVRTC_PROD_FALLBACK=1 to disable."
    )
    return True

