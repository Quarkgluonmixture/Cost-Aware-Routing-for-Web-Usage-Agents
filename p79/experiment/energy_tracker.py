from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


# ---------------------------------------------------------------------------
# Hardware TDP profiles (fallback when real measurement is unavailable)
# Adapted from external_code/tracker.py (Aiden Yiliu Li, Apache-2.0)
# ---------------------------------------------------------------------------
HARDWARE_PROFILES = {
    "m2": {"idle": 5.0, "load": 22.0},
    "m2_max": {"idle": 10.0, "load": 50.0},
    "cpu_intel_i7": {"idle": 10.0, "load": 65.0},
    "cpu_intel_i9": {"idle": 15.0, "load": 125.0},
    "cpu_amd_5900x": {"idle": 10.0, "load": 100.0},
    "rtx_4090": {"idle": 30.0, "load": 400.0},
    "a100": {"idle": 50.0, "load": 300.0},
    # B-320 (/stress A1.9 Mode A F1 + Mode C #2 OOB, 2026-05-16): config yaml
    # canonical key "a100_pcie_40gb" was not in this dict pre-fix → silent
    # `.get(key, HARDWARE_PROFILES["m2"])` fallback → laptop m2 profile (22W
    # load) reported for A100 paper-grade fire (300W load) → ~14× energy/CO2
    # under-quote on profile_fallback path. PCIe variant idle/load aligned
    # with "a100" baseline (PCIe TDP is the same 300W as SXM4 40GB; only
    # SXM4-80GB raises to 400W). See `configs/exp_v2_base.yaml:79`.
    "a100_pcie_40gb": {"idle": 50.0, "load": 300.0},
    "h100": {"idle": 60.0, "load": 500.0},
    # DGX Spark (GB10 Grace Blackwell): ~300W GPU TDP + ~65W Grace CPU
    "dgx_spark": {"idle": 40.0, "load": 365.0},
    # GB200 / Blackwell-based DGX
    "gb200": {"idle": 150.0, "load": 1200.0},
}


# ---------------------------------------------------------------------------
# Regional carbon intensity (gCO2/kWh)
# Primary sources: IEA 2023, ElectricityMaps static data
# Adapted from external_code/electricity_maps.py + data_source.py
# ---------------------------------------------------------------------------
REGION_INTENSITY_G_PER_KWH: Dict[str, float] = {
    # World average (IEA 2023)
    "world": 475.0,
    # United States
    "usa": 367.0, "us": 367.0,
    "us-east-1": 367.0, "us-east-2": 367.0,
    "us-west-1": 200.0, "us-west-2": 200.0,
    # Europe
    "eu-west-1": 296.0,   # Ireland
    "eu-west-2": 257.0,   # UK
    "eu-west-3": 85.0,    # France (nuclear-heavy)
    "eu-central-1": 385.0,  # Germany
    "eu-north-1": 41.0,   # Sweden
    "eu-south-1": 390.0,  # Italy
    "germany": 385.0, "france": 85.0, "uk": 257.0,
    "sweden": 41.0, "norway": 29.0, "ireland": 296.0,
    "netherlands": 380.0, "spain": 195.0, "poland": 773.0,
    # Asia-Pacific
    "ap-northeast-1": 462.0,  # Japan
    "ap-northeast-2": 415.0,  # South Korea
    "ap-south-1": 632.0,      # India
    "ap-southeast-1": 408.0,  # Singapore
    "ap-southeast-2": 510.0,  # Australia
    "japan": 462.0, "south_korea": 415.0, "india": 632.0,
    "china": 531.0, "singapore": 408.0, "australia": 510.0,
    "taiwan": 509.0, "hong_kong": 531.0,
    # Americas
    "sa-east-1": 85.0,    # Brazil
    "ca-central-1": 110.0,  # Canada
    "brazil": 85.0, "canada": 110.0,
    "mexico": 436.0, "argentina": 310.0,
    # Middle East / Africa
    "me-south-1": 513.0,  # Bahrain
    "af-south-1": 928.0,  # South Africa
    "south_africa": 928.0, "uae": 513.0,
}


# ---------------------------------------------------------------------------
# NVIDIAPowerReader — real GPU power via pynvml
# Adapted from external_code/tracker.py (Aiden Yiliu Li, Apache-2.0)
# ---------------------------------------------------------------------------
class NVIDIAPowerReader:
    """Read NVIDIA GPU instantaneous power draw via pynvml.

    Gracefully unavailable if pynvml is not installed or no GPU is present.
    Uses device index 0 (single-GPU assumption for DGX Spark GB10).
    """

    def __init__(self) -> None:
        self.available = False
        self._pynvml = None
        self._handle = None
        self._nvml_initialized = False
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._pynvml = pynvml
            if pynvml.nvmlDeviceGetCount() > 0:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.available = True
        except Exception:
            pass

    def get_power(self) -> Optional[float]:
        """Return current GPU power draw in Watts, or None on failure."""
        if not self.available or self._pynvml is None:
            return None
        try:
            mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
            return float(mw) / 1000.0  # milliwatts → Watts
        except Exception:
            return None

    def get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Return GPU name and total memory (MB), or None on failure."""
        if not self.available or self._pynvml is None:
            return None
        try:
            name = self._pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {"name": name, "memory_total_mb": float(mem.total) / (1024 * 1024)}
        except Exception:
            return None

    def shutdown(self) -> None:
        if self._nvml_initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False


# ---------------------------------------------------------------------------
# RAPLReader — CPU package power via Linux powercap interface
# Adapted from external_code/tracker.py (Aiden Yiliu Li, Apache-2.0)
# No-op on aarch64 / non-Intel-AMD platforms (path simply won't exist).
# ---------------------------------------------------------------------------
class RAPLReader:
    """Read CPU power via Linux RAPL (Intel/AMD only).

    Returns None on all non-Linux / aarch64 platforms.
    """

    _RAPL_ROOT = "/sys/class/powercap/intel-rapl"

    def __init__(self) -> None:
        import os

        self.available = False
        self._energy_file: Optional[str] = None
        self._last_energy: Optional[int] = None
        self._last_ts: Optional[float] = None
        try:
            if not os.path.exists(self._RAPL_ROOT):
                return
            pkg0 = f"{self._RAPL_ROOT}/intel-rapl:0/energy_uj"
            if os.path.exists(pkg0):
                self._energy_file = pkg0
                self.available = True
        except Exception:
            pass

    def get_power(self) -> Optional[float]:
        """Return instantaneous CPU package power in Watts, or None."""
        if not self.available or self._energy_file is None:
            return None
        try:
            # B-341 (/stress A1.9 Mode A F10, 2026-05-16): A1.8 B-288 sibling
            # — kernel mid-write race could leave the /sys/class/powercap
            # energy_uj file with non-UTF-8 bytes; bare `open(.., "r")`
            # raises UnicodeDecodeError → caught by outer `except` and
            # silently returns None → fallback to broken profile path.
            # `errors="replace"` makes the read robust to transient kernel
            # bytes; `int()` on the result still fails fast if non-digit.
            with open(self._energy_file, "r", errors="replace") as f:
                energy_uj = int(f.read().strip())
            now = time.monotonic()
            if self._last_energy is not None and self._last_ts is not None:
                dt = now - self._last_ts
                if dt > 0:
                    power = (energy_uj - self._last_energy) / 1e6 / dt  # µJ → J → W
                    self._last_energy = energy_uj
                    self._last_ts = now
                    return float(power)
            self._last_energy = energy_uj
            self._last_ts = now
            return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# EnergyEstimate — step-level result
# ---------------------------------------------------------------------------
@dataclass
class EnergyEstimate:
    kwh: Optional[float]
    co2e_kg: Optional[float]
    power_watts: Optional[float]
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kwh": self.kwh,
            "co2e_kg": self.co2e_kg,
            "power_watts": self.power_watts,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# LightweightEnergyTracker
# ---------------------------------------------------------------------------
class LightweightEnergyTracker:
    """Tracks energy per step.

    Measurement priority:
      1. Real GPU power via pynvml + background sampling (source="pynvml")
      2. psutil CPU% × hardware profile TDP (source="psutil_profile")
      3. Profile TDP × 0.6 utilisation (source="profile_fallback")
      4. Fixed watts from config (source="fixed_power")
      5. kwh_per_step constant (source="kwh_per_step")

    Config keys (all optional, safe defaults):
      enabled, kwh_per_step, co2e_kg_per_kwh, hardware_profile, region,
      use_psutil, fixed_power_watts,
      use_pynvml (bool, default True),
      sample_interval_s (float, default 0.5).
    """

    def __init__(self, energy_cfg: Dict[str, Any]) -> None:
        self.enabled = bool(energy_cfg.get("enabled", False))
        self.kwh_per_step = energy_cfg.get("kwh_per_step")
        self.co2e_kg_per_kwh = energy_cfg.get("co2e_kg_per_kwh")
        self.fixed_power_watts = energy_cfg.get("fixed_power_watts")
        self.hardware_profile = str(energy_cfg.get("hardware_profile", "m2"))
        # B-320 (/stress A1.9 Mode A F1 + Mode C #2 OOB, 2026-05-16): fail-loud
        # on unknown hardware_profile when energy is enabled. Pre-fix the
        # `.get(key, HARDWARE_PROFILES["m2"])` fallback path in
        # `_estimate_power_watts` silently coerced any unknown key to m2
        # laptop profile (22W vs A100 300W → ~14× energy under-quote). A
        # mis-typed yaml key would land in production undetected.
        if self.enabled and self.hardware_profile not in HARDWARE_PROFILES:
            raise ValueError(
                f"hardware_profile={self.hardware_profile!r} not in "
                f"HARDWARE_PROFILES (known keys: {sorted(HARDWARE_PROFILES.keys())}). "
                "Add an entry to HARDWARE_PROFILES or fix the yaml key. "
                "Silent fallback to m2 laptop profile is paper-grade-broken."
            )
        # B-336 (/stress A1.9 Mode A F7, 2026-05-16): `kwh_per_step` mode is
        # duration-blind — returns the same kwh value regardless of step
        # latency. Paper §3 reports energy as a function of step inference
        # work; duration-blind fixed-rate is incompatible. Deprecated:
        # explicit `kwh_per_step` config raises here (paper-grade fail-loud)
        # rather than silently producing duration-independent numbers.
        # Backward compat: config keys remain in DEFAULT_CONFIG + yaml schema
        # at value None (no behavior change for default null).
        if self.enabled and self.kwh_per_step is not None:
            raise ValueError(
                f"kwh_per_step={self.kwh_per_step!r} is deprecated per B-336. "
                "Mode was duration-blind (returned same kWh regardless of step "
                "latency), incompatible with paper §3 per-step energy claim. "
                "Set kwh_per_step: null in yaml + use pynvml-based measurement."
            )
        self.use_psutil = bool(energy_cfg.get("use_psutil", True))
        self.region = str(energy_cfg.get("region", "world")).lower()

        intensity_g = energy_cfg.get("carbon_intensity_g_per_kwh")
        if intensity_g is None:
            intensity_g = REGION_INTENSITY_G_PER_KWH.get(
                self.region, REGION_INTENSITY_G_PER_KWH["world"]
            )
        self.carbon_intensity_g_per_kwh = float(intensity_g)

        # pynvml / sampling config
        self._use_pynvml = bool(energy_cfg.get("use_pynvml", True))
        self._sample_interval = float(energy_cfg.get("sample_interval_s", 0.5))

        # sampling state
        self._nvidia_reader: Optional[NVIDIAPowerReader] = None
        self._rapl_reader: Optional[RAPLReader] = None
        # Bounded deque caps memory + makes append O(1) (was list+rebuild O(N) per sample).
        # 5 minutes / sample_interval = max sample count.
        _max_samples = max(60, int(300.0 / max(self._sample_interval, 0.05)))
        self._power_samples: Deque[Tuple[float, float]] = deque(maxlen=_max_samples)
        self._sample_lock = threading.Lock()
        self._sample_thread: Optional[threading.Thread] = None
        self._sampling_active = False

        # Prime psutil.cpu_percent: the first call with interval=0.0 always
        # returns 0.0 (no prior baseline) → first step would severely
        # underestimate CPU power. Discard that initial reading here so
        # estimate_step's subsequent calls return real per-call deltas.
        if self.use_psutil and psutil is not None:
            try:
                psutil.cpu_percent(interval=0.0)
            except Exception:
                pass

        if self.enabled and self._use_pynvml:
            self._nvidia_reader = NVIDIAPowerReader()
            self._rapl_reader = RAPLReader()
            if self._nvidia_reader.available:
                self._start_sampling()

    # ------------------------------------------------------------------
    # Background sampling
    # ------------------------------------------------------------------

    def _start_sampling(self) -> None:
        self._sampling_active = True
        self._sample_thread = threading.Thread(
            target=self._sampling_loop,
            daemon=True,
            name="p79-energy-sampler",
        )
        self._sample_thread.start()

    def _sampling_loop(self) -> None:
        while self._sampling_active:
            t = time.monotonic()
            gpu_w = self._nvidia_reader.get_power() if self._nvidia_reader else None
            if gpu_w is not None:
                rapl_w = self._rapl_reader.get_power() if self._rapl_reader else None
                total_w = gpu_w + (rapl_w or 0.0)
                # deque(maxlen=...) auto-evicts oldest sample → O(1) append,
                # no per-sample list rebuild (was O(N) per sample → O(N²) cumulative).
                with self._sample_lock:
                    self._power_samples.append((t, total_w))
            time.sleep(self._sample_interval)

    def _average_measured_power(
        self,
        duration_seconds: float,
        step_start_monotonic: Optional[float] = None,
    ) -> Tuple[Optional[float], int]:
        """Average sampled power over the step window.

        B-321 (/stress A1.9 Mode A F2 OOB, 2026-05-16): when
        `step_start_monotonic` is provided, the window is strictly bound to
        `[step_start, step_start + duration_seconds]` — covers only samples
        taken DURING inference. Pre-fix `cutoff = now - duration - 1` returned
        mean over ALL samples in window regardless of whether the step was
        actually running, so a fast step (200ms latency, 500ms sample
        interval) averaged mostly pre-step idle samples → A100 inference
        burst (~300W) reported as idle (~50W) → paper §3 per-step energy
        decomposition systematically biased toward idle.

        Returns (avg_power, sample_count). `sample_count` lets caller emit
        `energy_window_partial` flag when sample density too low.

        Legacy callers (no `step_start_monotonic`) fall back to pre-fix
        sliding window for backwards compat (zero behavior change).
        """
        if step_start_monotonic is not None:
            # B-321 strict window bound: only samples during [start, end].
            end = step_start_monotonic + max(duration_seconds, 0.0)
            with self._sample_lock:
                recent = [w for ts, w in self._power_samples
                          if step_start_monotonic <= ts <= end]
        else:
            # Legacy path (pre-B-321 behavior).
            cutoff = time.monotonic() - max(duration_seconds, 0.0) - 1.0
            with self._sample_lock:
                recent = [w for ts, w in self._power_samples if ts >= cutoff]
        if not recent:
            return None, 0
        return sum(recent) / len(recent), len(recent)

    # ------------------------------------------------------------------
    # Profile-based fallback
    # ------------------------------------------------------------------

    def _estimate_power_watts(self) -> EnergyEstimate:
        if self.fixed_power_watts is not None:
            return EnergyEstimate(
                kwh=None, co2e_kg=None,
                power_watts=float(self.fixed_power_watts),
                source="fixed_power",
            )

        profile = HARDWARE_PROFILES.get(self.hardware_profile, HARDWARE_PROFILES["m2"])
        idle = float(profile["idle"])
        load = float(profile["load"])

        if self.use_psutil and psutil is not None:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=0.0))
                util = max(0.0, min(cpu_percent / 100.0, 1.0))
                power = idle + (load - idle) * util
                return EnergyEstimate(
                    kwh=None, co2e_kg=None, power_watts=power, source="psutil_profile"
                )
            except Exception:
                pass

        power = idle + (load - idle) * 0.6
        return EnergyEstimate(
            kwh=None, co2e_kg=None, power_watts=power, source="profile_fallback"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_step(
        self,
        duration_seconds: float,
        step_start_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return energy estimate for one step of `duration_seconds` seconds.

        B-321 (/stress A1.9 Mode A F2 OOB, 2026-05-16): `step_start_monotonic`
        param added. When provided (recommended for paper-grade fire), the
        pynvml sample window is strictly bound to inference period — fixes
        fast-step idle-sample contamination. When None (legacy callers), the
        pre-fix sliding window is used for backwards compat (zero behavior
        change for legacy paths). Emits `window_sample_count` +
        `energy_window_partial` for paper §3 transparency.
        """
        if not self.enabled:
            return EnergyEstimate(
                kwh=None, co2e_kg=None, power_watts=None, source="disabled"
            ).as_dict()

        if self.kwh_per_step is not None:
            kwh = float(self.kwh_per_step)
            co2 = (
                float(kwh) * float(self.co2e_kg_per_kwh)
                if self.co2e_kg_per_kwh is not None
                else float(kwh) * self.carbon_intensity_g_per_kwh / 1000.0
            )
            return EnergyEstimate(
                kwh=kwh, co2e_kg=co2, power_watts=None, source="kwh_per_step"
            ).as_dict()

        # Try real measurement first; fall back to profile estimation.
        # B-321: pass step_start_monotonic for strict window bound.
        measured_watts, sample_count = self._average_measured_power(
            duration_seconds, step_start_monotonic=step_start_monotonic
        )
        if measured_watts is not None:
            power_watts = measured_watts
            source = "pynvml"
        else:
            power_est = self._estimate_power_watts()
            power_watts = float(power_est.power_watts or 0.0)
            source = power_est.source
            sample_count = 0

        duration_hours = max(float(duration_seconds), 0.0) / 3600.0
        kwh = power_watts * duration_hours / 1000.0
        co2 = (
            kwh * float(self.co2e_kg_per_kwh)
            if self.co2e_kg_per_kwh is not None
            else kwh * self.carbon_intensity_g_per_kwh / 1000.0
        )

        out = EnergyEstimate(
            kwh=kwh, co2e_kg=co2, power_watts=power_watts, source=source
        ).as_dict()
        # B-321: paper §3 transparency — flag window sample density so
        # downstream analyzer can compute `energy_window_partial_rate`
        # per cell + filter low-density steps from energy comparisons.
        out["window_sample_count"] = int(sample_count)
        out["energy_window_partial"] = bool(
            source == "pynvml" and sample_count < 2
        )
        return out

    @property
    def gpu_info(self) -> Optional[Dict[str, Any]]:
        """Return GPU name/memory info if pynvml is available."""
        if self._nvidia_reader is not None:
            return self._nvidia_reader.get_gpu_info()
        return None

    @property
    def measurement_source(self) -> str:
        """Report which measurement source `estimate_step` will use.

        Must mirror the priority chain in `estimate_step` exactly:
        kwh_per_step → pynvml → fixed_power → psutil_profile → profile_fallback.
        """
        if self.kwh_per_step is not None:
            return "kwh_per_step"
        if self._nvidia_reader is not None and self._nvidia_reader.available:
            return "pynvml"
        if self.fixed_power_watts is not None:
            return "fixed_power"
        if self.use_psutil and psutil is not None:
            return "psutil_profile"
        return "profile_fallback"

    def close(self) -> None:
        """Stop background sampling thread and release NVML resources."""
        self._sampling_active = False
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=2.0)
            self._sample_thread = None
        if self._nvidia_reader is not None:
            self._nvidia_reader.shutdown()
            self._nvidia_reader = None
