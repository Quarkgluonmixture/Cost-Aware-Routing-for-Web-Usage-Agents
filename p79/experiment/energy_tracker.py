from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


# Adapted from external_code/tracker.py + data_source.py (Aiden Yiliu Li, Apache-2.0)
HARDWARE_PROFILES = {
    "m2": {"idle": 5.0, "load": 22.0},
    "m2_max": {"idle": 10.0, "load": 50.0},
    "cpu_intel_i7": {"idle": 10.0, "load": 65.0},
    "cpu_intel_i9": {"idle": 15.0, "load": 125.0},
    "cpu_amd_5900x": {"idle": 10.0, "load": 100.0},
    "rtx_4090": {"idle": 30.0, "load": 400.0},
    "a100": {"idle": 50.0, "load": 300.0},
    "h100": {"idle": 60.0, "load": 500.0},
    # DGX Spark (GH200 Grace Hopper): ~900W GPU TDP + ~150W Grace CPU
    "dgx_spark": {"idle": 120.0, "load": 1050.0},
    # GB200 / Blackwell-based DGX
    "gb200": {"idle": 150.0, "load": 1200.0},
}


REGION_INTENSITY_G_PER_KWH = {
    "world": 475.0,
    "us-east-1": 367.0,
    "us-west-2": 367.0,
    "eu-west-1": 296.0,
    "eu-central-1": 385.0,
    "ap-south-1": 632.0,
}


@dataclass
class EnergyEstimate:
    kwh: Optional[float]
    co2e_kg: Optional[float]
    power_watts: Optional[float]
    source: str

    def as_dict(self) -> Dict[str, Optional[float] | str]:
        return {
            "kwh": self.kwh,
            "co2e_kg": self.co2e_kg,
            "power_watts": self.power_watts,
            "source": self.source,
        }


class LightweightEnergyTracker:
    def __init__(self, energy_cfg: Dict[str, Any]):
        self.enabled = bool(energy_cfg.get("enabled", False))
        self.kwh_per_step = energy_cfg.get("kwh_per_step")
        self.co2e_kg_per_kwh = energy_cfg.get("co2e_kg_per_kwh")

        self.fixed_power_watts = energy_cfg.get("fixed_power_watts")
        self.hardware_profile = str(energy_cfg.get("hardware_profile", "m2"))
        self.use_psutil = bool(energy_cfg.get("use_psutil", True))
        self.region = str(energy_cfg.get("region", "world")).lower()

        intensity_g = energy_cfg.get("carbon_intensity_g_per_kwh")
        if intensity_g is None:
            intensity_g = REGION_INTENSITY_G_PER_KWH.get(self.region, REGION_INTENSITY_G_PER_KWH["world"])
        self.carbon_intensity_g_per_kwh = float(intensity_g)

    def _estimate_power_watts(self) -> EnergyEstimate:
        if self.fixed_power_watts is not None:
            return EnergyEstimate(kwh=None, co2e_kg=None, power_watts=float(self.fixed_power_watts), source="fixed_power")

        profile = HARDWARE_PROFILES.get(self.hardware_profile, HARDWARE_PROFILES["m2"])
        idle = float(profile["idle"])
        load = float(profile["load"])

        if self.use_psutil and psutil is not None:
            try:
                cpu_percent = float(psutil.cpu_percent(interval=0.0))
                util = max(0.0, min(cpu_percent / 100.0, 1.0))
                power = idle + (load - idle) * util
                return EnergyEstimate(kwh=None, co2e_kg=None, power_watts=power, source="psutil_profile")
            except Exception:
                pass

        # Fallback: assume medium utilization.
        power = idle + (load - idle) * 0.6
        return EnergyEstimate(kwh=None, co2e_kg=None, power_watts=power, source="profile_fallback")

    def estimate_step(self, duration_seconds: float) -> Dict[str, Optional[float] | str]:
        if not self.enabled:
            return EnergyEstimate(kwh=None, co2e_kg=None, power_watts=None, source="disabled").as_dict()

        if self.kwh_per_step is not None:
            kwh = float(self.kwh_per_step)
            co2 = (
                float(kwh) * float(self.co2e_kg_per_kwh)
                if self.co2e_kg_per_kwh is not None
                else float(kwh) * self.carbon_intensity_g_per_kwh / 1000.0
            )
            return EnergyEstimate(kwh=kwh, co2e_kg=co2, power_watts=None, source="kwh_per_step").as_dict()

        power_est = self._estimate_power_watts()
        power_watts = float(power_est.power_watts or 0.0)
        duration_hours = max(float(duration_seconds), 0.0) / 3600.0
        kwh = power_watts * duration_hours / 1000.0
        co2 = (
            kwh * float(self.co2e_kg_per_kwh)
            if self.co2e_kg_per_kwh is not None
            else kwh * self.carbon_intensity_g_per_kwh / 1000.0
        )

        return EnergyEstimate(kwh=kwh, co2e_kg=co2, power_watts=power_watts, source=power_est.source).as_dict()
