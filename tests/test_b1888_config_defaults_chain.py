"""B-1888 — `defaults:` inheritance must recurse, not stop at one level.

`load_experiment_config` used to read each `defaults:` entry with a bare
`yaml.safe_load`, so a base config's OWN `defaults:` was never followed — it was
merged in as an inert `defaults` KEY. Single-level chains (every VWA
per-condition config -> exp_v2_base.yaml) were unaffected, which is why the gap
survived two months unnoticed. Two-level chains were silently broken: all WA
configs inherit exp_v2_wa_base.yaml, which itself declares exp_v2_base.yaml, so
every WA config resolved WITHOUT the base layer — no `backends.local_4b.type`
(the launch crash that surfaced this), no model path, no OOM guard, no token
pricing, no carbon intensity.

The guard that matters most here is the immutability one: fixing the recursion
must not perturb any already-fired VWA config.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from p79.experiment.config import load_experiment_config

VWA_SAMPLE = "configs/exp_v2_B1_dom_reddit.yaml"
WA_SAMPLE = "configs/exp_v2_B1_dom_wa_reddit.yaml"


def test_defaults_chain_vwa_single_level_unchanged():
    """A single-level chain resolves exactly as it always did.

    Asserted structurally rather than by golden hash: the VWA config declares
    only `max_new_tokens` + `temperature` under `backends.local_4b`, so every
    other key present proves the base layer merged, and the two it does declare
    prove the per-condition layer still wins.
    """
    cfg = load_experiment_config(VWA_SAMPLE)
    backend = cfg["backends"][cfg["backends"]["default_backend"]]

    assert backend["type"] == "local_qwen"                     # from base
    assert backend["path"] == "Qwen/Qwen3-VL-4B-Instruct"      # from base
    assert backend["min_free_vram_gb"] == 12                   # from base
    assert backend["temperature"] == 0.0                       # per-condition override
    assert cfg["experiment"]["benchmark"] == "visualwebarena"


def test_defaults_chain_wa_two_level_inherits_base():
    """The B-1888 regression: WA -> wa_base -> base must reach the base layer."""
    cfg = load_experiment_config(WA_SAMPLE)
    backend = cfg["backends"][cfg["backends"]["default_backend"]]

    # These live ONLY in exp_v2_base.yaml, two levels up.
    assert backend["type"] == "local_qwen", "base layer not inherited (B-1888)"
    assert backend["path"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert backend["min_free_vram_gb"] == 12

    # The middle layer must still win where it overrides.
    assert cfg["experiment"]["benchmark"] == "webarena"
    assert cfg["env"]["benchmark"] == "webarena"


def test_every_config_resolves_a_dispatchable_default_backend():
    """No config may resolve to a backend without `type`.

    `p79/backends/factory.py` dispatches explicitly and raises on a missing
    `type`; this is the crash B-1888 produced at WA launch. Sweeping every
    config keeps a third inheritance level (or a new benchmark family) from
    reintroducing it silently.
    """
    import glob

    missing = []
    for path in sorted(glob.glob("configs/*.yaml")):
        cfg = load_experiment_config(path)
        backends = cfg.get("backends") or {}
        name = backends.get("default_backend")
        if not name:
            continue
        if not (backends.get(name) or {}).get("type"):
            missing.append(f"{path} (default_backend={name})")
    assert not missing, "configs resolving to a typeless backend:\n  " + "\n  ".join(missing)


def test_defaults_chain_detects_cycle(tmp_path):
    """A cyclic `defaults:` reference must raise, not recurse forever."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(textwrap.dedent(f"""\
        defaults:
          - {b}
        experiment:
          name: a
    """))
    b.write_text(textwrap.dedent(f"""\
        defaults:
          - {a}
        experiment:
          name: b
    """))
    with pytest.raises(ValueError, match="Circular"):
        load_experiment_config(str(a))


def test_defaults_key_does_not_leak_into_resolved_config(tmp_path):
    """`defaults:` is inheritance plumbing, not config data.

    Pre-fix, a base config's own `defaults:` list survived into the merged
    result as a stray key (that was the whole bug, seen from the other side).
    """
    base = tmp_path / "base.yaml"
    mid = tmp_path / "mid.yaml"
    leaf = tmp_path / "leaf.yaml"
    base.write_text("experiment:\n  name: base\n  phase: phase1\n")
    mid.write_text(f"defaults:\n  - {base}\nexperiment:\n  name: mid\n")
    leaf.write_text(f"defaults:\n  - {mid}\nexperiment:\n  name: leaf\n")

    cfg = load_experiment_config(str(leaf))
    assert "defaults" not in cfg
    assert cfg["experiment"]["name"] == "leaf"     # leaf wins
    assert cfg["experiment"]["phase"] == "phase1"  # reached through 2 levels
