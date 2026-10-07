"""The null sampler and the decomposition behind the label-supply test.

Catches: a curveball trade that changes a task's total or a (mode, run) column's success count
(the null would then no longer condition on difficulty and arm SR, and its p-values would be
about the wrong model), and an ANOVA that mislabels a reproducible mode preference as noise or
invents interaction where every mode behaves the same.
"""
import numpy as np

from scripts.analysis.task_mode_interaction_null import _to_matrix, _to_rows, anova, curveball


def test_curveball_keeps_every_margin():
    rng = np.random.default_rng(0)
    Y = (rng.random((60, 6, 2)) < 0.3).astype(float)
    rows = _to_rows(Y)
    curveball(rows, 5000, rng)
    Z = _to_matrix(rows, 12)
    assert (Z.reshape(60, -1).sum(1) == Y.reshape(60, -1).sum(1)).all()
    assert (Z.reshape(60, -1).sum(0) == Y.reshape(60, -1).sum(0)).all()
    assert not np.array_equal(Z, Y)          # it actually moved


def test_anova_separates_reproducible_preference_from_none():
    # Each task prefers one mode, identically on both runs: all interaction, zero noise.
    n = 48
    Y = np.zeros((n, 6, 2))
    for i in range(n):
        Y[i, i % 6, :] = 1
    d = anova(Y)
    assert d["noise"] == 0 and d["interaction"] > 0.1
    # Same per-task totals but the preferred mode differs between runs: no reproducible part.
    Y2 = np.zeros((n, 6, 2))
    for i in range(n):
        Y2[i, i % 6, 0] = 1
        Y2[i, (i + 3) % 6, 1] = 1
    d2 = anova(Y2)
    assert d2["interaction"] < 0 < d2["noise"]
