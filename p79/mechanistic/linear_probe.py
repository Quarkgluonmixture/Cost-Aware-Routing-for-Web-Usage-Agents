"""Per-layer logistic regression probe with k-fold CV.

Tests whether mirage label is linearly decodable from hidden states at each
transformer layer. AUROC curve over layers reveals where mirage info is encoded.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def linear_probe_per_layer(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Train per-layer LR with stratified k-fold CV; return AUROC curve.

    Args:
        hidden_states: (n_samples, n_layers, hidden_dim)
        labels: (n_samples,) binary 0/1
        n_folds: CV folds
        seed: RNG seed for fold split
        C: LR L2 regularization (1/lambda)
        max_iter: LR solver iterations

    Returns:
        {
            "auroc_mean": list[float] length n_layers,
            "auroc_std": list[float] length n_layers,
            "auroc_per_fold": list[list[float]] (n_layers, n_folds),
            "best_layer": int,
            "best_auroc": float,
            "n_samples": int,
            "n_pos": int,
            "n_layers": int,
            "hidden_dim": int,
        }
    """
    n_samples, n_layers, hidden_dim = hidden_states.shape
    if labels.shape[0] != n_samples:
        raise ValueError(f"labels.shape[0] {labels.shape[0]} != n_samples {n_samples}")
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == n_samples:
        raise ValueError(f"degenerate labels: {n_pos}/{n_samples} positive")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    auroc_per_fold = np.zeros((n_layers, n_folds), dtype=np.float64)

    for layer_idx in range(n_layers):
        X = hidden_states[:, layer_idx, :]  # (n_samples, hidden_dim)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(
                C=C, max_iter=max_iter, solver="lbfgs", random_state=seed,
            )
            clf.fit(X_train, labels[train_idx])
            y_score = clf.predict_proba(X_test)[:, 1]
            auroc_per_fold[layer_idx, fold_idx] = roc_auc_score(labels[test_idx], y_score)
        if (layer_idx + 1) % 5 == 0:
            logger.info(
                f"Layer {layer_idx + 1}/{n_layers}: "
                f"AUROC {auroc_per_fold[layer_idx].mean():.4f} ± {auroc_per_fold[layer_idx].std():.4f}"
            )

    auroc_mean = auroc_per_fold.mean(axis=1)
    auroc_std = auroc_per_fold.std(axis=1)
    best_layer = int(auroc_mean.argmax())

    return {
        "auroc_mean": auroc_mean.tolist(),
        "auroc_std": auroc_std.tolist(),
        "auroc_per_fold": auroc_per_fold.tolist(),
        "best_layer": best_layer,
        "best_auroc": float(auroc_mean[best_layer]),
        "best_auroc_std": float(auroc_std[best_layer]),
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
    }


def plot_auroc_curve(
    probe_results: dict,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot per-layer AUROC ± std as a curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    auroc_mean = np.array(probe_results["auroc_mean"])
    auroc_std = np.array(probe_results["auroc_std"])
    best_layer = probe_results["best_layer"]
    n_samples = probe_results["n_samples"]
    n_pos = probe_results["n_pos"]
    layers = np.arange(len(auroc_mean))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, auroc_mean, marker="o", lw=1.5, label="AUROC mean")
    ax.fill_between(
        layers, auroc_mean - auroc_std, auroc_mean + auroc_std,
        alpha=0.25, label="±1 std (5-fold CV)",
    )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance (0.5)")
    ax.axvline(
        best_layer, color="red", ls=":", lw=0.8,
        label=f"best layer {best_layer} ({auroc_mean[best_layer]:.3f})",
    )
    ax.set_xlabel("Layer index (0 = embedding output, ≥1 = post-transformer-block)")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0.45, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    if title is None:
        title = f"Per-layer linear probe AUROC (N={n_samples}, n_pos={n_pos})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved AUROC curve → {save_path}")
