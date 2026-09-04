"""Turn experiment records into the headline comparison: AUC + bootstrap CI per
detector, and an ROC plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mats.experiment import ItemRecord
from mats.metrics import bootstrap_auc, roc_auc, roc_curve

_DETECTORS = (("diffuse", "diffuse"), ("entropy", "entropy"), ("llm_monitor", "monitor"))


def _eligible(records: list[ItemRecord]) -> list[ItemRecord]:
    return [r for r in records if r.eligible]


def summarize(records: list[ItemRecord]) -> dict:
    eligible = _eligible(records)
    labels = np.array([r.label for r in eligible])
    summary = {
        "n_total_records": len(records),
        "n_eligible": len(eligible),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "detectors": {},
    }
    for name, attr in _DETECTORS:
        scores = np.array([getattr(r, attr) for r in eligible], dtype=float)
        lo, hi = bootstrap_auc(labels, scores)
        summary["detectors"][name] = {
            "auc": roc_auc(labels, scores),
            "ci95": [lo, hi],
        }
    return summary


def plot_roc(records: list[ItemRecord], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = _eligible(records)
    labels = np.array([r.label for r in eligible])
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, attr in _DETECTORS:
        scores = np.array([getattr(r, attr) for r in eligible], dtype=float)
        fpr, tpr = roc_curve(labels, scores)
        ax.plot(fpr, tpr, marker=".", label=f"{name} (AUC={roc_auc(labels, scores):.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Unverbalized-cue detection")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
