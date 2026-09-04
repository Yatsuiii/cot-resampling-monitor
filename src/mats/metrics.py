"""ROC / PR metrics with bootstrap confidence intervals.

Self-contained (numpy only) so the Kaggle runner needs no scikit-learn. AUC is
the Mann-Whitney statistic with tie-aware average ranks; the bootstrap resamples
items with replacement to put an interval on it, which matters here because
n is small (40).
"""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average ranks within tied groups.
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    thresholds = np.concatenate([[np.inf], np.unique(scores)[::-1]])
    tpr, fpr = [], []
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(float((predicted & (labels == 1)).sum()) / n_pos)
        fpr.append(float((predicted & (labels == 0)).sum()) / n_neg)
    return np.array(fpr), np.array(tpr)


def bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, *, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    index = np.arange(len(labels))
    draws = []
    for _ in range(n):
        pick = rng.choice(index, size=len(index), replace=True)
        value = roc_auc(labels[pick], scores[pick])
        if not np.isnan(value):
            draws.append(value)
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
