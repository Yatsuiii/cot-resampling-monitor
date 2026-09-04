import numpy as np

from mats.metrics import bootstrap_auc, roc_auc, roc_curve


def test_auc_perfect_separation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(labels, scores) == 1.0


def test_auc_reversed_scores_is_zero():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(labels, scores) == 0.0


def test_auc_all_tied_is_half():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc(labels, scores) == 0.5


def test_auc_nan_without_both_classes():
    assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_roc_curve_spans_unit_square():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr = roc_curve(labels, scores)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 30 + [1] * 30)
    scores = np.concatenate([rng.normal(0, 1, 30), rng.normal(1.5, 1, 30)])
    point = roc_auc(labels, scores)
    lo, hi = bootstrap_auc(labels, scores, n=500, seed=1)
    assert lo <= point <= hi
