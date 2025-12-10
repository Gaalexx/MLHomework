from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def accuracy_score_manual(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must be equal")

    return float((y_true == y_pred).mean())


def precision_score_manual(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def recall_score_manual(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def f1_score_manual(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


def _binary_clf_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    desc_score_indices = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[desc_score_indices]
    y_score_sorted = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    return fps.astype(float), tps.astype(float)


def roc_auc_score_manual(
    y_true: Iterable[int], y_score: Iterable[float]
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length")

    if not (np.any(y_true == 0) and np.any(y_true == 1)):
        raise ValueError("ROC AUC is undefined if y_true has only one class")

    fps, tps = _binary_clf_curve(y_true, y_score)

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    return float(np.trapz(tpr, fpr))


def pr_auc_score_manual(
    y_true: Iterable[int], y_score: Iterable[float]
) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have same length")

    if not np.any(y_true == 1):
        raise ValueError("PR AUC is undefined if there are no positive samples")

    fps, tps = _binary_clf_curve(y_true, y_score)
    P = tps[-1]

    recall = tps / P
    precision = np.divide(
        tps,
        tps + fps,
        out=np.ones_like(tps, dtype=float),
        where=(tps + fps) != 0,
    )

    return float(np.trapz(precision, recall))
