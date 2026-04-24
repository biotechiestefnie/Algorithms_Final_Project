"""
test_evaluate_results.py

Pytest suite validating statistical evaluation utilities in evaluate_results.py.
Covers:
- Confusion matrix construction
- Per-class precision, recall, and F1-score
- Overall accuracy computation
- Log-likelihood summary statistics
- Edge cases involving empty inputs and zero-division behavior
"""

import numpy as np
import pytest

from src.evaluate_results import (
    compute_confusion_matrix,
    compute_class_metrics,
    compute_overall_accuracy,
    summarize_loglikelihoods,
)


def test_compute_confusion_matrix_basic():
    """
    Verify confusion matrix counts true/predicted pairs correctly
    """
    class_labels = ["A", "B", "C"]

    # results: (true, predicted, sequence)
    results = [
        ("A", "A", "seq1"),
        ("A", "B", "seq2"),
        ("B", "B", "seq3"),
        ("C", "A", "seq4"),
    ]

    # Expected matrix:
    # rows = true, cols = predicted
    # A: [1,1,0]
    # B: [0,1,0]
    # C: [1,0,0]
    expected = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0]
    ])

    result = compute_confusion_matrix(results, class_labels)
    assert np.array_equal(result, expected)


def test_compute_confusion_matrix_empty():
    """
    Empty results list should yield an all-zero confusion matrix
    """

    class_labels = ["A", "B"]
    results = []

    expected = np.zeros((2, 2), dtype=int)
    result = compute_confusion_matrix(results, class_labels)

    assert np.array_equal(result, expected)


def test_compute_class_metrics_basic():
    """
    Check precision, recall, and F1 for a simple confusion matrix
    """
    # Confusion matrix:
    # TP for A = 3, FP = 1, FN = 1
    conf = np.array([
        [3, 1],
        [1, 4]
    ])
    labels = ["A", "B"]

    metrics = compute_class_metrics(conf, labels)

    # A:
    # precision = 3 / (3+1) = 0.75
    # recall    = 3 / (3+1) = 0.75
    # f1        = 0.75
    assert metrics["A"]["precision"] == pytest.approx(0.75)
    assert metrics["A"]["recall"] == pytest.approx(0.75)
    assert metrics["A"]["f1"] == pytest.approx(0.75)

    # B:
    # precision = 4 / (4+1) = 0.8
    # recall    = 4 / (4+1) = 0.8
    # f1        = 0.8
    assert metrics["B"]["precision"] == pytest.approx(0.8)
    assert metrics["B"]["recall"] == pytest.approx(0.8)
    assert metrics["B"]["f1"] == pytest.approx(0.8)


def test_compute_class_metrics_zero_division():
    """
    Classes with no predicted or true positives should yield zero metrics
    """
    conf = np.array([
        [0, 0],
        [0, 0]
    ])
    labels = ["A", "B"]

    metrics = compute_class_metrics(conf, labels)

    assert metrics["A"]["precision"] == 0.0
    assert metrics["A"]["recall"] == 0.0
    assert metrics["A"]["f1"] == 0.0
    assert metrics["B"]["precision"] == 0.0
    assert metrics["B"]["recall"] == 0.0
    assert metrics["B"]["f1"] == 0.0


def test_compute_overall_accuracy_basic():
    """
    Verify accuracy = trace(conf_matrix) / total
    """
    conf = np.array([
        [3, 1],
        [2, 4]
    ])
    # correct = 3 + 4 = 7
    # total = 10
    expected = 0.7

    result = compute_overall_accuracy(conf)
    assert result == pytest.approx(expected)


def test_compute_overall_accuracy_empty():
    """
    Zero-total confusion matrix should yield accuracy = 0.0
    """

    conf = np.zeros((3, 3), dtype=int)
    assert compute_overall_accuracy(conf) == 0.0


def test_summarize_loglikelihoods_basic():
    """
    Check mean/variance and separation for simple score lists
    """
    pos = [1.0, 2.0, 3.0]
    neg = [-1.0, -2.0, -3.0]

    stats = summarize_loglikelihoods(pos, neg)

    assert stats["pos_mean"] == pytest.approx(2.0)
    assert stats["neg_mean"] == pytest.approx(-2.0)
    assert stats["pos_var"] == pytest.approx(np.var(pos))
    assert stats["neg_var"] == pytest.approx(np.var(neg))
    assert stats["separation"] == pytest.approx(4.0)  # 2 - (-2)


def test_summarize_loglikelihoods_empty_lists():
    """
    Empty score lists should produce NaN means/vars; separation should be NaN
    """
    pos = []
    neg = []

    stats = summarize_loglikelihoods(pos, neg)

    # np.mean([]) and np.var([]) both return NaN
    assert np.isnan(stats["pos_mean"])
    assert np.isnan(stats["neg_mean"])
    assert np.isnan(stats["pos_var"])
    assert np.isnan(stats["neg_var"])
    assert np.isnan(stats["separation"])
