"""
threshold_utils.py

Utility functions for computing per-class reject thresholds for
Markov-model classification.
"""

import numpy as np
from per_base_likelihood import per_base_likelihood


def compute_thresholds(training_data, models, k, percentile=1):
    """
    Compute per-class reject thresholds using a percentile cutoff.
    Any sequence scoring below this percentile (within its true class)
    will be rejected during classification.

    Parameters:
        training_data : dict[class_label -> list of sequences]
        models        : dict[class_label -> trained model]
        k             : k-mer length
        percentile    : percentile cutoff (default = 1)

    Returns:
        dict[class_label -> threshold]
    """

    thresholds = {}

    for class_label, seqs in training_data.items():
        scores = [per_base_likelihood(seq, models[class_label], k) for seq in seqs]
        threshold = np.percentile(scores, percentile)
        thresholds[class_label] = threshold

    return thresholds
