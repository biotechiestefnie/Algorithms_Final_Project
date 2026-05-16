"""
threshold_utils.py

Utility functions for computing per-class reject thresholds for
Markov-model classification.
"""

import numpy as np


def seq_loglik(seq, k, model_for_class_and_k, unseen_penalty):
    total = 0.0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        total += model_for_class_and_k.get(kmer, unseen_penalty)

    return total


def compute_thresholds(models, training_data, percentile=5):
    """
    Compute per-class reject thresholds using the SAME per-base
    log-likelihood scale used during classification
    Parameters:
        models        : dict[k -> dict[class_label -> model]]
        training_data : dict[class_label -> list of sequences]
        percentile    : percentile cutoff (e.g., 5 or 10)
    Returns:
        dict[k -> dict[class_label -> per-base LL threshold]]
    """

    thresholds = {}

    for k, class_models in models.items():
        thresholds[k] = {}

        for cls, kmer_logps in class_models.items():
            unseen = kmer_logps["_UNSEEN_"]

            per_base_scores = []
            for seq in training_data[cls]:
                total_ll = seq_loglik(seq, k, kmer_logps, unseen)

                # Calculate via per base likelihood to minimize length influence
                per_base_ll = total_ll / len(seq)

                per_base_scores.append(per_base_ll)

            per_base_scores_sorted = sorted(per_base_scores)
            idx = max(0, int(len(per_base_scores_sorted) * (percentile / 100)) - 1)

            thresholds[k][cls] = per_base_scores_sorted[idx]

    return thresholds
