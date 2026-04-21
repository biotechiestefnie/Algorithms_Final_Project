"""
per_base_likelihood.py

This module provides the normalized log-likelihood function for Run 2.
It only computes the per-base log-likelihood of a sequence under a
trained Markov model.

The model is represented as:
    dict[str, float]  mapping k-mer → log probability

Unseen k-mers are handled using the model["_UNSEEN_"] value produced by
Laplace smoothing in kmer_utils.normalize_counts().
"""

import math  # for log calculations


def per_base_likelihood(seq, model, k):
    """
    Compute the total log-likelihood of a DNA sequence under a trained Markov model.
    The sequence is decomposed into overlapping kmers. Each kmer contributes its
    log-probability from the model. Unseen kmers use the model["_UNSEEN_"] value.
    Parameters:
        seq (str): DNA sequence to evaluate
        model (dict[str, float]): trained Markov model mapping kmer → log probability
        k (int): kmer length
    Returns:
        float: normalized log-likelihood (total log-likelihood divided by sequence length)
    """

    if not isinstance(seq, str):
        raise TypeError(f"seq must be a string, got {type(seq)}")

    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k)}")

    if "_UNSEEN_" not in model:
        raise KeyError("Model is missing '_UNSEEN_' key for unseen k-mers.")

    L = len(seq)
    if L < k:
        raise ValueError(
            f"Sequence length {L} is shorter than k={k}; cannot compute likelihood."
        )

    total = 0.0

    for i in range(L - k + 1):
        kmer = seq[i:i + k]
        total += model.get(kmer, model["_UNSEEN_"])

    return total / L
