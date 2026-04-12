"""
Unit tests for the functional Markov model implementation.

These tests verify:
- Correct counting of (k+1)-mers by count_kmers().
- Correct Laplace-smoothed probability estimation by estimate_transition_probs().
- Correct computation of log-likelihood by log_likelihood().

All tests use small, deterministic sequences to ensure clarity and reproducibility.
"""

from src.markov_model import count_kmers, estimate_transition_probs, log_likelihood
import math


def test_count_kmers_basic():
    """
    Test that count_kmers() correctly counts (k+1)-mers in simple sequences
    Asserts nested dictionary of prefix → next_base counts matches expected values
    Parameters
        None
    Returns
        None
    """
    sequence = "AAC"
    k = 1

    # Expected transitions:
    # A -> A
    # A -> C
    counts = count_kmers(sequence, k)

    assert counts["A"]["A"] == 1
    assert counts["A"]["C"] == 1

    # No other prefixes should exist
    assert "C" not in counts


def test_estimate_transition_probs_laplace():
    """
    Test that estimate_transition_probs() applies Laplace smoothing correctly
    Asserts prefix probabilities sum to 1.0, unseen transitions receive non-zero pr
    Parameters
        None
    Returns
        None
    """
    sequence = "AAC"
    k = 1

    # Compute raw counts
    counts = count_kmers(sequence, k)

    # Compute smoothed probabilities
    model = estimate_transition_probs(counts, k, smoothing=1)

    # For prefix "A", outgoing probabilities must sum to 1.0
    total_prob = sum(model["A"].values())
    assert abs(total_prob - 1.0) < 1e-9

    # Unseen transitions must have non-zero probability
    assert model["A"]["G"] > 0
    assert model["A"]["T"] > 0


def test_log_likelihood_simple_sequence():
    """
    Test that log_likelihood() computes correct log probability
    for simple sequence under trained model
    Asserts that computed log-likelihood matches expected sum of log transition probs
    Parameters
        None
    Returns
        None
    """
    sequence = "AAC"
    k = 1

    # Train model on the same sequence
    counts = count_kmers(sequence, k)
    model = estimate_transition_probs(counts, k, smoothing=1)

    # Manually compute expected log-likelihood:
    # Transitions: A->A, A->C
    p_AA = model["A"]["A"]
    p_AC = model["A"]["C"]
    expected_logL = math.log(p_AA) + math.log(p_AC)

    # Compute using function
    computed_logL = log_likelihood(sequence, model, k)

    assert abs(computed_logL - expected_logL) < 1e-12  # Fallback probability-> epsilon
