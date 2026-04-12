from src.markov_model import count_kmers, estimate_transition_probs, log_likelihood
import math

def test_perfect_match_higher_score():
    # Extract raw kmer frequency counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts to probabilities using smoothing
    model = estimate_transition_probs(counts, k=1)

    ll_match = log_likelihood("AAC", model, k=1)
    ll_mismatch = log_likelihood("AGG", model, k=1)
    assert ll_match > ll_mismatch

def test_unseen_prefix_uniform():
    # Extract raw kmer freq counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts into probs with smoothing
    model = estimate_transition_probs(counts, k=1)

    ll = log_likelihood("TTT", model, k=1)
    expected = 2 * math.log(0.25)
    assert abs(ll - expected) < 1e-9

def test_missing_next_base_fallback():
    # Extract raw kmer freq counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts into probs with smoothing
    model = estimate_transition_probs(counts, k=1)

    ll = log_likelihood("AG", model, k=1)
    assert abs(ll - math.log(1e-12)) < 1e-9
