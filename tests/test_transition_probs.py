from src.markov_model import count_kmers, estimate_transition_probs

def test_probs_sum_to_one():
    # Extract raw kmer frequency counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts to probabilities with smoothing
    model = estimate_transition_probs(counts, k=1)

    assert abs(sum(model["A"].values()) - 1.0) < 1e-9

def test_seen_vs_unseen():
    # extract raw kmer frequency counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts to probabilities with smoothing
    model = estimate_transition_probs(counts, k=1)

    assert model["A"]["A"] > model["A"]["G"]
    assert model["A"]["A"] > model["A"]["T"]

def test_all_positive():
    # Extract raw kmer frequency counts
    counts = count_kmers("AAC", k=1)
    # Convert raw counts to probabilities with smoothing
    model = estimate_transition_probs(counts, k=1)

    for p in model["A"].values():
        assert p > 0
