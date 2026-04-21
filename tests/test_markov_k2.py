from src.per_base_likelihood import count_kmers, estimate_transition_probs

def test_k2_prefixes_exist():
    counts = count_kmers("AACA", k=2)
    model = estimate_transition_probs(counts, k=2)
    assert "AA" in model
    assert "AC" in model
    assert "CA" in model

def test_k2_probs_sum_to_one():
    counts = count_kmers("AACA", k=2)
    model = estimate_transition_probs(counts, k=2)
    for prefix in model:
        assert abs(sum(model[prefix].values()) - 1.0) < 1e-9
