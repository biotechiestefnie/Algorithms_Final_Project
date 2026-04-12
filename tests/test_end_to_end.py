from src.markov_model import count_kmers, estimate_transition_probs, log_likelihood

def test_model_ranking():
    counts_a = count_kmers("AACACAA", k=1)
    counts_b = count_kmers("GGGCGGG", k=1)

    model_a = estimate_transition_probs(counts_a, k=1)
    model_b = estimate_transition_probs(counts_b, k=1)

    seq = "AACAA"
    ll_a = log_likelihood(seq, model_a, k=1)
    ll_b = log_likelihood(seq, model_b, k=1)

    assert ll_a > ll_b
