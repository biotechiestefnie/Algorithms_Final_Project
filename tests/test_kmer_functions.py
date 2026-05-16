"""
test_kmer_functions.py

Pytest suite validating core k‑mer utilities in kmer_utils.py:
- count_kmers: correct extraction and counting of overlapping k‑mers
- aggregate_kmer_counts: correct accumulation of k‑mer counts across sequences
- normalize_counts: correct Laplace‑smoothed log‑probability computation
"""



import math
import pytest

from src.kmer_utils import count_kmers, aggregate_kmer_counts, normalize_counts


def test_count_kmers_basic():
    """
    Verify count_kmers extracts all overlapping k‑mers from a simple sequence
    """
    # simple sequence with overlapping kmers
    seq = "ACGTAC"
    k = 3
    result = count_kmers(seq, k)

    # expected kmers:
    # ACG, CGT, GTA, TAC
    expected = {
        "ACG": 1,
        "CGT": 1,
        "GTA": 1,
        "TAC": 1
    }

    assert result == expected


def test_count_kmers_repeated():
    """
    Ensure repeated overlapping k‑mers are counted correctly
    """

    seq = "AAAAA"
    k = 2
    result = count_kmers(seq, k)

    # kmers: AA, AA, AA, AA
    expected = {"AA": 4}
    assert result == expected


def test_count_kmers_invalid_short_sequence():
    """
    count_kmers should raise ValueError when sequence length < k
    """

    seq = "AC"
    k = 3

    with pytest.raises(ValueError):
        count_kmers(seq, k)


def test_aggregate_kmer_counts_basic():
    """
    Validate aggregation of k‑mer counts across multiple sequences"""

    seqs = ["ACGT", "CGTA"]
    k = 2

    # seq1: AC, CG, GT
    # seq2: CG, GT, TA
    # totals:
    # AC:1, CG:2, GT:2, TA:1
    result = aggregate_kmer_counts(seqs, k)

    expected = {
        "AC": 1,
        "CG": 2,
        "GT": 2,
        "TA": 1
    }

    assert result == expected


def test_aggregate_kmer_counts_empty_list():
    """
    Empty sequence list should yield an empty k‑mer count dictionary
    """

    seqs = []
    k = 3

    # no sequences → no kmers
    result = aggregate_kmer_counts(seqs, k)
    assert result == {}  # empty dict expected


def test_normalize_counts_basic():
    """
    Check Laplace‑smoothed log‑probabilities for observed and unseen k‑mers
    """

    counts = {"AA": 3, "AC": 1}
    k = 2

    # vocab size = 4^2 = 16
    # total_count = 4
    # denom = 4 + 16 = 20

    # P(AA) = (3+1)/20 = 4/20 = 0.2
    # P(AC) = (1+1)/20 = 2/20 = 0.1
    # P(unseen) = 1/20 = 0.05

    result = normalize_counts(counts, k)

    assert math.isclose(result["AA"], math.log(0.2))
    assert math.isclose(result["AC"], math.log(0.1))
    assert math.isclose(result["_UNSEEN_"], math.log(0.05))


def test_normalize_counts_all_unseen():
    """
    When no k‑mers are observed, unseen probability should equal 1 / vocab_size
    """

    counts = {}
    k = 1

    # vocab size = 4^1 = 4
    # total_count = 0
    # denom = 4

    # all kmers unseen → each has prob = 1/4 = 0.25
    result = normalize_counts(counts, k)

    assert math.isclose(result["_UNSEEN_"], math.log(0.25))


def test_normalize_counts_probability_sum_check():
    """
    Exponentiated log‑probs should form valid probabilities (<1, >0)
    """

    counts = {"AA": 1, "AC": 1}
    k = 2

    # vocab size = 16
    # total_count = 2
    # denom = 18

    result = normalize_counts(counts, k)

    # check that all log-probs exponentiate to valid probabilities
    probs = [math.exp(v) for v in result.values()]

    # sum of probabilities for observed + unseen *should* be < 1
    # because unseen covers many kmers not listed individually
    assert all(0 < p < 1 for p in probs)

