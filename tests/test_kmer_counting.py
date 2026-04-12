# Import module
from src.markov_model import count_kmers

def test_k1_simple():
    """
    Tests that the kmer counting algorithm works as expected.
    Parameters:
            None
    Returns:
        None
    """
    counts = count_kmers("AAC", k=1)
    assert counts == {"A": {"A": 1, "C": 1}}


def test_k2_simple():
        counts = count_kmers("AACA", k=2)
        assert counts == {"AA": {"C": 1}, "AC": {"A": 1}}


def test_empty_sequence():
        assert count_kmers("", k=1) == {}


def test_shorter_than_k():
        assert count_kmers("A", k=2) == {}
