"""
kmer_utils.py

Utility functions for:
- Counting k-mers in individual sequences
- Aggregating k-mer counts across multiple sequences
- Converting raw counts into smoothed log-probabilities

"""


# Import packages
import math  # For logarithms

# Import modules
from collections import defaultdict  # For automatic int initialization



def count_kmers(seq, k):
    """
    Count all k-mers in single DNA sequence
    Parameters:
        seq (str): DNA sequence composed of A/C/G/T characters
        k (int): length of each k-mer
    Returns:
        dict[str, int]: mapping of kmer → count within the sequence
    """

    # initialize dictionary that defaults to 0 for unseen keys
    counts = defaultdict(int)

    n = len(seq)  # compute sequence length
    if n < k:  # if sequence shorter than k, invalid input for kmer counting
        raise ValueError(  # raise explicit error
            f"Sequence length {n} is shorter than k={k}; cannot extract k-mers."
        )

    for i in range(n - k + 1):  # iterate over all valid kmer start positions
        kmer = seq[i:i+k]  # extract kmer substring
        counts[kmer] += 1  # increment count for kmer

    return dict(counts)  # return default dict of counts


def aggregate_kmer_counts(seqs, k):
    """
    Aggregate kmer counts across list of DNA sequences
    Parameters:
        seqs (list[str]): list of DNA sequences
        k (int): kmer length
    Returns:
        dict[str, int]: mapping of kmer → total count across all sequences
    """

    total = defaultdict(int)  # initialize global count dict

    for seq in seqs:  # iterate through each seq
        seq_counts = count_kmers(seq, k)  # call count kmers for seq
        for kmer, c in seq_counts.items():  # iterate through kmers in sequence
            total[kmer] += c  # add counts to global total

    return dict(total)  # return default dict of total counts


def normalize_counts(counts, k):
    """
    Convert raw kmer counts into normalized log-probabilities using
    Laplace smoothing:
        P(kmer) = (count(kmer) + 1) / (total_counts + V)
        where V = 4^k (size of k-mer pop)
    Parameters:
        counts (dict[str, int]): raw kmer counts
        k (int): kmer length
    Returns:
        dict[str, float]: mapping of k-mer → log probability
                          includes a special "_UNSEEN_" key for unseen k-mers
    """

    alphabet = ["A", "C", "G", "T"]  # define DNA alphabet
    vocab_size = len(alphabet) ** k  # compute number possible kmers

    total_count = sum(counts.values())  # sum all observed kmer counts
    denom = total_count + vocab_size  # denominator for Laplace smoothing

    log_probs = {}  # initialize dictionary for log probabilities

    for kmer, raw_count in counts.items():  # iterate through observed kmers
        prob = (raw_count + 1) / denom  # apply Laplace smoothing
        log_probs[kmer] = math.log(prob)  # store log probability

    unseen_prob = 1 / denom  # probability for unseen kmers
    log_probs["_UNSEEN_"] = math.log(unseen_prob)  # store log probability for unseen

    return log_probs  # return dictionary of log probabilities
