"""
training_utils.py

This module contains the shared training logic used by both Prototype Run 1 (raw
log-likelihood) and Run 2 (per-base normalized log-likelihood), as well as for
the final run. There are no modifications from the prototype run impacting this
script. It provides functions for training k-mer Markov models from labeled DNA
sequence data.

The functions in this module do not perform any scoring or classification.
They only:
    - count k-mers across sequences
    - apply Laplace smoothing
    - convert counts into log-probabilities
    - train one model per class label

Because training is identical for all runs, this module is imported by
both pipelines without introducing any cross-contamination between the
raw and normalized scoring logic.
"""


from kmer_utils import aggregate_kmer_counts, normalize_counts


def train_markov_model(seqs, k):
    """
    Train a Markov model by counting kmers across all sequences and converting
    those counts into smoothed log-probabilities
    Parameters:
        seqs (list[str]): list of DNA sequences used for training
        k (int): k-mer length
    Returns:
        dict[str, float]: mapping of k-mer → log probability, including "_UNSEEN_"
    """

    if not isinstance(k, int):  # ensure k is an integer
        raise TypeError(f"k must be an integer, got {type(k)}")  # raise explicit type error

    if not isinstance(seqs, list):  # ensure seqs a list or error
        raise TypeError(f"seqs must be a list of strings, got {type(seqs)}")

    raw_counts = aggregate_kmer_counts(seqs, k)  # count all kmers across sequences
    model = normalize_counts(raw_counts, k)  # convert raw counts to log-probabilities

    return model  # return trained model dictionary


def train_all_models(training_data, k):
    """
    Train one Markov model per class label
    Parameters:
        training_data (dict[str, list[str]]): mapping of class_label → list of DNA sequences for that class
        k (int): kmer length
    Returns:
        dict[str, dict[str, float]]: mapping of class_label → trained Markov model (kmer → log probability)
    Raises:
        TypeError: if training_data not a dict
        TypeError: if k not an int
    """

    if not isinstance(training_data, dict):
        raise TypeError(f"training_data must be a dict, got {type(training_data)}")

    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k)}")

    models = {}

    for class_label, seqs in training_data.items():
        model = train_markov_model(seqs, k)
        models[class_label] = model

    return models