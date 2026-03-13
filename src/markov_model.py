# src/markov_model.py

# Import packages
import math  # for computing log probabilities

# Import modules
from collections import defaultdict  # for nested dictionaries with default values


def count_kmers(sequence, k):
    """
    Count all (k+1)-mers in sequence
    For a k-order Markov model, we need counts of:
        prefix (length k) -> next base
    Parameters:
        sequence (str): DNA sequence consisting of characters A, C, G, T
        k (int): Markov order; number of previous bases used as context
    Returns:
        dict: Nested dictionary where:
              counts[prefix][next_base] = integer count
              prefix is a string of length k
              next_base is a single character (A, C, G, T)
    """

    # Create a nested dictionary:
    # Outer keys = prefixes of length k
    # Inner keys = next bases, values = counts
    counts = defaultdict(lambda: defaultdict(int))

    # Loop over valid positions where a (k+1)-mer exists
    # range stops at len(sequence) - k so that sequence[i+k] is valid
    for i in range(len(sequence) - k):

        # Extract prefix of length k starting at position i
        prefix = sequence[i:i+k]

        # Extract base immediately following prefix
        next_base = sequence[i+k]

        # Increment count for prefix → next_base transition
        counts[prefix][next_base] += 1

    # Return nested dictionary of counts
    return counts



def estimate_transition_probs(kmer_counts, k, smoothing=1):
    """
    Convert k-mer counts into conditional probabilities with Laplace smoothing.
    For each prefix of length k:
        P(next_base | prefix) = (count + smoothing) / (total + 4*smoothing)
    Parameters:
        kmer_counts (dict): Nested dictionary mapping:
                            prefix (str of length k) →
                                dict of next_base → count
        k (int): Markov order (not used directly here but included for clarity)
        smoothing (int): Laplace smoothing constant (default = 1)
    Returns:
        dict: Nested dictionary where:
              model[prefix][next_base] = probability (float)
    """

    # Initialize dictionary to store final probability model
    model = {}

    # Iterate over each prefix and dictionary of next-base counts
    for prefix, next_base_counts in kmer_counts.items():

        # Initialize inner dictionary for prefix
        model[prefix] = {}

        # Compute denominator for Laplace smoothing:
        # total observed counts + 4*smoothing (one for each base)
        total = sum(next_base_counts.values()) + 4 * smoothing

        # Iterate over all possible next bases in DNA
        for base in ["A", "C", "G", "T"]:

            # Get observed count for this base (0 if unseen)
            count = next_base_counts.get(base, 0)

            # Apply Laplace smoothing, compute probability
            model[prefix][base] = (count + smoothing) / total

    # Return full probability model
    return model



def log_likelihood(sequence, model, k):
    """
    Compute log-likelihood of sequence under trained k-order Markov model
    Uses:
        log P(sequence) = sum over positions of log P(x_i | prefix)
    If prefix not seen during training, revert to: uniform probability = 1/4
    to avoid log(0) collapse
    Parameters:
        sequence (str): DNA sequence to evaluate
        model (dict): Nested dictionary of conditional probabilities
        k (int): Markov order
    Returns:
        float: log-likelihood of the sequence under the model
    """

    # Initialize running log-likelihood
    logL = 0.0

    # Precompute log(1/4) for unseen prefixes
    uniform = math.log(1/4)

    # Loop over all valid positions where prefix + next base exists
    for i in range(len(sequence) - k):

        # Extract prefix of length k
        prefix = sequence[i:i+k]

        # Extract next base
        next_base = sequence[i+k]

        # If prefix exists in model, use probability distribution
        if prefix in model:

            # Get probability of next_base; fallback epsilon value if missing
            prob = model[prefix].get(next_base, 1e-12)

            # Add log(probability) to running total
            logL += math.log(prob)

        # If prefix never seen during training
        else:
            # Use uniform probability log(1/4)
            logL += uniform

    # Return total log-likelihood
    return logL
