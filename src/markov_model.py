

# Import packages
import math  # for computing log probabilities

# Import modules
from collections import defaultdict  # for nested dictionaries with default values


def count_kmers(sequences, k):
    """
    Count all (k+1)-mers across a list of sequences.
    For a k-order Markov model, we need counts of:
        prefix (length k) -> next base
    Parameters:
        sequences (list[str]): list of DNA sequences (strings)
        k (int): Markov order
    Returns:
        dict: Nested dictionary where:
              counts[prefix][next_base] = integer count
    """

    # Nested dictionary:
    # Outer keys = prefixes length k
    # Inner keys = next bases, values = counts
    counts = defaultdict(lambda: defaultdict(int))

    # Loop over each sequence in list
    for seq in sequences:

        # Loop over valid positions with (k+1)-mers
        for i in range(len(seq) - k):

            # Extract prefix length k
            prefix = seq[i:i+k]

            # Extract base immediately following prefix
            next_base = seq[i+k]

            # Increment count for prefix → next_base transition
            counts[prefix][next_base] += 1

    return counts



def estimate_transition_probs(kmer_counts, k, smoothing=1):
    """
    Convert k-mer counts into conditional probabilities with Laplace smoothing
    For each prefix length k:
        P(next_base | prefix) = (count + smoothing) / (total + 4*smoothing)
    Parameters:
        kmer_counts (dict): Nested dictionary mapping:
                            prefix (str of length k) → dict of next_base → count
        k (int): Markov order (not used directly here but included for clarity)
        smoothing (int): Laplace smoothing constant (default = 1)
    Returns:
        model (dict): Nested dictionary where model[prefix][next_base] = probability (float)
    """

    # Initialize dictionary to store final probability model
    model = {}

    # Iterate over each prefix and dictionary next-base counts
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
        float: log-likelihood of sequence under model
    """

    # Initialize running log-likelihood
    logL = 0.0

    # Precompute log(1/4) for unseen prefixes
    uniform = math.log(1/4)

    # Loop over all valid positions where prefix + next base exists
    for i in range(len(sequence) - k):

        # Extract prefix length k
        prefix = sequence[i:i+k]

        # Extract next base
        next_base = sequence[i+k]

        # If prefix exists, use probability distribution
        if prefix in model:

            # If next_base never observed, use fallback epsilon value
            if next_base not in model[prefix] or model[prefix][next_base] == 0:
                logL += math.log(1e-12)
            else:
                # Add log(probability) to running total
                logL += math.log(model[prefix][next_base])

        # If prefix never seen during training
        else:
            # Use uniform probability log(1/4)
            logL += uniform

    # Return total log-likelihood
    return logL


def classify_sequence(sequence, models, k):
    """
    Classify a sequence by computing log-likelihood under each class model
    Parameters:
        sequence (str): DNA sequence to classify
        models (dict): models[class_label][k] = model_dict
        k (int): Markov order
    Returns:
        (best_class, best_logL)
    """

    # Initialize variables for best class and logL
    best_class = None
    best_logL = float("-inf")

    for class_label in models[k]:
        model_for_k = models[k][class_label]

        if model_for_k is None:
            continue  # no model for k

        logL = log_likelihood(sequence, model_for_k, k)

        if logL > best_logL:
            best_logL = logL
            best_class = class_label

    return best_class, best_logL

