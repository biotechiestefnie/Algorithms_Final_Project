from markov_model import count_kmers, estimate_transition_probs



def train_markov(seqs, k, alpha):
    """
    Train a k-order Markov model from a list of sequences (strings).
    Returns a dict: prefix -> {next_base -> probability} transition probs
    """

    # Count (k+1)-mers across all sequences
    kmer_counts = count_kmers(seqs, k)

    # Convert counts to transition probabilities with Laplace smoothing
    probs = estimate_transition_probs(kmer_counts, k, alpha)

    return probs



def train_all_models(training, k_values, alpha):
    """
    Train Markov models for each class and each k-value
    Parameters:
        training (dict): training data dictionary of class labels: sequences
        k_values (list int): k value orders to train models with
        alpha (float): LaPlace Smoothing Constant (pseudocount added to every transition
                       when estimating)
    Returns:
        models[k][class_label] = probability model
    """

    # If no available data for training model, raise error
    if not training:
        raise ValueError("No training data provided")

    # If LaPlace Smoothing pseudocount equal to zero or negative, return error
    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    # Outer dict keyed by k
    models = {}  # Initialize dictionary for models

    # Iterate through k values
    for k in k_values:

        # Set boundaries to prevent overgeneralization or underfitting
        if k < 1 or k > 6:
            # Raise error if not within boundaries
            raise ValueError("k must be between 1 and 6")

        # Assign k to models
        models[k] = {}

        for class_label, seqs in training.items():

            # If no seqs available for training, raise error
            if not seqs:
                raise ValueError(f"No sequences provided for class '{class_label}'")

            # Edge case: all sequences too short
            if all(len(seq) <= k for seq in seqs):
                models[k][class_label] = {}
                continue

            # Train model for this class and k
            probs = train_markov(seqs, k, alpha)
            models[k][class_label] = probs

    return models

