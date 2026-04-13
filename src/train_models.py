from markov_model import count_kmers, estimate_transition_probs



def train_markov(seqs, k, alpha):
    """
    Train a k-order Markov model from a list of sequences (strings).
    Returns a dict: prefix -> {next_base -> probability}
    """

    # Count (k+1)-mers across all sequences
    kmer_counts = count_kmers(seqs, k)

    # Convert counts to transition probabilities with Laplace smoothing
    probs = estimate_transition_probs(kmer_counts, k, alpha)

    return probs



def train_all_models(training_data, k_values, alpha):
    """
    Train Markov models for each class and each k-value.
    Parameters:
        training_data (dict): class_label -> [seqs]
        k_values (list): list of k orders to train
        alpha (float): Laplace smoothing constant
    Returns:
        dict: models[class_label][k] = probability model
    """

    if not training_data:
        raise ValueError("No training data provided")

    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    # Outer dict keyed by class_label
    models = {}

    # Loop over each class and its sequences
    for class_label, seqs in training_data.items():

        if not seqs:
            raise ValueError(f"No sequences provided for class '{class_label}'")

        # Initialize inner dict for this class
        models[class_label] = {}

        # Train a model for each k
        for k in k_values:

            if k < 1 or k > 6:
                raise ValueError("k must be between 1 and 6")

            # Edge case: sequences too short for this k
            if all(len(seq) <= k for seq in seqs):
                models[class_label][k] = {}
                continue

            # Train model for this class and k
            probs = train_markov(seqs, k, alpha)
            models[class_label][k] = probs

    return models
