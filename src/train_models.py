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
        dict: models[k][class_label] = probability model
    """

    if not training_data:
        raise ValueError("No training data provided")

    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    models = {}

    for k in k_values:
        if k < 1 or k > 6:
            raise ValueError("k must be between 1 and 6")

        models[k] = {}

        # Iterate over each class and its sequences
        for class_label, seqs in training_data.items():

            if not seqs:
                raise ValueError(f"No sequences provided for class '{class_label}'")

            # Extract feature name from class label (e.g., promoter_positive -> promoter)
            feature = class_label.split("_")[0]

            # Check sequence length edge case
            if all(len(seq) <= k for seq in seqs):
                # Undefined mathematically — track as empty model
                models[k][class_label] = {}
                continue

            # Train model for this class and k
            probs = train_markov(seqs, k, alpha)
            models[k][class_label] = probs

    return models
