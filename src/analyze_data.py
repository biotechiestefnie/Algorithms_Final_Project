
import csv
from markov_model import classify_sequence

def classify_test_set(test_sequences, models, k):
    """
    Classify each test sequence using trained models for a specific k.
    Returns a list of (sequence, predicted_class, score) tuples.
    """

    # Edge case: k not trained
    # models structured as: models[k][class_label] = model_dict
    if k not in models:
        raise ValueError("No models available for this k")

    # Edge case: no class models exist for k
    # This means models[k] = {} (empty dict)
    if not models[k]:
        raise ValueError("No class models available for classification")

    # Initialize results list for classification outputs
    results = []

    # Main classification loop
    # test_sequences = list of raw DNA strings
    for seq in test_sequences:

        # Skip sequences too short for this k
        if len(seq) <= k:
            results.append((seq, None, float("-inf")))
            continue

        # classify_sequence() returns (predicted_class, log_likelihood_score)
        predicted_class, score = classify_sequence(seq, models[k], k)

        # Append tuple containing sequence, predicted class, score
        results.append((seq, predicted_class, score))

    # Return full list of classification results
    return results


def compute_accuracy(results):
    """
    Compute overall classification accuracy as a percentage
    rounded to 4 decimal places.
    Parameters:
        results (list): list of result dicts from classify_test_set()
                        Each dict must contain: "true_class" , "predicted_class"
    Returns:
        float: accuracy percentage (0.0 to 100.0), rounded to 4 decimals
    """

    # Edge case: no results provided
    # Avoid division by zero; return 0.0% accuracy
    if not results:
        return 0.0

    # Count true class predictions
    correct = sum(
        1 for r in results
        if r["true_class"] == r["predicted_class"]
    )

    # Total number classified sequences
    total = len(results)

    # Compute accuracy as percentage
    accuracy_pct = (correct / total) * 100.0

    # Round to 4 decimal places for reporting consistency
    return round(accuracy_pct, 4)


def compute_summary_stats(results):
    """
    Compute summary statistics from classification results
    Parameters:
        results (list): list of result dicts from classify_test_set()
                        Each dict contains: "true_class", "predicted_class",
                                            "log_likelihood",  "sequence"
    Returns:
        dict: summary statistics including: "total", "unclassified",
                                            "neg_inf", "class_counts"
    """

    # Edge case: no results
    # Return zeros and empty class_counts
    if not results:
        return {
            "total": 0,
            "unclassified": 0,
            "neg_inf": 0,
            "class_counts": {}
        }

    # Initialize dict for statistics
    stats = {
        "total": len(results),  # Total sequences processed

        # Count sequences where predicted_class is None
        "unclassified": sum(
            1 for r in results if r["predicted_class"] is None
        ),

        # Count sequences where score == -inf
        "neg_inf": sum(
            1 for r in results if r["log_likelihood"] == float("-inf")
        )
    }

    # Initialize class counts dictionary
    class_counts = {}

    # Count times each predicted class appears
    for r in results:
        cls = r["predicted_class"]

        # Skip None predictions (unclassified)
        if cls is None:
            continue

        # Initialize count for new class
        if cls not in class_counts:
            class_counts[cls] = 0

        # Increment count
        class_counts[cls] += 1

    stats["class_counts"] = class_counts

    return stats


def confusion_matrix(results):
    """
    Build confusion matrix from classification results for per k reporting
    Parameters:
        results (list): output of classify_test_set()
    Returns:
        dict of dicts: true_class -> predicted_class -> count
    """

    # collect all class labels
    classes = sorted(set(r["true_class"] for r in results))

    # initialize matrix
    matrix = {c: {p: 0 for p in classes} for c in classes}

    for r in results:
        t = r["true_class"]
        p = r["predicted_class"]

        # Skip unclassified sequences
        if p is None:
            continue

        matrix[t][p] += 1

    return matrix


def evaluate_model_performance(all_results, true_labels):
    """
    Compute advanced evaluation metrics across all k values, including:
    accuracy vs k, confusion matrices, and per-class likelihood distributions
    Parameters:
        all_results (dict): map k -> list of tuples:
                                     (sequence, predicted_class, score, class_scores)
        true_labels (dict): map sequence -> true_class
    Returns:
        dict containing: "accuracy_vs_k", "confusion_matrices", "likelihood_distributions" for all orders k
    """

    metrics = {}  # initialize metrics dictionary

    # Accuracy versus k
    accuracy_vs_k = {}  # initialize accuracy dictionary

    for k in all_results:

        # Edge case: no results for this k
        if not all_results[k]:
            continue

        correct = 0  # count correct predictions
        total = len(all_results[k])  # total sequences for k

        for (seq, pred, _, _) in all_results[k]:

            # Edge case: missing true label
            if seq not in true_labels:
                raise ValueError(f"True label missing for sequence: {seq}")

            # Increment if prediction matches truth
            if pred == true_labels[seq]:
                correct += 1

        # compute accuracy for k
        accuracy_vs_k[k] = correct / total if total > 0 else 0.0

    metrics["accuracy_vs_k"] = accuracy_vs_k

    # Confusion matrices for all k
    confusion_matrices = {}  # initialize confusion matrix dictionary

    # collect all class labels from true_labels
    class_set = sorted(set(true_labels.values()))

    for k in all_results:

        # initialize confusion matrix for k:
        # matrix[true][pred] = count
        matrix = {true: {pred: 0 for pred in class_set} for true in class_set}

        for (seq, pred, _, _) in all_results[k]:

            # Edge case: missing true label
            if seq not in true_labels:
                raise ValueError(f"True label missing for sequence: {seq}")

            true = true_labels[seq]

            # Edge case: predicted_class is None
            if pred is None:
                continue  # no increment for any cell

            # Edge case: predicted class not recognized
            if pred not in matrix[true]:
                raise ValueError(f"Predicted class '{pred}' not recognized")

            # increment confusion matrix cell
            matrix[true][pred] += 1

        confusion_matrices[k] = matrix

    metrics["confusion_matrices"] = confusion_matrices

    # Per class likelihood distributions
    likelihood_distributions = {}  # initialize likelihood dictionary

    for k in all_results:

        per_class = {}  # per_class[class_label] = list of likelihoods

        for (seq, pred, score, class_scores) in all_results[k]:

            # Edge case: missing class_scores
            if class_scores is None:
                raise ValueError(f"Class scores missing for sequence: {seq}")

            # Append each class score to corresponding  list
            for class_label in class_scores:

                if class_label not in per_class:
                    per_class[class_label] = []

                per_class[class_label].append(class_scores[class_label])

        likelihood_distributions[k] = per_class

    metrics["likelihood_distributions"] = likelihood_distributions

    return metrics


def write_output_files(results, stats):
    """
    Write classification results and summary statistics to output files
    for downstream analysis
    Parameters:
        results (list): list of (sequence, predicted_class, score) OR
                        list of dicts from classify_test_set()
        stats (dict): summary statistics from compute_summary_stats()
    Returns:
        None
    """

    # Write per-sequence results to csv
    # Edge case: results empty -> writes header only
    with open("per_sequence_results.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(["sequence", "predicted_class", "score"])

        # Write each result row
        for r in results:
            # Support both tuple format and dict format
            if isinstance(r, dict):
                seq = r["sequence"]
                cls = r["predicted_class"]
                score = r["log_likelihood"]

            else:
                seq, cls, score = r

            writer.writerow([seq, cls, score])

    #  Write summary statistics to TXT
    # Edge case: stats empty -> writes zeros
    with open("summary_stats.txt", "w") as f:

        # Total sequences processed
        f.write("Total Sequences: " + str(stats.get("total", 0)) + "\n")

        # Number unclassified
        f.write("Unclassified: " + str(stats.get("unclassified", 0)) + "\n")

        # Number -inf scores
        f.write("Negative Infinite Scores: " + str(stats.get("neg_inf", 0)) + "\n")

        # Class counts section
        f.write("Class Counts:\n")

        class_counts = stats.get("class_counts", {})

        # Edge case: no class counts -> writes nothing under header
        for class_label in class_counts:
            f.write("  " + class_label + ": " + str(class_counts[class_label]) + "\n")

    return

