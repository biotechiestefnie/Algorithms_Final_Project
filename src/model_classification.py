"""
model_classification.py

This module generates full classification results including:
- True label
- Predicted label
- Log-likelihood under every class model
- Best-scoring class
- Sequence itself

These results are required for evaluation, plotting, and output writing
"""

from per_base_likelihood import per_base_likelihood


def _score_all_models(seq, models, k):
    """
    Internal helper to compute log-likelihood under each class model.
    Removes duplicated logic from generate_full_results.
    """
    scores = {}
    for class_label, model in models.items():
        scores[class_label] = per_base_likelihood(seq, model, k)
    return scores


def classify_sequence(seq, models, k):
    """
    Classify a single DNA sequence by selecting the class whose Markov model
    assigns the highest log-likelihood.
    This wrapper is kept for convenience after removing classifier.py.
    """
    scores = _score_all_models(seq, models, k)
    return max(scores, key=scores.get)


def classify_with_reject(seq, models, k, thresholds):
    """
    Classification with a reject option using per-class thresholds.
    Parameters:
        seq (str): DNA sequence to classify
        models (dict[str, dict[str, float]]): class_label → Markov model
        k (int): k-mer length
        thresholds (dict[str, float]): class_label → threshold for that class
    Returns:
        (predicted_label, scores)
            predicted_label: str ("promoter", "repeat", or "neither")
            scores: dict[class_label → log-likelihood]
    """

    scores = {cls: per_base_likelihood(seq, model, k)
              for cls, model in models.items()}

    # Best class by LL
    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]

    # Reject if below that class's threshold
    if best_score < thresholds[best_class]:
        return "neither", scores

    return best_class, scores


def generate_full_results(test_data, models, k, threshold):
    """
    Generate full classification results for all sequences in test set
    Parameters:
        test_data (dict[str, list[str]]):
            mapping of true_label → list of sequences
        models (dict[str, dict[str, float]]):
            mapping of class_label → trained Markov model
        k (int): kmer length
        threshold (dict): reject thresholds for each class from classify_with_reject
    Returns:
        list[dict]:
            list of result dictionaries, each containing:
                - true_label
                - predicted_label
                - scores (dict[class_label → log-likelihood])
                - sequence
    """

    results = []

    for true_label, seqs in test_data.items():
        for seq in seqs:

            predicted_label, scores = classify_with_reject(seq, models, k, threshold)

            results.append({
                "true_label": true_label,
                "predicted_label": predicted_label,
                "scores": scores,
                "sequence": seq
            })

    return results
