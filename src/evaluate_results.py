"""
evaluate_results.py

This module provides statistical evaluation utilities for analyzing the
performance of a multi-class Markov model classifier.

It computes:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix
- Log-likelihood separation statistics
- plotting utilities for visualizing class separation

"""


# Import packages
import numpy as np # for numerical operations


def compute_confusion_matrix(results, class_labels):
    """
    Compute a confusion matrix from classification results
    Parameters:
        results (list[tuple[str, str, str]]):
            list of (true_label, predicted_label, sequence)
        class_labels (list[str]):
            ordered list of class names
    Returns:
        np.ndarray: confusion matrix of shape (num_classes, num_classes)
                    rows = true labels
                    cols = predicted labels
    """


    n = len(class_labels)  # number of classes
    matrix = np.zeros((n, n), dtype=int)  # initialize confusion matrix

    label_to_idx = {}
    for i, label in enumerate(class_labels):
        label_to_idx[label] = i

    for true_label, predicted_label, _ in results:  # iterate through results
        i = label_to_idx[true_label]  # row index
        j = label_to_idx[predicted_label]  # column index
        matrix[i, j] += 1  # increment confusion count

    return matrix  # return confusion matrix


def compute_class_metrics(conf_matrix, class_labels):
    """
    Compute precision, recall, and F1-score for each class
    Parameters:
        conf_matrix (np.ndarray): confusion matrix
        class_labels (list[str]): ordered list of class names
    Returns:
        dict[str, dict[str, float]]:
            mapping of class_label → {precision, recall, f1}
    """

    metrics = {}  # initialize metrics dictionary

    for idx, label in enumerate(class_labels):  # iterate through classes
        tp = conf_matrix[idx, idx]  # true positives
        fp = conf_matrix[:, idx].sum() - tp  # false positives
        fn = conf_matrix[idx, :].sum() - tp  # false negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # compute precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # compute recall

        if precision + recall > 0:  # avoid division by zero
            f1 = 2 * precision * recall / (precision + recall)  # compute F1
        else:
            f1 = 0.0  # fallback if undefined

        metrics[label] = {  # store metrics for this class
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics  # return dictionary of metrics


def compute_overall_accuracy(conf_matrix):
    """
    Compute overall classification accuracy
    Parameters:
        conf_matrix (np.ndarray): confusion matrix
    Returns:
        float: accuracy value
    """

    correct = np.trace(conf_matrix)  # sum of diagonal elements
    total = conf_matrix.sum()  # total number of predictions
    return correct / total if total > 0 else 0.0  # compute accuracy safely


def compute_per_class_accuracy(conf_matrix, class_labels):
    """
    Compute per-class accuracy values.
    Parameters:
        conf_matrix (np.ndarray): confusion matrix
        class_labels (list[str]): ordered list of class names
    Returns:
        dict[str, float]: mapping class_label → accuracy
    """
    n = len(class_labels)
    accuracies = {}
    total = conf_matrix.sum()

    for idx, label in enumerate(class_labels):
        tp = conf_matrix[idx, idx]
        fp = conf_matrix[:, idx].sum() - tp
        fn = conf_matrix[idx, :].sum() - tp
        tn = total - (tp + fp + fn)
        acc = (tp + tn) / total if total > 0 else 0.0
        accuracies[label] = acc

    return accuracies


def summarize_loglikelihoods(pos_scores, neg_scores):
    """
    Compute summary statistics for  log-likelihoods
    Parameters:
        pos_scores (list[float]): log-likelihoods for true class sequences
        neg_scores (list[float]): log-likelihoods for non-class sequences
    Returns:
        dict[str, float]: summary statistics including means and variances
    """

    return {  # return dictionary of summary statistics
        "pos_mean": np.mean(pos_scores),
        "pos_var": np.var(pos_scores),
        "neg_mean": np.mean(neg_scores),
        "neg_var": np.var(neg_scores),
        "separation": np.mean(pos_scores) - np.mean(neg_scores)
    }
