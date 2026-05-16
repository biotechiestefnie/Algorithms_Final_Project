"""
output_writer.py

This module writes all output files produced by the pipeline.
It performs no computation and no plotting.
It only saves:
- classification results
- evaluation metrics
- confusion matrices
- log-likelihood summary statistics
"""

import os
import csv
import json
import numpy as np


def ensure_dir(path):
    """
    Create directory if it does not already exist.
    Parameters:
        path (str): directory path to create
    """
    os.makedirs(path, exist_ok=True)  # create directory safely


def write_classification_results(results, out_path):
    """
    Write full classification results to a CSV file.
    Parameters:
        results (list[dict]): output of generate_full_results()
        out_path (str): path to output CSV file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    with open(out_path, "w", newline="") as f:  # open file for writing
        writer = csv.writer(f)  # create CSV writer
        writer.writerow(["true_label", "predicted_label", "sequence", "scores"])  # write header row

        for r in results:  # iterate through result dictionaries
            writer.writerow([
                r["true_label"],  # write true class label
                r["predicted_label"],  # write predicted class label
                r["sequence"],  # write sequence string
                json.dumps(r["scores"])  # convert score dictionary to JSON string
            ])


def write_metrics_csv(metrics, accuracy, out_path):
    """
    Write per-class metrics and overall accuracy to CSV.
    Parameters:
        metrics (dict[str, dict]): precision/recall/f1 per class
        accuracy (float): overall accuracy value
        out_path (str): path to output CSV file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    with open(out_path, "w", newline="") as f:  # open file for writing
        writer = csv.writer(f)  # create CSV writer
        writer.writerow(["class", "precision", "recall", "f1"])  # write header row

        for label, m in metrics.items():  # iterate through class metrics
            writer.writerow([
                label,  # class name
                m["precision"],  # precision value
                m["recall"],  # recall value
                m["f1"]  # F1 score
            ])

        writer.writerow(["overall_accuracy", accuracy, "", ""])  # write accuracy row


def write_metrics_json(metrics, accuracy, out_path):
    """
    Write metrics and accuracy to a JSON file.
    Parameters:
        metrics (dict[str, dict]): per-class metrics
        accuracy (float): overall accuracy
        out_path (str): path to output JSON file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    data = {
        "metrics": metrics,  # store metrics dictionary
        "overall_accuracy": accuracy  # store accuracy value
    }

    with open(out_path, "w") as f:  # open file for writing
        json.dump(data, f, indent=4)  # write JSON with indentation


def write_metrics_summary(metrics, accuracy, out_path):
    """
    Write a human-readable text summary of metrics.
    Parameters:
        metrics (dict[str, dict]): per-class metrics
        accuracy (float): overall accuracy
        out_path (str): path to output TXT file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    with open(out_path, "w") as f:  # open file for writing
        f.write("Classification Metrics Summary\n\n")  # write title

        for label, m in metrics.items():  # iterate through class metrics
            f.write(f"{label}:\n")  # write class name
            f.write(f"  Precision: {m['precision']:.4f}\n")  # write precision
            f.write(f"  Recall:    {m['recall']:.4f}\n")  # write recall
            f.write(f"  F1 Score:  {m['f1']:.4f}\n\n")  # write F1 score

        f.write(f"Overall Accuracy: {accuracy:.4f}\n")  # write accuracy value


def write_confusion_matrix_csv(conf_matrix, class_labels, out_path):
    """
    Write confusion matrix to CSV.
    Parameters:
        conf_matrix (np.ndarray): confusion matrix array
        class_labels (list[str]): ordered list of class names
        out_path (str): path to output CSV file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    with open(out_path, "w", newline="") as f:  # open file for writing
        writer = csv.writer(f)  # create CSV writer
        writer.writerow([""] + class_labels)  # write header row with class labels

        for label, row in zip(class_labels, conf_matrix):  # iterate through rows
            writer.writerow([label] + list(row))  # write row label and row values


def write_confusion_matrix_npy(conf_matrix, out_path):
    """
    Save confusion matrix as a NumPy .npy file.
    Parameters:
        conf_matrix (np.ndarray): confusion matrix array
        out_path (str): path to output .npy file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists
    np.save(out_path, conf_matrix)  # save array to file


def write_loglikelihood_stats(stats_dict, out_path):
    """
    Write log-likelihood summary statistics to CSV.
    Parameters:
        stats_dict (dict[str, dict]): class → summary statistics
        out_path (str): path to output CSV file
    """
    ensure_dir(os.path.dirname(out_path))  # ensure directory exists

    with open(out_path, "w", newline="") as f:  # open file for writing
        writer = csv.writer(f)  # create CSV writer
        writer.writerow(["class", "pos_mean", "pos_var", "neg_mean", "neg_var", "separation"])  # header row

        for label, s in stats_dict.items():  # iterate through classes
            writer.writerow([
                label,  # class name
                s["pos_mean"],  # mean positive log-likelihood
                s["pos_var"],  # variance of positive log-likelihoods
                s["neg_mean"],  # mean negative log-likelihood
                s["neg_var"],  # variance of negative log-likelihoods
                s["separation"]  # mean difference between pos and neg
            ])
