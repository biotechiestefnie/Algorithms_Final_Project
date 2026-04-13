import os
import csv
import json


def get_output_path(category, dataset, filename, base_dir="results"):
    """
    Build full output path for a given category, dataset, and filename.
    Example:
        get_output_path("summary_stats", "prototype", "run1_summary.txt")
        -> results/summary_stats/prototype/run1_summary.txt
    Assumes directory structure already exists.
    """
    return os.path.join(base_dir, category, dataset, filename)


def write_per_sequence_results(results, dataset, run_name, base_dir="results"):
    """
    Write per-sequence classification results to csv.
    Output:
        results/per_sequence_classification/<dataset>/<run_name>_results.csv
    Columns:
        sequence, predicted_class, score
    """

    # FIXED: no datatype, no feature — use run_name
    filename = f"{run_name}_results.csv"
    path = get_output_path("per_sequence_classification", dataset, filename, base_dir)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "predicted_class", "score"])

        # FIXED: results are 4-tuples (seq, pred, score, class_scores)
        for (seq, pred, score, _) in results:
            writer.writerow([seq, pred, score])


def write_summary_stats(stats, dataset, run_name, base_dir="results"):
    """
    Write summary statistics to txt.
    Output:
        results/summary_stats/<dataset>/<run_name>_summary.txt
    """

    filename = f"{run_name}_summary.txt"
    path = get_output_path("summary_stats", dataset, filename, base_dir)

    with open(path, "w") as f:
        f.write(f"Total sequences: {stats['total']}\n")
        f.write(f"Unclassified: {stats['unclassified']}\n")
        f.write(f"Negative infinity scores: {stats['neg_inf']}\n")
        f.write("Class counts:\n")

        for cls, count in stats["class_counts"].items():
            f.write(f"  {cls}: {count}\n")


def write_confusion_matrix(matrix, dataset, k, run_name, base_dir="results"):
    """
    Write confusion matrix to csv.
    Output:
        results/confusion_matrices/<dataset>/<run_name>_k<k>.csv
    """

    filename = f"{run_name}_k{k}.csv"
    path = get_output_path("confusion_matrices", dataset, filename, base_dir)

    classes = sorted(matrix.keys())

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # header row
        writer.writerow(["true \\ pred"] + classes)

        # matrix rows
        for true_cls in classes:
            row = [true_cls] + [matrix[true_cls][pred_cls] for pred_cls in classes]
            writer.writerow(row)


def write_accuracy_curve(accuracy_vs_k, dataset, run_name, base_dir="results"):
    """
    Write accuracy vs k to csv.
    Output:
        results/accuracy_curves/<dataset>/<run_name>_accuracy.csv
    """

    filename = f"{run_name}_accuracy.csv"
    path = get_output_path("accuracy_curves", dataset, filename, base_dir)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "accuracy"])

        for k in sorted(accuracy_vs_k.keys()):
            writer.writerow([k, accuracy_vs_k[k]])


def write_likelihood_distributions(likelihoods, dataset, run_name, base_dir="results"):
    """
    Write per-class likelihood distributions to JSON.
    Output:
        results/logs/<dataset>/<run_name>_likelihoods.json
    """

    filename = f"{run_name}_likelihoods.json"
    path = get_output_path("logs", dataset, filename, base_dir)

    with open(path, "w") as f:
        json.dump(likelihoods, f, indent=2)


def write_all_outputs(results, stats, metrics, dataset, run_name, base_dir="results"):
    """
    High-level wrapper to write all output files for a run
    Calls:
        write_per_sequence_results()
        write_summary_stats()
        write_confusion_matrix() for each k
        write_accuracy_curve()
        write_likelihood_distributions()
    Assumes the directory structure already exists.
    """

    # per-sequence results
    write_per_sequence_results(results, dataset, run_name, base_dir)

    # summary stats
    write_summary_stats(stats, dataset, run_name, base_dir)

    # confusion matrices (one per k)
    for k, matrix in metrics["confusion_matrices"].items():
        write_confusion_matrix(matrix, dataset, k, run_name, base_dir)

    # accuracy curve
    write_accuracy_curve(metrics["accuracy_vs_k"], dataset, run_name, base_dir)

    # likelihood distributions
    write_likelihood_distributions(metrics["likelihood_distributions"], dataset, run_name, base_dir)
