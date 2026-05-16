from evaluate_results import (
    compute_confusion_matrix,
    compute_class_metrics,
    compute_overall_accuracy,
    compute_per_class_accuracy,
    summarize_loglikelihoods
)

class_labels_final = ["promoters", "exons", "introns", "repeats"]


def compute_tp_fp_fn_tn(conf_matrix, results_k, class_labels):
    """
    Compute TP, FP, FN, TN for each class.
    Rejects count as:
        FN if true_label == class
        TN if true_label != class
    """

    tp_fp_fn_tn = {}

    total = conf_matrix.sum() + sum(
        1 for r in results_k if r["predicted_label"] == "reject"
    )

    for idx, cls in enumerate(class_labels):
        tp = conf_matrix[idx, idx]
        fp = conf_matrix[:, idx].sum() - tp
        fn = conf_matrix[idx, :].sum() - tp
        tn = total - (tp + fp + fn)

        for r in results_k:
            if r["predicted_label"] == "reject":
                if r["true_label"] == cls:
                    fn += 1
                else:
                    tn += 1

        tp_fp_fn_tn[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn
        }

    return tp_fp_fn_tn


def compute_ll_means_across_k(results_by_k_final, class_labels):
    ll_means = {}

    for k, results_k in results_by_k_final.items():
        ll_means[k] = {}

        for cls in class_labels:
            scores = [
                float(r["scores"][cls])
                for r in results_k
                if r["predicted_label"] != "reject"
            ]

            ll_means[k][cls] = sum(scores) / len(scores) if scores else None

    return ll_means


def evaluate_final_run(results_by_k_final, class_labels=None):
    """
    Function to evaluate classification results in final run for all classes across k values
    Parameters:
       results_by_k_final: classification results from final run by k value
       class_labels: True labels for each class
    Returns:
         all results in final classification evaluation
    """

    if class_labels is None:
        class_labels = class_labels_final

    all_metrics = {}

    for k, results_k in results_by_k_final.items():
        formatted_results = [
            (r["true_label"], r["predicted_label"], None)
            for r in results_k
            if r["predicted_label"] != "reject"
        ]

        conf_matrix = compute_confusion_matrix(formatted_results, class_labels)
        class_metrics = compute_class_metrics(conf_matrix, class_labels)
        per_class_acc = compute_per_class_accuracy(conf_matrix, class_labels)
        overall_acc = compute_overall_accuracy(conf_matrix)
        mean_ll_across_k = compute_ll_means_across_k(results_by_k_final, class_labels)
        tp_fp_fn_tn = compute_tp_fp_fn_tn(conf_matrix, results_k, class_labels)

        all_metrics[k] = {
            "confusion_matrix": conf_matrix,
            "class_metrics": class_metrics,
            "per_class_accuracy": per_class_acc,
            "overall_accuracy": overall_acc,
            "tp_fp_fn_tn": tp_fp_fn_tn
        }

    all_metrics["ll_scores_across_k"] = compute_ll_scores_across_k(
        results_by_k_final, class_labels
    )

    return all_metrics
