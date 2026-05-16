"""
plot_results.py

This module provides visualization utilities for inspecting the outputs of
evaluate_results.py. It contains only plotting functions and produces
publication‑quality figures for interpreting model behavior.

The plots in this module include:
- Confusion matrix heatmaps
- Positive vs negative log‑likelihood histograms
- Log‑likelihood comparison plots
- Multiclass log‑likelihood separation plots
- Violin plots of log‑likelihood distributions across classes

All functions in this module assume that numerical evaluations (confusion
matrix, per‑class metrics, log‑likelihood summaries, etc.) have already been
computed by evaluate_results.py. No statistical calculations are performed
here; this module is strictly for visualization.
"""


# Import modules
import matplotlib.pyplot as plt
import os


def ensure_dir(path):
    """
    Create directory if it does not already exist for plots to exist
    Parameters:
        path (str): directory path to create
    """
    os.makedirs(path, exist_ok=True)  # create directory safely


def plot_confusion_matrix(conf_matrix, class_labels):
    """
    Plot a confusion matrix using matplotlib
    Parameters:
        conf_matrix (np.ndarray): confusion matrix
        class_labels (list[str]): ordered list of class names
    """

    fig, ax = plt.subplots(figsize=(8, 6))  # create figure
    im = ax.imshow(conf_matrix, cmap="Blues")  # display matrix as heatmap

    plt.colorbar(im)  # add colorbar

    ax.set_xticks(range(len(class_labels)))  # set x ticks
    ax.set_yticks(range(len(class_labels)))  # set y ticks
    ax.set_xticklabels(class_labels)  # label x ticks
    ax.set_yticklabels(class_labels)  # label y ticks

    plt.xlabel("Predicted Label")  # x-axis label
    plt.ylabel("True Label")  # y-axis label
    plt.title("Confusion Matrix")  # plot title

    plt.tight_layout()  # adjust layout
    plt.show()  # display plot


def plot_pos_likelihoods_over_k(pos_scores_by_k, class_label):
    """
    Plot the distribution of positive-class log-likelihoods across k values.

    Parameters:
        pos_scores_by_k (dict[int, list[float]]):
            mapping of k → list of log-likelihoods for true class sequences
        class_label (str): class being evaluated
    """

    fig, ax = plt.subplots(figsize=(12, 5))

    ks = sorted(pos_scores_by_k.keys())
    data = [pos_scores_by_k[k] for k in ks]

    ax.boxplot(data, labels=ks, showmeans=True)

    ax.set_xlabel("k value")
    ax.set_ylabel("Positive-class Log-Likelihood")
    ax.set_title(f"{class_label}: Positive Log-Likelihood Distribution Across k")
    plt.tight_layout()
    plt.show()


def plot_loglikelihood_histogram(pos_scores, neg_scores, class_label):
    """
    Plot log-likelihood histograms for true vs negative sequences
    Parameters:
        pos_scores (list[float]): log-likelihoods for true class sequences
        neg_scores (list[float]): log-likelihoods for non-class sequences
        class_label (str): name of the class being evaluated
    """

    fig, ax = plt.subplots(figsize=(12, 4))  # create figure

    plt.hist(pos_scores, bins=50, alpha=0.6, label=f"{class_label} (true)")  # positives
    plt.hist(neg_scores, bins=50, alpha=0.6, label=f"{class_label} (other classes)")  # negatives

    plt.xlabel("Log-Likelihood")  # x-axis label
    plt.ylabel("Count")  # y-axis label
    plt.title(f"Log-Likelihood Distribution for {class_label}")  # title
    plt.legend()  # add legend
    plt.tight_layout()  # adjust layout
    plt.show()  # display plot


def plot_multiclass_separation(all_scores):
    """
    Plot log-likelihood distributions for all classes
    Parameters:
        all_scores (dict[str, list[float]]): mapping of class_label → list of log-likelihoods
    """

    fig, ax = plt.subplots(figsize=(12, 5))  # create figure

    for label, scores in all_scores.items():  # iterate through classes
        plt.hist(scores, bins=50, alpha=0.5, label=label)  # plot histogram

    plt.xlabel("Log-Likelihood")  # x-axis label
    plt.ylabel("Count")  # y-axis label
    plt.title("Log-Likelihood Distributions Across All Classes")  # title
    plt.legend()  # add legend
    plt.tight_layout()  # adjust layout
    plt.show()  # display plot


def plot_violin(all_scores):
    """
    Plot violin plots for log-likelihood distributions across classes.
    Parameters:
        all_scores (dict[str, list[float]]):
            mapping of class_label → list of log-likelihoods
    """

    fig, ax = plt.subplots(figsize=(10, 6))  # create figure

    data = [scores for scores in all_scores.values()]  # extract score lists
    labels = list(all_scores.keys())  # extract class labels

    ax.violinplot(data, showmeans=True)  # create violin plot
    ax.set_xticks(range(1, len(labels) + 1))  # set x ticks
    ax.set_xticklabels(labels)  # label x ticks

    plt.ylabel("Log-Likelihood")  # y-axis label
    plt.title("Log-Likelihood Violin Plot Across Classes")  # title
    plt.tight_layout()  # adjust layout
    plt.show()  # display plot


def save_class_plot(fig, class_label, out_dir):
    """
    Save a per-class plot to a PNG file.
    Parameters:
        fig (matplotlib.figure.Figure): figure object to save
        class_label (str): class name used for filename
        out_dir (str): directory where PNG will be saved
    """
    ensure_dir(out_dir)  # ensure directory exists
    out_path = os.path.join(out_dir, f"{class_label}.png")  # build file path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")  # save figure
    plt.close(fig)  # close figure to free memory


def save_global_plot(fig, filename, out_dir):
    """
    Save a global plot (not tied to a single class) to a PNG file
    Parameters:
        fig (matplotlib.figure.Figure): figure object to save
        filename (str): name of output PNG file
        out_dir (str): directory where PNG will be saved
    """
    ensure_dir(out_dir)  # ensure directory exists
    out_path = os.path.join(out_dir, filename)  # build file path
    fig.savefig(out_path, dpi=300, bbox_inches="tight")  # save figure
    plt.close(fig)  # close figure to free memory
