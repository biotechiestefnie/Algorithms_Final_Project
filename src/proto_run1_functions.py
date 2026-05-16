"""

proto_run1_functions.py

This module preserves the original functions used to generate the results for the
first prototype run. In that initial implementation, log‑likelihoods were computed
in their raw, cumulative form, meaning the total log‑probability increased in magnitude
with sequence length. Thresholds for positive classification were defined as the 1st
percentile of each class’s raw log‑likelihood distribution, and any sequence scoring
below that boundary was rejected.

Because raw log‑likelihoods scale directly with sequence length, the resulting score
distributions for positive and negative datasets overlapped heavily. This prevented the
model from establishing meaningful separation between classes and caused the reject
option to fail. Nevertheless, these original functions are preserved here so that the
first prototype run can be reproduced exactly in the notebook, allowing a clear
demonstration of the progression of my analysis and the justification for subsequent
methodological changes.

Jupyter Notebook executes code using the most recently defined version of each function.
Updating the scoring logic later in the notebook overwrites the earlier definitions,
and restarting the kernel clears all previous results. As a result, it was not possible
to display both the original (raw‑likelihood) run and the updated (length‑normalized)
run within the same session using a single set of function names. Attempts to isolate
the runs by renaming variables were insufficient because the underlying function
definitions had already changed.

To resolve this, I created a dedicated script containing the original raw‑likelihood
functions. The notebook now imports these functions explicitly for the first run,
ensuring that the original behavior—and therefore the original results—can be reproduced
cleanly and consistently alongside the updated implementation.
"""

import numpy as np


def log_likelihood_raw(seq, model, k):
    total = 0.0
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in model:
            total += model[kmer]
        else:
            total += model["_UNSEEN_"]
    return total


def compute_thresholds_raw(training_data, models, k, percentile=1):
    thresholds = {}

    for class_label, seqs in training_data.items():
        scores = [log_likelihood_raw(seq, models[class_label], k)
                  for seq in seqs]

        threshold = np.percentile(scores, percentile)
        thresholds[class_label] = threshold

    return thresholds


def classify_sequence_raw(seq, models, k):
    """
    Classify a single DNA sequence by selecting the class whose Markov model
    assigns the highest raw log-likelihood.
    """

    scores = _score_all_models_raw(seq, models, k)
    return max(scores, key=scores.get)


def _score_all_models_raw(seq, models, k):
    """
    Internal helper to compute log-likelihood under each class model.
    Removes duplicated logic from generate_full_results.
    """
    scores = {}
    for class_label, model in models.items():
        scores[class_label] = log_likelihood_raw(seq, model, k)
    return scores



def classify_with_reject_raw(seq, models, k, thresholds):
    scores = {
        cls: log_likelihood_raw(seq, model, k)
        for cls, model in models.items()
    }

    best_class = max(scores, key=scores.get)
    best_score = scores[best_class]

    if best_score < thresholds[best_class]:
        return "neither", scores

    return best_class, scores


def generate_full_results_raw(test_data, models, k, thresholds):
    """
    Generate full classification results for Run 1 using raw log-likelihood scoring.
    Mirrors results_generator.generate_full_results but calls the raw functions.
    """
    results = []

    for true_label, seqs in test_data.items():
        for seq in seqs:
            pred, scores = classify_with_reject_raw(seq, models, k, thresholds)
            results.append({
                "true_label": true_label,
                "predicted_label": pred,
                "scores": scores,
                "sequence": seq
            })

    return results

