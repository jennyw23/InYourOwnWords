"""Interpretation fidelity metrics.

Fidelity in the paper is the harmonic mean of recall and specificity. hypothesaes
releases differ on what their "f1" metric means, so patch_hypothesaes() makes the
scoring explicit and version-independent.
"""

import numpy as np

FIDELITY_METRIC = "recall_specificity_hmean"


def compute_metrics(annotations, labels, activations):
    """Recall, precision, specificity, F1, recall/specificity harmonic mean, and correlation."""
    if not (1 in labels and 0 in labels):
        return {"recall": 0.0, "precision": 0.0, "specificity": 0.0, "f1": 0.0,
                FIDELITY_METRIC: 0.0, "correlation": 0.0}

    annotations = np.asarray(annotations).astype(bool)
    labels = np.asarray(labels).astype(bool)

    tp = np.sum(annotations & labels)
    fp = np.sum(annotations & ~labels)
    fn = np.sum(~annotations & labels)
    tn = np.sum(~annotations & ~labels)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    rs = 2 * recall * specificity / (recall + specificity) if (recall + specificity) > 0 else 0.0
    correlation = np.corrcoef(activations, annotations)[0, 1] if len(np.unique(annotations)) > 1 else 0.0

    return {"recall": recall, "precision": precision, "specificity": specificity, "f1": f1,
            FIDELITY_METRIC: rs, "correlation": correlation}


def patch_hypothesaes():
    """Make NeuronInterpreter.score_interpretations return every metric above."""
    from hypothesaes.interpret_neurons import NeuronInterpreter

    def _compute_metrics(self, annotations, labels, activations):
        return compute_metrics(annotations, labels, activations)

    NeuronInterpreter._compute_metrics = _compute_metrics
