from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics_binary(eval_prediction):

    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [ p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [ l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average='binary'),
        "recall": recall_score(true_labels, true_predictions, average='binary'),
        "f1_score": f1_score(true_labels, true_predictions, pos_label=1),
    }


def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [ p for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]
    true_labels = [ l for prediction, label in zip(predictions, labels) for (p, l) in zip(prediction, label) if l != -100]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions, average='micro'),
        "recall": recall_score(true_labels, true_predictions, average='micro'),
        "f1_score": f1_score(true_labels, true_predictions, average='micro'),
    }