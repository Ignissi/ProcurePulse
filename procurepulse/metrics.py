"""Считает метрики двух голов модели и верхней части очереди."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking_metrics(labels, scores):
    k = min(100, len(labels))
    order = np.argsort(-scores, kind="stable")
    top_labels = labels[order[:k]]

    positives = labels.sum()
    prevalence = labels.mean()

    return {
        "k": k,
        "precision_at_100": float(top_labels.mean()),
        "recall_at_100": float(top_labels.sum() / max(positives, 1)),
        "lift_at_100": float(top_labels.mean() / max(prevalence, 1e-6)),
    }


def evaluate(frame, log_prices, probabilities):
    metrics = {}
    price_mask = frame["price_mask"].to_numpy()
    outcome_mask = frame["outcome_mask"].to_numpy()

    metrics["price_count"] = int(price_mask.sum())
    metrics["outcome_count"] = int(outcome_mask.sum())

    if price_mask.any():
        actual = frame.loc[price_mask, "observed_price"].to_numpy()
        predicted_log = log_prices[price_mask]
        predicted = np.expm1(np.clip(predicted_log, 0, 30))

        errors = predicted - actual
        log_errors = predicted_log - np.log1p(actual)
        metrics["price_mae"] = float(np.abs(errors).mean())
        metrics["price_rmse"] = float(np.sqrt(np.square(errors).mean()))
        metrics["price_log_mae"] = float(np.abs(log_errors).mean())

    if outcome_mask.any():
        labels = frame.loc[outcome_mask, "outcome"].to_numpy(dtype=int)
        scores = probabilities[outcome_mask]

        metrics["outcome_rate"] = float(labels.mean())
        metrics["brier"] = float(brier_score_loss(labels, scores))

        if len(np.unique(labels)) == 2:
            metrics["average_precision"] = float(average_precision_score(labels, scores))
            metrics["roc_auc"] = float(roc_auc_score(labels, scores))

            if len(np.unique(scores)) > 1:
                metrics.update(ranking_metrics(labels, scores))

    return metrics
