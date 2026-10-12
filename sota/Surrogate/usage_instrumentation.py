"""Non-invasive metrics for evolved surrogate-usage strategies."""

from functools import wraps
import logging

import numpy as np
from scipy.stats import kendalltau

logger = logging.getLogger(__name__)


def instrument_surrogate_usage(optimizer):
    """Observe ensemble training without replacing the evolved new_epoch method."""
    if getattr(optimizer, "_llmge_metrics_installed", False):
        return optimizer

    original_get_ensemble = optimizer._get_ensemble

    @wraps(original_get_ensemble)
    def observed_get_ensemble(*args, **kwargs):
        ensemble = original_get_ensemble(*args, **kwargs)
        original_fit = ensemble.fit

        @wraps(original_fit)
        def observed_fit(*fit_args, **fit_kwargs):
            result = original_fit(*fit_args, **fit_kwargs)
            try:
                test_data = getattr(optimizer, "test_data", None)
                targets = np.asarray(getattr(optimizer, "test_accuracies", []), dtype=float).reshape(-1)
                if not test_data or not targets.size:
                    return result

                predictions = np.asarray(ensemble.query(test_data), dtype=float)
                if predictions.ndim == 2:
                    predictions = predictions.mean(axis=0)
                predictions = predictions.reshape(-1)
                if predictions.shape != targets.shape:
                    raise ValueError(f"Prediction shape {predictions.shape} != target shape {targets.shape}")

                tau, _ = kendalltau(targets, predictions)
                mse = np.mean((targets - predictions) ** 2)
                if not (np.isfinite(tau) and np.isfinite(mse)):
                    raise ValueError(f"Non-finite evaluation metrics: tau={tau}, mse={mse}")

                optimizer.surrogate_test_metrics.append(float(tau))
                optimizer.surrogate_test_metrics_mse.append(float(mse))
            except Exception:
                # Diagnostic errors must not override evolved search behavior.
                logger.exception("Could not collect surrogate metrics after ensemble fit")
            return result

        ensemble.fit = observed_fit
        return ensemble

    optimizer._get_ensemble = observed_get_ensemble
    optimizer._llmge_metrics_installed = True
    return optimizer
