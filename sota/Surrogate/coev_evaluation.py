"""Coevolution-only observation of ensemble fits; no search-policy replacement."""
import types
import numpy as np
from scipy.stats import kendalltau


def observe_usage_optimizer(optimizer):
    """Keep evolved new_epoch; measure the existing objectives after ensemble fits.

    Wrapping the ensemble factory/fit observes training without controlling
    candidate generation, acquisition, selection, or architecture evaluation.
    """
    get_ensemble = optimizer._get_ensemble

    def observed_ensemble(self, *args, **kwargs):
        ensemble = get_ensemble(*args, **kwargs)
        fit = ensemble.fit

        def observed_fit(*fit_args, **fit_kwargs):
            result = fit(*fit_args, **fit_kwargs)
            predictions = np.asarray(ensemble.query(self.test_data))
            mean_predictions = np.mean(predictions, axis=0)
            tau, _ = kendalltau(self.test_accuracies, mean_predictions)
            mse = np.mean((np.array(self.test_accuracies) - mean_predictions) ** 2)
            self.surrogate_test_metrics.append(tau)
            self.surrogate_test_metrics_mse.append(mse)
            print(f"[Coevolution] Surrogate Test Kendall Tau = {tau:.4f}, MSE = {mse:.6f}")
            return result

        ensemble.fit = observed_fit
        return ensemble

    optimizer._get_ensemble = types.MethodType(observed_ensemble, optimizer)
    return optimizer
