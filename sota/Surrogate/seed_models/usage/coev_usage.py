"""Coevolution seed: only the surrogate usage strategy is evolvable."""
import numpy as np
from scipy.stats import norm
from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.discrete.bananas.optimizer import Bananas

# --OPTION--
# Evolve usage only; the paired surrogate is injected through predictor_cls/kwargs.
class SurrogateUsageOptimizer(Bananas):
    """Standard BANANAS: fit ensemble, score candidates, select, evaluate."""

    def __init__(self, config, zc_api=None, predictor_cls=None, predictor_kwargs=None):
        super().__init__(config, zc_api=zc_api)
        self.predictor_cls = predictor_cls
        self.predictor_kwargs = predictor_kwargs or {}

    def _get_train(self):
        return [model.arch for model in self.train_data], [model.accuracy for model in self.train_data]

    def _get_ensemble(self):
        ensemble = Ensemble(
            num_ensemble=self.num_ensemble,
            ss_type=self.ss_type,
            predictor_type=self.predictor_type,
            zc=self.zc,
            zc_only=self.zc_only,
            config=self.config,
        )
        if self.predictor_cls is not None:
            ensemble.ensemble = [self.predictor_cls(**self.predictor_kwargs)
                                 for _ in range(self.num_ensemble)]
        return ensemble

    def _acquisition(self, ytrain):
        def score(architectures, info=None):
            architectures = architectures if isinstance(architectures, list) else [architectures]
            values = np.asarray(self.ensemble.query(architectures, info))
            mean, std = np.mean(values, axis=0), np.std(values, axis=0)
            if self.acq_fn_type == "its": return np.random.normal(mean, std)
            if self.acq_fn_type == "ucb": return mean + 0.5 * std
            if self.acq_fn_type == "ei":
                std = np.maximum(std / 5.0, 1e-12); z = (mean - max(ytrain)) / std
                return std * (z * norm.cdf(z) + norm.pdf(z))
            return mean
        return score

    def _get_best_candidates(self, candidates, acq_fn):
        scores = np.asarray(acq_fn([candidate.arch for candidate in candidates]))
        return [candidates[index] for index in np.argsort(scores)[-self.k:]]

    def new_epoch(self, epoch):
        if epoch < self.num_init:
            model = self._sample_new_model(); self._remove_from_test_set(model.arch_hash); self._set_scores(model)
            return
        if not self.next_batch:
            xtrain, ytrain = self._get_train()
            self.ensemble = self._get_ensemble(); self.ensemble.fit(xtrain, ytrain)
            self.next_batch = self._get_best_candidates(self._get_new_candidates(ytrain), self._acquisition(ytrain))
        model = self.next_batch.pop(); self._remove_from_test_set(model.arch_hash); self._set_scores(model)
