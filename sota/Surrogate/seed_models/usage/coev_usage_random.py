"""Coevolution usage seed: random candidate selection (non-BANANAS acquisition).

Swap into ``src/cfg/surrogate_coev.py`` POPULATIONS['usage']['seed_file'] to evolve
a random-search policy instead of ``coev_usage.py`` (BANANAS acquisition).
"""
import numpy as np
from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.discrete.bananas.optimizer import Bananas

# --OPTION--
# Evolve usage only; the paired surrogate is injected through predictor_cls/kwargs.
class SurrogateUsageOptimizer(Bananas):
    """Random top-k among candidates; ensemble still fit for τ/MSE diagnostics."""

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
            return np.zeros(len(architectures), dtype=float)
        return score

    def _get_best_candidates(self, candidates, acq_fn):
        k = min(self.k, len(candidates))
        if k <= 0:
            return []
        indices = np.random.choice(len(candidates), size=k, replace=False)
        return [candidates[int(i)] for i in indices]

    def new_epoch(self, epoch):
        if epoch < self.num_init:
            model = self._sample_new_model(); self._remove_from_test_set(model.arch_hash); self._set_scores(model)
            return
        if not self.next_batch:
            xtrain, ytrain = self._get_train()
            self.ensemble = self._get_ensemble(); self.ensemble.fit(xtrain, ytrain)
            self.next_batch = self._get_best_candidates(
                self._get_new_candidates(ytrain), self._acquisition(ytrain)
            )
        model = self.next_batch.pop(); self._remove_from_test_set(model.arch_hash); self._set_scores(model)
