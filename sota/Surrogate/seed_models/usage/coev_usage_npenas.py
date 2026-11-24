"""NPENAS usage seed with an injected, separately evolved surrogate.

Reuse NASLib NPENAS mutation proposals and exploitation-only selection while
retaining BANANAS evaluator bookkeeping and ensemble-observation hooks.
"""
import numpy as np
from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.discrete.bananas.optimizer import Bananas

# --OPTION--
# Evolve usage only; the paired surrogate is injected through predictor_cls/kwargs.
from naslib.optimizers.discrete.npenas.optimizer import Npenas


class SurrogateUsageOptimizer(Bananas):
    """NPENAS: mutate top observed architectures and rank by predicted mean."""

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

    def _get_new_candidates(self, ytrain):
        # NPENAS always mutates top observed architectures, independently of
        # the BANANAS acq_fn_optimization setting inherited from the evaluator.
        if not ytrain:
            return [self._sample_new_model() for _ in range(self.num_candidates)]
        candidates = Npenas._get_new_candidates(self, ytrain)
        # NASLib rounds mutation counts down; small candidate budgets may
        # otherwise yield an empty batch and fail at next_batch.pop().
        if not candidates:
            candidates = [self._sample_new_model() for _ in range(self.num_candidates)]
        return candidates

    def _acquisition(self, ytrain):
        def score(architectures, info=None):
            architectures = architectures if isinstance(architectures, list) else [architectures]
            values = np.asarray(self.ensemble.query(architectures, info))
            return np.mean(values, axis=0)
        return score

    def _get_best_candidates(self, candidates, acq_fn):
        if not candidates or self.k <= 0:
            return []
        scores = np.asarray(acq_fn([candidate.arch for candidate in candidates]))
        return [candidates[index] for index in np.argsort(scores)[-self.k:]]

    def new_epoch(self, epoch):
        if epoch < self.num_init:
            model = self._sample_new_model()
            self._remove_from_test_set(model.arch_hash)
            self._set_scores(model)
            return
        if not self.next_batch:
            xtrain, ytrain = self._get_train()
            self.ensemble = self._get_ensemble()
            self.ensemble.fit(xtrain, ytrain)
            self.next_batch = self._get_best_candidates(self._get_new_candidates(ytrain), self._acquisition(ytrain))
        model = self.next_batch.pop()
        self._remove_from_test_set(model.arch_hash)
        self._set_scores(model)
