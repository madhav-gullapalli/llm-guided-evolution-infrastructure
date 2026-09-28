"""BANANAS usage seed for USE evolution.

The surrogate is concrete fixed code in this artifact.  The marked optimizer
is the independently evolvable program passed to the unchanged LLMGE flow.
"""

import time
import numpy as np
import torch
from scipy.stats import norm

from naslib.predictors.trees.xgb import XGBoost
from naslib.predictors.ensemble import Ensemble
from naslib.optimizers.discrete.bananas.optimizer import Bananas


DEFAULT_SURROGATE_CONFIG = {
    "name": "xgboost",
    "embedding_col": "codellama_python_7b_pytorch_code_exclude_helper_embedding",
    "use_pca": False, "pca_components": 128, "ss_type": "nasbench201",
    "hparams_from_file": False, "nthread": 4, "device": "cuda",
    "tree_method": "hist", "num_layers": 3, "layer_width": 128,
    "batch_size": 32, "lr": 1e-3, "epochs": 200, "loss": "mse",
}


def get_surrogate_config(base_cfg=None, **runtime_constants):
    cfg = DEFAULT_SURROGATE_CONFIG.copy() if base_cfg is None else base_cfg.copy()
    cfg.update({key: value for key, value in runtime_constants.items() if value is not None})
    return cfg


class CustomXGBoost(XGBoost):
    def __init__(self, **kwargs):
        base_keys = {"encoding_type", "ss_type", "zc", "zc_only", "hpo_wrapper", "hparams_from_file"}
        super().__init__(**{key: value for key, value in kwargs.items() if key in base_keys})
        self.custom_hyperparams = {key: value for key, value in kwargs.items() if key not in base_keys}
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        self.hyperparams.update(self.custom_hyperparams)
        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)
        print(f"[CustomXGBoost] Training completed in {time.time() - start:.2f} seconds.")
        return result


SURROGATE_REGISTRY = {"xgboost": CustomXGBoost}


def build_predictor_kwargs(cfg):
    if cfg["name"] not in SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {cfg['name']}")
    return {
        "base_predictor_cls": SURROGATE_REGISTRY[cfg["name"]],
        "corpus_path": cfg["corpus_path"], "embedding_col": cfg["embedding_col"],
        "use_pca": cfg["use_pca"], "pca_components": cfg["pca_components"],
        "ss_type": cfg["ss_type"], "hparams_from_file": cfg["hparams_from_file"],
        "nthread": cfg["nthread"], "device": cfg["device"], "tree_method": cfg["tree_method"],
    }


# --OPTION--
# Evolution-mode constraint: evolve only this usage strategy.  Preserve the
# concrete surrogate definition above unchanged.
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
