import time
from naslib.optimizers.discrete.bananas.optimizer import Bananas
from naslib.predictors.ensemble import Ensemble
from scipy.stats import norm
import torch
import numpy as np

from naslib.predictors.mlp import MLPPredictor
from naslib.predictors.trees.xgb import XGBoost

# Shared helpers live above the first option marker so mutation/crossover
# do not overwrite the factory when a family-specific block is edited.
SURROGATE_REGISTRY = {}
ALLOWED_SURROGATE_NAMES = ("mlp", "xgboost")
_CLASS_ALIASES = {
    "xgboost": ("CustomXGBoost",),
    "mlp": ("CustomMLP",),
}


def register_surrogate(name, cls):
    """Add or replace one family without wiping the other."""
    SURROGATE_REGISTRY[name] = cls


def get_surrogate_config(base_cfg=None, **runtime_constants):
    cfg = DEFAULT_SURROGATE_CONFIG.copy() if base_cfg is None else base_cfg.copy()
    cfg.update({key: value for key, value in runtime_constants.items() if value is not None})
    return cfg


def resolve_predictor_cls(name):
    if name in SURROGATE_REGISTRY:
        return SURROGATE_REGISTRY[name]
    for alias in _CLASS_ALIASES.get(name, ()):
        cls = globals().get(alias)
        if cls is not None:
            return cls
    raise ValueError(
        f"Unknown surrogate: {name}. Allowed names: {ALLOWED_SURROGATE_NAMES}. "
        "Mutate DEFAULT_SURROGATE_CONFIG['name'] to 'mlp' or 'xgboost'."
    )


def build_predictor_kwargs(cfg):
    name = cfg["name"]
    if name not in ALLOWED_SURROGATE_NAMES:
        raise ValueError(
            f"Unknown surrogate: {name}. Allowed names: {ALLOWED_SURROGATE_NAMES}."
        )
    if "corpus_path" not in cfg:
        raise ValueError("Missing required runtime config: corpus_path")

    predictor_kwargs = {
        "base_predictor_cls": resolve_predictor_cls(name),
        "corpus_path": cfg["corpus_path"],
        "embedding_col": cfg["embedding_col"],
        "use_pca": cfg["use_pca"],
        "pca_components": cfg["pca_components"],
    }

    if name == "xgboost":
        predictor_kwargs.update(
            {
                "ss_type": cfg["ss_type"],
                "hparams_from_file": cfg["hparams_from_file"],
                "nthread": cfg["nthread"],
                "device": cfg["device"],
                "tree_method": cfg["tree_method"],
                "max_depth": cfg.get("max_depth", 6),
                "min_child_weight": cfg.get("min_child_weight", 1),
                "colsample_bytree": cfg.get("colsample_bytree", 1),
                "learning_rate": cfg.get("xgb_learning_rate", cfg.get("learning_rate", 0.3)),
                "colsample_bylevel": cfg.get("colsample_bylevel", 1),
            }
        )
    elif name == "mlp":
        predictor_kwargs.update(
            {
                "num_layers": cfg["num_layers"],
                "layer_width": cfg["layer_width"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "loss": cfg["loss"],
            }
        )

    return predictor_kwargs

# --OPTION--
# Evolution-mode constraint: concrete BANANAS usage seed retained in every variant.
# Preserve SurrogateUsageOptimizer; do not alter surrogate-usage behavior here.
class SurrogateUsageOptimizer(Bananas):
    """Concrete standard-BANANAS usage seed; retained in every variant."""

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
            predictions = np.asarray(self.ensemble.query(architectures, info))
            mean, std = np.mean(predictions, axis=0), np.std(predictions, axis=0)
            if self.acq_fn_type == "its":
                return np.random.normal(mean, std)
            if self.acq_fn_type == "ucb":
                return mean + 0.5 * std
            if self.acq_fn_type == "ei":
                std = np.maximum(std / 5.0, 1e-12)
                z = (mean - max(ytrain)) / std
                return std * (z * norm.cdf(z) + norm.pdf(z))
            return mean
        return score

    def _get_best_candidates(self, candidates, acq_fn):
        info = None
        if self.zc and len(self.train_data) <= self.max_zerocost:
            for candidate in candidates:
                candidate.zc_scores = self.query_zc_scores(candidate.arch)
            info = [{"zero_cost_scores": candidate.zc_scores} for candidate in candidates]
        scores = np.asarray(acq_fn([candidate.arch for candidate in candidates], info))
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
            candidates = self._get_new_candidates(ytrain)
            self.next_batch = self._get_best_candidates(candidates, self._acquisition(ytrain))
        model = self.next_batch.pop()
        self._remove_from_test_set(model.arch_hash)
        self._set_scores(model)


# --OPTION--
# LLM choice block: switch the active family by setting name to "mlp" or "xgboost".
# Do not redefine get_surrogate_config, build_predictor_kwargs, or SURROGATE_REGISTRY here.
DEFAULT_SURROGATE_CONFIG = {
    "name": "mlp",
    "embedding_col": "codellama_python_7b_pytorch_code_exclude_helper_embedding",
    "use_pca": False,
    "pca_components": 128,
    "ss_type": "nasbench201",
    "hparams_from_file": False,
    "nthread": 4,
    "device": "cuda",
    "tree_method": "hist",
    "max_depth": 6,
    "min_child_weight": 1,
    "colsample_bytree": 1,
    "xgb_learning_rate": 0.3,
    "colsample_bylevel": 1,
    "num_layers": 3,
    "layer_width": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "epochs": 200,
    "loss": "mse",
}

# --OPTION--
# XGBoost family. Mutating this block changes tree-model behavior only.
class CustomXGBoost(XGBoost):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomXGBoost] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(xtrain, ytrain, train_info, params, **kwargs)

        print(f"[CustomXGBoost] Training completed in {time.time() - start:.2f} seconds.")
        return result


register_surrogate("xgboost", CustomXGBoost)

# --OPTION--
# MLP family. Mutating this block changes neural-regressor behavior only.
class CustomMLP(MLPPredictor):
    def __init__(self, **kwargs):
        base_valid_args = [
            "encoding_type",
            "ss_type",
            "zc",
            "zc_only",
            "hpo_wrapper",
            "hparams_from_file",
            "config",
        ]
        base_args = {k: v for k, v in kwargs.items() if k in base_valid_args}
        self.custom_hyperparams = {k: v for k, v in kwargs.items() if k not in base_valid_args}

        super().__init__(**base_args)

        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        print(f"[CustomMLP] Hyperparams set: {self.hyperparams}")

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        start = time.time()
        if self.hyperparams is None:
            self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)

        result = super().fit(
            xtrain,
            ytrain,
            train_info=train_info,
            epochs=self.hyperparams["epochs"],
            loss=self.hyperparams["loss"],
            **kwargs,
        )
        print(f"[CustomMLP] Training completed in {time.time() - start:.2f} seconds.")
        return result


register_surrogate("mlp", CustomMLP)
