import time

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
