"""MLP surrogate seed; the usage policy is supplied by its paired module."""
from naslib.predictors.mlp import MLPPredictor

DEFAULT_SURROGATE_CONFIG = {
    "name": "mlp",
    "embedding_col": "codellama_python_7b_pytorch_code_exclude_helper_embedding",
    "use_pca": False,
    "pca_components": 128,
    "ss_type": "nasbench201",
    "hparams_from_file": False,
    "num_layers": 3,
    "layer_width": 128,
    "batch_size": 32,
    "lr": 1e-3,
    "regularization": 0.0,
    "epochs": 200,
    "loss": "mse",
}

# --OPTION--
# Evolve only surrogate construction and training; preserve the public factories.
def get_surrogate_config(base_cfg=None, **runtime_constants):
    cfg = DEFAULT_SURROGATE_CONFIG.copy()
    if base_cfg is not None:
        cfg.update(base_cfg)
    cfg.update({key: value for key, value in runtime_constants.items() if value is not None})
    return cfg


class CustomMLP(MLPPredictor):
    def __init__(self, epochs=200, loss="mse", **kwargs):
        base_keys = ("encoding_type", "ss_type", "hpo_wrapper", "hparams_from_file")
        super().__init__(**{key: value for key, value in kwargs.items() if key in base_keys})
        self.custom_hyperparams = {
            key: kwargs[key]
            for key in ("num_layers", "layer_width", "batch_size", "lr", "regularization")
            if key in kwargs
        }
        self.hyperparams = self.default_hyperparams.copy()
        self.hyperparams.update(self.custom_hyperparams)
        self.epochs = epochs
        self.loss = loss

    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        self.hyperparams = (self.hyperparams or self.default_hyperparams).copy()
        self.hyperparams.update(self.custom_hyperparams)
        kwargs.setdefault("epochs", self.epochs)
        kwargs.setdefault("loss", self.loss)
        return super().fit(xtrain, ytrain, train_info=train_info, **kwargs)


SURROGATE_REGISTRY = {"mlp": CustomMLP}
PREDICTOR_KWARG_KEYS = {
    "mlp": ("ss_type", "hparams_from_file", "num_layers", "layer_width",
            "batch_size", "lr", "regularization", "epochs", "loss"),
}


def build_predictor_kwargs(cfg):
    name = cfg["name"]
    if name not in SURROGATE_REGISTRY:
        raise ValueError(f"Unknown surrogate: {name}")
    if "corpus_path" not in cfg:
        raise ValueError("Missing required runtime config: corpus_path")
    if cfg["loss"] not in ("mse", "mae"):
        raise ValueError("MLP loss must be 'mse' or 'mae'")
    predictor_kwargs = {
        "base_predictor_cls": SURROGATE_REGISTRY[name],
        "corpus_path": cfg["corpus_path"],
        "embedding_col": cfg["embedding_col"],
        "use_pca": cfg["use_pca"],
        "pca_components": cfg["pca_components"],
    }
    predictor_kwargs.update({key: cfg[key] for key in PREDICTOR_KWARG_KEYS[name]})
    return predictor_kwargs
