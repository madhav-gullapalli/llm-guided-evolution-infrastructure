"""MLP seed factories, real numeric training, and mutation context."""
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'sota/Surrogate'))
    return importlib.import_module('sota.Surrogate.seed_models.surrogate.coev_surrogate_mlp')


def test_mlp_factory_and_directory_discovery(seed):
    from src.cfg.surrogate_coev import POPULATIONS
    from src.seed_models import discover_seed_models

    cfg = seed.get_surrogate_config({'epochs': 2}, corpus_path='runtime.csv', loss=None)
    assert cfg['name'] == 'mlp' and cfg['loss'] == 'mse'
    assert cfg['epochs'] == 2
    kwargs = seed.build_predictor_kwargs(cfg)
    assert kwargs['base_predictor_cls'] is seed.CustomMLP
    assert kwargs['corpus_path'] == 'runtime.csv'
    assert kwargs['num_layers'] == 3 and kwargs['layer_width'] == 128
    assert kwargs['epochs'] == 2 and kwargs['loss'] == 'mse'
    assert seed.DEFAULT_SURROGATE_CONFIG['epochs'] == 200
    names = [Path(path).name for path in discover_seed_models(REPO / POPULATIONS['surrogate']['seed_dir'])]
    assert names == ['coev_surrogate.py', 'coev_surrogate_mlp.py']
    with pytest.raises(ValueError, match='corpus_path'):
        seed.build_predictor_kwargs(seed.get_surrogate_config())
    with pytest.raises(ValueError, match='MLP loss'):
        seed.build_predictor_kwargs({**cfg, 'loss': 'invalid'})


def test_mlp_fits_numeric_embeddings(seed, monkeypatch):
    import naslib.predictors.mlp as mlp

    monkeypatch.setattr(mlp, 'device', torch.device('cpu'))
    cfg = seed.get_surrogate_config(corpus_path='test.csv', num_layers=1, layer_width=4,
                                    epochs=1, batch_size=4, regularization=0.0)
    kwargs = seed.build_predictor_kwargs(cfg)
    predictor_cls = kwargs.pop('base_predictor_cls')
    for key in ['corpus_path', 'embedding_col', 'use_pca', 'pca_components']:
        kwargs.pop(key)
    predictor = predictor_cls(encoding_type=None, **kwargs)
    xs = np.arange(12, dtype=np.float32).reshape(4, 3) / 12
    ys = np.array([0.1, 0.3, 0.6, 0.8])
    assert np.isfinite(predictor.fit(xs, ys))
    predictions = predictor.query(xs[:2])
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert len(predictor.model.layers) == 1
    assert predictor.model.layers[0].out_features == 4


@pytest.mark.parametrize("descendant", [False, True])
def test_mlp_mutation_uses_its_own_contract(seed, monkeypatch, tmp_path, descendant):
    import run_improved as runner

    monkeypatch.setattr(runner, 'PROMPT_GLOB', 'templates/naslib/general/*.txt')
    source = tmp_path / 'model_descendant.py'
    source.write_text(Path(seed.__file__).read_text())
    reference = REPO / 'sota/Surrogate/seed_models/surrogate/coev_surrogate.py'
    template, _ = runner.generate_template(
        0, -1, None, runner.SOTA_ROOT, str(reference) if descendant else seed.__file__,
        str(REPO), 'llama3', source_path=str(source) if descendant else None)
    assert 'SURROGATE COMPONENT CONTRACT (MLP)' in template
    assert 'epochs and loss to fit' in template
    assert 'You are editing NASLib\'s XGBoost subclass' not in template
