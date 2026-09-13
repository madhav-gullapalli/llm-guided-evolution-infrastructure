import glob
import pandas as pd
import os

os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
import numpy as np
import logging
from scipy.stats import norm
import sys
import time
import types
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cfg.constants_surrogate import (
    MODEL as DEFAULT_MODEL_MODULE,
    SURROGATE_CORPUS_PATH,
    SURROGATE_DATASET,
    SURROGATE_RESULTS_DIR,
    SURROGATE_RUN_DIR,
    SURROGATE_SEARCH_SPACE,
    VARIANT_DIR as DEFAULT_VARIANT_DIR,
)

from trajectory_plots import surrogate_data_plots
# --- PARSE CUSTOM ARGUMENTS FIRST (before any NASLib imports) ---
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=242, help='Random seed for reproducibility')
parser.add_argument('--run_baselines', action='store_true', help='Run baseline experiments')
parser.add_argument('--surrogate', type=str, default=None, help='Override the surrogate family defined in the model module')
parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
parser.add_argument('--debug', action='store_true', help='stores trajectory_log and candudate_Log', default=True)
parser.add_argument('--model', type=str, default=DEFAULT_MODEL_MODULE, help='Model module name (LLM-GE evolved variant)')
parser.add_argument('--variant_dir', type=str, default=DEFAULT_VARIANT_DIR, help='Directory where LLM-GE writes model variants')
parser.add_argument('--epochs', type=int, default=100, help='Number of BANANAS search epochs')
custom_args, remaining = parser.parse_known_args()

# Update sys.argv to only contain args that NASLib's parser understands
# NASLib parser accepts --seed, so we keep it
sys.argv = [sys.argv[0], '--seed', str(custom_args.seed)] + remaining

# NOW import NASLib (which will parse the cleaned sys.argv)
from naslib import utils 
from naslib.utils import get_dataset_api, create_exp_dir
from naslib.utils.encodings import EncodingType
from naslib.search_spaces import NasBench201SearchSpace 
from naslib.optimizers import RegularizedEvolution, Bananas, Npenas
from naslib.defaults.trainer import Trainer

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.gp import VarSparseGPPredictor, GPPredictor
from naslib.predictors.llm_enhanced_201 import LLM_NB201_Predictor 

from naslib.optimizers.discrete.bananas import optimizer as bananas_opt
from naslib.optimizers.discrete.bananas import acquisition_functions as acq_funcs

import matplotlib.pyplot as plt
import seaborn as sns
import json

# Store custom args for use later in the script
args = custom_args

# --- LOAD EVOLVED MODEL CONFIG ---
import importlib
_script_dir = Path(__file__).parent.resolve()
_variant_dir = args.variant_dir if os.path.isabs(args.variant_dir) else str(_script_dir / args.variant_dir)
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
sys.path.append(_variant_dir)
_model_module = importlib.import_module(args.model)
surrogate_config = _model_module.get_surrogate_config(corpus_path=SURROGATE_CORPUS_PATH)
if args.surrogate is not None:
    surrogate_config["name"] = args.surrogate

try:
    gene_id = args.model.split('model_')[1]
except:
    gene_id = 'seed'

# --- CONFIGURATION ---
config = utils.get_config_from_args(config_type="nas")
config.dataset = SURROGATE_DATASET
config.search_space = SURROGATE_SEARCH_SPACE 
config.optimizer = ""
config.search.seed = args.seed
config.seed = args.seed
config.save_arch_weights = False
config.search.num_init = 50
config.search.k = 10
config.search.epochs = args.epochs
config.search.num_candidates = 100
config.out_dir = os.path.join(SURROGATE_RUN_DIR, gene_id)
config.debug_predictor = True
config.search.num_ensemble = 3
config.search.num_arches_to_mutate = 16
config.search.max_mutations = 3
config.search.checkpoint_freq = 10000

RUN_BASELINES = args.run_baselines
RUN_ALL = True
SURROGATE = args.surrogate  # Options: 'xgboost' or 'mlp'
NUM_TRIALS = args.trials
DEBUG = args.debug

print("RUN_BASELINES:", RUN_BASELINES)
print("SURROGATE:", SURROGATE or surrogate_config["name"])

if RUN_ALL:
    print("Running Both Baselines and LLM method.")
elif RUN_BASELINES:
    print("Running Baselines only.")
else:
    print("Running LLM method only.")

# config printing
print("Experiment Configuration:")
print(f"Dataset: {config.dataset}")
print(f"Search Space: {config.search_space}")
print(f"Seed: {config.search.seed}")
print(f"Number of Initial Samples: {config.search.num_init}")
print(f"Expansion Size (k): {config.search.k}")
print(f"Number of Search Epochs: {config.search.epochs}")
print(f"Number of Candidates: {config.search.num_candidates}")
print(f"Size of Ensemble: {config.search.num_ensemble}")

def set_config_save():
    global config
    config.save = os.path.join(config.out_dir, config.search_space, config.dataset, config.optimizer, str(config.search.seed))

def write_config_to_file():
    global config
    config_save_path = os.path.join(config.save, "config.txt")
    with open(config_save_path, "w") as f:
        f.write(str(config))
    print(f"Configuration saved to {config_save_path}")

original_acq_factory = acq_funcs.acquisition_function

def batch_acquisition_function(ensemble, ytrain, acq_fn_type="its", explore_factor=0.5, ei_calibration_factor=5.0):
    """
    Batched wrapper that returns a function capable of processing 
    a list of architectures in one shot.
    """
    
    # Define helper to handle batch statistics (N, Ensemble_Size) -> (N,)
    def get_stats(preds): 
        return np.mean(preds, axis=0), np.std(preds, axis=0)

    # 2. Define the ACTUAL execution logic (The thing that runs on GPU)
    def batched_executor(archs, info=None):
        t0 = time.time()
        
        # A. Batch Inference (The Bottleneck)
        # ensemble.query passes the full list 'archs' to the predictor
        # This triggers ONE gpu call instead of 2000
        preds = ensemble.query(archs, info) 
        
        # B. Acquisition Math
        mean, std = get_stats(preds)
        
        if acq_fn_type == "its":
            scores = np.random.normal(mean, std)
        elif acq_fn_type == "ucb":
            scores = mean + explore_factor * std
        elif acq_fn_type == "ei":
            fs = std / ei_calibration_factor
            gam = (mean - ytrain.max()) / fs
            scores = fs * (gam * norm.cdf(gam) + norm.pdf(gam))
        else:
            # Fallback for unknown types (rare)
            scores = original_acq_factory(ensemble, ytrain, acq_fn_type)(archs, info)

        t1 = time.time()
        print(f"[Batch Acquisition] Processed {len(archs)} candidates in {t1-t0:.4f}s")
        
        return scores

    # 3. Return the batched callable
    return batched_executor

def _get_best_candidates_batch(self, candidates, acq_fn):
    from naslib.search_spaces.core.query_metrics import Metric
    print(f"candidate count: {len(candidates)}, candidates: {candidates}")
    info = [{'zero_cost_scores': c.zc_scores} for c in candidates] \
        if self.zc and len(self.train_data) <= self.max_zerocost else None

    preds = self.ensemble.query([c.arch for c in candidates], info)
    mean_preds = np.mean(preds, axis=0)

    acq_values = acq_fn([c.arch for c in candidates], info)

    selected_indices = np.argsort(acq_values)[-self.k:]
    selected = [candidates[i] for i in selected_indices]

    # --- LOG ALL CANDIDATES ---
    if not hasattr(self, "candidate_log" ):
        self.candidate_log = []
    def compute_nb201_index(op_indices):
        index = 0
        for i, op in enumerate(op_indices):
            index += int(op) * (5 ** i)
        return index
    for i, c in enumerate(candidates):

        arch_index = compute_nb201_index(c.arch.op_indices)
        true_accuracy = c.arch.query(
            metric=Metric.VAL_ACCURACY,
            dataset=config.dataset,
            dataset_api=self.dataset_api
        )
        self.candidate_log.append({
            "generation": int(self.current_epoch),
            "arch_index": int(arch_index),
            "op_indices": list(map(int, c.arch.op_indices)),
            "predicted_accuracy": float(mean_preds[i]),
            "acquisition_value": float(acq_values[i]),
            "selected": bool(i in selected_indices),
            "true_accuracy": float(true_accuracy) if true_accuracy is not None else None
        })

    return selected

acq_funcs.acquisition_function = batch_acquisition_function
bananas_opt.acquisition_function = batch_acquisition_function
bananas_opt.Bananas._get_best_candidates = _get_best_candidates_batch
print("Patched Bananas optimizer for batch acquisition function.")

# --- LOAD API ---
dataset_api = get_dataset_api(config.search_space, config.dataset)

# --- MONKEY PATCH TO DEBUG TIMING ---
def debug_new_epoch(self, epoch):
    import time
    from scipy.stats import kendalltau
    self.current_epoch = epoch

    if epoch < self.num_init:
        model = self._sample_new_model()
        self._set_scores(model)
    else:
        if len(self.next_batch) == 0:
            # 1. Train
            print("[Debug] Starting Fit...")
            t0 = time.time()
            xtrain, ytrain = self._get_train()
            self.ensemble = self._get_ensemble()
            
            # Setup semi-supervised if needed (omitted for brevity as self.semi usually False)
            
            self.ensemble.fit(xtrain, ytrain)
            t1 = time.time()
            print(f"[Debug] Fit finished in {t1-t0:.2f}s")

            # 2. Acquisition Setup
            acq_fn = bananas_opt.acquisition_function(
                ensemble=self.ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type
            )

            # 3. Candidate Generation
            print(f"[Debug] Generating {self.num_candidates} candidates...")
            t2 = time.time()
            candidates = self._get_new_candidates(ytrain=ytrain)
            t3 = time.time()
            print(f"[Debug] Generated candidates in {t3-t2:.2f}s")

            # 4. Selection (Acquisition)
            print("[Debug] Selecting best candidates...")
            self.next_batch = self._get_best_candidates(candidates, acq_fn)
            t4 = time.time()
            print(f"[Debug] Selection finished in {t4-t3:.2f}s")
            
            # Compute Kendall Tau on test set (only when ensemble is retrained)
            pred_scores = self.ensemble.query(self.test_data)  # Shape: (num_ensemble, num_test)
            # Aggregate ensemble predictions by taking mean across ensemble members
            mean_pred_scores = np.mean(pred_scores, axis=0)  # Shape: (num_test,)
            
            # Calculate Kendall Tau (Ranking Correlation)
            tau, _ = kendalltau(self.test_accuracies, mean_pred_scores)
            
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((np.array(self.test_accuracies) - mean_pred_scores) ** 2)
            
            # Log the metrics
            self.surrogate_test_metrics.append(tau)
            self.surrogate_test_metrics_mse.append(mse)
            print(f"Epoch {epoch}: Surrogate Test Kendall Tau = {tau:.4f}, MSE = {mse:.6f}")

        # 5. Evaluation
        print(f"[Debug] Evaluating architecture {len(self.next_batch)}...")
        t5 = time.time()
        model = self.next_batch.pop()
        self._set_scores(model)
        if not hasattr(self, "trajectory"):
            self.trajectory = []

        self.trajectory.append({
            "op_indices": getattr(model, "op_indices", None),
            "generation": epoch,
            "predicted_accuracy": getattr(model, "predicted_accuracy", None),
            "true_accuracy": model.accuracy
        })
        print(f"[Debug] Evaluation finished in {time.time()-t5:.2f}s")

# Apply the patch
bananas_opt.Bananas.new_epoch = debug_new_epoch
print("Monkey-patched Bananas.new_epoch for profiling.")




# =============================================================================
# STRICTLY MINIMAL PATCH: SPEED + ACCURACY FIX
# =============================================================================
import torch
import torch.nn as nn
from naslib.search_spaces.core import primitives as core_ops
from naslib.search_spaces.nasbench201 import primitives as nb201_ops
import naslib.search_spaces.nasbench201.conversions as conversions_mod
from naslib.search_spaces import NasBench201SearchSpace

print("[Patch] Applying Minimal Fixes...")

# -----------------------------------------------------------------------------
# 1. SPEED FIX: Replace the Heavy Primitives
# -----------------------------------------------------------------------------
# We replace the __init__ methods of the heavy classes. 

def fast_relu_init(self, C_in, C_out, kernel_size, stride=1, affine=True, track_running_stats=True, **kwargs):
    super(core_ops.ReLUConvBN, self).__init__(locals())
    self.kernel_size = kernel_size  # Required for get_op_name
    self.stride = stride
    self.C_in = C_in
    self.C_out = C_out
    # Speedup: No nn.Conv2d created here.

def fast_avg_pool_init(self, kernel_size, stride=1, affine=True, **kwargs):
    super(core_ops.AvgPool1x1, self).__init__(locals())
    self.kernel_size = kernel_size  # Required for get_op_name
    self.stride = stride

def fast_stem_init(self, C_in, C_out, **kwargs):
    super(core_ops.Stem, self).__init__(locals())
    self.C_in = C_in
    self.C_out = C_out

def fast_resnet_init(self, C_in, C_out, stride, affine=True, track_running_stats=True, **kwargs):
    super(nb201_ops.ResNetBasicblock, self).__init__(locals())
    self.stride = stride
    self.C_in = C_in
    self.C_out = C_out

# Apply the swaps in-place.
core_ops.ReLUConvBN.__init__ = fast_relu_init
core_ops.AvgPool1x1.__init__ = fast_avg_pool_init
core_ops.Stem.__init__ = fast_stem_init
nb201_ops.ResNetBasicblock.__init__ = fast_resnet_init

print("[Patch] Primitives are now lightweight. Graph creation should be <0.001s.")

# =============================================================================
# OPTIMIZED PATCH: HOLLOW SEARCH SPACE (SAFE MODE)
# =============================================================================
def fast_hollow_init(self, n_classes=10, in_channels=3):
    """
    Replacement __init__ that skips building the graph topology.
    """
    # Initialize parent without heavy setup
    super(NasBench201SearchSpace, self).__init__()
    
    # Essential attributes for NASLib
    self.num_classes = n_classes
    self.op_indices = None
    self.max_epoch = 199
    self.in_channels = in_channels
    self.space_name = "nasbench201"
    self.labeled_archs = None
    
    # CRITICAL: Disables graph interaction
    self.instantiate_model = False 
    self.sample_without_replacement = False

# Override get_op_indices to never look at the (non-existent) graph
def safe_get_op_indices(self):
    if self.op_indices is None:
        # Fallback if accessed before setting (should not happen in Bananas)
        raise ValueError("Attempted to access op_indices of uninitialized Hollow Graph")
    return self.op_indices

# Apply patches
NasBench201SearchSpace.__init__ = fast_hollow_init
NasBench201SearchSpace.get_op_indices = safe_get_op_indices

print("[Patch] NasBench201SearchSpace is now hollow and safe.")


for i in range(NUM_TRIALS):
    print(f"\n\n=== TRIAL {i+1}/{NUM_TRIALS} ===")
    config.search.seed = args.seed*(i+1)
    config.seed = config.search.seed
    
    def run_experiment(optimizer_name, predictor_cls=None, predictor_kwargs=None):
        p_name = predictor_cls.__name__ if predictor_cls else "Default"
        print(f"\n\n>>> RUNNING: {optimizer_name} (Predictor: {p_name}) <<<")
        
        if p_name == "LLM_NB201_Predictor":
            base_pred = predictor_kwargs['base_predictor_cls'].__name__
            config.optimizer = f"{p_name}_{base_pred}_{optimizer_name}"
        else:
            config.optimizer = f"{p_name}_{optimizer_name}"
        set_config_save()

        # 1. Select Optimizer
        if optimizer_name == "rea":
            optimizer = RegularizedEvolution(config)
        elif optimizer_name == "bananas":
            optimizer = Bananas(config)
        elif optimizer_name == "npenas":
            optimizer = Npenas(config)
        
        # 2. Setup Search Space (NB201)
        search_space = NasBench201SearchSpace()
        optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

        
        # 3. INJECT CUSTOM PREDICTOR
        if predictor_cls is not None:
            print(f"Injecting Custom Predictor: {p_name}")
            
            if optimizer_name == "bananas":
                # --- MONKEY PATCHING ENSEMBLE ---
                def _get_custom_ensemble(self):
                    ensemble = Ensemble(num_ensemble=self.num_ensemble, ss_type=self.ss_type, predictor_type=self.predictor_type, config=self.config, zc=self.zc)
                    # CHANGE: Create 3 instances matching the given custom predictor
                    ensemble.ensemble = [
                        predictor_cls(**predictor_kwargs) 
                        for _ in range(self.num_ensemble)
                    ]
                    # try:
                    #     print("Ensemble Predictor Hyperparameters:")
                    #     print(ensemble.ensemble[0].default_hyperparams)
                    # except:
                    #     print("Predictor has no default_hyperparams attribute.")
                    return ensemble

                optimizer._get_ensemble = types.MethodType(_get_custom_ensemble, optimizer)
                # --- END MONKEY PATCHING ---
            else:
                optimizer.predictor = predictor_cls(**predictor_kwargs)
        
        # 4. Run Search
        create_exp_dir(config.save)
        create_exp_dir(config.save + "/search")
        create_exp_dir(config.save + "/eval")
        write_config_to_file()
        trainer = Trainer(optimizer, config, lightweight_output=True)
        trainer.search() 


        # Access the optimizer instance from the trainer
        optimizer = trainer.optimizer
        #save trajectory if available
        if hasattr(optimizer, "trajectory") and DEBUG:
            trajectory_path = os.path.join(
                config.save,
                f"trajectory_trial_{i}_seed_{config.search.seed}.json"
            )
            with open(trajectory_path, "w") as f:
                json.dump(optimizer.trajectory, f, indent=2)
            print(f"Trajectory saved to {trajectory_path}")
        else:
            print("No trajectory found on optimizer.")
        if hasattr(optimizer, "candidate_log") and DEBUG:
            candidate_path = os.path.join(
                config.save,
                f"candidate_log_trial_{i}_seed_{config.search.seed}.json"
            )
            
            with open(candidate_path, "w") as f:
                json.dump(optimizer.candidate_log, f, indent=2)
            surrogate_data_plots(config.save, sel = False)
            surrogate_data_plots(config.save, sel = True)
            print(f"Candidate log saved to {candidate_path}")
        else:
            print("No candidate_log found on optimizer.")
        # Retrieve the stored metrics
        if hasattr(optimizer, 'surrogate_test_metrics') and len(optimizer.surrogate_test_metrics) > 0:
            metrics_tau = optimizer.surrogate_test_metrics
            metrics_mse = optimizer.surrogate_test_metrics_mse if hasattr(optimizer, 'surrogate_test_metrics_mse') else []
            epochs = list(range(len(metrics_tau)))

            # Save metrics to JSON file for later analysis
            # Create a unique experiment identifier
            predictor_type = p_name if p_name != "LLM_NB201_Predictor" else f"LLM_{predictor_kwargs['base_predictor_cls'].__name__}"
            experiment_id = f"{predictor_type}_{optimizer_name}"
            
            metrics_data = {
                'experiment_id': experiment_id,
                'optimizer_type': optimizer_name,
                'predictor_class': p_name,
                'base_predictor': predictor_kwargs.get('base_predictor_cls').__name__ if p_name == "LLM_NB201_Predictor" else None,
                'surrogate_type': SURROGATE,
                'uses_llm': p_name == "LLM_NB201_Predictor",
                'dataset': config.dataset,
                'seed': config.search.seed,
                'trial': i,
                'epochs': epochs,
                'kendall_tau': metrics_tau,
                'mse': metrics_mse,
                'config_str': config.optimizer
            }
            
            metrics_path = os.path.join(config.save, f"surrogate_metrics_trial_{i}_seed_{config.search.seed}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"Surrogate metrics saved to {metrics_path}")

            # Plot individual trial with dual panels
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot Kendall Tau
            ax1.plot(epochs, metrics_tau, marker='o', linestyle='-', color='b', label='Kendall Tau')
            ax1.set_title(f'Surrogate Kendall Tau on Test Set ({config.optimizer})')
            ax1.set_xlabel('Search Iterations (Model Updates)')
            ax1.set_ylabel('Kendall Tau')
            ax1.grid(True)
            ax1.legend()
            
            # Plot MSE (if available)
            if metrics_mse:
                ax2.plot(epochs, metrics_mse, marker='s', linestyle='-', color='r', label='MSE')
                ax2.set_title(f'Surrogate MSE on Test Set ({config.optimizer})')
                ax2.set_xlabel('Search Iterations (Model Updates)')
                ax2.set_ylabel('MSE')
                ax2.grid(True)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'MSE data not available', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(config.save, f"surrogate_test_metrics_trial_{i}_seed_{config.search.seed}.png")
            plt.savefig(plot_path)
            print(f"Surrogate metrics plot saved to {plot_path}")
            plt.close()
    
            return trainer.optimizer.history


# ----------------------------
# Run experiment
# ----------------------------

if not RUN_BASELINES or RUN_ALL:
    start_time = time.time()

    history = run_experiment(
        "bananas",
        predictor_cls=LLM_NB201_Predictor,
        predictor_kwargs=_model_module.build_predictor_kwargs(surrogate_config),
    )

    # run_experiment saves surrogate metrics to a JSON inside config.out_dir
    import glob as _glob
    metrics_files = sorted(_glob.glob(f"{config.out_dir}/**/surrogate_metrics_*.json", recursive=True))
    if metrics_files:
        import json as _json
        with open(metrics_files[-1]) as _f:
            _m = _json.load(_f)
        final_kendall_tau = _m["kendall_tau"][-1] if _m["kendall_tau"] else 0.0
        final_mse = _m["mse"][-1] if _m["mse"] else 0.0
    else:
        print("Warning: no surrogate metrics JSON found — defaulting to worst-case values.")
        final_kendall_tau = 0.0
        final_mse = float('inf')

    # Write LLM-GE results file
    os.makedirs(SURROGATE_RESULTS_DIR, exist_ok=True)
    filename = os.path.abspath(os.path.join(SURROGATE_RESULTS_DIR, f'{gene_id}_results.txt'))
    with open(filename, 'w') as _f:
        _f.write(f"{final_kendall_tau},{final_mse}")

    print(f"Results written to {filename}")
    print(f"  Kendall Tau: {final_kendall_tau:.4f}")
    print(f"  MSE:         {final_mse:.6f}")
    print('=' * 120); print('job done'); print('=' * 120)
