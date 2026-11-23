import collections
import logging
import torch
import copy
import numpy as np
import random
from scipy.stats import kendalltau

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import (
    acquisition_function,
)

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.zerocost import ZeroCost
from naslib.predictors.utils.encodings import encode_spec

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import AttrDict, count_parameters_in_MB, get_train_val_loaders
from naslib.utils.log import log_every_n_seconds

# for the test set caching and kendall tau calculation
import pickle
import os

from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_naslib, convert_naslib_to_str, convert_str_to_op_indices, convert_naslib_to_op_indices
logger = logging.getLogger(__name__)


class Bananas(MetaOptimizer):

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api=None):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 1000

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        self.test_size = None  # Will be set in adapt_search_space
        self.test_data = []
        self.test_accuracies = []
        self.test_hashes = set()  # Store hashes of test architectures to avoid sampling them
        self.surrogate_test_metrics = [] # store KT over time
        self.surrogate_test_metrics_mse = [] # store MSE over time
        self.test_min_accuracy = -1

        self.zc = config.search.zc if hasattr(config.search, 'zc') else None
        self.semi = "semi" in self.predictor_type 
        self.zc_api = zc_api
        self.use_zc_api = config.search.use_zc_api if hasattr(
            config.search, 'use_zc_api') else False
        self.zc_names = config.search.zc_names if hasattr(
            config.search, 'zc_names') else None
        self.zc_only = config.search.zc_only if hasattr(
            config.search, 'zc_only') else False
        
        self.load_labeled = config.search.load_labeled if hasattr(
            config.search, 'load_labeled') else False

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Bananas is currently only implemented for benchmarks."

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(
                self.config, mode="train")
        if self.semi:
            self.unlabeled = []
        
        # Determine test set size and generation strategy based on search space
        max_test_size = 1000
        presampled_op_indices = None
        
        if self.ss_type == 'nasbench201':
            # NB201: Use exhaustive enumeration (5^6 = 15625 total architectures)
            logger.info("NB201 detected: using exhaustive architecture enumeration for test set")
            
            # Enumerate all architectures
            arch_iterator = self.search_space.get_arch_iterator(dataset_api)
            all_op_indices = []
            
            # Filter for validity (same logic as in graph.py)
            def is_valid_arch(op_indices):
                op_list = list(op_indices)
                return not ((op_list[0] == op_list[1] == op_list[2] == 1) or
                            (op_list[2] == op_list[4] == op_list[5] == 1))
            
            for op_indices in arch_iterator:
                if is_valid_arch(op_indices):
                    all_op_indices.append(list(op_indices))
            
            # Shuffle and take min(space_size, max_test_size)
            import random as rand
            rand.shuffle(all_op_indices)
            self.test_size = min(len(all_op_indices), max_test_size)
            presampled_op_indices = all_op_indices[:self.test_size]
            logger.info(f"Enumerated {len(all_op_indices)} valid NB201 architectures, using {self.test_size} for test set")
        else:
            # NB101, NB301, etc: Use random sampling with hash-set collision checking
            self.test_size = max_test_size
            logger.info(f"Using random sampling with collision detection for test set size {self.test_size}")
        
        # Create a unique filename for this search space + dataset combo
        cache_filename = f"fixed_test_set_{self.search_space.get_type()}_{self.dataset}.pkl"

        if os.path.exists(cache_filename):
            print(f"Loading fixed test set from cache: {cache_filename}")
            try:
                with open(cache_filename, 'rb') as f:
                    arch_representations, self.test_accuracies = pickle.load(f)
                self.test_data = []
                self.test_hashes = set()
                
                for arch_repr in arch_representations:
                    arch = self.search_space.clone()
                    
                    # Handle different architecture representations
                    if self.ss_type == 'nasbench201':
                        # NB201: uses op_indices
                        if hasattr(arch, 'instantiate_model') and not arch.instantiate_model:
                            arch.op_indices = arch_repr
                        else:
                            convert_op_indices_to_naslib(arch_repr, arch)
                    elif self.ss_type == 'nasbench301':
                        # NB301: uses compact representation
                        arch.compact = arch_repr
                    elif self.ss_type == 'nasbench101':
                        # NB101: uses spec tuple representation
                        from naslib.search_spaces.nasbench101.conversions import convert_tuple_to_spec
                        arch.set_spec(arch_repr)
                    else:
                        # Other spaces: try op_indices first, then compact
                        if hasattr(arch, 'op_indices'):
                            arch.op_indices = arch_repr
                        elif hasattr(arch, 'compact'):
                            arch.compact = arch_repr
                        else:
                            logger.warning(f"Unknown architecture representation for {self.ss_type}")
                    
                    self.test_data.append(arch)
                    self.test_hashes.add(arch.get_hash())
            except Exception as e:
                print(f"Failed to load cache ({e}), regenerating...")
                self.test_data = [] # Trigger regeneration below
        else:
            # Initialize lists if cache didn't exist
            self.test_data = []

        # Generate if we didn't load successfully
        if not self.test_data:
            print(f"Generating new fixed test set ({self.test_size} samples)...")
            self.test_accuracies = []
            self.test_hashes = set()
            
            if presampled_op_indices is not None:
                # NB201: Use presampled architectures from enumeration
                for op_indices in presampled_op_indices:
                    arch = self.search_space.clone()
                    arch.set_op_indices(op_indices)
                    
                    # Parse the architecture if needed
                    if self.search_space.instantiate_model == True:
                        arch.parse()
                    
                    # Query ground truth
                    acc = arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
                    
                    # Filter out architectures below minimum accuracy threshold
                    if acc < self.test_min_accuracy:
                        continue
                    
                    print(f"Generated test architecture with {self.performance_metric}: {acc:.4f}") 
                    self.test_data.append(arch)
                    self.test_accuracies.append(acc)
                    self.test_hashes.add(arch.get_hash())
                    
                    # Stop if we've reached desired test set size
                    if len(self.test_data) >= self.test_size:
                        break
            else:
                # Other spaces: Random sampling with hash collision checking
                attempts = 0
                max_attempts = self.test_size * 10  # Prevent infinite loops
                
                while len(self.test_data) < self.test_size and attempts < max_attempts:
                    arch = self.search_space.clone()
                    arch.sample_random_architecture(dataset_api=self.dataset_api)
                    
                    # Parse the architecture if needed
                    if self.search_space.instantiate_model == True:
                        arch.parse()
                    
                    arch_hash = arch.get_hash()
                    
                    # Skip if we've already sampled this architecture
                    if arch_hash in self.test_hashes:
                        attempts += 1
                        continue
                    
                    # Query ground truth
                    acc = arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
                    
                    # Filter out architectures below minimum accuracy threshold
                    if acc < self.test_min_accuracy:
                        attempts += 1
                        continue
                    
                    print(f"Generated test architecture {len(self.test_data)+1}/{self.test_size} with {self.performance_metric}: {acc:.4f}") 
                    self.test_data.append(arch)
                    self.test_accuracies.append(acc)
                    self.test_hashes.add(arch_hash)
                    attempts += 1
                
                if len(self.test_data) < self.test_size:
                    logger.warning(f"Could only generate {len(self.test_data)} unique architectures out of {self.test_size} requested")

            # Save to cache for next time
            print(f"Saving fixed test set ({len(self.test_data)} architectures) to {cache_filename}")
            with open(cache_filename, 'wb') as f:
                arch_representations = []
                for arch in self.test_data:
                    # Get architecture representation based on search space type
                    if self.ss_type == 'nasbench201':
                        arch_representations.append(arch.get_op_indices())
                    elif self.ss_type == 'nasbench301':
                        arch_representations.append(arch.get_compact())
                    elif self.ss_type == 'nasbench101':
                        # NB101: use tuple representation from get_hash()
                        arch_representations.append(arch.get_hash())
                    else:
                        # Fallback: try op_indices first, then compact
                        if hasattr(arch, 'get_op_indices'):
                            arch_representations.append(arch.get_op_indices())
                        elif hasattr(arch, 'get_compact'):
                            arch_representations.append(arch.get_compact())
                        else:
                            logger.warning(f"Cannot extract architecture representation for {self.ss_type}")
                pickle.dump((arch_representations, self.test_accuracies), f)
        
        print(f'[Bananas Optimizer] Test set generation complete: {len(self.test_data)} architectures')

    def get_zero_cost_predictors(self):
        return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}

    def query_zc_scores(self, arch):
        zc_scores = {}
        zc_methods = self.get_zero_cost_predictors()
        arch_hash = arch.get_hash()
        for zc_name, zc_method in zc_methods.items():

            if self.use_zc_api and str(arch_hash) in self.zc_api:
                score = self.zc_api[str(arch_hash)][zc_name]['score']
            else:
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query(arch, dataloader=zc_method.train_loader)

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[zc_name] = score

        return zc_scores

    def _remove_from_test_set(self, arch_hash):
        """Remove architecture from test set if it exists there."""
        if arch_hash in self.test_hashes:
            # Find and remove from test_data and test_accuracies
            for idx, test_arch in enumerate(self.test_data):
                if test_arch.get_hash() == arch_hash:
                    logger.info(f"Removing architecture {arch_hash} from test set (entered training population)")
                    self.test_hashes.remove(arch_hash)
                    self.test_data.pop(idx)
                    self.test_accuracies.pop(idx)
                    break

    def _set_scores(self, model):

        if self.use_zc_api and str(model.arch_hash) in self.zc_api:
            model.accuracy = self.zc_api[str(model.arch_hash)]['val_accuracy']
        else:
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

        if self.zc and len(self.train_data) <= self.max_zerocost:
            model.zc_scores = self.query_zc_scores(model.arch)

        self.train_data.append(model)
        self._update_history(model)

    def _sample_new_model(self):
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(
            dataset_api=self.dataset_api, load_labeled=self.load_labeled)
        model.arch_hash = model.arch.get_hash()
        
        if self.search_space.instantiate_model == True:
            model.arch.parse()
        return model

    def _get_train(self):
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    def _get_ensemble(self):
        ensemble = Ensemble(num_ensemble=self.num_ensemble,
                            ss_type=self.ss_type,
                            predictor_type=self.predictor_type,
                            zc=self.zc,
                            zc_only=self.zc_only,
                            config=self.config)

        return ensemble

    def _get_new_candidates(self, ytrain):
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == 'random_sampling':

            for _ in range(self.num_candidates):
                # self.search_space.sample_random_architecture(dataset_api=self.dataset_api, load_labeled=self.sample_from_zc_api) # FIXME extend to Zero Cost case
                model = self._sample_new_model()
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )
                candidates.append(model)

        elif self.acq_fn_optimization == 'mutation':
            # mutate the k best architectures by x
            best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
            best_archs = [self.train_data[i].arch for i in best_arch_indices]
            candidates = []
            for arch in best_archs:
                for _ in range(int(self.num_candidates / len(best_archs) / self.max_mutations)):
                    candidate = arch.clone()
                    for __ in range(int(self.max_mutations)):
                        arch = self.search_space.clone()
                        arch.mutate(candidate, dataset_api=self.dataset_api)
                        if self.search_space.instantiate_model == True:
                            arch.parse()
                        candidate = arch

                    model = torch.nn.Module()
                    model.arch = candidate
                    model.arch_hash = candidate.get_hash()
                    candidates.append(model)

        else:
            logger.info('{} is not yet supported as a acq fn optimizer'.format(
                self.encoding_type))
            raise NotImplementedError()

        return candidates

    def new_epoch(self, epoch):

        if epoch < self.num_init:
            model = self._sample_new_model()
            self._remove_from_test_set(model.arch_hash)
            self._set_scores(model)
        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain, ytrain = self._get_train()
                ensemble = self._get_ensemble()

                if self.semi:
                    # create unlabeled data and pass it to the predictor
                    while len(self.unlabeled) < len(xtrain):
                        model = self._sample_new_model()

                        if self.zc and len(self.train_data) <= self.max_zerocost:
                            model.zc_scores = self.query_zc_scores(model.arch)

                        self.unlabeled.append(model)

                    ensemble.set_pre_computations(
                        unlabeled=[m.arch for m in self.unlabeled]
                    )

                if self.zc and len(self.train_data) <= self.max_zerocost:
                    # pass the zero-cost scores to the predictor
                    train_info = {'zero_cost_scores': [
                        m.zc_scores for m in self.train_data]}
                    ensemble.set_pre_computations(xtrain_zc_info=train_info)

                    if self.semi:
                        unlabeled_zc_info = {'zero_cost_scores': [
                            m.zc_scores for m in self.unlabeled]}
                        ensemble.set_pre_computations(
                            unlabeled_zc_info=unlabeled_zc_info)

                ensemble.fit(xtrain, ytrain)

                # Predict scores for the test set
                pred_scores = ensemble.query(self.test_data)
                
                # Calculate Kendall Tau (Ranking Correlation)
                tau, _ = kendalltau(self.test_accuracies, pred_scores)
                
                # Calculate MSE (Mean Squared Error)
                mse = np.mean((np.array(self.test_accuracies) - np.array(pred_scores)) ** 2)
                
                # Log the metrics
                self.surrogate_test_metrics.append(tau)
                self.surrogate_test_metrics_mse.append(mse)
                logger.info(f"Epoch {epoch}: Surrogate Test Kendall Tau = {tau:.4f}, MSE = {mse:.6f}")
                # -----------------------------------------------
                # define an acquisition function
                acq_fn = acquisition_function(
                    ensemble=ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type
                )

                # optimize the acquisition function to output k new architectures
                candidates = self._get_new_candidates(ytrain=ytrain)

                self.next_batch = self._get_best_candidates(candidates, acq_fn)

            # train the next architecture chosen by the neural predictor
            model = self.next_batch.pop()
            self._remove_from_test_set(model.arch_hash)
            self._set_scores(model)

    def _get_best_candidates(self, candidates, acq_fn):
        if self.zc and len(self.train_data) <= self.max_zerocost:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch)

            values = [acq_fn(model.arch, [{'zero_cost_scores': model.zc_scores}]) for model in candidates]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        choices = [candidates[i] for i in sorted_indices[-self.k:]]

        return choices

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self, report_incumbent=True):
        if report_incumbent:
            best_arch = self.get_final_architecture()
        else:
            best_arch = self.train_data[-1].arch
        
        if self.search_space.space_name != "nasbench301":
            return (
                best_arch.query(
                    Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                best_arch.query(
                    Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                best_arch.query(
                    Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                best_arch.query(
                    Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
                ),
            )
        else:
            return (
                -1, 
                best_arch.query(
                    Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                best_arch.query(
                    Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                best_arch.query(
                    Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
                ),
            ) 

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        if self.search_space.space_name != "nasbench301":
            return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        else:
            return -1

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == 'nasbench301':
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch
