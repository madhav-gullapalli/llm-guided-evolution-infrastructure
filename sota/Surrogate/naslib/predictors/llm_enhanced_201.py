import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

# NASLib Utilities for NASBench201
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str
from naslib.search_spaces.nasbench201.conversions import convert_op_indices_to_str

# ==========================================
# 1. PRE-COMPUTED EMBEDDING CACHE LOADER (SINGLETON)
# ==========================================
class EmbeddingCacheLoader:
    """
    Singleton loader for pre-computed embeddings from CSV/Pickle file for NASBench201.
    Replaces online LLM inference with O(1) cache lookups.
    Implemented as singleton to avoid loading the corpus multiple times.

    The raw COLE CSV is ~5GB of text (code + several embedding columns). Loading it
    with pandas OOMs gene-eval jobs. We stream only arch_string + the requested
    column, then write a compact npz next to the evaluator for later runs.
    """
    _instance = None
    _cache = {}
    _CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "embedding_cache"

    def __new__(cls, corpus_path=None, embedding_col='codellama_python_7b_pytorch_code_embedding'):
        if cls._instance is None:
            cls._instance = super(EmbeddingCacheLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, corpus_path=None, embedding_col='codellama_python_7b_pytorch_code_embedding'):
        if self._initialized:
            return

        if corpus_path is None:
            raise ValueError("corpus_path must be provided on first initialization")

        self.corpus_path = corpus_path
        self.embedding_col = embedding_col
        self._load_corpus()
        self._initialized = True

    def _npz_path(self) -> Path:
        key = hashlib.sha1(f"{self.corpus_path}|{self.embedding_col}".encode()).hexdigest()[:16]
        return self._CACHE_DIR / f"nb201_{key}.npz"

    @staticmethod
    def _parse_embedding(val):
        if isinstance(val, (np.ndarray, list)):
            return np.asarray(val, dtype=np.float32)

        val_str = str(val).strip()
        if "," in val_str:
            try:
                return np.asarray(json.loads(val_str), dtype=np.float32)
            except Exception:
                pass

        cleaned = val_str.replace("[", "").replace("]", "").replace("\n", " ").strip()
        parsed = np.fromstring(cleaned, sep=" ", dtype=np.float32)
        if parsed.size:
            return parsed
        from ast import literal_eval
        return np.asarray(literal_eval(val_str), dtype=np.float32)

    def _load_npz(self, path: Path):
        print(f"[Cache Loader] Loading compact cache {path} ...", flush=True)
        data = np.load(path, allow_pickle=True)
        keys = data["arch_strings"]
        embs = data["embeddings"]
        EmbeddingCacheLoader._cache = {str(k): embs[i] for i, k in enumerate(keys)}
        print(f"[Cache Loader] Successfully loaded {len(EmbeddingCacheLoader._cache)} embeddings.", flush=True)

    def _save_npz(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(EmbeddingCacheLoader._cache.keys()), dtype=object)
        embs = np.stack(list(EmbeddingCacheLoader._cache.values()))
        tmp = path.with_name(path.stem + "_tmp.npz")
        np.savez(tmp, arch_strings=keys, embeddings=embs)
        os.replace(tmp, path)
        print(f"[Cache Loader] Wrote compact cache ({embs.nbytes / 1e6:.1f} MB arrays) to {path}", flush=True)

    def _load_csv_streaming(self, csv_path: str):
        import csv

        print(
            f"[Cache Loader] Streaming {csv_path} column {self.embedding_col!r} "
            "(do not pandas-read the full 4.8GB file)...",
            flush=True,
        )
        cache = {}
        with open(csv_path, newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            try:
                arch_idx = header.index("arch_string")
                emb_idx = header.index(self.embedding_col)
            except ValueError as exc:
                raise ValueError(
                    f"CSV missing required column. Have {header}. Need arch_string and {self.embedding_col}."
                ) from exc

            for i, row in enumerate(reader, start=1):
                cache[row[arch_idx]] = self._parse_embedding(row[emb_idx])
                if i % 1000 == 0:
                    print(f"[Cache Loader] ... {i} architectures", flush=True)

        EmbeddingCacheLoader._cache = cache
        print(f"[Cache Loader] Successfully loaded {len(cache)} embeddings from CSV.", flush=True)

    def _load_corpus(self):
        """Load embeddings from compact npz, pickle, or streamed CSV."""
        npz_path = self._npz_path()
        print(f"[Cache Loader] Requested corpus: {self.corpus_path}", flush=True)
        print(f"[Cache Loader] Compact cache path: {npz_path}", flush=True)

        try:
            if npz_path.is_file():
                self._load_npz(npz_path)
                return

            src = self.corpus_path
            if src.endswith(".npz"):
                self._load_npz(Path(src))
                return
            if src.endswith(".pkl"):
                import pandas as pd

                print(f"[Cache Loader] Loading pickle {src} ...", flush=True)
                df = pd.read_pickle(src)
                EmbeddingCacheLoader._cache = {
                    k: self._parse_embedding(v)
                    for k, v in zip(df["arch_string"], df[self.embedding_col])
                }
                print(
                    f"[Cache Loader] Successfully loaded {len(EmbeddingCacheLoader._cache)} embeddings.",
                    flush=True,
                )
            else:
                self._load_csv_streaming(src)

            self._save_npz(npz_path)
        except Exception as e:
            print(f"[Cache Loader] ERROR loading corpus: {e}", flush=True)
            traceback.print_exc()
            EmbeddingCacheLoader._cache = {}

    def get_embedding(self, arch_string):
        """Retrieve embedding for a single architecture string"""
        if arch_string not in EmbeddingCacheLoader._cache:
            raise ValueError(f"Architecture not found in cache: {arch_string}")
        return EmbeddingCacheLoader._cache[arch_string]

    def get_embeddings(self, arch_strings):
        """Retrieve embeddings for multiple architecture strings"""
        embeddings = []
        for arch_str in arch_strings:
            embeddings.append(self.get_embedding(arch_str))
        return np.vstack(embeddings)


# ==========================================
# 2. LLM-ENHANCED PREDICTOR FOR NASBENCH201
# ==========================================
class LLM_NB201_Predictor:
    """
    Wrapper for any NASLib predictor that uses pre-computed LLM embeddings
    from NASBench201 architectures instead of standard encodings.
    
    Based on the working architecture from LLM_NB301_Predictor but adapted
    for NASBench201 with cache-based embedding retrieval.
    """
    
    def __init__(self, base_predictor_cls, corpus_path, 
                 embedding_col='codellama_python_7b_pytorch_code_embedding',
                 use_pca=True, pca_components=128, **kwargs):
        """
        Args:
            base_predictor_cls: The predictor class to wrap (e.g., VarSparseGPPredictor)
            corpus_path: Path to CSV/PKL file with pre-computed embeddings
            embedding_col: Column name containing embeddings
            use_pca (bool): Whether to apply PCA dimensionality reduction
            pca_components (int): Target number of PCA components
            **kwargs: Additional arguments passed to base_predictor_cls
        """
        # print(f"[LLM NB201 Predictor] Initializing with {base_predictor_cls.__name__}...")
        
        # Initialize cache loader
        self.cache_loader = EmbeddingCacheLoader(corpus_path, embedding_col)
        
        # Initialize base predictor with encoding_type=None (we handle encoding)
        self.predictor = base_predictor_cls(encoding_type=None, **kwargs)
        
        # PCA configuration
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None
        
        self._pca_warning_printed = False
        # print(f"[LLM NB201 Predictor] Initialization complete.")
    
    def _get_embeddings_from_cache(self, architectures):
        """
        Convert NASLib architecture objects to embeddings via cache lookup.
        Uses convert_naslib_to_str for NASBench201 architecture strings.
        """
        # Convert architectures to strings
        # arch_strings = [convert_naslib_to_str(arch) for arch in architectures]
        arch_strings = []
        for arch in architectures:
            # Use the fast, graph-free conversion
            if hasattr(arch, 'op_indices') and arch.op_indices is not None:
                s = convert_op_indices_to_str(arch.op_indices)
            else:
                # Fallback (should not be reached with hollow patch)
                s = convert_naslib_to_str(arch)
            arch_strings.append(s)
        
        # Retrieve from cache
        embeddings = self.cache_loader.get_embeddings(arch_strings)
        
        return embeddings
    
    def fit(self, xtrain, ytrain, train_info=None, **kwargs):
        """
        Fit the predictor on training data.
        
        Workflow:
        1. Convert architectures to embeddings via cache lookup
        2. Apply PCA (fit and transform)
        3. Pass transformed embeddings to base predictor
        """
        # Step 1: Get embeddings from cache
        xtrain_emb = self._get_embeddings_from_cache(xtrain)
        
        # Step 2: Apply PCA (fit and transform)
        if self.use_pca:
            n_samples = len(xtrain_emb)
            current_dims = min(self.pca_components, n_samples)
            
            if current_dims < self.pca_components and not self._pca_warning_printed:
                print(f"[LLM NB201 Predictor] WARNING: Reducing PCA to {current_dims} components (requested {self.pca_components}, have {n_samples} samples)")
                self._pca_warning_printed = True
            
            self.pca = PCA(n_components=current_dims)
            xtrain_emb = self.pca.fit_transform(xtrain_emb)
        
        # Step 3: Train base predictor
        return self.predictor.fit(xtrain_emb, ytrain, train_info, **kwargs)
    
    def query(self, xtest, info=None, *args, **kwargs):
        """
        Query the predictor on test data.
        
        Workflow:
        1. Convert architectures to embeddings via cache lookup
        2. Apply PCA transform (using fitted PCA from training)
        3. Query base predictor
        """
        # Step 1: Get embeddings from cache
        xtest_emb = self._get_embeddings_from_cache(xtest)
        
        # Step 2: Apply PCA transform (using fitted transform)
        if self.use_pca and self.pca is not None:
            xtest_emb = self.pca.transform(xtest_emb)
        
        # Step 3: Query base predictor
        return self.predictor.query(xtest_emb, info, *args, **kwargs)
    
    def get_model(self, **kwargs):
        """Delegate to base predictor"""
        return self.predictor.get_model(**kwargs)
    
    def __getattr__(self, name):
        """Forward any other attribute access to the base predictor"""
        return getattr(self.predictor, name)
