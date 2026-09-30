#!/usr/bin/env python3
"""Stream the COLE CSV into the compact npz used by EmbeddingCacheLoader."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

CORPUS = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_corrected.csv"
COL = "codellama_python_7b_pytorch_code_exclude_helper_embedding"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "embedding_cache"


def parse_embedding(val: str) -> np.ndarray:
    val_str = str(val).strip()
    try:
        return np.asarray(json.loads(val_str), dtype=np.float32)
    except Exception:
        cleaned = val_str.replace("[", "").replace("]", "").replace("\n", " ").strip()
        return np.fromstring(cleaned, sep=",", dtype=np.float32)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{CORPUS}|{COL}".encode()).hexdigest()[:16]
    out = CACHE_DIR / f"nb201_{key}.npz"
    print(f"writing {out}", flush=True)
    t0 = time.time()
    keys = []
    embs = []
    with open(CORPUS, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        arch_idx = header.index("arch_string")
        emb_idx = header.index(COL)
        for i, row in enumerate(reader, start=1):
            keys.append(row[arch_idx])
            embs.append(parse_embedding(row[emb_idx]))
            if i % 500 == 0:
                print(f"{i} rows  elapsed={time.time() - t0:.1f}s", flush=True)
    embs_arr = np.stack(embs)
    keys_arr = np.array(keys, dtype=object)
    tmp = out.with_name(out.stem + "_tmp.npz")
    np.savez(tmp, arch_strings=keys_arr, embeddings=embs_arr)
    os.replace(tmp, out)
    print(
        f"done n={len(keys)} shape={embs_arr.shape} bytes={out.stat().st_size} "
        f"elapsed={time.time() - t0:.1f}s path={out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
