from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from psdnet.features import save_json


def split_indices_train_val_test(
    num_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    if n_train <= 0 or n_val <= 0 or n_train + n_val >= num_samples:
        raise ValueError(
            f"Invalid split for num_samples={num_samples}: train={n_train}, val={n_val}"
        )
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    if test_indices.shape[0] <= 0:
        raise ValueError("Test split is empty; reduce train/val ratios.")
    return train_indices, val_indices, test_indices


def save_split(
    output_path: str | Path,
    *,
    dataset_pkl: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    extra: dict | None = None,
) -> dict:
    payload = {
        "dataset_pkl": dataset_pkl,
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "num_samples": int(train_indices.shape[0] + val_indices.shape[0] + test_indices.shape[0]),
        "num_train": int(train_indices.shape[0]),
        "num_val": int(val_indices.shape[0]),
        "num_test": int(test_indices.shape[0]),
        "train_indices": train_indices.astype(int).tolist(),
        "val_indices": val_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
    }
    if extra:
        payload.update(extra)
    save_json(output_path, payload)
    return payload


def load_split(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["train_indices"] = np.asarray(payload["train_indices"], dtype=np.int64)
    payload["val_indices"] = np.asarray(payload["val_indices"], dtype=np.int64)
    payload["test_indices"] = np.asarray(payload["test_indices"], dtype=np.int64)
    return payload
