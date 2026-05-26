from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from psdnet.dataset import PSDRegressionDataset, prepare_regression_arrays
from psdnet.eeg_data import load_eeg_data_from_pkl
from psdnet.features import (
    DEFAULT_BAND_NAMES,
    clip_transformed_model_outputs,
    compute_real_psd_metrics,
    fit_target_transform,
    inverse_transform_target_array,
    save_json,
    transform_target_array,
)
from psdnet.model import create_model
from splits import load_split, save_split, split_indices_train_val_test

MODEL_TYPE = "psdnet_v2"


class WeightedRegressionLoss(nn.Module):
    def __init__(self, loss_type: str, weights):
        super().__init__()
        self.loss_type = loss_type
        self.register_buffer("w", torch.tensor(list(weights), dtype=torch.float32).view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return ((pred - target) ** 2 * self.w).mean()
        return (F.smooth_l1_loss(pred, target, reduction="none") * self.w).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else config


@torch.no_grad()
def predict(model, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for inputs, batch_targets in dataloader:
        inputs = inputs.to(device)
        batch_predictions = model(inputs).cpu().numpy()
        predictions.append(batch_predictions)
        targets.append(batch_targets.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = inputs.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def evaluate_split(
    model,
    inputs: np.ndarray,
    transformed_targets: np.ndarray,
    indices: np.ndarray,
    target_transform: dict,
    band_names: list[str],
    batch_size: int,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        PSDRegressionDataset(inputs, transformed_targets, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    pred_t, tgt_t = predict(model, loader, device)
    pred_t = clip_transformed_model_outputs(pred_t, target_transform)
    pred = inverse_transform_target_array(pred_t, target_transform)
    tgt = inverse_transform_target_array(tgt_t, target_transform)
    return compute_real_psd_metrics(pred, tgt, band_names=band_names)


def build_model(n_bands: int, cfg: dict, target_space: str) -> nn.Module:
    return create_model(
        model_type=MODEL_TYPE,
        n_bands=n_bands,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        num_segments=cfg["num_segments"],
        nonnegative_output=target_space == "linear",
    )


def create_split_if_needed(config: dict, output_dir: Path) -> dict:
    split_path = output_dir / "split.json"
    if split_path.exists():
        return load_split(split_path)
    eeg_data = load_eeg_data_from_pkl(config["paths"]["dataset_pkl"])
    train_idx, val_idx, test_idx = split_indices_train_val_test(
        eeg_data.get_sample_count(),
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=config["split"]["seed"],
    )
    return save_split(
        split_path,
        dataset_pkl=config["paths"]["dataset_pkl"],
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        seed=config["split"]["seed"],
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        extra={
            "dataset_name": eeg_data.dataset_name,
            "num_classes": int(eeg_data.get_label_count()),
            "sampling_rate": float(eeg_data.sampling_rate),
        },
    )


def train_psdnet(config: dict, split_payload: dict, output_dir: Path, device: torch.device) -> dict:
    run_dir = output_dir / f"psd_{MODEL_TYPE}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    teacher = config["teacher"]
    data_cfg = config["data"]
    train_cfg = config["psd_training"]
    band_names = teacher.get("band_names", DEFAULT_BAND_NAMES)
    bands_hz = teacher.get("bands_hz")

    eeg_data = load_eeg_data_from_pkl(config["paths"]["dataset_pkl"])
    inputs, targets = prepare_regression_arrays(
        eeg_data=eeg_data,
        bands_hz=bands_hz,
        welch_nperseg=int(teacher["welch_nperseg"]),
        log_psd=bool(teacher["log_psd"]),
        psd_eps=float(teacher["psd_eps"]),
        center_input=bool(data_cfg["center_input"]),
        time_points=data_cfg.get("time_points"),
        target_mode=teacher.get("target_mode", "absolute"),
    )
    train_indices = split_payload["train_indices"]
    val_indices = split_payload["val_indices"]
    test_indices = split_payload["test_indices"]

    target_transform = fit_target_transform(
        targets[train_indices],
        target_space=teacher["target_space"],
        psd_scale=1.0,
        psd_eps=float(teacher["psd_eps"]),
    )
    transformed_targets = transform_target_array(targets, target_transform)

    train_loader = DataLoader(
        PSDRegressionDataset(inputs, transformed_targets, train_indices),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        PSDRegressionDataset(inputs, transformed_targets, val_indices),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model_cfg = {
        "hidden_dim": int(train_cfg["hidden_dim"]),
        "dropout": float(train_cfg["dropout"]),
        "num_segments": int(train_cfg["num_segments"]),
    }
    model = build_model(targets.shape[1], model_cfg, teacher["target_space"]).to(device)
    band_loss_weights = [float(x.strip()) for x in str(train_cfg["band_loss_weights"]).split(",") if x.strip()]
    criterion = WeightedRegressionLoss(train_cfg["loss_type"], band_loss_weights).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    best_val_mse = float("inf")
    best_epoch = 0
    best_metrics = None
    history = []

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_pred_t, val_tgt_t = predict(model, val_loader, device)
        val_pred_t = clip_transformed_model_outputs(val_pred_t, target_transform)
        val_pred = inverse_transform_target_array(val_pred_t, target_transform)
        val_tgt = inverse_transform_target_array(val_tgt_t, target_transform)
        val_metrics = compute_real_psd_metrics(val_pred, val_tgt, band_names=band_names)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **val_metrics})
        print(
            f"[PSDNet] epoch {epoch}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.6f} val_mae={val_metrics['overall_mae']:.6f}"
        )
        if val_metrics["overall_mse"] < best_val_mse:
            best_val_mse = val_metrics["overall_mse"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": {
                        "model_type": MODEL_TYPE,
                        "n_bands": targets.shape[1],
                        "hidden_dim": model_cfg["hidden_dim"],
                        "dropout": model_cfg["dropout"],
                        "num_segments": model_cfg["num_segments"],
                        "nonnegative_output": teacher["target_space"] == "linear",
                    },
                    "teacher_config": {
                        "band_names": band_names,
                        "bands_hz": bands_hz,
                        "welch_nperseg": int(teacher["welch_nperseg"]),
                        "log_psd": bool(teacher["log_psd"]),
                        "psd_eps": float(teacher["psd_eps"]),
                        "target_mode": teacher.get("target_mode", "absolute"),
                        "target_space": teacher["target_space"],
                        "target_transform": target_transform,
                    },
                    "data_config": {
                        "center_input": bool(data_cfg["center_input"]),
                        "time_points": data_cfg.get("time_points"),
                    },
                    "split_path": str(output_dir / "split.json"),
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_metrics,
                },
                checkpoint_dir / "best.pt",
            )

    if not (checkpoint_dir / "best.pt").exists():
        raise RuntimeError("No checkpoint saved for PSDNet")
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device)["model_state_dict"])
    metrics_train = evaluate_split(
        model, inputs, transformed_targets, train_indices, target_transform, band_names, int(train_cfg["batch_size"]), device
    )
    metrics_val = evaluate_split(
        model, inputs, transformed_targets, val_indices, target_transform, band_names, int(train_cfg["batch_size"]), device
    )
    metrics_test = evaluate_split(
        model, inputs, transformed_targets, test_indices, target_transform, band_names, int(train_cfg["batch_size"]), device
    )

    summary = {
        "model_type": MODEL_TYPE,
        "dataset_pkl": config["paths"]["dataset_pkl"],
        "num_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "history": history,
    }
    save_json(run_dir / "psd_metrics.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PSDNet for band-power estimation.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["split"]["seed"]))
    device = torch.device(args.device)

    split_payload = create_split_if_needed(config, output_dir)
    summary = train_psdnet(config, split_payload, output_dir, device)
    save_json(output_dir / "psd_estimation_summary.json", {"model": summary})


if __name__ == "__main__":
    main()
