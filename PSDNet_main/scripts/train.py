from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


class WeightedRegressionLoss(nn.Module):
    def __init__(self, loss_type: str, weights: Sequence[float]):
        super().__init__()
        if loss_type not in {"mse", "smoothl1"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.loss_type = loss_type
        self.register_buffer("w", torch.tensor(list(weights), dtype=torch.float32).view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return ((pred - target) ** 2 * self.w).mean()
        return (F.smooth_l1_loss(pred, target, reduction="none") * self.w).mean()


PROJECT_DIR = REPO_ROOT
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config.example.yaml"


def get_config_value(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else config


def create_parser(config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PSDNet on EEGData pickle.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--train_pkl",
        type=str,
        default=get_config_value(config, "paths", "train_pkl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_config_value(config, "paths", "output_dir"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=get_config_value(config, "training", "batch_size", default=64),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=get_config_value(config, "training", "epochs", default=20),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=get_config_value(config, "training", "lr", default=1e-3),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=get_config_value(config, "training", "weight_decay", default=1e-4),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=get_config_value(config, "training", "num_workers", default=0),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=get_config_value(config, "training", "seed", default=42),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=get_config_value(config, "data", "val_ratio", default=0.2),
    )
    parser.add_argument(
        "--center_input",
        type=lambda x: str(x).lower() == "true",
        default=get_config_value(config, "data", "center_input", default=True),
    )
    parser.add_argument(
        "--time_points",
        type=int,
        default=get_config_value(config, "data", "time_points"),
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=get_config_value(config, "model", "hidden_dim", default=64),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=get_config_value(config, "model", "dropout", default=0.1),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=get_config_value(config, "model", "model_type", default="psdnet"),
        choices=["psdnet", "psdnet_v2"],
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=get_config_value(config, "model", "num_segments", default=8),
    )
    parser.add_argument(
        "--welch_nperseg",
        type=int,
        default=get_config_value(config, "teacher", "welch_nperseg", default=256),
    )
    parser.add_argument(
        "--log_psd",
        type=lambda x: str(x).lower() == "true",
        default=get_config_value(config, "teacher", "log_psd", default=False),
    )
    parser.add_argument(
        "--psd_eps",
        type=float,
        default=get_config_value(config, "teacher", "psd_eps", default=1e-12),
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default=get_config_value(config, "teacher", "target_mode", default="absolute"),
        choices=["absolute", "relative"],
    )
    parser.add_argument(
        "--target_space",
        type=str,
        default=get_config_value(config, "teacher", "target_space", default="linear"),
        choices=["linear", "log_train_linear_eval"],
    )
    parser.add_argument(
        "--psd_scale",
        type=float,
        default=get_config_value(config, "teacher", "psd_scale", default=1.0),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "smoothl1"],
    )
    parser.add_argument(
        "--band_loss_weights",
        type=str,
        default="1,1,1,1,1",
        help="Comma-separated per-band weights (length must match number of bands).",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_point = int(num_samples * (1.0 - val_ratio))
    if split_point <= 0 or split_point >= num_samples:
        raise ValueError(f"Invalid val_ratio={val_ratio} for num_samples={num_samples}")
    return indices[:split_point], indices[split_point:]


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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


@torch.no_grad()
def predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    targets = []
    for inputs, batch_targets in dataloader:
        inputs = inputs.to(device)
        batch_predictions = model(inputs).cpu().numpy()
        predictions.append(batch_predictions)
        targets.append(batch_targets.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    parser = create_parser(config)
    args = parser.parse_args()

    set_seed(args.seed)

    train_pkl = Path(args.train_pkl)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    band_names = get_config_value(config, "teacher", "band_names", default=DEFAULT_BAND_NAMES)
    bands_hz = get_config_value(config, "teacher", "bands_hz")

    eeg_data = load_eeg_data_from_pkl(str(train_pkl))
    inputs, targets = prepare_regression_arrays(
        eeg_data=eeg_data,
        bands_hz=bands_hz,
        welch_nperseg=args.welch_nperseg,
        log_psd=args.log_psd,
        psd_eps=args.psd_eps,
        center_input=args.center_input,
        time_points=args.time_points,
        target_mode=args.target_mode,
    )

    train_indices, val_indices = split_indices(inputs.shape[0], args.val_ratio, args.seed)
    target_transform = fit_target_transform(
        targets[train_indices],
        target_space=args.target_space,
        psd_scale=args.psd_scale,
        psd_eps=args.psd_eps,
    )
    transformed_targets = transform_target_array(targets, target_transform)
    train_dataset = PSDRegressionDataset(inputs, transformed_targets, train_indices)
    val_dataset = PSDRegressionDataset(inputs, transformed_targets, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = create_model(
        model_type=args.model_type,
        n_bands=targets.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_segments=args.num_segments,
        nonnegative_output=args.target_space == "linear",
    ).to(device)
    band_loss_weights = [float(x.strip()) for x in args.band_loss_weights.split(",") if x.strip()]
    if len(band_loss_weights) != targets.shape[1]:
        raise ValueError(
            f"band_loss_weights length {len(band_loss_weights)} != n_bands {targets.shape[1]}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = WeightedRegressionLoss(args.loss_type, band_loss_weights).to(device)

    history = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_metrics = None

    print(f"Dataset: {eeg_data.dataset_name}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Sampling rate: {eeg_data.sampling_rate}")
    print(f"Bands: {band_names}")
    print(f"Target mode: {args.target_mode}")
    print(f"Target space: {args.target_space}")
    print(f"Model type: {args.model_type}")
    print(f"Loss type: {args.loss_type} band_weights={band_loss_weights}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_predictions, val_targets = predict(model, val_loader, device)
        val_predictions = clip_transformed_model_outputs(val_predictions, target_transform)
        val_predictions = inverse_transform_target_array(val_predictions, target_transform)
        val_targets = inverse_transform_target_array(val_targets, target_transform)
        val_metrics = compute_real_psd_metrics(val_predictions, val_targets, band_names=band_names)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            **val_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} "
            f"val_mse={val_metrics['overall_mse']:.6f} "
            f"val_mae={val_metrics['overall_mae']:.6f}"
        )
        if val_metrics["overall_mse"] < best_val_mse:
            best_val_mse = val_metrics["overall_mse"]
            best_epoch = epoch
            best_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_args": {
                        "model_type": args.model_type,
                        "n_bands": targets.shape[1],
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "num_segments": args.num_segments,
                        "nonnegative_output": args.target_space == "linear",
                    },
                    "teacher_config": {
                        "band_names": band_names,
                        "bands_hz": bands_hz,
                        "welch_nperseg": args.welch_nperseg,
                        "log_psd": args.log_psd,
                        "psd_eps": args.psd_eps,
                        "target_mode": args.target_mode,
                        "target_space": args.target_space,
                        "psd_scale": args.psd_scale,
                        "target_transform": target_transform,
                    },
                    "data_config": {
                        "center_input": args.center_input,
                        "time_points": args.time_points,
                    },
                    "loss_type": args.loss_type,
                    "band_loss_weights": band_loss_weights,
                    "train_pkl": str(train_pkl),
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_metrics,
                },
                checkpoint_dir / "best.pt",
            )

    save_json(
        output_dir / "train_metrics.json",
        {
            "dataset_name": eeg_data.dataset_name,
            "sampling_rate": eeg_data.sampling_rate,
            "train_pkl": str(train_pkl),
            "num_samples": int(inputs.shape[0]),
            "num_train_samples": int(train_indices.shape[0]),
            "num_val_samples": int(val_indices.shape[0]),
            "band_names": band_names,
            "bands_hz": bands_hz,
            "target_mode": args.target_mode,
            "target_space": args.target_space,
            "psd_scale": args.psd_scale,
            "target_transform": target_transform,
            "model_type": args.model_type,
            "loss_type": args.loss_type,
            "band_loss_weights": band_loss_weights,
            "best_epoch": best_epoch,
            "best_val_metrics": best_metrics,
            "history": history,
        },
    )


if __name__ == "__main__":
    main()
