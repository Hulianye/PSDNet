from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psdnet.dataset import PSDRegressionDataset, prepare_channel_regression_arrays, prepare_regression_arrays
from psdnet.eeg_data import load_eeg_data_from_pkl, resample_eeg_data
from psdnet.features import (
    DEFAULT_BAND_NAMES,
    apply_linear_calibration_per_band,
    clip_transformed_model_outputs,
    compute_calibration_metrics,
    compute_real_psd_metrics,
    decompose_bandpower_targets,
    fit_linear_calibration_per_band,
    inverse_transform_target_array,
    reconstruct_bandpower_from_decomposition,
    save_json,
    transform_target_array,
)
from psdnet.model import create_model


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
    parser = argparse.ArgumentParser(description="Evaluate PSDNet checkpoint on EEGData pickle.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--test_pkl",
        type=str,
        default=get_config_value(config, "paths", "test_pkl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path(get_config_value(config, "paths", "output_dir")) / "checkpoints" / "best.pt"),
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
        "--num_workers",
        type=int,
        default=get_config_value(config, "training", "num_workers", default=0),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--calibration_fit_pkl",
        type=str,
        default="",
        help="If set, fit per-band linear calibration on this pickle (train) and report test metrics after applying.",
    )
    parser.add_argument(
        "--include_calibration",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="If true, include same-split calibration diagnostics in eval_metrics.json (not used as primary metrics).",
    )
    return parser


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


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(pre_args.config)
    parser = create_parser(config)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    teacher_config = checkpoint.get("teacher_config", {})
    data_config = checkpoint.get("data_config", {})
    model_args = checkpoint["model_args"]

    band_names = teacher_config.get("band_names", get_config_value(config, "teacher", "band_names", default=DEFAULT_BAND_NAMES))
    bands_hz = teacher_config.get("bands_hz", get_config_value(config, "teacher", "bands_hz"))

    eeg_data = load_eeg_data_from_pkl(args.test_pkl)
    target_sfreq = teacher_config.get("target_sfreq")
    resample_meta = None
    if target_sfreq is not None:
        eeg_data, resample_meta = resample_eeg_data(eeg_data, float(target_sfreq))
    channel_mode = teacher_config.get("channel_mode") == "single_channel"
    input_time_points = int(teacher_config.get("input_time_points", 1024))

    if channel_mode:
        inputs, targets, _meta = prepare_channel_regression_arrays(
            eeg_data=eeg_data,
            bands_hz=bands_hz,
            welch_nperseg=int(teacher_config.get("welch_nperseg", get_config_value(config, "teacher", "welch_nperseg", default=256))),
            log_psd=bool(teacher_config.get("log_psd", get_config_value(config, "teacher", "log_psd", default=False))),
            psd_eps=float(teacher_config.get("psd_eps", get_config_value(config, "teacher", "psd_eps", default=1e-12))),
            center_input=bool(data_config.get("center_input", get_config_value(config, "data", "center_input", default=True))),
            input_time_points=input_time_points,
            target_mode=teacher_config.get("target_mode", get_config_value(config, "teacher", "target_mode", default="absolute")),
        )
    else:
        inputs, targets = prepare_regression_arrays(
            eeg_data=eeg_data,
            bands_hz=bands_hz,
            welch_nperseg=int(teacher_config.get("welch_nperseg", get_config_value(config, "teacher", "welch_nperseg", default=256))),
            log_psd=bool(teacher_config.get("log_psd", get_config_value(config, "teacher", "log_psd", default=False))),
            psd_eps=float(teacher_config.get("psd_eps", get_config_value(config, "teacher", "psd_eps", default=1e-12))),
            center_input=bool(data_config.get("center_input", get_config_value(config, "data", "center_input", default=True))),
            time_points=data_config.get("time_points", get_config_value(config, "data", "time_points")),
            target_mode=teacher_config.get("target_mode", get_config_value(config, "teacher", "target_mode", default="absolute")),
        )
    target_decomposition = teacher_config.get("target_decomposition", "none")
    psd_eps = float(teacher_config.get("psd_eps", 1e-12))
    target_transform = teacher_config.get(
        "target_transform",
        {
            "target_space": teacher_config.get("target_space", "linear"),
            "psd_scale": teacher_config.get("psd_scale", 1.0),
            "psd_eps": psd_eps,
        },
    )
    linear_eval_targets = targets
    if target_decomposition == "relative_total":
        model_targets = decompose_bandpower_targets(linear_eval_targets, psd_eps=psd_eps)
    else:
        model_targets = transform_target_array(targets, target_transform)

    dataset = PSDRegressionDataset(inputs, model_targets)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = create_model(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions, _model_targets = predict(model, dataloader, device)
    if target_decomposition == "relative_total":
        predictions = reconstruct_bandpower_from_decomposition(
            predictions,
            psd_eps=psd_eps,
            transform_config=target_transform,
        )
        eval_targets = linear_eval_targets
    else:
        predictions = clip_transformed_model_outputs(predictions, target_transform)
        predictions = inverse_transform_target_array(predictions, target_transform)
        eval_targets = inverse_transform_target_array(_model_targets, target_transform)
    metrics = compute_real_psd_metrics(predictions, eval_targets, band_names=band_names)

    eval_payload: dict = {
        "dataset_name": eeg_data.dataset_name,
        "sampling_rate": eeg_data.sampling_rate,
        "test_pkl": args.test_pkl,
        "band_names": band_names,
        "bands_hz": bands_hz,
        "checkpoint": args.checkpoint,
        "target_mode": teacher_config.get("target_mode", "absolute"),
        "target_decomposition": target_decomposition,
        "target_space": target_transform.get("target_space", "linear"),
        "psd_scale": float(target_transform.get("psd_scale", 1.0)),
        "target_transform": target_transform,
        "model_type": model_args.get("model_type", "psdnet"),
        "channel_mode": "single_channel" if channel_mode else "trial_mean",
        "input_time_points": input_time_points if channel_mode else None,
        "primary_metrics_note": "Report overall_mae/rmse/mape/smape and per_band_* below as model output without target-domain calibration.",
        **metrics,
    }
    if target_sfreq is not None:
        eval_payload["target_sfreq"] = float(target_sfreq)
    if resample_meta is not None:
        eval_payload["resample_meta"] = resample_meta
    if args.include_calibration:
        calibration = compute_calibration_metrics(predictions, eval_targets, band_names=band_names)
        eval_payload["calibration"] = calibration
        eval_payload["calibration_note"] = (
            "Diagnostic only: calibration fits on the same split as metrics; not used as primary results."
        )

    if args.calibration_fit_pkl:
        eeg_fit = load_eeg_data_from_pkl(args.calibration_fit_pkl)
        if channel_mode:
            inputs_fit, targets_fit, _mf = prepare_channel_regression_arrays(
                eeg_data=eeg_fit,
                bands_hz=bands_hz,
                welch_nperseg=int(teacher_config.get("welch_nperseg", get_config_value(config, "teacher", "welch_nperseg", default=256))),
                log_psd=bool(teacher_config.get("log_psd", get_config_value(config, "teacher", "log_psd", default=False))),
                psd_eps=float(teacher_config.get("psd_eps", get_config_value(config, "teacher", "psd_eps", default=1e-12))),
                center_input=bool(data_config.get("center_input", get_config_value(config, "data", "center_input", default=True))),
                input_time_points=input_time_points,
                target_mode=teacher_config.get("target_mode", get_config_value(config, "teacher", "target_mode", default="absolute")),
            )
        else:
            inputs_fit, targets_fit = prepare_regression_arrays(
                eeg_data=eeg_fit,
                bands_hz=bands_hz,
                welch_nperseg=int(teacher_config.get("welch_nperseg", get_config_value(config, "teacher", "welch_nperseg", default=256))),
                log_psd=bool(teacher_config.get("log_psd", get_config_value(config, "teacher", "log_psd", default=False))),
                psd_eps=float(teacher_config.get("psd_eps", get_config_value(config, "teacher", "psd_eps", default=1e-12))),
                center_input=bool(data_config.get("center_input", get_config_value(config, "data", "center_input", default=True))),
                time_points=data_config.get("time_points", get_config_value(config, "data", "time_points")),
                target_mode=teacher_config.get("target_mode", get_config_value(config, "teacher", "target_mode", default="absolute")),
            )
        transformed_fit = transform_target_array(targets_fit, target_transform)
        fit_dataset = PSDRegressionDataset(inputs_fit, transformed_fit)
        fit_loader = DataLoader(
            fit_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        pred_fit_t, tgt_fit_t = predict(model, fit_loader, device)
        pred_fit_t = clip_transformed_model_outputs(pred_fit_t, target_transform)
        pred_fit = inverse_transform_target_array(pred_fit_t, target_transform)
        tgt_fit = inverse_transform_target_array(tgt_fit_t, target_transform)
        train_calib_params = fit_linear_calibration_per_band(pred_fit, tgt_fit, band_names=band_names)
        pred_test_calibrated = apply_linear_calibration_per_band(predictions, train_calib_params, band_names=band_names)
        metrics_train_calib = compute_real_psd_metrics(pred_test_calibrated, eval_targets, band_names=band_names)
        eval_payload["calibration_fit_pkl"] = args.calibration_fit_pkl
        eval_payload["train_fit_linear_calibration_params"] = train_calib_params
        eval_payload["metrics_after_train_fit_linear_calibration"] = metrics_train_calib

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "eval_metrics.json", eval_payload)
    np.savez(
        output_dir / "eval_predictions.npz",
        predictions=predictions,
        targets=eval_targets,
    )

    print(f"Dataset: {eeg_data.dataset_name}")
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"overall_mse={metrics['overall_mse']:.6f}")
    print(f"overall_mae={metrics['overall_mae']:.6f}")
    print(f"overall_rmse={metrics['overall_rmse']:.6f}")
    print(f"overall_mape={metrics['overall_mape']:.6f}")
    print(f"pearson_mean={metrics['pearson_mean']:.6f}")


if __name__ == "__main__":
    main()
