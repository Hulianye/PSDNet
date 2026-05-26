from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_speed import timed_gpu_welch, timed_psdnet, warmup_gpu_welch, warmup_psdnet
from psdnet.dataset import preprocess_inputs
from psdnet.eeg_data import load_eeg_data_from_pkl
from psdnet.features import build_psd_targets, compute_real_psd_metrics, inverse_transform_target_array, save_json
from psdnet.model import create_model
from splits import load_split

MODEL_TYPE = "psdnet_v2"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PSDNet inference speed vs CPU Welch.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["paths"]["output_dir"])
    split_payload = load_split(output_dir / "split.json")
    checkpoint_path = output_dir / f"psd_{MODEL_TYPE}" / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher_config = checkpoint["teacher_config"]
    data_config = checkpoint["data_config"]
    model_args = checkpoint["model_args"]

    eeg_data = load_eeg_data_from_pkl(config["paths"]["dataset_pkl"])
    inputs = preprocess_inputs(
        eeg_data.eeg_data,
        center_input=bool(data_config.get("center_input", True)),
        time_points=data_config.get("time_points"),
    ).astype(np.float32)
    test_indices = split_payload["test_indices"]
    inputs = inputs[test_indices]

    bands_hz = teacher_config.get("bands_hz")
    welch_nperseg = int(teacher_config.get("welch_nperseg", 256))
    log_psd = bool(teacher_config.get("log_psd", False))
    psd_eps = float(teacher_config.get("psd_eps", 1e-12))
    band_names = teacher_config.get("band_names")
    target_transform = teacher_config.get("target_transform", {})
    device = torch.device(args.device)

    speed_cfg = config["speed_benchmark"]
    matrix_results = []
    for num_samples in speed_cfg["num_samples_list"]:
        subset_n = min(int(num_samples), inputs.shape[0])
        subset = inputs[:subset_n]
        for batch_size in speed_cfg["batch_sizes"]:
            cpu_start = time.perf_counter()
            cpu_targets = build_psd_targets(
                eeg_nct=subset,
                sampling_rate=eeg_data.sampling_rate,
                bands_hz=bands_hz,
                welch_nperseg=welch_nperseg,
                log_psd=log_psd,
                psd_eps=psd_eps,
            )
            cpu_elapsed = time.perf_counter() - cpu_start

            warmup_gpu_welch(
                subset,
                eeg_data.sampling_rate,
                bands_hz,
                welch_nperseg,
                log_psd,
                psd_eps,
                device,
                batch_size,
                int(speed_cfg["warmup_batches"]),
            )
            gpu_targets, gpu_elapsed = timed_gpu_welch(
                subset,
                eeg_data.sampling_rate,
                bands_hz,
                welch_nperseg,
                log_psd,
                psd_eps,
                device,
                batch_size,
            )

            model = create_model(**model_args).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            warmup_psdnet(
                model,
                subset,
                batch_size,
                device,
                False,
                int(speed_cfg["warmup_batches"]),
            )
            psd_pred_t, psd_elapsed = timed_psdnet(model, subset, batch_size, device, False)
            psd_pred = inverse_transform_target_array(psd_pred_t, target_transform)
            psd_metrics = compute_real_psd_metrics(psd_pred, cpu_targets, band_names=band_names)

            matrix_results.append(
                {
                    "num_samples": subset_n,
                    "batch_size": int(batch_size),
                    "cpu_welch_total_sec": float(cpu_elapsed),
                    "cpu_welch_sec_per_sample": float(cpu_elapsed / subset_n),
                    "gpu_welch_total_sec": float(gpu_elapsed),
                    "gpu_welch_sec_per_sample": float(gpu_elapsed / subset_n),
                    "psdnet_total_sec": float(psd_elapsed),
                    "psdnet_sec_per_sample": float(psd_elapsed / subset_n),
                    "psdnet_speedup_vs_cpu": float(cpu_elapsed / max(psd_elapsed, 1e-12)),
                    "gpu_welch_speedup_vs_cpu": float(cpu_elapsed / max(gpu_elapsed, 1e-12)),
                    "psdnet_speedup_vs_gpu_welch": float(gpu_elapsed / max(psd_elapsed, 1e-12)),
                    "psdnet_vs_cpu_overall_mae": psd_metrics["overall_mae"],
                    "psdnet_vs_cpu_pearson_mean": psd_metrics["pearson_mean"],
                }
            )
            print(
                f"num_samples={subset_n} batch={batch_size} "
                f"psdnet_speedup_vs_cpu={matrix_results[-1]['psdnet_speedup_vs_cpu']:.2f}x"
            )

    save_json(output_dir / "speed_benchmark.json", {"results": matrix_results})


if __name__ == "__main__":
    main()
