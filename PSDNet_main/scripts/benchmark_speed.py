from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psdnet.dataset import preprocess_inputs
from psdnet.eeg_data import load_eeg_data_from_pkl
from psdnet.features import (
    build_psd_targets,
    build_psd_targets_torch,
    compute_real_psd_metrics,
    inverse_transform_target_array,
)
from psdnet.model import create_model


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU Welch vs GPU Welch vs PSDNet inference speed."
    )
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="If set, write benchmark metrics to this JSON file.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use autocast FP16 on CUDA for model forward only.",
    )
    parser.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        help="Wrap model with torch.compile (PyTorch 2+).",
    )
    parser.add_argument(
        "--warmup_batches",
        type=int,
        default=3,
        help="CUDA warmup batches for GPU Welch and PSDNet timing.",
    )
    parser.add_argument(
        "--skip_cpu_welch",
        action="store_true",
        help="Skip CPU SciPy Welch reference (faster smoke test).",
    )
    return parser


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_model(
    model,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    outputs = []
    for start in range(0, inputs.shape[0], batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = model(batch)
        else:
            out = model(batch)
        outputs.append(out.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def warmup_gpu_welch(
    inputs: np.ndarray,
    sampling_rate: float,
    bands_hz: list[list[float]] | None,
    welch_nperseg: int,
    log_psd: bool,
    psd_eps: float,
    device: torch.device,
    batch_size: int,
    num_warmup_batches: int,
) -> None:
    warmup_n = min(inputs.shape[0], batch_size * max(1, num_warmup_batches))
    if warmup_n <= 0:
        return
    _ = build_psd_targets_torch(
        eeg_nct=inputs[:warmup_n],
        sampling_rate=sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
        device=device,
        batch_size=batch_size,
    )
    cuda_sync(device)


def warmup_psdnet(
    model,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    num_warmup_batches: int,
) -> None:
    warmup_n = min(inputs.shape[0], batch_size * max(1, num_warmup_batches))
    if warmup_n <= 0:
        return
    _ = run_model(model, inputs[:warmup_n], batch_size, device, use_amp)
    cuda_sync(device)


def timed_gpu_welch(
    inputs: np.ndarray,
    sampling_rate: float,
    bands_hz: list[list[float]] | None,
    welch_nperseg: int,
    log_psd: bool,
    psd_eps: float,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    cuda_sync(device)
    start = time.perf_counter()
    targets = build_psd_targets_torch(
        eeg_nct=inputs,
        sampling_rate=sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
        device=device,
        batch_size=batch_size,
    )
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    return targets, elapsed


def timed_psdnet(
    model,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, float]:
    cuda_sync(device)
    start = time.perf_counter()
    predictions = run_model(model, inputs, batch_size, device, use_amp)
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    return predictions, elapsed


def _metrics_subset(metrics: dict, prefix: str) -> dict:
    return {
        f"{prefix}_overall_mae": metrics["overall_mae"],
        f"{prefix}_overall_rmse": metrics["overall_rmse"],
        f"{prefix}_overall_mape": metrics["overall_mape"],
        f"{prefix}_pearson_mean": metrics["pearson_mean"],
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    eeg_data = load_eeg_data_from_pkl(args.pkl_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    teacher_config = checkpoint.get("teacher_config", {})
    data_config = checkpoint.get("data_config", {})
    model_args = checkpoint["model_args"]

    inputs = preprocess_inputs(
        eeg_data.eeg_data,
        center_input=bool(data_config.get("center_input", True)),
        time_points=data_config.get("time_points"),
    ).astype(np.float32)
    num_samples = min(int(args.num_samples), inputs.shape[0])
    inputs = inputs[:num_samples]

    bands_hz = teacher_config.get("bands_hz")
    welch_nperseg = int(teacher_config.get("welch_nperseg", 256))
    log_psd = bool(teacher_config.get("log_psd", False))
    psd_eps = float(teacher_config.get("psd_eps", 1e-12))
    band_names = teacher_config.get("band_names")

    cpu_welch_targets = None
    cpu_welch_elapsed = None
    if not args.skip_cpu_welch:
        cpu_start = time.perf_counter()
        cpu_welch_targets = build_psd_targets(
            eeg_nct=inputs,
            sampling_rate=eeg_data.sampling_rate,
            bands_hz=bands_hz,
            welch_nperseg=welch_nperseg,
            log_psd=log_psd,
            psd_eps=psd_eps,
        )
        cpu_welch_elapsed = time.perf_counter() - cpu_start

    device = torch.device(args.device)
    warmup_batches = int(args.warmup_batches)
    warmup_gpu_welch(
        inputs=inputs,
        sampling_rate=eeg_data.sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
        device=device,
        batch_size=args.batch_size,
        num_warmup_batches=warmup_batches,
    )
    gpu_welch_targets, gpu_welch_elapsed = timed_gpu_welch(
        inputs=inputs,
        sampling_rate=eeg_data.sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
        device=device,
        batch_size=args.batch_size,
    )

    model = create_model(**model_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if getattr(args, "compile_model", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    target_transform = teacher_config.get(
        "target_transform",
        {
            "target_space": teacher_config.get("target_space", "linear"),
            "psd_scale": teacher_config.get("psd_scale", 1.0),
            "psd_eps": teacher_config.get("psd_eps", 1e-12),
        },
    )

    warmup_psdnet(
        model=model,
        inputs=inputs,
        batch_size=args.batch_size,
        device=device,
        use_amp=bool(getattr(args, "amp", False)),
        num_warmup_batches=warmup_batches,
    )
    transformed_predictions, psdnet_elapsed = timed_psdnet(
        model=model,
        inputs=inputs,
        batch_size=args.batch_size,
        device=device,
        use_amp=bool(getattr(args, "amp", False)),
    )
    psdnet_targets = inverse_transform_target_array(transformed_predictions, target_transform)

    result: dict = {
        "num_samples": int(num_samples),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "amp": bool(getattr(args, "amp", False)),
        "compile_model": bool(getattr(args, "compile_model", False)),
        "warmup_batches": warmup_batches,
        "gpu_welch_total_sec": float(gpu_welch_elapsed),
        "gpu_welch_sec_per_sample": float(gpu_welch_elapsed / num_samples),
        "psdnet_total_sec": float(psdnet_elapsed),
        "psdnet_sec_per_sample": float(psdnet_elapsed / num_samples),
    }

    if cpu_welch_elapsed is not None:
        result["cpu_welch_total_sec"] = float(cpu_welch_elapsed)
        result["cpu_welch_sec_per_sample"] = float(cpu_welch_elapsed / num_samples)
        result["gpu_welch_speedup_vs_cpu"] = float(cpu_welch_elapsed / max(gpu_welch_elapsed, 1e-12))
        result["psdnet_speedup_vs_cpu"] = float(cpu_welch_elapsed / max(psdnet_elapsed, 1e-12))
        gpu_welch_vs_cpu = compute_real_psd_metrics(gpu_welch_targets, cpu_welch_targets, band_names=band_names)
        psdnet_vs_cpu = compute_real_psd_metrics(psdnet_targets, cpu_welch_targets, band_names=band_names)
        result.update(_metrics_subset(gpu_welch_vs_cpu, "gpu_welch_vs_cpu"))
        result.update(_metrics_subset(psdnet_vs_cpu, "psdnet_vs_cpu"))

    result["psdnet_speedup_vs_gpu_welch"] = float(gpu_welch_elapsed / max(psdnet_elapsed, 1e-12))
    if cpu_welch_targets is not None:
        psdnet_vs_gpu_welch = compute_real_psd_metrics(psdnet_targets, gpu_welch_targets, band_names=band_names)
        result.update(_metrics_subset(psdnet_vs_gpu_welch, "psdnet_vs_gpu_welch"))

    # Legacy keys for backward compatibility
    if cpu_welch_elapsed is not None:
        result["welch_total_sec"] = result["cpu_welch_total_sec"]
        result["welch_sec_per_sample"] = result["cpu_welch_sec_per_sample"]
        result["model_total_sec"] = result["psdnet_total_sec"]
        result["model_sec_per_sample"] = result["psdnet_sec_per_sample"]
        result["speedup"] = result["psdnet_speedup_vs_cpu"]
        result["overall_mae"] = result["psdnet_vs_cpu_overall_mae"]
        result["overall_rmse"] = result["psdnet_vs_cpu_overall_rmse"]
        result["overall_mape"] = result["psdnet_vs_cpu_overall_mape"]
        result["pearson_mean"] = result["psdnet_vs_cpu_pearson_mean"]

    return result


def main() -> None:
    args = create_parser().parse_args()
    result = run_benchmark(args)

    num_samples = result["num_samples"]
    print(f"num_samples={num_samples}")
    print(f"batch_size={result['batch_size']}")
    print(f"device={result['device']}")
    if "cpu_welch_total_sec" in result:
        print(f"cpu_welch_total_sec={result['cpu_welch_total_sec']:.6f}")
        print(f"cpu_welch_sec_per_sample={result['cpu_welch_sec_per_sample']:.8f}")
    print(f"gpu_welch_total_sec={result['gpu_welch_total_sec']:.6f}")
    print(f"gpu_welch_sec_per_sample={result['gpu_welch_sec_per_sample']:.8f}")
    print(f"psdnet_total_sec={result['psdnet_total_sec']:.6f}")
    print(f"psdnet_sec_per_sample={result['psdnet_sec_per_sample']:.8f}")
    if "gpu_welch_speedup_vs_cpu" in result:
        print(f"gpu_welch_speedup_vs_cpu={result['gpu_welch_speedup_vs_cpu']:.4f}")
        print(f"psdnet_speedup_vs_cpu={result['psdnet_speedup_vs_cpu']:.4f}")
    print(f"psdnet_speedup_vs_gpu_welch={result['psdnet_speedup_vs_gpu_welch']:.4f}")
    if "gpu_welch_vs_cpu_overall_mape" in result:
        print(f"gpu_welch_vs_cpu_mape={result['gpu_welch_vs_cpu_overall_mape']:.6f}")
        print(f"psdnet_vs_cpu_mape={result['psdnet_vs_cpu_overall_mape']:.6f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
