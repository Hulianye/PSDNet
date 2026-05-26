from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import signal as scipy_signal


DEFAULT_BANDS_HZ = [
    [1.0, 4.0],
    [4.0, 8.0],
    [8.0, 13.0],
    [13.0, 30.0],
    [30.0, 45.0],
]

DEFAULT_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def compute_channel_band_psd(
    eeg_ct: np.ndarray,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
) -> np.ndarray:
    bands_hz = DEFAULT_BANDS_HZ if bands_hz is None else bands_hz
    eeg_ct = np.asarray(eeg_ct, dtype=np.float64)
    num_channels = eeg_ct.shape[0]
    outputs = np.zeros((num_channels, len(bands_hz)), dtype=np.float32)

    for ch_idx in range(num_channels):
        channel = eeg_ct[ch_idx]
        nperseg = min(int(welch_nperseg), channel.shape[-1])
        if nperseg < 4:
            continue
        freqs, psd = scipy_signal.welch(
            channel,
            fs=float(sampling_rate),
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=max(256, nperseg),
        )
        for band_idx, (low, high) in enumerate(bands_hz):
            mask = (freqs >= float(low)) & (freqs < float(high))
            if np.any(mask):
                outputs[ch_idx, band_idx] = float(np.trapz(psd[mask], freqs[mask]))

    if log_psd:
        outputs = np.log(outputs + float(psd_eps))
    return outputs


def compute_global_band_psd(
    eeg_ct: np.ndarray,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
) -> np.ndarray:
    channel_psd = compute_channel_band_psd(
        eeg_ct=eeg_ct,
        sampling_rate=sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
    )
    return channel_psd.mean(axis=0).astype(np.float32)


def build_psd_targets(
    eeg_nct: np.ndarray,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
) -> np.ndarray:
    eeg_nct = np.asarray(eeg_nct, dtype=np.float32)
    targets = np.zeros((eeg_nct.shape[0], len(DEFAULT_BANDS_HZ if bands_hz is None else bands_hz)), dtype=np.float32)
    for sample_idx in range(eeg_nct.shape[0]):
        targets[sample_idx] = compute_global_band_psd(
            eeg_ct=eeg_nct[sample_idx],
            sampling_rate=sampling_rate,
            bands_hz=bands_hz,
            welch_nperseg=welch_nperseg,
            log_psd=log_psd,
            psd_eps=psd_eps,
        )
    return targets


def _resolve_welch_segment_params(
    time_len: int,
    welch_nperseg: int,
    noverlap: int | None = None,
    nfft: int | None = None,
) -> tuple[int, int, int]:
    nperseg = min(int(welch_nperseg), int(time_len))
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = max(256, nperseg)
    return nperseg, int(noverlap), int(nfft)


def _welch_bandpower_torch_core(
    eeg_nct: torch.Tensor,
    sampling_rate: float,
    bands_hz: list[list[float]],
    nperseg: int,
    noverlap: int,
    nfft: int,
    log_psd: bool,
    psd_eps: float,
) -> torch.Tensor:
    """Welch bandpower for ``eeg_nct`` shaped ``[N, C, T]``."""
    num_samples, num_channels, _ = eeg_nct.shape
    num_bands = len(bands_hz)
    outputs = torch.zeros(
        (num_samples, num_channels, num_bands),
        device=eeg_nct.device,
        dtype=eeg_nct.dtype,
    )
    if nperseg < 4:
        return outputs

    step = nperseg - noverlap
    windows = eeg_nct.unfold(dimension=-1, size=nperseg, step=step)
    windows = windows - windows.mean(dim=-1, keepdim=True)

    win = torch.hann_window(nperseg, periodic=False, device=eeg_nct.device, dtype=eeg_nct.dtype)
    scale = 1.0 / (float(sampling_rate) * win.pow(2).sum())
    windows = windows * win

    fft = torch.fft.rfft(windows, n=nfft, dim=-1)
    psd = (fft.abs() ** 2) * scale
    if nfft % 2 == 0:
        psd[..., 1:-1] = psd[..., 1:-1] * 2
    else:
        psd[..., 1:] = psd[..., 1:] * 2
    psd = psd.mean(dim=-2)

    freqs = torch.fft.rfftfreq(nfft, d=1.0 / float(sampling_rate)).to(eeg_nct.device)
    for band_idx, (low, high) in enumerate(bands_hz):
        mask = (freqs >= float(low)) & (freqs < float(high))
        if torch.any(mask):
            outputs[..., band_idx] = torch.trapz(psd[..., mask], freqs[mask], dim=-1)

    if log_psd:
        outputs = torch.log(outputs + float(psd_eps))
    return outputs


def welch_bandpower_gpu(
    x: np.ndarray | torch.Tensor,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    noverlap: int | None = None,
    nfft: int | None = None,
    log_psd: bool = False,
    psd_eps: float = 1e-12,
    device: str | torch.device = "cuda",
    batch_size: int | None = None,
    return_numpy: bool = True,
) -> np.ndarray | torch.Tensor:
    """
    GPU batch Welch bandpower aligned with ``compute_channel_band_psd`` (SciPy).

    Input ``x``:
      - ``[N, C, T]`` -> output ``[N, C, B]``
      - ``[M, T]`` -> output ``[M, B]`` (single channel per row)
    """
    bands_hz = DEFAULT_BANDS_HZ if bands_hz is None else bands_hz
    squeeze_channel = False
    if isinstance(x, np.ndarray):
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    else:
        x_t = x.to(dtype=torch.float32)
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(1)
        squeeze_channel = True
    if x_t.ndim != 3:
        raise ValueError(f"Expected x with 2 or 3 dims, got shape {tuple(x_t.shape)}")

    torch_device = torch.device(device)
    time_len = x_t.shape[-1]
    nperseg, noverlap_resolved, nfft_resolved = _resolve_welch_segment_params(
        time_len, welch_nperseg, noverlap=noverlap, nfft=nfft
    )
    num_samples = x_t.shape[0]
    if batch_size is None:
        batch_size = num_samples

    out_parts: list[torch.Tensor] = []
    for start in range(0, num_samples, batch_size):
        batch = x_t[start : start + batch_size].to(torch_device, non_blocking=True)
        out_parts.append(
            _welch_bandpower_torch_core(
                batch,
                sampling_rate=sampling_rate,
                bands_hz=bands_hz,
                nperseg=nperseg,
                noverlap=noverlap_resolved,
                nfft=nfft_resolved,
                log_psd=log_psd,
                psd_eps=psd_eps,
            )
        )
    outputs = torch.cat(out_parts, dim=0)
    if squeeze_channel:
        outputs = outputs.squeeze(1)
    if return_numpy:
        return outputs.cpu().numpy()
    return outputs


def compute_channel_band_psd_torch(
    eeg_nct: torch.Tensor,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
) -> torch.Tensor:
    """Batch Welch bandpower per channel; aligns with scipy welch (hann, density, detrend constant)."""
    bands_hz = DEFAULT_BANDS_HZ if bands_hz is None else bands_hz
    eeg_nct = eeg_nct.to(dtype=torch.float32)
    time_len = eeg_nct.shape[-1]
    nperseg, noverlap, nfft = _resolve_welch_segment_params(time_len, welch_nperseg)
    return _welch_bandpower_torch_core(
        eeg_nct,
        sampling_rate=sampling_rate,
        bands_hz=bands_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        log_psd=log_psd,
        psd_eps=psd_eps,
    )


def build_psd_targets_torch(
    eeg_nct: np.ndarray | torch.Tensor,
    sampling_rate: float,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
    device: str | torch.device = "cuda",
    batch_size: int = 128,
) -> np.ndarray:
    """GPU batched Welch targets (N, B), channel-averaged like build_psd_targets."""
    if isinstance(eeg_nct, np.ndarray):
        eeg_nct = torch.from_numpy(np.asarray(eeg_nct, dtype=np.float32))
    else:
        eeg_nct = eeg_nct.to(dtype=torch.float32)

    channel_psd = welch_bandpower_gpu(
        eeg_nct,
        sampling_rate=sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
        device=device,
        batch_size=batch_size,
        return_numpy=False,
    )
    assert isinstance(channel_psd, torch.Tensor)
    return channel_psd.mean(dim=1).cpu().numpy()


RELATIVE_TOTAL_SHAPE_DIM = 5
RELATIVE_TOTAL_OUTPUT_DIM = 6


def decompose_bandpower_targets(
    power: np.ndarray,
    psd_eps: float = 1e-12,
) -> np.ndarray:
    """Split linear bandpower (N, B) into relative shape (N, B) + log_total (N, 1)."""
    power = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    eps = float(psd_eps)
    total = np.maximum(power.sum(axis=1, keepdims=True), eps)
    shape = (power / total).astype(np.float32)
    log_total = np.log(total.squeeze(-1) + eps).astype(np.float32)
    return np.concatenate([shape, log_total[:, np.newaxis]], axis=1).astype(np.float32)


def fit_relative_total_transform(
    decomp_targets: np.ndarray,
    psd_eps: float = 1e-12,
    margin: float = 2.0,
) -> dict:
    """Fit log_total clip bounds from training decomposition targets."""
    decomp_targets = np.asarray(decomp_targets, dtype=np.float64)
    log_total = decomp_targets[:, RELATIVE_TOTAL_SHAPE_DIM]
    return {
        "target_decomposition": "relative_total",
        "psd_eps": float(psd_eps),
        "n_shape_bands": RELATIVE_TOTAL_SHAPE_DIM,
        "output_dim": RELATIVE_TOTAL_OUTPUT_DIM,
        "log_total_clip_lo": float(np.min(log_total) - margin),
        "log_total_clip_hi": float(np.max(log_total) + margin),
    }


def reconstruct_bandpower_from_decomposition(
    decomp: np.ndarray,
    psd_eps: float = 1e-12,
    transform_config: dict | None = None,
) -> np.ndarray:
    """Reconstruct linear bandpower (N, B) from model output: shape logits + log_total."""
    decomp = np.asarray(decomp, dtype=np.float64)
    eps = float(psd_eps)
    if transform_config is not None:
        eps = float(transform_config.get("psd_eps", eps))
    shape_logits = decomp[:, :RELATIVE_TOTAL_SHAPE_DIM]
    log_total = decomp[:, RELATIVE_TOTAL_SHAPE_DIM]
    if transform_config is not None:
        lo = transform_config.get("log_total_clip_lo")
        hi = transform_config.get("log_total_clip_hi")
        if lo is not None and hi is not None:
            log_total = np.clip(log_total, float(lo), float(hi))
    shape_logits = shape_logits - shape_logits.max(axis=1, keepdims=True)
    exp_shape = np.exp(shape_logits)
    shape = exp_shape / np.maximum(exp_shape.sum(axis=1, keepdims=True), eps)
    max_log = float(np.log(np.finfo(np.float32).max)) * 0.99
    log_total = np.minimum(log_total, max_log)
    total = np.exp(log_total)
    power = shape * total[:, np.newaxis]
    return np.clip(power, 0.0, np.finfo(np.float32).max).astype(np.float32)


def apply_target_mode(targets: np.ndarray, target_mode: str = "absolute") -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if target_mode == "absolute":
        return targets
    if target_mode == "relative":
        return (targets - targets.mean(axis=1, keepdims=True)).astype(np.float32)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def fit_target_transform(
    targets: np.ndarray,
    target_space: str = "linear",
    psd_scale: float = 1.0,
    psd_eps: float = 1e-12,
) -> dict[str, float | list[float] | str]:
    targets = np.asarray(targets, dtype=np.float32)
    if target_space == "linear":
        return {
            "target_space": target_space,
            "psd_scale": float(psd_scale),
            "psd_eps": float(psd_eps),
        }
    if target_space == "log_train_linear_eval":
        z = np.log(targets / float(psd_scale) + float(psd_eps)).astype(np.float64)
        z_lo = np.min(z, axis=0)
        z_hi = np.max(z, axis=0)
        slack_lo = 10.0
        margin_hi = 8.0
        cap_hi = 15.0
        hi_arr = np.minimum(z_hi + margin_hi, cap_hi)
        return {
            "target_space": target_space,
            "psd_scale": float(psd_scale),
            "psd_eps": float(psd_eps),
            "log_pred_clip_lo": (z_lo - slack_lo).astype(np.float64).tolist(),
            "log_pred_clip_hi": hi_arr.astype(np.float64).tolist(),
        }
    raise ValueError(f"Unknown target_space: {target_space}")


def transform_target_array(targets: np.ndarray, transform_config: dict) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    target_space = transform_config.get("target_space", "linear")
    psd_scale = float(transform_config.get("psd_scale", 1.0))
    psd_eps = float(transform_config.get("psd_eps", 1e-12))
    if target_space == "linear":
        return targets.astype(np.float32)
    if target_space == "log_train_linear_eval":
        return np.log(targets / psd_scale + psd_eps).astype(np.float32)
    raise ValueError(f"Unknown target_space: {target_space}")


def inverse_transform_target_array(targets: np.ndarray, transform_config: dict) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    target_space = transform_config.get("target_space", "linear")
    psd_scale = float(transform_config.get("psd_scale", 1.0))
    psd_eps = float(transform_config.get("psd_eps", 1e-12))
    if target_space == "linear":
        return np.maximum(targets, 0.0).astype(np.float32)
    if target_space == "log_train_linear_eval":
        max_log = float(np.log(np.finfo(np.float32).max)) * 0.99
        clipped = np.minimum(targets.astype(np.float64), max_log)
        return (np.maximum(np.exp(clipped) - psd_eps, 0.0) * psd_scale).astype(np.float32)
    raise ValueError(f"Unknown target_space: {target_space}")


def clip_transformed_model_outputs(predictions: np.ndarray, transform_config: dict) -> np.ndarray:
    """Clip model outputs in transformed space before inverse (stabilizes log_train_linear_eval)."""
    if transform_config.get("target_space") != "log_train_linear_eval":
        return np.asarray(predictions, dtype=np.float32)
    pred = np.asarray(predictions, dtype=np.float32)
    n_b = pred.shape[1]
    lo = transform_config.get("log_pred_clip_lo", -35.0)
    hi = transform_config.get("log_pred_clip_hi", 14.0)
    lo_a = np.asarray(lo, dtype=np.float32).reshape(-1)
    hi_a = np.asarray(hi, dtype=np.float32).reshape(-1)
    if lo_a.size == 1:
        lo_a = np.full(n_b, float(lo_a[0]), dtype=np.float32)
    if hi_a.size == 1:
        hi_a = np.full(n_b, float(hi_a[0]), dtype=np.float32)
    if lo_a.shape[0] != n_b or hi_a.shape[0] != n_b:
        raise ValueError(
            f"log_pred_clip bounds length {lo_a.shape[0]}/{hi_a.shape[0]} != n_bands {n_b}"
        )
    return np.clip(pred, lo_a.reshape(1, -1), hi_a.reshape(1, -1))


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    band_names: list[str] | None = None,
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")

    band_names = DEFAULT_BAND_NAMES if band_names is None else band_names
    errors = predictions - targets
    metrics = {
        "overall_mse": float(np.mean(errors ** 2)),
        "overall_mae": float(np.mean(np.abs(errors))),
        "per_band_mse": {},
        "per_band_mae": {},
        "per_band_pearson": {},
    }

    pearsons = []
    for band_idx in range(predictions.shape[1]):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
        pred_band = predictions[:, band_idx]
        target_band = targets[:, band_idx]
        metrics["per_band_mse"][band_name] = float(np.mean((pred_band - target_band) ** 2))
        metrics["per_band_mae"][band_name] = float(np.mean(np.abs(pred_band - target_band)))
        if pred_band.std() == 0.0 or target_band.std() == 0.0:
            pearson = 0.0
        else:
            pearson = float(np.corrcoef(pred_band, target_band)[0, 1])
        metrics["per_band_pearson"][band_name] = pearson
        pearsons.append(pearson)

    metrics["pearson_mean"] = float(np.mean(pearsons))
    return metrics


def compute_real_psd_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    band_names: list[str] | None = None,
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    band_names = DEFAULT_BAND_NAMES if band_names is None else band_names
    eps = 1e-12
    errors = predictions - targets
    abs_targets = np.maximum(np.abs(targets), eps)
    denom = np.maximum(np.abs(predictions) + np.abs(targets), eps)

    metrics = {
        "overall_mse": float(np.mean(errors ** 2)),
        "overall_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "overall_mae": float(np.mean(np.abs(errors))),
        "overall_mape": float(np.mean(np.abs(errors) / abs_targets)),
        "overall_smape": float(np.mean(2.0 * np.abs(errors) / denom)),
        "per_band_mse": {},
        "per_band_rmse": {},
        "per_band_mae": {},
        "per_band_mape": {},
        "per_band_smape": {},
        "per_band_pearson": {},
    }

    pearsons = []
    for band_idx in range(predictions.shape[1]):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
        pred_band = predictions[:, band_idx]
        target_band = targets[:, band_idx]
        error_band = pred_band - target_band
        abs_target_band = np.maximum(np.abs(target_band), eps)
        denom_band = np.maximum(np.abs(pred_band) + np.abs(target_band), eps)
        mse = float(np.mean(error_band ** 2))
        metrics["per_band_mse"][band_name] = mse
        metrics["per_band_rmse"][band_name] = float(np.sqrt(mse))
        metrics["per_band_mae"][band_name] = float(np.mean(np.abs(error_band)))
        metrics["per_band_mape"][band_name] = float(np.mean(np.abs(error_band) / abs_target_band))
        metrics["per_band_smape"][band_name] = float(np.mean(2.0 * np.abs(error_band) / denom_band))
        if pred_band.std() == 0.0 or target_band.std() == 0.0:
            pearson = 0.0
        else:
            pearson = float(np.corrcoef(pred_band, target_band)[0, 1])
        metrics["per_band_pearson"][band_name] = pearson
        pearsons.append(pearson)
    metrics["pearson_mean"] = float(np.mean(pearsons))
    return metrics


def fit_linear_calibration_per_band(
    predictions: np.ndarray,
    targets: np.ndarray,
    band_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit y ~= slope * x + intercept per band (numpy polyfit)."""
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    band_names = DEFAULT_BAND_NAMES if band_names is None else band_names
    linear_params: dict[str, dict[str, float]] = {}
    for band_idx in range(predictions.shape[1]):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
        pred_band = predictions[:, band_idx]
        target_band = targets[:, band_idx]
        if pred_band.std() == 0.0:
            slope = 0.0
            intercept = float(target_band.mean())
        else:
            try:
                slope, intercept = np.polyfit(pred_band, target_band, deg=1)
                slope = float(slope)
                intercept = float(intercept)
            except np.linalg.LinAlgError:
                slope, intercept = 1.0, 0.0
        linear_params[band_name] = {"slope": slope, "intercept": intercept}
    return linear_params


def apply_linear_calibration_per_band(
    predictions: np.ndarray,
    per_band_params: dict[str, dict[str, float]],
    band_names: list[str] | None = None,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    band_names = DEFAULT_BAND_NAMES if band_names is None else band_names
    out = np.zeros_like(predictions, dtype=np.float32)
    for band_idx in range(predictions.shape[1]):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
        params = per_band_params[band_name]
        pred_band = predictions[:, band_idx]
        out[:, band_idx] = params["slope"] * pred_band + params["intercept"]
    return out


def compute_calibration_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    band_names: list[str] | None = None,
) -> dict:
    predictions = np.asarray(predictions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    band_names = DEFAULT_BAND_NAMES if band_names is None else band_names

    bias = (predictions - targets).mean(axis=0, keepdims=True)
    bias_corrected = predictions - bias
    bias_metrics = compute_regression_metrics(bias_corrected, targets, band_names=band_names)

    linear_params_raw = fit_linear_calibration_per_band(predictions, targets, band_names=band_names)
    linear_params = {}
    for band_idx in range(predictions.shape[1]):
        band_name = band_names[band_idx] if band_idx < len(band_names) else f"band_{band_idx}"
        p = linear_params_raw[band_name].copy()
        p["bias"] = float(bias[0, band_idx])
        linear_params[band_name] = p
    linearly_calibrated = apply_linear_calibration_per_band(predictions, linear_params_raw, band_names=band_names)

    linear_metrics = compute_regression_metrics(linearly_calibrated, targets, band_names=band_names)
    return {
        "bias_correction": {
            "metrics": bias_metrics,
            "per_band_bias": {
                band_names[idx] if idx < len(band_names) else f"band_{idx}": float(bias[0, idx])
                for idx in range(predictions.shape[1])
            },
        },
        "linear_calibration": {
            "metrics": linear_metrics,
            "per_band_params": linear_params,
        },
    }
