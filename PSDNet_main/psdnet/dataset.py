from __future__ import annotations

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import Dataset

from .features import apply_target_mode, build_psd_targets, compute_channel_band_psd, welch_bandpower_gpu


def preprocess_inputs(eeg_nct: np.ndarray, center_input: bool = True, time_points: int | None = None) -> np.ndarray:
    eeg_nct = np.asarray(eeg_nct, dtype=np.float32)
    if time_points is not None:
        eeg_nct = eeg_nct[:, :, : int(time_points)]
    if center_input:
        eeg_nct = eeg_nct - eeg_nct.mean(axis=-1, keepdims=True)
    return eeg_nct.astype(np.float32)


def prepare_regression_arrays(
    eeg_data,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
    center_input: bool = True,
    time_points: int | None = None,
    target_mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    inputs = preprocess_inputs(
        eeg_nct=eeg_data.eeg_data,
        center_input=center_input,
        time_points=time_points,
    )
    targets = build_psd_targets(
        eeg_nct=inputs,
        sampling_rate=eeg_data.sampling_rate,
        bands_hz=bands_hz,
        welch_nperseg=welch_nperseg,
        log_psd=log_psd,
        psd_eps=psd_eps,
    )
    targets = apply_target_mode(targets, target_mode=target_mode)
    return inputs, targets


def resample_1d_to_length(signal_1d: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a 1D waveform to ``n_out`` samples (float32)."""
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    if x.size == n_out:
        return x.astype(np.float32)
    y = scipy_signal.resample(x, int(n_out), axis=-1)
    return y.astype(np.float32)


def prepare_channel_regression_arrays(
    eeg_data,
    bands_hz: list[list[float]] | None = None,
    welch_nperseg: int = 256,
    log_psd: bool = True,
    psd_eps: float = 1e-12,
    center_input: bool = True,
    input_time_points: int = 1024,
    target_mode: str = "absolute",
    target_backend: str = "scipy",
    target_device: str = "cuda",
    target_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | list]]:
    """Expand each trial channel into one sample: inputs (N*C,1,T_fix), targets (N*C,B)."""
    if target_backend not in ("scipy", "torch_gpu"):
        raise ValueError(f"Unknown target_backend: {target_backend}")

    eeg_nct = preprocess_inputs(eeg_data.eeg_data, center_input=center_input, time_points=None)
    eeg_nct = np.asarray(eeg_nct, dtype=np.float32)
    num_trials, num_ch, _ = eeg_nct.shape
    sr = float(eeg_data.sampling_rate)
    names = list(eeg_data.channel_names) if eeg_data.channel_names is not None else [f"ch{k}" for k in range(num_ch)]

    rows = num_trials * num_ch
    t_fix = int(input_time_points)
    inputs_out = np.zeros((rows, 1, t_fix), dtype=np.float32)
    trial_idx = np.zeros(rows, dtype=np.int32)
    channel_idx = np.zeros(rows, dtype=np.int32)
    channel_names: list[str] = []

    if target_backend == "torch_gpu":
        bandpower_nct = welch_bandpower_gpu(
            eeg_nct,
            sampling_rate=sr,
            bands_hz=bands_hz,
            welch_nperseg=welch_nperseg,
            log_psd=log_psd,
            psd_eps=psd_eps,
            device=target_device,
            batch_size=target_batch_size,
        )
        targets_out = np.asarray(bandpower_nct, dtype=np.float32).reshape(rows, -1)
    else:
        targets_list: list[np.ndarray] = []
        row = 0
        for trial_i in range(num_trials):
            for ch_j in range(num_ch):
                raw_ch = eeg_nct[trial_i, ch_j, :]
                psd_row = compute_channel_band_psd(
                    np.asarray(raw_ch, dtype=np.float64).reshape(1, -1),
                    sampling_rate=sr,
                    bands_hz=bands_hz,
                    welch_nperseg=welch_nperseg,
                    log_psd=log_psd,
                    psd_eps=psd_eps,
                )[0]
                targets_list.append(np.asarray(psd_row, dtype=np.float32))
                row += 1
        targets_out = np.stack(targets_list, axis=0)

    row = 0
    for trial_i in range(num_trials):
        for ch_j in range(num_ch):
            raw_ch = eeg_nct[trial_i, ch_j, :]
            rs = resample_1d_to_length(raw_ch, t_fix)
            rs = rs - float(rs.mean())
            inputs_out[row, 0, :] = rs
            trial_idx[row] = trial_i
            channel_idx[row] = ch_j
            ch_name = names[ch_j] if ch_j < len(names) else f"ch{ch_j}"
            channel_names.append(str(ch_name))
            row += 1
    targets_out = apply_target_mode(targets_out, target_mode=target_mode)
    meta: dict[str, np.ndarray | list] = {
        "source_id": np.zeros(rows, dtype=np.int32),
        "dataset_name": np.array([str(eeg_data.dataset_name)] * rows, dtype=object),
        "trial_index": trial_idx,
        "channel_index": channel_idx,
        "channel_name": np.array(channel_names, dtype=object),
    }
    return inputs_out, targets_out, meta


class PSDRegressionDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, indices: np.ndarray | None = None):
        self.inputs = np.asarray(inputs, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.indices = np.arange(self.inputs.shape[0]) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])
        return (
            torch.from_numpy(self.inputs[sample_idx]),
            torch.from_numpy(self.targets[sample_idx]),
        )
