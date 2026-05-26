import io
import os
import pickle
from math import gcd

import numpy as np
from scipy import signal as scipy_signal


class EEGData:
    def __init__(
        self,
        dataset_name,
        eeg_data,
        subject_ids,
        channel_names,
        sampling_rate,
        labels=None,
        dataset_type="classification",
        regression_values=None,
        img_feature=None,
        split_method="LOSO",
        is_binary=False,
        usage=None,
    ):
        self.dataset_name = dataset_name
        self.eeg_data = np.array(eeg_data)
        self.subject_ids = np.array(subject_ids)
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.labels = None if labels is None else np.array(labels)
        self.dataset_type = dataset_type
        self.regression_values = None if regression_values is None else np.array(regression_values)
        self.img_feature = None if img_feature is None else np.array(img_feature)
        self.split_method = split_method
        self.is_binary = is_binary
        self.usage = None if usage is None else np.array(usage)

    def get_sample_count(self):
        return self.eeg_data.shape[0]

    def get_channel_count(self):
        return self.eeg_data.shape[1]

    def get_time_point_count(self):
        return self.eeg_data.shape[2]

    def get_duration(self):
        return self.get_time_point_count() / self.sampling_rate

    def get_subject_unique_count(self):
        return len(np.unique(self.subject_ids))

    def get_label_count(self):
        if self.labels is None:
            return 0
        return len(np.unique(self.labels))

    def __str__(self):
        result = (
            f"Dataset: {self.dataset_name}\n"
            f"Samples: {self.get_sample_count()}, Channels: {self.get_channel_count()}, Time points: {self.get_time_point_count()}\n"
            f"Sampling rate: {self.sampling_rate} Hz, Duration per sample: {self.get_duration():.2f} s\n"
            f"Subjects: {self.get_subject_unique_count()}, Task: {self.dataset_type}\n"
        )
        if self.labels is not None:
            result += f"Labels: {self.get_label_count()} classes, {list(np.unique(self.labels))}\n"
        return result


class _BenchmarkEEGDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils.EEGDataLoader" and name == "EEGData":
            return EEGData
        return super().find_class(module, name)


def load_eeg_data_from_pkl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "rb") as f:
        data = f.read()
    eeg_data_obj = _BenchmarkEEGDataUnpickler(io.BytesIO(data)).load()
    if not isinstance(eeg_data_obj, EEGData):
        raise TypeError(f"Loaded object is not EEGData: {type(eeg_data_obj)}")
    return eeg_data_obj


def resample_eeg_data(eeg_data: EEGData, target_sfreq: float) -> tuple[EEGData, dict]:
    """Resample trial time axis to ``target_sfreq``; returns new EEGData and metadata."""
    orig_sfreq = float(eeg_data.sampling_rate)
    target_sfreq = float(target_sfreq)
    meta = {
        "dataset_name": eeg_data.dataset_name,
        "original_sfreq": orig_sfreq,
        "target_sfreq": target_sfreq,
        "resampled": False,
        "original_time_points": int(eeg_data.eeg_data.shape[-1]),
        "resampled_time_points": int(eeg_data.eeg_data.shape[-1]),
    }
    if abs(orig_sfreq - target_sfreq) < 1e-6:
        return eeg_data, meta

    eeg = np.asarray(eeg_data.eeg_data, dtype=np.float64)
    up = int(round(target_sfreq * 1000))
    down = int(round(orig_sfreq * 1000))
    g = gcd(up, down)
    up //= g
    down //= g
    eeg_rs = scipy_signal.resample_poly(eeg, up, down, axis=-1).astype(np.float32)
    meta["resampled"] = True
    meta["resampled_time_points"] = int(eeg_rs.shape[-1])

    out = EEGData(
        dataset_name=eeg_data.dataset_name,
        eeg_data=eeg_rs,
        subject_ids=eeg_data.subject_ids,
        channel_names=eeg_data.channel_names,
        sampling_rate=target_sfreq,
        labels=getattr(eeg_data, "labels", None),
        dataset_type=getattr(eeg_data, "dataset_type", "classification"),
        regression_values=getattr(eeg_data, "regression_values", None),
        img_feature=getattr(eeg_data, "img_feature", None),
        split_method=getattr(eeg_data, "split_method", "LOSO"),
        is_binary=getattr(eeg_data, "is_binary", False),
        usage=getattr(eeg_data, "usage", None),
    )
    return out, meta
