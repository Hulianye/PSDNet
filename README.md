# PSDNet

PSDNet is a neural network for fast EEG band-power (PSD) estimation. This repository contains the model implementation and scripts to reproduce the main PSD estimation and speed experiments from the course paper.

## Repository layout

```text
PSDNet_main/
├── psdnet/              # core model and utilities
├── scripts/             # training, evaluation, speed benchmark
├── configs/             # BNCI2014001 / BNCI2014004 experiment configs
├── data/                # local dataset pickles (not included)
└── outputs/             # generated experiment outputs
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation

Place preprocessed dataset pickle files under `data/`:

- `data/BNCI2014001.pkl`
- `data/BNCI2014004.pkl`

Each pickle should follow the format expected by `psdnet.eeg_data.load_eeg_data_from_pkl` (EEG trials, labels, sampling rate, dataset name).

## Reproduce main experiments

Run PSDNet training and speed benchmark on both datasets:

```bash
python scripts/run_main_experiments.py
```

This will:

1. Train PSDNet (`psdnet_v2`) on BNCI2014001 and BNCI2014004
2. Benchmark inference speed vs CPU Welch on the held-out test split
3. Write `outputs/main_experiment_summary.json` and print a summary table

Options:

```bash
python scripts/run_main_experiments.py --datasets BNCI2014001
python scripts/run_main_experiments.py --skip_train   # speed only
python scripts/run_main_experiments.py --skip_speed   # training only
```

## Individual commands

Train PSDNet on one dataset:

```bash
python scripts/train_psd.py --config configs/BNCI2014001.yaml
```

Speed benchmark (requires a trained checkpoint):

```bash
python scripts/run_speed.py --config configs/BNCI2014001.yaml
```

Ad-hoc training with a custom config:

```bash
python scripts/train.py --config config.example.yaml --train_pkl data/BNCI2014001.pkl --output_dir outputs/custom_run
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --config config.example.yaml \
  --checkpoint outputs/BNCI2014001/psd_psdnet_v2/checkpoints/best.pt \
  --test_pkl data/BNCI2014001.pkl \
  --output_dir outputs/eval_run
```

## Paper reference results

| Dataset | MAE | RMSE | Pearson | CPU (ms/sample) | PSDNet (ms/sample) | Speedup |
|---------|-----|------|---------|-----------------|-------------------|---------|
| BNCI2014001 | 2.8503 | 7.9105 | 0.8313 | 4.98 | 0.20 | 25.48× |
| BNCI2014004 | 0.2989 | 0.6588 | 0.9289 | 0.70 | 0.03 | 20.73× |

Exact numbers may vary slightly across hardware and random seeds.

## License

MIT License. See [LICENSE](LICENSE).
