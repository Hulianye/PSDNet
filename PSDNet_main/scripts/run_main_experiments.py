from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "BNCI2014001": REPO_ROOT / "configs" / "BNCI2014001.yaml",
    "BNCI2014004": REPO_ROOT / "configs" / "BNCI2014004.yaml",
}


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_paper_speed_row(results: list[dict], num_samples: int, batch_size: int) -> dict | None:
    for row in results:
        if int(row["num_samples"]) == num_samples and int(row["batch_size"]) == batch_size:
            return row
    return None


def collect_dataset_summary(dataset_name: str, config_path: Path) -> dict:
    import yaml

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = REPO_ROOT / config["paths"]["output_dir"]

    psd_metrics = load_json(output_dir / "psd_psdnet_v2" / "psd_metrics.json")
    speed_payload = load_json(output_dir / "speed_benchmark.json")
    speed_cfg = config["speed_benchmark"]
    paper_num_samples = int(speed_cfg["num_samples_list"][-1])
    paper_batch_size = int(speed_cfg["batch_sizes"][-1])
    speed_row = pick_paper_speed_row(speed_payload["results"], paper_num_samples, paper_batch_size)
    if speed_row is None:
        raise RuntimeError(
            f"No speed result for num_samples={paper_num_samples}, batch_size={paper_batch_size} "
            f"in {output_dir / 'speed_benchmark.json'}"
        )

    test_metrics = psd_metrics["metrics_test"]
    return {
        "dataset": dataset_name,
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "psd_estimation": {
            "mae": test_metrics["overall_mae"],
            "rmse": test_metrics["overall_rmse"],
            "pearson": test_metrics["pearson_mean"],
        },
        "speed": {
            "num_samples": speed_row["num_samples"],
            "batch_size": speed_row["batch_size"],
            "cpu_ms_per_sample": speed_row["cpu_welch_sec_per_sample"] * 1000.0,
            "psdnet_ms_per_sample": speed_row["psdnet_sec_per_sample"] * 1000.0,
            "speedup_vs_cpu": speed_row["psdnet_speedup_vs_cpu"],
        },
    }


def print_summary_table(summary: dict) -> None:
    print("\n=== PSDNet main experiment summary ===")
    print(f"{'Dataset':<14} {'MAE':>8} {'RMSE':>8} {'Pearson':>8} {'CPU ms':>10} {'PSDNet ms':>11} {'Speedup':>8}")
    for row in summary["datasets"]:
        psd = row["psd_estimation"]
        speed = row["speed"]
        print(
            f"{row['dataset']:<14} "
            f"{psd['mae']:8.4f} {psd['rmse']:8.4f} {psd['pearson']:8.4f} "
            f"{speed['cpu_ms_per_sample']:10.2f} {speed['psdnet_ms_per_sample']:11.2f} "
            f"{speed['speedup_vs_cpu']:7.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PSDNet main experiments on BNCI2014001 and BNCI2014004.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_speed", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    python = sys.executable
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for name in selected:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {', '.join(DATASETS)}")
        config_path = DATASETS[name]
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config: {config_path}")

        if not args.skip_train:
            run([python, str(REPO_ROOT / "scripts" / "train_psd.py"), "--config", str(config_path), "--device", args.device])
        if not args.skip_speed:
            run([python, str(REPO_ROOT / "scripts" / "run_speed.py"), "--config", str(config_path), "--device", args.device])

    summary = {"datasets": [collect_dataset_summary(name, DATASETS[name]) for name in selected]}
    summary_path = REPO_ROOT / "outputs" / "main_experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary_table(summary)
    print(f"\nSaved summary to {summary_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
