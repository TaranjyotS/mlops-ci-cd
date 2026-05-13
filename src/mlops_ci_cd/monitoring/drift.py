from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = ["feature1", "feature2"]


def compute_drift(reference_path: str, current_path: str, report_path: str, threshold: float = 0.25) -> dict[str, object]:
    """Compute simple mean-shift drift signals for numeric features."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)
    report: dict[str, object] = {"threshold": threshold, "features": {}, "drift_detected": False}

    for column in FEATURE_COLUMNS:
        ref_mean = float(reference[column].mean())
        cur_mean = float(current[column].mean())
        ref_std = float(reference[column].std() or 1.0)
        normalized_shift = abs(cur_mean - ref_mean) / ref_std
        drifted = normalized_shift > threshold
        report["features"][column] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "normalized_shift": normalized_shift,
            "drifted": drifted,
        }
        report["drift_detected"] = bool(report["drift_detected"] or drifted)

    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/raw/train.csv")
    parser.add_argument("--current", required=True)
    parser.add_argument("--report", default="reports/drift_report.json")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    report = compute_drift(args.reference, args.current, args.report, args.threshold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
