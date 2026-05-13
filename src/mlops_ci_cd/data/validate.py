from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"feature1", "feature2", "target"}
REPORT_PATH = Path("reports/ge_validation.json")


def _write_report(success: bool, checks: list[dict[str, object]], output_path: Path = REPORT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"success": success, "checks": checks}, indent=2), encoding="utf-8")


def validate_csv(csv_path: str, report_path: str | Path = REPORT_PATH) -> bool:
    """Validate core data quality expectations for the training dataset."""
    df = pd.read_csv(csv_path)
    checks: list[dict[str, object]] = []

    def add_check(name: str, success: bool, details: object = None) -> None:
        checks.append({"name": name, "success": success, "details": details})

    add_check("minimum_row_count", len(df) >= 50, {"rows": len(df), "minimum": 50})
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    add_check("required_columns", not missing_columns, {"missing": missing_columns})

    if not missing_columns:
        add_check("no_nulls", not df[list(REQUIRED_COLUMNS)].isna().any().any())
        add_check("feature1_range", bool(df["feature1"].between(0, 10).all()))
        add_check("feature2_range", bool(df["feature2"].between(0, 10).all()))
        add_check("target_binary", set(df["target"].dropna().unique()).issubset({0, 1}))
        add_check("target_has_two_classes", df["target"].nunique() == 2)

    success = all(bool(check["success"]) for check in checks)
    _write_report(success, checks, Path(report_path))
    if not success:
        failed = [check["name"] for check in checks if not check["success"]]
        raise SystemExit(f"Data validation failed: {failed}. See {report_path}")

    print("Data validation passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/train.csv", help="Path to training CSV")
    parser.add_argument("--report", default=str(REPORT_PATH), help="Path to validation report JSON")
    args = parser.parse_args()
    validate_csv(args.data, args.report)


if __name__ == "__main__":
    main()
