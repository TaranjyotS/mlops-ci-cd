from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import great_expectations as gx
    from great_expectations.data_context import EphemeralDataContext
    from great_expectations.data_context.types.base import DataContextConfig
except Exception:  # pragma: no cover
    gx = None
    EphemeralDataContext = None
    DataContextConfig = None


SUITE_NAME = "train_suite"


def _ephemeral_context():
    """Create a YAML-free Great Expectations context.

    This avoids config-schema issues on different GE versions and works in CI.
    """
    if EphemeralDataContext is None or DataContextConfig is None:
        raise RuntimeError("great-expectations is not installed")

    config = DataContextConfig(
        config_version=3.0,
        datasources={},
        stores={},
        expectations_store_name=None,
        validations_store_name=None,
        checkpoint_store_name=None,
    )
    return EphemeralDataContext(project_config=config)


def validate_csv(path: str) -> None:
    df = pd.read_csv(path)

    # Basic safety checks (even if GE isn't available)
    required_cols = {"feature1", "feature2", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Validation failed: missing columns: {sorted(missing)}")
    if df["target"].isna().any():
        raise SystemExit("Validation failed: target contains nulls")
    if len(df) < 50:
        raise SystemExit("Validation failed: too few rows (<50)")

    # Great Expectations checks (optional but recommended)
    if gx is None:
        Path("reports").mkdir(exist_ok=True)
        (Path("reports") / "ge_validation.json").write_text(
            "{\"success\": true, \"note\": \"GE not installed; basic validation only.\"}",
            encoding="utf-8",
        )
        print("⚠️ great-expectations not available; basic validation passed.")
        return

    context = _ephemeral_context()

    # Datasource + asset + batch request (stable across 0.18.x)
    ds = context.sources.add_pandas(name="pandas")
    asset = ds.add_dataframe_asset(name="train_df")
    batch_request = asset.build_batch_request(dataframe=df)

    # Ensure suite exists
    existing = {s.name for s in context.list_expectation_suites()}
    if SUITE_NAME not in existing:
        suite = context.add_expectation_suite(expectation_suite_name=SUITE_NAME)
    else:
        suite = context.get_expectation_suite(SUITE_NAME)

    # Expectations
    suite.add_expectation(gx.expectations.ExpectTableRowCountToBeBetween(min_value=50, max_value=None))
    for col in ["feature1", "feature2", "target"]:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    context.save_expectation_suite(suite)

    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=SUITE_NAME)
    results = validator.validate()

    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "ge_validation.json").write_text(results.json(indent=2), encoding="utf-8")

    if not results.success:
        raise SystemExit("Great Expectations validation failed. See reports/ge_validation.json")

    print("✅ Data validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/train.csv", help="Path to training CSV")
    args = parser.parse_args()
    validate_csv(args.data)


if __name__ == "__main__":
    main()
