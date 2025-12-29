from __future__ import annotations

import argparse
import json
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
    """
    Create a Great Expectations EphemeralDataContext with in-memory stores.
    This avoids loading any YAML config.

    Returns:
        EphemeralDataContext: An ephemeral Great Expectations data context.
    """
    from great_expectations.data_context import EphemeralDataContext
    from great_expectations.data_context.types.base import (
        DataContextConfig,
        InMemoryStoreBackendDefaults,
    )

    config = DataContextConfig(
        config_version=3.0,
        datasources={},
        store_backend_defaults=InMemoryStoreBackendDefaults(),
        expectations_store_name="expectations_store",
        validations_store_name="validations_store",
        checkpoint_store_name="checkpoint_store",
    )
    return EphemeralDataContext(project_config=config)


def validate_csv(csv_path: str) -> None:
    """
    Docstring for validate_csv
    
    Parameters:
        csv_path: Path to the CSV file to validate.
    """

    df = pd.read_csv(csv_path)

    context = _ephemeral_context()

    # Add a pandas datasource + dataframe asset
    datasource = context.sources.add_pandas(name="pandas_ds")
    asset = datasource.add_dataframe_asset(name="train_df")

    batch_request = asset.build_batch_request(dataframe=df)

    # Create or load suite
    suite_name = "train_suite"
    existing = {s.name for s in context.list_expectation_suites()}
    if suite_name not in existing:
        context.add_expectation_suite(expectation_suite_name=suite_name)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )

    # Fluent expectations API (stable in GE 0.18.x)
    validator.expect_table_row_count_to_be_between(min_value=50, max_value=None)

    first_col = df.columns[0] if len(df.columns) > 0 else None
    if first_col is None:
        raise SystemExit("Validation failed: dataset has zero columns.")

    validator.expect_column_values_to_not_be_null(column=first_col)

    # Save suite + validate
    validator.save_expectation_suite(discard_failed_expectations=False)
    results = validator.validate()

    # Save report
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ge_validation.json").write_text(
        json.dumps(results.to_json_dict(), indent=2),
        encoding="utf-8",
    )

    if not results.success:
        raise SystemExit("Great Expectations validation failed. See reports/ge_validation.json")

    print("✅ Great Expectations validation passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/train.csv", help="Path to training CSV")
    args = parser.parse_args()
    validate_csv(args.data)


if __name__ == "__main__":
    main()
