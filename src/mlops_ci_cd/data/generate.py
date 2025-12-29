from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(out_path: str, n_rows: int = 1000, seed: int = 42) -> Path:
    """Generate a simple synthetic binary classification dataset.
    This is intentionally lightweight so the pipeline is runnable anywhere.

    Parameters:
        out_path: Output CSV file path.
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        Path: The path to the generated CSV file.
    """
    rng = np.random.default_rng(seed)

    feature1 = rng.uniform(0, 10, size=n_rows)
    feature2 = rng.uniform(0, 10, size=n_rows)

    # Simple target correlated with features + noise
    logits = 0.35 * feature1 - 0.2 * feature2 + rng.normal(0, 1.0, size=n_rows)
    prob = 1 / (1 + np.exp(-logits))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({"feature1": feature1, "feature2": feature2, "target": target})

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/train.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=1000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = generate_dataset(args.out, n_rows=args.rows, seed=args.seed)
    print(f"✅ Generated dataset: {out}")


if __name__ == "__main__":
    main()
