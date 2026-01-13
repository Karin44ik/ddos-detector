import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Show CSV columns and sample rows"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV file"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display"
    )

    args = parser.parse_args()
    csv_path = Path(args.input)

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[✓] File: {csv_path}")
    print(f"[✓] Shape: {df.shape}")
    print(f"[✓] Columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}")

    print("\n[✓] Sample rows:")
    print(df.head(args.rows))


if __name__ == "__main__":
    main()
