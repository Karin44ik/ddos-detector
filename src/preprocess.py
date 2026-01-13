# src/preprocess.py
from pathlib import Path
from typing import Tuple

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Обрізає пробіли та уніфікує назви колонок."""
    return df.rename(columns=lambda c: str(c).strip())


def to_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Перетворює фрейм у (X, y):
    - якщо є 'Label' (CICIDS) → робить 'label' (BENIGN=0, інакше=1)
    - лишає лише числові фічі
    """
    df = normalize_columns(df)

    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "Label" in df.columns:
        df["attack_cat"] = df["Label"].astype(str)
        df["label"] = (df["attack_cat"].str.upper() != "BENIGN").astype(int)
        label_col = "label"

    if label_col is None:
        raise ValueError("No 'label'/'Label' column found in dataframe.")

    non_numeric = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df.drop(columns=[label_col] + non_numeric, errors="ignore")
    y = df[label_col].astype(int)
    return X, y


def load_xy(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Зчитує CSV та повертає (X, y)."""
    df = pd.read_csv(csv_path)
    return to_xy(df)


if __name__ == "__main__":
    # Швидка перевірка: читаємо train.csv, друкуємо розміри
    path = Path("data/processed/train.csv")
    if path.exists():
        X_, y_ = load_xy(path)
        print("X:", X_.shape, "y:", y_.shape)
    else:
        print("data/processed/train.csv not found")
