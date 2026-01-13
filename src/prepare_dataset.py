import sys
from pathlib import Path
from typing import List, Optional
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

"""
prepare_dataset.py — створення компактного датасету для DoS/DDoS (UNSW‑NB15 / CICIDS2017)
-------------------------------------------------------------------------------------------
- Беремо CSV з data/raw/*.csv, авто‑визначаємо тип датасету (або вказуємо --dtype)
- Чистимо до числових фіч, балансуємо класи, формуємо підмножину (за замовч. 10k)
- Зберігаємо: data/processed/<base>_10k.csv, train.csv, test.csv (80/20, stratified)

Запуск із кореня проєкту (Windows/PowerShell):
    python src/prepare_dataset.py --rows 10000 --dtype CICIDS
або без явного типу (буде авто‑детект):
    python src/prepare_dataset.py --rows 10000

Примітки:
- Використовуємо «терпиме» читання CSV: engine='python', on_bad_lines='skip', encoding_errors='ignore'
- Для CICIDS2017: 'Label' → binary 'label' (BENIGN=0, інше=1); оригінальний текст у 'attack_cat'
- Для UNSW‑NB15: очікуємо 'label' (0/1); 'attack_cat' зберігаємо, якщо є
"""

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """CSV reader, стійкий до «битих» рядків/символів."""
    kwargs = dict(engine="python", on_bad_lines="skip")
    if nrows is not None:
        kwargs["nrows"] = nrows
    try:
        return pd.read_csv(path, encoding_errors="ignore", **kwargs)
    except TypeError:
        return pd.read_csv(path, **kwargs)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Обрізає пробіли на початку/вкінці назв колонок і уніфікує дублікати."""
    df = df.rename(columns=lambda c: str(c).strip())
    # Якщо випадково є дублікати назв після strip — зробимо унікальними
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for c in df.columns:
        base = c
        if base in seen:
            seen[base] += 1
            new_cols.append(f"{base}.{seen[base]}")
        else:
            seen[base] = 0
            new_cols.append(base)
    df.columns = new_cols
    return df


def detect_dataset(files: List[Path]) -> str:
    """Повертає 'UNSW' або 'CICIDS' за структурою колонок."""
    for f in files:
        try:
            head = _read_csv(f, nrows=5)
            head = _normalize_columns(head)
        except Exception:
            continue
        cols = [str(c).strip() for c in head.columns]
        cols_lower = {c.lower() for c in cols}

        if "label" in cols_lower and "attack_cat" in cols_lower:
            return "UNSW"
        if "label" in cols_lower and "attack_cat" not in cols_lower:
            return "CICIDS"
        if "label" in cols or "Label" in cols:
            return "CICIDS"

    raise RuntimeError("Could not detect dataset type. Ensure CSVs are in data/raw and readable.")


def load_all_csv(files: List[Path]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = _read_csv(f)
            df = _normalize_columns(df)
            print(f"[+] Loaded {f.name} shape={df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"[!] Skipped {f.name}: {e}")
    if not dfs:
        raise RuntimeError("No CSV files could be loaded from data/raw (check ZIPs/permissions).")
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[+] Concatenated shape: {df_all.shape}")
    return df_all


def clean_unsw(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower(): c for c in df.columns}
    if "label" not in df.columns and "label" in lower_map:
        df = df.rename(columns={lower_map["label"]: "label"})
    if "attack_cat" not in df.columns and "attack_cat" in lower_map:
        df = df.rename(columns={lower_map["attack_cat"]: "attack_cat"})

    drop_like = {"id", "srcip", "sport", "dstip", "dsport", "stime", "ltime"}
    to_drop = [c for c in df.columns if c.lower() in drop_like]
    df = df.drop(columns=to_drop, errors="ignore")

    keep = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "label" not in keep:
        keep.append("label")
    if "attack_cat" in df.columns and "attack_cat" not in keep:
        keep.append("attack_cat")
    df = df[keep]

    num_cols = [c for c in df.columns if c not in {"label", "attack_cat"}]
    if num_cols:
        df = df.dropna(subset=num_cols)

    df["label"] = df["label"].astype(int)
    if "attack_cat" not in df.columns:
        df["attack_cat"] = df["label"].map({0: "BENIGN", 1: "ATTACK"})
    return df


def clean_cicids(df: pd.DataFrame) -> pd.DataFrame:
    # нормалізуємо назви колонок — у CICIDS часто є лідируючі пробіли
    df = _normalize_columns(df)

    # прибираємо явні не-фічі (IP/час/порти), які часто є рядковими
    drop_like = {
        "flow id",
        "source ip",
        "destination ip",
        "timestamp",
        "source port",
        "destination port",
    }
    to_drop = [c for c in df.columns if c.lower().strip() in drop_like]
    df = df.drop(columns=to_drop, errors="ignore")

    # шукаємо колонку 'Label' незалежно від регістру/пробілів
    label_col = None
    for c in df.columns:
        if str(c).strip().lower() == "label":
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("CICIDS CSV must contain 'Label' column")

    # оригінальна категорія та бінарна мітка
    df["attack_cat"] = df[label_col].astype(str)
    df["label"] = (df["attack_cat"].str.upper().str.strip() != "BENIGN").astype(int)

    # тільки числові фічі
    feat_cols = [
        c
        for c in df.columns
        if c not in {label_col, "attack_cat", "label"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feat_cols:
        raise RuntimeError("No numeric features found after cleaning CICIDS CSV.")
    df = df[feat_cols + ["attack_cat", "label"]]
    df = df.dropna(subset=feat_cols)
    return df


def stratified_sample(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain 'label' column")
    rows = min(rows, len(df))

    labels = set(df["label"].unique())
    if labels == {0, 1}:
        per = rows // 2
        df0 = df[df["label"] == 0].sample(n=min(per, (df["label"] == 0).sum()), random_state=42)
        df1 = df[df["label"] == 1].sample(n=min(per, (df["label"] == 1).sum()), random_state=42)
        out = pd.concat([df0, df1], axis=0)
        if len(out) < rows:
            rest = rows - len(out)
            pool = df.drop(index=out.index, errors="ignore")
            if len(pool) > 0:
                out = pd.concat(
                    [out, pool.sample(n=min(rest, len(pool)), random_state=42)], axis=0
                )
        return out.sample(frac=1.0, random_state=42).reset_index(drop=True)

    return df.sample(n=rows, random_state=42).reset_index(drop=True)


def train_test_save(df: pd.DataFrame, base_name: str) -> None:
    X = df.drop(columns=["label"])
    y = df["label"]
    X_tr, X_ts, y_tr, y_ts = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train = X_tr.copy()
    train["label"] = y_tr.values
    test = X_ts.copy()
    test["label"] = y_ts.values

    out_all = PROC_DIR / f"{base_name}_10k.csv"
    out_tr = PROC_DIR / "train.csv"
    out_ts = PROC_DIR / "test.csv"

    df.to_csv(out_all, index=False)
    train.to_csv(out_tr, index=False)
    test.to_csv(out_ts, index=False)

    print(f"[+] Saved: {out_all}")
    print(f"[+] Saved: {out_tr}")
    print(f"[+] Saved: {out_ts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact, clean subset for IDS datasets")
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Кількість рядків у підмножині (default 10000)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["UNSW", "CICIDS"],
        help="Примусовий тип датасету",
    )
    args = parser.parse_args()

    if not RAW_DIR.exists():
        print("[!] data/raw not found. Create it and put original CSVs inside.")
        sys.exit(1)

    files = sorted([p for p in RAW_DIR.glob("*.csv")])
    if not files:
        print("[!] No CSV files found in data/raw. Place dataset CSVs there and rerun.")
        sys.exit(1)

    dtype = args.dtype or detect_dataset(files)
    print(f"[+] Detected dataset type: {dtype}")

    df_all = load_all_csv(files)
    if dtype == "UNSW":
        df_clean = clean_unsw(df_all)
        base = "unsw_subset"
    else:
        df_clean = clean_cicids(df_all)
        base = "cicids_subset"

    print(f"[+] Cleaned shape: {df_clean.shape}")

    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"[+] Dropped {before - len(df_clean)} duplicate rows")

    df_small = stratified_sample(df_clean, args.rows)
    print(f"[+] Subset shape: {df_small.shape}")

    train_test_save(df_small, base)
    print("[✓] Done.")


if __name__ == "__main__":
    main()
