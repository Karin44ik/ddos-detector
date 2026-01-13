# src/detect_all.py

from pathlib import Path
import argparse

import joblib
import pandas as pd

from utils import apply_numeric_sanitizer


def load_features(csv_path: Path, feature_names, sanitizer):
    """Завантажити CSV і підготувати його під модель."""
    df = pd.read_csv(csv_path)

    # Викидаємо нечислові колонки
    non_numeric = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    X = df.drop(columns=non_numeric, errors="ignore")

    # Вирівнюємо порядок ознак під тренування
    X = X.reindex(columns=feature_names, fill_value=0)

    # Санітизація числових значень (NaN, inf, викиди тощо)
    X = apply_numeric_sanitizer(X, sanitizer)
    return X


def run_detection_for_file(csv_path: Path, model, feature_names, sanitizer):
    """Запустити детекцію для одного CSV-файла."""
    print(f"[+] Обробка файлу: {csv_path}")

    X = load_features(csv_path, feature_names, sanitizer)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out_path = csv_path.with_suffix(".pred.csv")
    pd.DataFrame(
        {"proba_attack": y_prob, "pred_label": y_pred}
    ).to_csv(out_path, index=False)

    print(f"    → результати збережено в: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Запуск детекції DoS/DDoS для всіх CSV у data/processed/"
        )
    )
    parser.add_argument(
        "--bundle",
        default="models/ddos_rf.pkl",
        help=(
            "Joblib-файл із моделлю та метаданими "
            "(за замовчуванням models/ddos_rf.pkl)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Каталог з підготовленими CSV (default: data/processed).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    bundle_path = Path(args.bundle)

    if not data_dir.exists():
        raise SystemExit(f"[!] Каталог не знайдено: {data_dir}")

    if not bundle_path.exists():
        raise SystemExit(f"[!] Модель не знайдено: {bundle_path}")

    # Завантажуємо модель і службову інформацію
    print(f"[+] Завантаження моделі з {bundle_path}")
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    sanitizer = bundle["sanitizer"]

    # Шукаємо всі CSV, крім уже предиктованих (*.pred.csv)
    csv_files = sorted(
        p for p in data_dir.glob("*.csv")
        if not p.name.endswith(".pred.csv")
    )

    if not csv_files:
        print(f"[!] У каталозі {data_dir} не знайдено CSV-файлів.")
        return

    print(f"[+] Знайдено {len(csv_files)} файлів для обробки:")
    for p in csv_files:
        print(f"    - {p.name}")

    # Проганяємо модель по кожному файлу
    for csv_path in csv_files:
        run_detection_for_file(csv_path, model, feature_names, sanitizer)

    print("[✓] Детекція завершена для всіх файлів.")


if __name__ == "__main__":
    main()
