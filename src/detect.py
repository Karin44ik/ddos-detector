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

    # Вирівнюємо порядок фіч під той, що був при тренуванні
    X = X.reindex(columns=feature_names, fill_value=0)

    # Санітизація числових значень (NaN, inf, викиди тощо)
    X = apply_numeric_sanitizer(X, sanitizer)
    return X


def main():
    parser = argparse.ArgumentParser(
        description="Predict DoS/DDoS labels for new CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Шлях до CSV з мережевими ознаками.",
    )
    parser.add_argument(
        "--bundle",
        default="models/ddos_rf.pkl",
        help="Joblib-файл із збереженою моделлю (за замовчуванням "
             "models/ddos_rf.pkl).",
    )
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    csv_path = Path(args.input)

    # Завантажуємо модель і супровідну інформацію
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    sanitizer = bundle["sanitizer"]

    # Готуємо фічі
    X = load_features(csv_path, feature_names, sanitizer)

    # Прогноз ймовірності атаки та фінальної мітки
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Зберігаємо результат поруч з вхідним файлом
    out_path = csv_path.with_suffix(".pred.csv")
    pd.DataFrame(
        {
            "proba_attack": y_prob,
            "pred_label": y_pred,
        }
    ).to_csv(out_path, index=False)

    print(f"[✓] Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
