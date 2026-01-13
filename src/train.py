# src/train.py
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from utils import (
    log,
    load_csv,
    save_model,
    class_distribution,
    plot_class_distribution,
    compute_feature_importance,
    plot_feature_importance,
    fit_numeric_sanitizer,
    apply_numeric_sanitizer,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def to_numeric_X(df: pd.DataFrame) -> pd.DataFrame:
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return df.drop(columns=non_numeric, errors="ignore")


def main():
    log("Завантаження train/test...")
    df_train = load_csv(DATA_DIR / "train.csv")
    df_test = load_csv(DATA_DIR / "test.csv")

    X_train = to_numeric_X(df_train.drop(columns=["label"]))
    y_train = df_train["label"].astype(int)
    X_test = to_numeric_X(df_test.drop(columns=["label"]))
    y_test = df_test["label"].astype(int)

    class_distribution(y_train).to_csv(REPORTS_DIR / "class_distribution.csv")
    plot_class_distribution(y_train, REPORTS_DIR / "class_distribution.png")

    log("Санітизація числових фіч...")
    san_stats = fit_numeric_sanitizer(X_train)
    X_train = apply_numeric_sanitizer(X_train, san_stats)
    X_test = apply_numeric_sanitizer(X_test, san_stats)

    log("Навчання RandomForest...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    log("Оцінювання...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    (REPORTS_DIR / "rf_report.txt").write_text(report, encoding="utf-8")

    bundle = {
        "model": model,
        "feature_names": list(X_train.columns),
        "sanitizer": san_stats,
    }
    save_model(bundle, MODEL_DIR / "ddos_rf.pkl")

    imp = compute_feature_importance(model, list(X_train.columns))
    imp.to_csv(REPORTS_DIR / "rf_feature_importance.csv", index=False)
    plot_feature_importance(
        imp, top_k=20, out_path=REPORTS_DIR / "rf_feature_importance_top20.png", title="RandomForest Feature Importance"
    )

    log("✅ Готово!")


if __name__ == "__main__":
    main()
