# src/evaluate.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report
from utils import (
    log,
    permutation_importance_df,
    plot_permutation_importance,
    apply_numeric_sanitizer,
)

DATA_DIR = Path("data/processed")
MODEL_PATH = Path("models/ddos_rf.pkl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_xy(csv_path: Path, feature_names=None, sanitizer=None):
    df = pd.read_csv(csv_path)
    y = df["label"].astype(int)
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    X = df.drop(columns=["label"] + non_numeric, errors="ignore")
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)
    if sanitizer is not None:
        X = apply_numeric_sanitizer(X, sanitizer)
    return X, y


def main():
    log("Завантаження моделі...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["feature_names"]
    sanitizer = bundle.get("sanitizer")

    log("Оцінювання на test.csv...")
    X_test, y_test = load_xy(DATA_DIR / "test.csv", feature_names=features, sanitizer=sanitizer)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    (REPORTS_DIR / "eval_report.txt").write_text(report, encoding="utf-8")
    print(report)

    log("Обчислення permutation importance...")
    imp = permutation_importance_df(model, X_test, y_test, features)
    imp.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_permutation_importance(imp, REPORTS_DIR / "permutation_importance.png")

    log("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
