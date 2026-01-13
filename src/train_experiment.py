# src/train_experiment.py
from pathlib import Path
import argparse
import json
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.svm import SVC

# XGBoost (опційно)
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from utils import (
    log,
    load_csv,
    fit_numeric_sanitizer,
    apply_numeric_sanitizer,
    compute_feature_importance,
    plot_feature_importance,
    class_distribution,
    plot_class_distribution,
)

DATA = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports/experiments")
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    nonnum = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return df.drop(columns=nonnum, errors="ignore")


def build_model(kind: str):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    if kind == "svm":
        # без окремого скейлера, бо санітизація + кліп вже стабілізують шкали
        return SVC(
            C=3.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    if kind == "xgb":
        assert HAS_XGB, "xgboost не встановлено: pip install xgboost"
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
        )
    raise ValueError("unknown model kind")


def get_prob(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # для SVM/інших з decision_function — нормалізуємо до [0,1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    # fallback
    return model.predict(X).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train & evaluate RF/SVM/XGB on same pipeline"
    )
    ap.add_argument("--model", choices=["rf", "svm", "xgb"], required=True)
    args = ap.parse_args()
    kind = args.model

    out_dir = REPORTS / kind
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) дані
    log("load train/test")
    df_tr = load_csv(DATA / "train.csv")
    df_ts = load_csv(DATA / "test.csv")

    y_tr = df_tr["label"].astype(int)
    y_ts = df_ts["label"].astype(int)
    X_tr = to_numeric(df_tr.drop(columns=["label"]))
    X_ts = to_numeric(df_ts.drop(columns=["label"]))

    # коротка діагностика класів
    class_distribution(y_tr).to_csv(out_dir / "train_class_dist.csv")
    plot_class_distribution(y_tr, out_dir / "train_class_dist.png")

    # ===== 2) санітизація (fit на train, apply до train/test)
    log("fit/apply sanitizer")
    san = fit_numeric_sanitizer(X_tr)
    X_tr = apply_numeric_sanitizer(X_tr, san)
    X_ts = apply_numeric_sanitizer(X_ts, san)

    # ===== 3) тренування
    log(f"train {kind}")
    mdl = build_model(kind)
    t0 = time.time()
    mdl.fit(X_tr, y_tr)
    train_sec = time.time() - t0

    # ===== 4) оцінка
    p = get_prob(mdl, X_ts)
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "model": kind,
        "accuracy": float(accuracy_score(y_ts, yhat)),
        "f1": float(f1_score(y_ts, yhat)),
        "roc_auc": float(roc_auc_score(y_ts, p)),
        "pr_auc": float(average_precision_score(y_ts, p)),
        "train_time_sec": round(train_sec, 3),
        "n_features": int(X_tr.shape[1]),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_ts.shape[0]),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_ts, yhat, digits=4), encoding="utf-8"
    )

    # ===== 5) важливість ознак (де можливо)
    try:
        imp = compute_feature_importance(mdl, list(X_tr.columns))
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
        plot_feature_importance(
            imp,
            top_k=20,
            out_path=out_dir / "feature_importance_top20.png",
            title=f"{kind.upper()} — Feature Importance (Top 20)",
        )
    except Exception:
        pass  # для SVM може не бути feature_importances_

    # ===== 6) збереження моделі (для інференсу)
    bundle = {"model": mdl, "feature_names": list(X_tr.columns), "sanitizer": san}
    import joblib  # локальний імпорт, щоб не ловити F401, якщо не використовуєш тут

    joblib.dump(bundle, MODELS / f"ddos_{kind}.pkl")
    log(f"done → {out_dir}")


if __name__ == "__main__":
    main()
