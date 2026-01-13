import sys
from pathlib import Path

# --- bootstrap: додаємо src у PYTHONPATH (щоб імпорти з src працювали) ---
ROOT = Path(__file__).resolve().parents[2]  # .../ddos-detector
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from utils import apply_numeric_sanitizer  # noqa: E402


def _find_label_column(df: pd.DataFrame) -> str:
    for name in ("Label", "label", "target", "y"):
        if name in df.columns:
            return name
    raise ValueError(
        "Не знайдено колонку мітки. Очікую 'Label' або 'label' у CSV."
    )


def _load_bundle(model_path: Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "Файл моделі має бути bundle-словником з ключем 'model'."
        )
    for k in ("feature_names", "sanitizer"):
        if k not in bundle:
            raise ValueError(f"У bundle немає ключа '{k}'.")
    return bundle


def _prepare_x_y(df: pd.DataFrame, feature_names, sanitizer):
    label_col = _find_label_column(df)
    y = df[label_col].astype(int).to_numpy()

    non_numeric = [
        c for c in df.columns
        if c != label_col and not pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df.drop(columns=non_numeric, errors="ignore")
    X = X.drop(columns=[label_col], errors="ignore")

    # вирівнюємо порядок і набір фіч як під час тренування
    X = X.reindex(columns=list(feature_names), fill_value=0)

    # санітизація inf/NaN/викидів згідно sanitizer, збереженого при тренуванні
    X = apply_numeric_sanitizer(X, sanitizer)
    return X, y, label_col


def _plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    fig.tight_layout()
    return fig


def _plot_roc(fpr, tpr, auc_val: float):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _plot_pr(precision, recall, ap: float):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(recall, precision, label=f"PR AUC (AP) = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return None

    imp = np.asarray(model.feature_importances_, dtype=float)
    idx = np.argsort(imp)[::-1][:20]
    top_names = np.asarray(feature_names)[idx]
    top_vals = imp[idx]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top_names))[::-1], top_vals[::-1])
    ax.set_yticks(range(len(top_names))[::-1])
    ax.set_yticklabels(top_names[::-1])
    ax.set_title("Top-20 Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Metrics", layout="wide")
    st.title("📊 Metrics: ROC / PR / Confusion Matrix")

    models_dir = ROOT / "models"
    processed_dir = ROOT / "data" / "processed"

    model_options = {
        "RandomForest (ddos_rf.pkl)": models_dir / "ddos_rf.pkl",
        "SVM (ddos_svm.pkl)": models_dir / "ddos_svm.pkl",
        "XGBoost (ddos_xgb.pkl)": models_dir / "ddos_xgb.pkl",
    }
    available = {k: v for k, v in model_options.items() if v.exists()}
    if not available:
        st.error("Не знайдено моделей у папці models/.")
        return

    model_label = st.selectbox("Оберіть модель", list(available.keys()))
    model_path = available[model_label]

    default_test = processed_dir / "test.csv"
    csv_path = st.text_input(
        "Шлях до CSV (має містити мітку Label/label)",
        value=str(default_test),
    )
    csv_path = Path(csv_path)

    if not csv_path.exists():
        st.warning("CSV не знайдено за вказаним шляхом.")
        return

    bundle = _load_bundle(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    sanitizer = bundle["sanitizer"]

    df = pd.read_csv(csv_path)
    X, y_true, label_col = _prepare_x_y(df, feature_names, sanitizer)

    if not hasattr(model, "predict_proba"):
        st.error(
            "Ця модель не підтримує predict_proba. "
            "Для ROC/PR потрібні ймовірності."
        )
        return

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Метрики
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("F1", f"{f1:.4f}")
    c3.metric("ROC-AUC", f"{roc_auc:.4f}")
    c4.metric("PR-AUC (AP)", f"{pr_auc:.4f}")

    # Графіки ROC/PR
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    left, right = st.columns(2)
    with left:
        st.pyplot(_plot_roc(fpr, tpr, roc_auc), clear_figure=True)
    with right:
        st.pyplot(_plot_pr(prec, rec, pr_auc), clear_figure=True)

    # Confusion Matrix + Report
    cm = confusion_matrix(y_true, y_pred)
    left, right = st.columns([1, 1.2])
    with left:
        st.pyplot(_plot_confusion_matrix(cm), clear_figure=True)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y_true, y_pred, digits=4)
        st.code(rep)

    # Feature importance (якщо є)
    st.subheader("Explainability")
    fig_imp = _plot_feature_importance(model, feature_names)
    if fig_imp is None:
        st.info("Для цієї моделі feature_importances_ недоступний (наприклад, SVM).")
    else:
        st.pyplot(fig_imp, clear_figure=True)

    # Збереження pred-файлу (опційно)
    if st.button("💾 Зберегти predictions поруч із CSV"):
        out = csv_path.with_suffix(".pred.csv")
        pd.DataFrame(
            {
                "proba_attack": y_prob,
                "pred_label": y_pred,
                "true_label": y_true,
            }
        ).to_csv(out, index=False)
        st.success(f"Збережено: {out}")


if __name__ == "__main__":
    main()
