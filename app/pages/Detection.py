from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# === Додаємо src у PYTHONPATH (щоб можна було імпортувати utils із src/) ===
ROOT = Path(__file__).resolve().parents[2]  # .../ddos-detector
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import apply_numeric_sanitizer  # noqa: E402


MODEL_PATH = ROOT / "models" / "ddos_rf.pkl"


def load_features(csv_path: Path, feature_names, sanitizer) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # залишаємо тільки числові колонки
    non_numeric = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    x = df.drop(columns=non_numeric, errors="ignore")

    # порядок фіч як у тренуванні
    x = x.reindex(columns=feature_names, fill_value=0)

    # санітизація (NaN/inf/занадто великі значення)
    x = apply_numeric_sanitizer(x, sanitizer)
    return x


@st.cache_resource
def load_model_bundle(model_path: Path):
    return joblib.load(model_path)


def main() -> None:
    st.set_page_config(page_title="DoS/DDoS Detection", layout="wide")
    st.title("🛡️ Виявлення DoS/DDoS атак (ML)")

    if not MODEL_PATH.exists():
        st.error(f"Не знайдено модель: {MODEL_PATH}")
        st.stop()

    bundle = load_model_bundle(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    sanitizer = bundle.get("sanitizer")

    st.info("Завантаж CSV з фічами (наприклад: train.csv / test.csv / cicids_subset_10k.csv).")

    uploaded = st.file_uploader("CSV файл", type=["csv"])

    threshold = st.slider("Поріг атаки (threshold)", 0.0, 1.0, 0.5, 0.01)

    if uploaded is None:
        st.stop()

    # Streamlit дає file-like обʼєкт → читаємо напряму в pandas
    df_raw = pd.read_csv(uploaded)

    st.subheader("Попередній перегляд даних")
    st.dataframe(df_raw.head(20), use_container_width=True)

    # тимчасово збережемо у файл (щоб зручно передати у load_features)
    tmp_path = ROOT / "data" / "processed" / "_tmp_uploaded.csv"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(tmp_path, index=False)

    x = load_features(tmp_path, feature_names, sanitizer)

    st.subheader("Підготовлені ознаки для моделі")
    st.write(f"Shape: {x.shape}")
    st.dataframe(x.head(10), use_container_width=True)

    if st.button("🔍 Запустити детекцію"):
        proba = model.predict_proba(x)[:, 1]
        pred = (proba >= threshold).astype(int)

        result = pd.DataFrame(
            {
                "proba_attack": proba,
                "pred_label": pred,
            }
        )

        st.subheader("Результат")
        st.write(
            f"Кількість рядків: {len(result)} | "
            f"Attack=1: {int((pred == 1).sum())} | "
            f"Normal=0: {int((pred == 0).sum())}"
        )
        st.dataframe(result.head(50), use_container_width=True)

        out_name = Path(uploaded.name).with_suffix(".pred.csv").name
        out_path = ROOT / "data" / "processed" / out_name
        result.to_csv(out_path, index=False)

        st.success(f"✅ Збережено: {out_path}")

        st.download_button(
            label="⬇️ Завантажити pred.csv",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name=out_name,
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
