from __future__ import annotations

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # .../ddos-detector
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"


def main() -> None:
    st.set_page_config(page_title="DDoS Detector", layout="wide")
    st.title("🛡️ DDoS Detector — Streamlit UI")

    st.markdown(
        """
Цей інтерфейс дозволяє:
- запускати детекцію на CSV (фічі трафіку);
- переглядати метрики/звіти та порівняння моделей;
- відкривати згенеровані графіки з папки `reports/`.
"""
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("📁 Дані")
        st.write("`data/processed/`")
        if DATA.exists():
            files = sorted([p.name for p in DATA.glob("*.csv")])
            st.write(f"CSV файлів: {len(files)}")
            if files:
                st.code("\n".join(files[:10]))
        else:
            st.warning("Папка data/processed не знайдена.")

    with c2:
        st.subheader("🧠 Моделі")
        st.write("`models/`")
        if MODELS.exists():
            models = sorted([p.name for p in MODELS.glob("*.pkl")])
            st.write(f"PKL файлів: {len(models)}")
            if models:
                st.code("\n".join(models))
        else:
            st.warning("Папка models не знайдена.")

    with c3:
        st.subheader("📊 Звіти")
        st.write("`reports/`")
        if REPORTS.exists():
            imgs = sorted([p.name for p in REPORTS.glob("*.png")])
            st.write(f"PNG файлів: {len(imgs)}")
            if imgs:
                st.code("\n".join(imgs[:10]))
        else:
            st.warning("Папка reports не знайдена.")

    st.info(
        "Сторінки зліва (About / Detection / Metrics / Compare / Visuals) "
        "з’являються автоматично з папки app/pages/."
    )


if __name__ == "__main__":
    main()
