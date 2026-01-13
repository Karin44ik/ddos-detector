from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def main() -> None:
    st.set_page_config(page_title="Compare", layout="wide")
    st.title("📊 Compare models")

    csv_path = REPORTS / "models_comparison.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"Немає: {csv_path}")

    st.subheader("Графіки порівняння")
    imgs = [
        REPORTS / "cmp_f1.png",
        REPORTS / "cmp_roc_auc.png",
        REPORTS / "cmp_pr_auc.png",
        REPORTS / "cmp_train_time_sec.png",
    ]

    any_img = False
    for p in imgs:
        if p.exists():
            any_img = True
            st.image(str(p), caption=p.name, use_container_width=True)

    if not any_img:
        st.info("Графіків cmp_*.png поки немає в папці reports/.")


if __name__ == "__main__":
    main()
