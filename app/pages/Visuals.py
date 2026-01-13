from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# --- bootstrap: додаємо src у PYTHONPATH (щоб імпорти з src працювали) ---
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402


def list_pngs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.png") if p.is_file()])


def main() -> None:
    st.set_page_config(page_title="Visuals", layout="wide")
    st.title("🖼️ Visuals: перегляд збережених графіків (reports/)")

    reports_dir = ROOT / "reports"
    exp_dir = reports_dir / "experiments"

    st.caption(f"Папка звітів: {reports_dir}")

    if not reports_dir.exists():
        st.error("Папка reports/ не знайдена. Спочатку запусти train/compare.")
        return

    # 1) Швидкий блок: порівняльні графіки з compare.py
    st.subheader("1) Порівняння моделей (compare.py)")

    cmp_files = [
        reports_dir / "cmp_roc_auc.png",
        reports_dir / "cmp_pr_auc.png",
        reports_dir / "cmp_f1.png",
        reports_dir / "cmp_train_time_sec.png",
    ]
    existing_cmp = [p for p in cmp_files if p.exists()]

    if not existing_cmp:
        st.info("Порівняльні графіки не знайдені. Запусти: python src/compare.py")
    else:
        cols = st.columns(2)
        for i, p in enumerate(existing_cmp):
            with cols[i % 2]:
                st.markdown(f"**{p.name}**")
                st.image(str(p), use_container_width=True)

    # 2) Експерименти по моделях (train_experiment.py)
    st.subheader("2) Експерименти (reports/experiments/<model>/)")

    if not exp_dir.exists():
        st.info("Папка experiments/ не знайдена. Запусти train_experiment.py.")
    else:
        models = []
        for name in ("rf", "svm", "xgb"):
            d = exp_dir / name
            if d.exists():
                models.append(name)

        if not models:
            st.info("Немає підпапок rf/svm/xgb у reports/experiments/.")
        else:
            selected = st.selectbox("Оберіть модель", models)

            model_dir = exp_dir / selected
            imgs = list_pngs(model_dir)

            if not imgs:
                st.warning(f"У {model_dir} немає .png файлів.")
            else:
                st.caption(f"Знайдено {len(imgs)} графіків у {model_dir}")
                for p in imgs:
                    st.markdown(f"**{selected} / {p.name}**")
                    st.image(str(p), use_container_width=True)

    # 3) Галерея всіх png з reports/
    st.subheader("3) Інші графіки в reports/")
    other_pngs = [
        p for p in list_pngs(reports_dir)
        if p.name not in {x.name for x in cmp_files}
    ]

    if not other_pngs:
        st.info("Інших png у reports/ немає.")
    else:
        # щоб не було дуже довго — зробимо мультиселект
        names = [p.name for p in other_pngs]
        chosen = st.multiselect("Показати файли", options=names, default=names[:3])

        for p in other_pngs:
            if p.name in chosen:
                st.markdown(f"**{p.name}**")
                st.image(str(p), use_container_width=True)


if __name__ == "__main__":
    main()
