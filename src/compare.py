# src/compare.py
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

EXP = Path("reports/experiments")
OUT = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics() -> pd.DataFrame:
    rows = []
    for sub in sorted(EXP.glob("*")):
        mfile = sub / "metrics.json"
        if not mfile.exists():
            continue
        data = json.loads(mfile.read_text(encoding="utf-8"))
        # страховка: якщо в JSON нема "model" — беремо ім’я папки
        data.setdefault("model", sub.name)
        rows.append(data)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_index()


def bar(df: pd.DataFrame, col: str, title: str, fname: str) -> None:
    ax = df[col].plot(kind="bar", rot=0, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title(title)
    ax.set_ylabel(col)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=140)
    plt.close()


def main() -> None:
    df = load_metrics()
    if df.empty:
        print("No metrics found in reports/experiments/*/metrics.json")
        return

    df_rounded = df[["accuracy", "f1", "roc_auc", "pr_auc", "train_time_sec"]].round(4)
    df_rounded.to_csv(OUT / "models_comparison.csv")
    print(df_rounded)

    bar(df_rounded, "f1", "F1-score (higher is better)", "cmp_f1.png")
    bar(df_rounded, "roc_auc", "ROC AUC", "cmp_roc_auc.png")
    bar(df_rounded, "pr_auc", "PR AUC", "cmp_pr_auc.png")
    bar(df_rounded, "train_time_sec", "Training time (sec, lower is better)", "cmp_train_time_sec.png")
    print("Saved comparison CSV and plots to reports/")


if __name__ == "__main__":
    main()
