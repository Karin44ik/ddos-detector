# src/utils.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


# ============ ЛОГІКА ============

def log(msg: str) -> None:
    """Вивід повідомлення з часом."""
    t = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{t} {msg}")


# ============ IO ============

def load_csv(path: Path) -> pd.DataFrame:
    log(f"Завантаження: {path.name}")
    df = pd.read_csv(path)
    log(f"→ {df.shape}")
    return df


def save_model(bundle: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    log(f"✅ Модель збережено: {path}")


# ============ АНАЛІТИКА КЛАСІВ ============

def class_distribution(y: pd.Series) -> pd.DataFrame:
    dist = y.value_counts(normalize=False).sort_index()
    df = pd.DataFrame({"count": dist, "percent": 100 * dist / len(y)})
    log(f"Class distribution:\n{df}")
    return df


def plot_class_distribution(y: pd.Series, out_path: Path) -> None:
    dist = y.value_counts()
    dist.plot(kind="bar")
    plt.title("Розподіл класів (Train)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log(f"Class distribution plot → {out_path}")


# ============ FEATURE IMPORTANCE ============

def compute_feature_importance(model, feature_names) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df
    raise ValueError("Модель не має атрибуту feature_importances_")


def plot_feature_importance(df: pd.DataFrame, top_k: int, out_path: Path, title: str) -> None:
    top = df.head(top_k)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log(f"Feature importance plot → {out_path}")


# ============ PERMUTATION IMPORTANCE ============

def permutation_importance_df(model, X, y, feature_names):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    df = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def plot_permutation_importance(df: pd.DataFrame, out_path: Path, top_k: int = 20) -> None:
    top = df.head(top_k)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title("Permutation Importance (Top 20)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    log(f"Permutation importance plot → {out_path}")


# ============ SANITIZER ============

def fit_numeric_sanitizer(X: pd.DataFrame) -> Dict[str, pd.Series]:
    """Обчислює статистики для очищення числових фіч."""
    Xn = X.replace([np.inf, -np.inf], np.nan)
    meds = Xn.median(numeric_only=True)
    low = Xn.quantile(0.001, numeric_only=True)
    high = Xn.quantile(0.999, numeric_only=True)
    return {"medians": meds, "low": low, "high": high}


def apply_numeric_sanitizer(X: pd.DataFrame, stats: Dict[str, pd.Series]) -> pd.DataFrame:
    """Застосовує очищення числових фіч."""
    Xc = X.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.fillna(stats["medians"])
    lo = stats["low"].reindex(Xc.columns, fill_value=None)
    hi = stats["high"].reindex(Xc.columns, fill_value=None)
    Xc = Xc.clip(lower=lo, upper=hi, axis=1)
    return Xc.astype("float32")
