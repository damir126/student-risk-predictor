"""Train, compare and evaluate models, then save the best one.

    python -m riskml.train                 # synthetic data
    python -m riskml.train --data my.csv   # your own data
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET, load

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing lives inside the pipeline, so the API applies exactly the same
    transformations as training (no train/serve skew, no leakage during CV)."""
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


CANDIDATES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, class_weight="balanced", random_state=0),
}


def best_threshold(y_true, proba) -> float:
    """Threshold that maximizes F1 (the default 0.5 is rarely optimal for imbalanced data)."""
    precision, recall, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def train(data_path: str | None = None, out_model: Path = MODEL_PATH, reports: Path = REPORTS,
          n: int = 5000, verbose: bool = True) -> dict:
    df = load(data_path, n=n)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    say = print if verbose else (lambda *a, **k: None)
    say(f"{len(df)} students, dropout rate {y.mean():.1%}; train {len(X_train)}, test {len(X_test)}")

    # ---- 1. model selection with 5-fold cross-validation on the training set only ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    comparison = {}
    for name, make in CANDIDATES.items():
        t0 = time.perf_counter()
        scores = cross_validate(build_pipeline(make()), X_train, y_train, cv=cv,
                                scoring=["roc_auc", "f1", "average_precision"])
        comparison[name] = {
            "roc_auc": round(scores["test_roc_auc"].mean(), 4),
            "roc_auc_std": round(scores["test_roc_auc"].std(), 4),
            "f1": round(scores["test_f1"].mean(), 4),
            "pr_auc": round(scores["test_average_precision"].mean(), 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        say(f"  {name:<20} ROC-AUC {comparison[name]['roc_auc']:.3f} +/- {comparison[name]['roc_auc_std']:.3f}"
            f"   F1 {comparison[name]['f1']:.3f}")
    best_name = max(comparison, key=lambda k: comparison[k]["roc_auc"])
    say(f"best model: {best_name}")

    # ---- 2. fit the best model, pick a threshold on a validation split (not on the test set) ----
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=1)
    threshold = best_threshold(y_val, build_pipeline(CANDIDATES[best_name]()).fit(X_fit, y_fit).predict_proba(X_val)[:, 1])
    model = build_pipeline(CANDIDATES[best_name]()).fit(X_train, y_train)

    # ---- 3. final, honest evaluation on the untouched test set ----
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    test_metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "threshold": round(threshold, 4),
        "report": classification_report(y_test, pred, target_names=["stays", "dropout"], output_dict=True),
    }
    say(classification_report(y_test, pred, target_names=["stays", "dropout"]))

    # ---- 4. explainability: permutation importance on the test set ----
    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    importance = dict(sorted(
        {col: round(float(v), 4) for col, v in zip(X_test.columns, imp.importances_mean)}.items(),
        key=lambda kv: -kv[1]))

    # ---- 5. save artifacts ----
    out_model.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": threshold, "model_name": best_name,
                 "features": NUMERIC + CATEGORICAL, "metrics": {k: v for k, v in test_metrics.items() if k != "report"}},
                out_model)
    summary = {"model": best_name, "cv_comparison": comparison, "test": test_metrics, "importance": importance}
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2))
    _plots(model, X_test, y_test, pred, importance, reports)
    say(f"model saved to {out_model}, reports in {reports}")
    return summary


def _plots(model, X_test, y_test, pred, importance, reports: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k:", label="random")
    axes[0].set_title("ROC curve (test set)")
    axes[0].legend(loc="lower right")
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["stays", "dropout"], ax=axes[1],
                                            colorbar=False, cmap="Blues")
    axes[1].set_title("Confusion matrix")
    top = list(importance.items())[:10][::-1]
    axes[2].barh([k for k, _ in top], [v for _, v in top], color="#4f46e5")
    axes[2].set_title("Permutation importance (drop in ROC-AUC)")
    fig.tight_layout()
    fig.savefig(reports / "evaluation.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the dropout risk model")
    p.add_argument("--data", help="CSV file (default: generate synthetic data)")
    p.add_argument("--n", type=int, default=5000, help="synthetic dataset size")
    args = p.parse_args()
    train(args.data, n=args.n)


if __name__ == "__main__":
    main()
