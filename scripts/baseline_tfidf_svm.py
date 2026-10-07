"""Baseline model: TF-IDF on the complaint text + scaled vitals + one-hot sex, LinearSVC.

Five-fold out-of-fold evaluation over the patient-grouped folds from
`poncol_data`, so every row is predicted exactly once by a model that never saw
that patient. Writes per-row probabilities and metrics.

    python scripts/baseline_tfidf_svm.py --csv poncol.csv --out runs/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from poncol_data import CATEGORICAL_FIELDS, NUMERIC_FIELDS, RANDOM_STATE, TEXT_FIELD, build_dataset  # noqa: E402


def decision_scores(pipeline: Pipeline, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Logits per class, clipped for binary-safe log-loss. LinearSVC has no predict_proba."""
    scores = pipeline.decision_function(X)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return np.clip(scores, -30.0, 30.0) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="poncol.csv")
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--coverage", type=float, default=0.75)
    parser.add_argument("--sample", type=int, default=0, help="0 keeps every row; must match the Jeff run")
    parser.add_argument("--group-field", default="patient", choices=["patient", "village", "text"])
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    dataset = build_dataset(
        args.csv, n_splits=args.splits, coverage=args.coverage,
        sample=args.sample, group_field=args.group_field, seed=args.seed,
    )
    classes = np.array(dataset.labels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fold_metrics: list[dict] = []
    start = time.time()

    for index in range(dataset.n_splits):
        X_train, y_train, X_val, y_val = dataset.fold(index)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FIELDS),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FIELDS),
                ("txt", TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2)), TEXT_FIELD),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LinearSVC(class_weight="balanced", random_state=args.seed, max_iter=5000, dual=False)),
            ]
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_val)
        scores = decision_scores(pipeline, X_val, classes)
        accuracy = float(accuracy_score(y_val, prediction))
        macro_f1 = float(f1_score(y_val, prediction, average="macro", zero_division=0))
        fold_metrics.append({"fold": index, "rows": int(len(y_val)), "accuracy": accuracy, "macro_f1": macro_f1})
        print(f"fold {index}: accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}  ({len(y_val)} rows)", flush=True)

        for position, (_, row) in enumerate(X_val.iterrows()):
            rows.append(
                {
                    "row_id": int(row.name) if isinstance(row.name, (int, np.integer)) else position,
                    "label": y_val.iloc[position],
                    "prediction": prediction[position],
                    "fold": index,
                    "scores": [float(value) for value in scores[position]],
                    "probs": [float(value) for value in np.exp(scores[position]) / np.exp(scores[position]).sum()],
                }
            )

    predictions = pd.DataFrame(rows)
    metrics = {
        "model": "tfidf_svm",
        "accuracy": float(accuracy_score(predictions["label"], predictions["prediction"])),
        "macro_f1": float(f1_score(predictions["label"], predictions["prediction"], average="macro", zero_division=0)),
        "log_loss": float(log_loss(predictions["label"], np.vstack(predictions["probs"]), labels=classes)),
        "seconds": round(time.time() - start, 1),
        "folds": fold_metrics,
        "dataset": dataset.report,
    }
    (out / "predictions.csv").write_text(
        predictions.drop(columns=["scores"]).to_csv(index=False)
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.save(out / "oof_scores.npy", np.vstack(predictions["scores"].to_numpy()))

    print("\n=== TF-IDF + LinearSVC, 5-fold out-of-fold ===")
    print(f"accuracy {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}  log-loss {metrics['log_loss']:.4f}")
    print(classification_report(predictions["label"], predictions["prediction"], zero_division=0))
    print(f"wrote {out}/metrics.json and {out}/predictions.csv")


if __name__ == "__main__":
    main()