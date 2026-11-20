"""Fit majority baseline, shrinkage LDA and RBF SVM, then evaluate held-out participants."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from download_data import REPO, atomic_output
from extract_features import load_split


def make_models():
    """Construct the fixed ML estimators used in the initial experiment."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "shrinkage_lda": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])),
        "rbf_svm": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=1024)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--features", type=Path, default=REPO / "data/processed/channels16/features50ms.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/metrics/ml")
    parser.add_argument("--models", nargs="+", choices=["majority_baseline", "shrinkage_lda", "rbf_svm"],
                        default=["majority_baseline", "shrinkage_lda", "rbf_svm"])
    args = parser.parse_args()
    split = load_split(args.splits)
    with np.load(args.features, allow_pickle=False) as data:
        if str(data["split_sha256"]) != hashlib.sha256(args.splits.read_bytes()).hexdigest():
            raise ValueError("Split changed since feature extraction. Extract features again.")
        X, y, groups = data["X"], data["y"], data["groups"]
    if X.ndim != 2 or not np.isfinite(X).all() or len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Invalid feature arrays.")
    if set(groups) != set(split["train"] + split["test"]):
        raise ValueError("Feature participants differ from split.")
    train = np.isin(groups, split["train"])
    test = np.isin(groups, split["test"])
    if (train & test).any() or set(np.unique(y[train])) != {0, 1} or set(np.unique(y[test])) != {0, 1}:
        raise ValueError("Invalid train/test data.")
    models = make_models()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in dict.fromkeys(args.models):
        started = time.monotonic()
        model = models[name]
        print(f"Training {name}: {train.sum()} samples", flush=True)
        model.fit(X[train], y[train])
        predicted = model.predict(X[test])
        result = {
            "model": name, "macro_f1": float(f1_score(y[test], predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y[test], predicted, labels=[0, 1]).tolist(),
            "matrix_rows": "true", "matrix_columns": "predicted", "label_order": ["standard", "target"],
            "train_participants": split["train"], "test_participants": split["test"],
            "train_samples": int(train.sum()), "test_samples": int(test.sum()),
            "feature_count": X.shape[1], "elapsed_seconds": time.monotonic() - started,
            "sklearn_version": sklearn.__version__, "parameters": str(model),
            "selection": "Fixed initial parameters; no tuning on held-out test participants.",
        }
        with atomic_output(args.output_dir / f"{name}.json") as output:
            output.write((json.dumps(result, indent=2) + "\n").encode())
        print(f"{name}: macro-F1={result['macro_f1']:.4f}\n{np.array(result['confusion_matrix'])}", flush=True)


if __name__ == "__main__":
    main()
