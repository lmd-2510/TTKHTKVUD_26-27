"""Create a participant-level split from completed per-participant epoch files."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
from download_data import REPO, SUBJECTS
from preprocess import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data/processed/channels16/after_ica")
    parser.add_argument("--output", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = {}
    names = times = None
    for subject in SUBJECTS:
        path = args.data_dir / f"{subject}.npz"
        with np.load(path, allow_pickle=False) as data:
            X, y = data["X"], data["y"]
            if X.ndim != 3 or X.shape[0] != len(y) or not np.isfinite(X).all():
                raise ValueError(f"{subject}: invalid epoch array")
            if set(np.unique(y)) != {0, 1}:
                raise ValueError(f"{subject}: both labels required")
            if str(data["subject"]) != subject:
                raise ValueError(f"{subject}: participant identity mismatch")
            current_names = data["channel_names"].tolist()
            if names is None:
                names, times = current_names, data["times_s"].copy()
            if names != current_names or not np.array_equal(times, data["times_s"]):
                raise ValueError(f"{subject}: channel order or time grid differs")
            records[subject] = {"file": str(path.resolve()), "standard": int((y == 0).sum()),
                                "target": int((y == 1).sum())}
    subjects = list(SUBJECTS)
    random.Random(args.seed).shuffle(subjects)
    result = {"status": "prepared_with_ica_review_status_in_quality_reports",
              "seed": args.seed, "train": sorted(subjects[:34]), "validation": [],
              "test": sorted(subjects[34:]), "participants": records}
    if args.output.exists():
        old = json.loads(args.output.read_text("utf-8"))
        if old.get("train") and (old["train"] != result["train"] or old["test"] != result["test"]):
            raise ValueError("Existing split differs. Choose another --output to create a new split.")
    write_json(args.output, result)
    print(f"Saved {args.output}: 34 train / 8 test participants.")
    print("ICA review status remains in each participant quality report.")


if __name__ == "__main__":
    main()
