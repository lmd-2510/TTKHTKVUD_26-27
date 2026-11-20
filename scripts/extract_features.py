"""Average EEG into 20 time windows per channel for classical ML."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from download_data import REPO, atomic_output


def load_split(path):
    split = json.loads(path.read_text("utf-8"))
    train, test = split["train"], split["test"]
    if (not train or not test or len(set(train + test)) != len(train + test)
            or split.get("validation")):
        raise ValueError("Expected disjoint nonempty train/test participant lists and empty validation.")
    return split


def window_features(X, times, window_ms=50):
    width = window_ms / 1000
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Window size must be finite and positive.')
    count = round(1 / width)
    if not np.isclose(count * width, 1):
        raise ValueError("Window size must divide 1000 ms.")
    if times.ndim != 1 or len(times) != X.shape[2] or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid epoch time grid.")
    if not np.isclose(times[0], 0) or not np.isclose(times[-1], 1):
        raise ValueError("Expected epochs spanning 0 through 1 second.")
    bins = []
    for index in range(count):
        start, end = index * width, (index + 1) * width
        # Internal boundaries belong to the following bin; include the final 1 s sample.
        mask = (times >= start - 1e-10) & (times < end - 1e-10)
        if index == count - 1:
            mask = (times >= start - 1e-10) & (times <= end + 1e-10)
        if not mask.any():
            raise ValueError("A feature window contains no samples.")
        bins.append(X[:, :, mask].mean(axis=2) * 1e6)
    return np.stack(bins, axis=2).reshape(len(X), -1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--output", type=Path, default=REPO / "data/processed/channels16/features50ms.npz")
    parser.add_argument("--window-ms", type=float, default=50)
    args = parser.parse_args()
    split = load_split(args.splits)
    arrays, labels, groups = [], [], []
    expected_names = expected_times = None
    for subject in split["train"] + split["test"]:
        record = split["participants"][subject]
        with np.load(record["file"], allow_pickle=False) as data:
            X, y = data["X"], data["y"]
            if X.ndim != 3 or y.shape != (len(X),) or not np.isfinite(X).all():
                raise ValueError(f"{subject}: invalid data shape or nonfinite signal.")
            if str(data["subject"]) != subject or set(np.unique(y)) != {0, 1}:
                raise ValueError(f"{subject}: identity or label mismatch.")
            if any(int((y == code).sum()) != record[label]
                   for code, label in enumerate(("standard", "target"))):
                raise ValueError(f"{subject}: counts differ from split inventory.")
            names, times = data["channel_names"].tolist(), data["times_s"]
            if expected_names is None:
                expected_names, expected_times = names, times.copy()
            if names != expected_names or not np.array_equal(times, expected_times):
                raise ValueError(f"{subject}: channel order or time grid differs.")
            arrays.append(window_features(X, times, args.window_ms))
            labels.append(y)
            groups.append(np.full(len(y), subject))
            print(f"{subject}: {len(y)} samples", flush=True)
    digest = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    with atomic_output(args.output) as output:
        np.savez(output, X=np.concatenate(arrays), y=np.concatenate(labels),
                 groups=np.concatenate(groups), channel_names=np.array(expected_names),
                 window_ms=args.window_ms, units="uV", split_sha256=digest)
    print(f"Features saved: {args.output} | dimensions: {arrays[0].shape[1]}")


if __name__ == "__main__":
    main()
