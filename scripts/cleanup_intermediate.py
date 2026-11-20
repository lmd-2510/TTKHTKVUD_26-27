"""Remove obsolete pilot data after validating final participant arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from download_data import REPO


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.digest()


def plan_cleanup(root):
    split = json.loads((root / "configs/splits.json").read_text("utf-8"))
    subjects = split["train"] + split["test"]
    if len(subjects) != 42 or len(set(subjects)) != 42:
        raise ValueError("Expected 42 distinct train/test participants.")
    final = root / "data/processed/channels16/after_ica"
    protected = set()
    for subject in subjects:
        record = split["participants"][subject]
        path = Path(record["file"]).resolve()
        if path != (final / f"{subject}.npz").resolve():
            raise ValueError(f"Unexpected current training file: {path}")
        with np.load(path, allow_pickle=False) as data:
            if (str(data["subject"]) != subject or data["X"].ndim != 3
                    or data["X"].shape[1:] != (16, 129)
                    or data["y"].shape != (len(data["X"]),)
                    or not np.isfinite(data["X"]).all()
                    or set(np.unique(data["y"])) != {0, 1}):
                raise ValueError(f"Invalid final data: {path}")
            if any(int((data["y"] == code).sum()) != record[label]
                   for code, label in enumerate(("standard", "target"))):
                raise ValueError(f"Final label counts differ from split: {subject}")
        protected.add(path)
    candidates = []
    for subject in subjects:
        for parent in ("data/processed", "data/processed/channels16"):
            path = root / parent / f"{subject}.npz"
            if path.is_file():
                candidates.append(path)
    pilot = root / "data/processed/ica_cleaned"
    canonical = root / "data/processed/ica_cleaned_all"
    # Delete only pilot files proven identical to their full-cohort counterparts.
    if pilot.exists():
        for path in sorted(pilot.glob("sub-*/*")):
            if path.is_file():
                counterpart = canonical / path.relative_to(pilot)
                if counterpart.is_file() and path.stat().st_size == counterpart.stat().st_size:
                    if digest(path) == digest(counterpart):
                        candidates.append(path)
    for path in candidates:
        resolved = path.resolve()
        if path.is_symlink() or root not in resolved.parents or resolved in protected:
            raise ValueError(f"Unexpected cleanup path: {path}")
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete the listed obsolete files.")
    args = parser.parse_args()
    root = REPO.resolve()
    candidates = plan_cleanup(root)
    size = sum(path.stat().st_size for path in candidates)
    for path in candidates:
        print(path.relative_to(root))
    print(f"{len(candidates)} obsolete files | {size / 1024**3:.3f} GiB")
    if args.apply:
        for path in candidates:
            path.unlink()
        pilot = root / "data/processed/ica_cleaned"
        if pilot.exists():
            for folder in sorted(pilot.glob("sub-*")):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
        print("Cleanup completed. Final data and report evidence retained.")
    else:
        print("Preview only. Run with --apply to delete these files.")


if __name__ == "__main__":
    main()
