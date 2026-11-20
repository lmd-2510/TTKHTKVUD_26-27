"""Compare model target recall across reaction-time groups on validation participants."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from download_data import REPO, atomic_output
from extract_features import load_split, window_features
from train_ml import make_models
from visualize import infer, load_selected, save_figure, save_json


def join_response_times(split, subjects, trials_path, original_sfreq):
    """Join by participant and original stimulus sample; verify the retained label."""
    by_subject = {subject: {} for subject in subjects}
    with trials_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            subject = row["subject"]
            if subject not in by_subject or row["stimulus_class"] not in ("standard", "target"):
                continue
            sample = round(float(row["stimulus_onset_s"]) * original_sfreq)
            if sample in by_subject[subject]:
                raise ValueError(f"Duplicate raw stimulus sample: {subject}/{sample}")
            rt = float(row["rt_ms"]) if row["rt_ms"] else np.nan
            by_subject[subject][sample] = (int(row["stimulus_code"]), rt)
    latencies, samples = [], []
    for subject in subjects:
        with np.load(split["participants"][subject]["file"], allow_pickle=False) as data:
            original = data["raw_event_samples"]
            if original.shape != data["y"].shape:
                raise ValueError(f"{subject}: invalid original event samples.")
            for sample, label in zip(original, data["y"]):
                match = by_subject[subject].get(int(sample))
                expected_code = 5 if label == 0 else 6
                if match is None or match[0] != expected_code:
                    raise ValueError(f"Cannot match retained event: {subject}/{sample}. Check sample origin and frequency.")
                latencies.append(match[1] if label == 1 else np.nan)
                samples.append(int(sample))
    return np.array(latencies), np.array(samples)


def reaction_groups(latencies, lower, upper):
    groups = np.full(len(latencies), "no_eligible_response", dtype="<U24")
    valid = np.isfinite(latencies)
    groups[valid & (latencies <= lower)] = "fast"
    groups[valid & (latencies > lower) & (latencies <= upper)] = "middle"
    groups[valid & (latencies > upper)] = "slow"
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--timing-dir", type=Path, default=REPO / "results/analysis/response_timing")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/analysis/response_models")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    import torch
    from train_dl import make_model
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Check the environment or use --device cpu.")

    split = load_split(args.splits)
    split_hash = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    checkpoints = {}
    for name in ("compact_eeg_cnn", "eeg_inception"):
        ckpt = torch.load(REPO / f"results/checkpoints/dl/{name}.pt",
                          map_location="cpu", weights_only=True)
        if ckpt["split_sha256"] != split_hash or ckpt["model"] != name:
            raise ValueError(f"{name}: checkpoint split/model mismatch.")
        checkpoints[name] = ckpt
    reference = checkpoints["compact_eeg_cnn"]
    fit_people, val_people = reference["fit_participants"], reference["validation_participants"]
    if (set(fit_people) & set(val_people)
            or set(fit_people + val_people) != set(split["train"])
            or set(fit_people + val_people) & set(split["test"])):
        raise ValueError("Invalid fitting/validation participant assignments.")
    for checkpoint in checkpoints.values():
        if (checkpoint["fit_participants"] != fit_people
                or checkpoint["validation_participants"] != val_people):
            raise ValueError("DL checkpoints use different participant assignments.")
    timing = json.loads((args.timing_dir / "summary.json").read_text("utf-8"))
    if set(timing["participants"]) != set(split["train"]):
        raise ValueError("Timing inventory does not cover exactly the training participants.")

    people = fit_people + val_people
    X, y, subjects, names, times = load_selected(split, people)
    metadata = json.loads((REPO / "data/raw/ds004621/task-oddball_eeg.json").read_text("utf-8"))
    latencies, original_samples = join_response_times(
        split, people, args.timing_dir / "trials.csv", float(metadata["SamplingFrequency"]))
    fit = np.isin(subjects, fit_people)
    validation = np.isin(subjects, val_people)
    usable_fit_rt = latencies[fit & (y == 1) & np.isfinite(latencies)]
    if len(usable_fit_rt) < 4:
        raise ValueError("Too few retained fitting targets with response times.")
    lower, upper = np.percentile(usable_fit_rt, [25, 75])
    if lower >= upper:
        raise ValueError("Degenerate reaction-time thresholds.")
    labels = reaction_groups(latencies[validation], lower, upper)
    y_val = y[validation]
    predictions = {}
    features = window_features(X * 1e-6, times)
    for name, model in make_models().items():
        print(f"Fitting {name} on {len(fit_people)} participants for validation analysis", flush=True)
        model.fit(features[fit], y[fit])
        predictions[name] = model.predict(features[validation])

    for name, checkpoint in checkpoints.items():
        if checkpoint["channel_names"] != names or not np.array_equal(checkpoint["times_s"], times):
            raise ValueError(f"{name}: EEG axes differ from checkpoint.")
        mean = np.array(checkpoint["normalization_mean_uv"], dtype=np.float32)
        std = np.array(checkpoint["normalization_std_uv"], dtype=np.float32)
        normalized = ((X[validation] - mean) / std).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Nonfinite normalized EEG.")
        model = make_model(name, len(names), len(times)).to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        predicted = infer(model, normalized, args.batch_size, args.device).argmax(1)
        score = f1_score(y_val, predicted, labels=[0, 1], average="macro", zero_division=0)
        if not np.isclose(score, checkpoint["validation_macro_f1"], atol=1e-6):
            raise ValueError(f"{name}: predictions do not reproduce saved validation macro-F1.")
        predictions[name] = predicted
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    order = ("fast", "middle", "slow", "no_eligible_response")
    results = {}
    for name, predicted in predictions.items():
        target_groups = {}
        for group in order:
            mask = (y_val == 1) & (labels == group)
            count = int(mask.sum())
            correct = int((predicted[mask] == 1).sum())
            target_groups[group] = {"targets": count, "correct": correct,
                                    "target_recall": correct/count if count else None}
        results[name] = {
            "validation_macro_f1": float(f1_score(y_val, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_val, predicted, labels=[0, 1]).tolist(),
            "target_groups": target_groups,
            "per_participant_target_recall": {
                subject: float((predicted[(subjects[validation] == subject) & (y_val == 1)] == 1).mean())
                for subject in val_people},
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", {
        "scope": "Exploratory validation analysis; no test participant predictions.",
        "fit_participants": fit_people, "validation_participants": val_people,
        "rt_boundaries_ms": {"fit_target_q25": float(lower), "fit_target_q75": float(upper)},
        "group_rule": "fast <= q25; middle > q25 and <= q75; slow > q75; missing separately",
        "matching": timing["matching_rule"], "split_sha256": split_hash,
        "ml_protocol": "Same fixed estimators refitted on the 30 DL fitting participants.",
        "limitations": "Validation selected the DL checkpoints; this is descriptive, not independent evaluation. RT-group differences can reflect participant composition and do not establish motor causation.",
        "models": results,
    })
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["model", "subject", "raw_event_sample", "true_label", "predicted_label",
                     "target_rt_ms", "target_rt_group"])
    for name, predicted in predictions.items():
        for subject, sample, actual, pred, rt, group in zip(
                subjects[validation], original_samples[validation], y_val, predicted,
                latencies[validation], labels):
            writer.writerow([name, subject, sample, int(actual), int(pred),
                             float(rt) if np.isfinite(rt) else "",
                             group if actual == 1 else "standard"])
    with atomic_output(args.output_dir / "predictions.csv") as stream:
        stream.write(buffer.getvalue().encode())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = .15
    x = np.arange(len(order))
    for index, (name, result) in enumerate(results.items()):
        values = [result["target_groups"][g]["target_recall"] for g in order]
        ax.bar(x + (index-2)*width, [v if v is not None else np.nan for v in values],
               width=width, label=name)
    counts = [results[next(iter(results))]["target_groups"][g]["targets"] for g in order]
    ax.set(xticks=x, xticklabels=[f"{g}\nn={n}" for g,n in zip(order,counts)],
           ylabel="Target recall", ylim=(0, 1.05),
           title="Response-time groups | four validation participants")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5,1.24), ncol=3)
    save_figure(fig, args.output_dir, "target_recall_by_response_time")
    print(f"RT boundaries from fitting targets: {lower:.1f}, {upper:.1f} ms")
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
