"""Describe stimulus-to-button timing in training participants using Oddball event tables."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from download_data import REPO, atomic_output, read_tsv
from extract_features import load_split


def match_responses(onsets, codes, min_rt=0.1, max_rt=1.0):
    order = np.argsort(onsets, kind="stable")
    onsets, codes = np.asarray(onsets)[order], np.asarray(codes)[order]
    responses = onsets[codes == 1]
    indices = np.flatnonzero(np.isin(codes, [5, 6, 7]))
    rows = []
    used = set()
    for position, index in enumerate(indices):
        onset = float(onsets[index])
        next_stimulus = float(onsets[indices[position + 1]]) if position + 1 < len(indices) else np.inf
        # Never attribute one button press to a following stimulus.
        stop = min(onset + max_rt, next_stimulus)
        first = int(np.searchsorted(responses, onset, side="left"))
        last = int(np.searchsorted(responses, stop, side="left"))
        candidates = list(range(first, last))
        used.update(candidates)
        valid = [i for i in candidates if responses[i] - onset >= min_rt - 1e-10]
        rows.append({
            "stimulus_onset_s": onset,
            "stimulus_code": int(codes[index]),
            "stimulus_class": {5: "standard", 6: "target", 7: "distractor"}[int(codes[index])],
            "rt_ms": float((responses[valid[0]] - onset) * 1000) if valid else None,
            "response_count_in_window": len(candidates),
            "early_response_count": sum(responses[i] - onset < min_rt - 1e-10 for i in candidates),
        })
    return rows, int(len(responses) - len(used))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--raw-dir", type=Path, default=REPO / "data/raw/ds004621")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/analysis/response_timing")
    args = parser.parse_args()
    split = load_split(args.splits)
    all_rows, people = [], {}
    for subject in split["train"]:
        path = args.raw_dir / subject / "eeg" / f"{subject}_task-oddball_events.tsv"
        onsets = [float(v) for v in read_tsv(path, "onset")]
        descriptions = read_tsv(path, "event_type")
        codes = []
        for text in descriptions:
            match = re.fullmatch(r"S\s*(\d+)", text.strip())
            codes.append(int(match[1]) if match else -1)
        if not np.isfinite(onsets).all():
            raise ValueError(f"{subject}: nonfinite event onset.")
        rows, unmatched = match_responses(onsets, codes)
        for row in rows:
            row["subject"] = subject
        people[subject] = {
            "unassigned_S1_events": unmatched,
            "stimuli": dict(Counter(row["stimulus_class"] for row in rows)),
            "stimuli_with_eligible_response": dict(Counter(row["stimulus_class"] for row in rows
                                                         if row["rt_ms"] is not None)),
        }
        all_rows.extend(rows)
    target = [row for row in all_rows if row["stimulus_code"] == 6]
    times = np.array([row["rt_ms"] for row in target if row["rt_ms"] is not None])
    if not len(times):
        raise ValueError("No eligible S1 response following a target in the specified window.")
    summary = {
        "participants": split["train"], "scope": "training participants only; raw event rows before epoch rejection",
        "matching_rule": "First S1 from 100 ms up to but excluding 1000 ms after stimulus, before the next S5/S6/S7.",
        "caveat": "Temporal association is not proof of correct task execution or a motor origin of model features.",
        "target_stimuli": len(target), "targets_with_eligible_response": len(times),
        "targets_without_eligible_response": len(target) - len(times),
        "target_rt_ms": {"q25": float(np.percentile(times, 25)), "median": float(np.median(times)),
                         "q75": float(np.percentile(times, 75))},
        "target_responses_700_to_800_ms": int(((times >= 700) & (times < 800)).sum()),
        "per_participant": people,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with atomic_output(args.output_dir / "summary.json") as stream:
        stream.write((json.dumps(summary, indent=2) + "\n").encode())
    with atomic_output(args.output_dir / "trials.csv") as stream:
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
        stream.write(buffer.getvalue().encode("utf-8"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(times, bins=np.arange(100, 1001, 25), color="#12645C", edgecolor="white")
    ax.axvspan(700, 800, alpha=.2, color="#BA6438", label="700–800 ms")
    ax.set(xlabel="First eligible button-response latency after target (ms)", ylabel="Number of trials",
           title="Training participants | raw Oddball events")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "target_response_times.png", dpi=160)
    plt.close(fig)
    print(f"Training participants: {len(split['train'])}")
    print(f"Target events with eligible response: {len(times)}/{len(target)}")
    print(f"Median target-response latency: {np.median(times):.1f} ms")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
