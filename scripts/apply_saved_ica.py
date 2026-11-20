"""Apply saved ICA and export continuous EEG for before/after inspection."""
import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from download_data import REPO, SUBJECTS
from preprocess import filter_eeg, write_json, resolve_report_path


def choose_components(info, threshold):
    if info["status"] == "manually_reviewed":
        return info["excluded"], "existing_manual_review"
    if not info.get("converged", False):
        return [], "review_required_nonconvergence"
    scores = info["scores_by_channel"]
    if not {"Fp1", "Fp2"} <= scores.keys():
        return [], "review_required_missing_proxies"
    left = scores["Fp1"]
    right = scores["Fp2"]
    candidates = set(left["candidates"]) & set(right["candidates"])
    selected = []
    for index in sorted(candidates):
        a, b = left["correlations"][index], right["correlations"][index]
        if a is not None and b is not None and abs(a) >= threshold and abs(b) >= threshold:
            selected.append(index)
    return selected, "automatic_frontal_proxy" if selected else "review_required_no_match"


def process(subject, qc_dir, output_dir, threshold):
    qc_dir = resolve_report_path(qc_dir)
    qc_path = qc_dir / f"{subject}.json"
    qc = json.loads(qc_path.read_text("utf-8"))
    cfg = qc["configuration"]
    settings = cfg["preprocessing"]
    info = qc["ica"]
    model_path = qc_dir / info["model_file"]
    ica = mne.preprocessing.read_ica(model_path)
    if len(ica.ch_names) != len(qc["ica_input_channels"]):
        raise ValueError("Saved ICA and QC channel counts disagree.")
    source = (REPO / cfg["paths"]["raw_dir"] / cfg["dataset"]["id"] /
              subject / "eeg" / f"{subject}_task-oddball_eeg.vhdr")
    raw = mne.io.read_raw_brainvision(source, preload=False)
    raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])).load_data()
    if qc["bad_channels"]:
        raw.drop_channels(qc["bad_channels"])
    if raw.ch_names != qc["ica_input_channels"]:
        raise ValueError("Source channel order differs from saved ICA run.")
    raw.set_eeg_reference("average")
    filter_eeg(raw, *settings["bandpass_hz"], settings["filter_order"])
    excluded, status = choose_components(info, threshold)
    if any(type(i) is not int or not 0 <= i < ica.n_components_ for i in excluded):
        raise ValueError("Component index outside saved ICA model.")
    plot_names = [name for name in ("Fp1", "Fp2", "Fz", "Cz", "Pz") if name in raw.ch_names]
    start = min(30.0, raw.times[-1] / 4)
    stop = min(start + 60, raw.times[-1])
    indices = raw.time_as_index([start, stop])
    before = raw.get_data(picks=plot_names, start=indices[0], stop=indices[1]) * 1e6
    ica.exclude = excluded
    if excluded:
        ica.apply(raw)
    after = raw.get_data(picks=plot_names, start=indices[0], stop=indices[1]) * 1e6
    folder = output_dir / subject
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(plot_names), 1, figsize=(14, 2.2 * len(plot_names)),
                             sharex=True, squeeze=False)
    times = raw.times[indices[0]:indices[1]]
    for ax, name, old, new in zip(axes[:, 0], plot_names, before, after):
        ax.plot(times, old, color="0.65", label="Before ICA", linewidth=0.7)
        ax.plot(times, new, color="#12645C", label="After ICA", linewidth=0.7)
        ax.set_ylabel(f"{name} (uV)")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Time in recording (s)")
    fig.suptitle(f"{subject} | excluded components: {excluded} | {status}")
    fig.tight_layout()
    fig.savefig(folder / "before_after.png", dpi=130)
    plt.close(fig)
    # Persist at the original sample rate to preserve annotation timing.
    # FIF single precision is adequate for EEG and limits disk usage.
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        destination = Path(temp) / f"{subject}-cleaned_raw.fif"
        raw.save(destination, fmt="single", split_size="1GB", overwrite=True)
        for path in Path(temp).iterdir():
            if (folder / path.name).exists():
                raise FileExistsError(f"Output already exists: {folder / path.name}")
        for path in Path(temp).iterdir():
            path.replace(folder / path.name)
    result = {
        "subject": subject, "status": status, "exclude": excluded,
        "reason": "Existing manual decision" if status == "existing_manual_review" else
                  f"Intersection of Fp1/Fp2 MNE candidates with absolute correlations >= {threshold}",
        "proxy_threshold": threshold, "source_ica": str(model_path),
        "channels": raw.ch_names, "sfreq": raw.info["sfreq"],
        "plot_interval_s": [start, stop],
        "plot_ptp_uv_before": dict(zip(plot_names, np.ptp(before, axis=1).tolist())),
        "plot_ptp_uv_after": dict(zip(plot_names, np.ptp(after, axis=1).tolist())),
        "qc_source": str(qc_path),
    }
    write_json(folder / "cleaning.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS)
    parser.add_argument("--qc-dir", type=Path,
                        default=REPO / "results/channels16/preprocessing_reports")
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/processed/ica_cleaned")
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    results, errors = {}, {}
    for subject in sorted(set(args.subjects or SUBJECTS)):
        try:
            print(f"Applying saved ICA: {subject}", flush=True)
            results[subject] = process(subject, args.qc_dir, args.output_dir, args.threshold)
            print(f"{subject}: {results[subject]['status']}, excluded={results[subject]['exclude']}", flush=True)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            errors[subject] = str(error)
            print(f"{subject}: ERROR {error}", flush=True)
    write_json(args.output_dir / "cleaning_summary.json", {"subjects": results, "errors": errors})
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
