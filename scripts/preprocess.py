"""Preprocess BrainVision Oddball EEG and write per-participant quality reports."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import tempfile
import warnings

import mne
import numpy as np
import yaml

from download_data import REPO, SUBJECTS, atomic_output, read_tsv


def write_json(path, data):
    with atomic_output(path) as output:
        output.write((json.dumps(data, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def stimulus_events(raw, events_path, codes):
    wanted = set(codes.values())
    sfreq = raw.info["sfreq"]
    offsets = read_tsv(events_path, "onset")
    event_types = read_tsv(events_path, "event_type")
    rows = []
    for onset, event_type in zip(offsets, event_types):
        match = re.fullmatch(r"S\s*(\d+)", event_type.strip())
        if match and int(match[1]) in wanted:
            seconds = float(onset)
            if not np.isfinite(seconds) or not 0 <= seconds < raw.n_times / sfreq:
                raise ValueError(f"Stimulus onset outside the recording: {onset}")
            sample = raw.first_samp + round(seconds * sfreq)
            rows.append((sample, 0, int(match[1])))
    if not rows:
        raise ValueError(f"No standard or target events in {events_path}")
    events = np.array(sorted(rows), dtype=int)
    if len(np.unique(events[:, 0])) != len(events):
        raise ValueError("Multiple candidate stimuli share one sample.")

    marker_ids = {}
    for description in set(raw.annotations.description):
        match = re.fullmatch(r"(?:Stimulus/)?S\s*(\d+)", description.strip())
        if match and int(match[1]) in wanted:
            marker_ids[description] = int(match[1])
    if not marker_ids:
        raise ValueError("No matching BrainVision stimulus markers found.")
    markers, _ = mne.events_from_annotations(raw, event_id=marker_ids)
    checks = {}
    for label, code in codes.items():
        listed = events[events[:, 2] == code, 0]
        marked = markers[markers[:, 2] == code, 0]
        if len(listed) != len(marked) or not len(listed):
            raise ValueError(f"{label}: {len(listed)} TSV events vs {len(marked)} BrainVision markers.")
        max_ms = float(np.max(np.abs(listed - marked)) * 1000 / sfreq)
        if max_ms > 2:
            raise ValueError(f"{label}: TSV and BrainVision markers differ by up to {max_ms:g} ms.")
        checks[label] = {"count": len(listed), "max_difference_ms": max_ms}
    return events, checks


def filter_eeg(raw, low, high, order):
    raw.filter(low, high, picks="eeg", method="iir", phase="zero",
               iir_params={"order": order, "ftype": "butter"})


def clean_ica(raw, settings, rate, review, ica_path):
    fit_raw = raw.copy().resample(rate, npad="auto")
    filter_eeg(fit_raw, settings["fit_highpass_hz"], None, settings["filter_order"])
    available = len(mne.pick_types(fit_raw.info, eeg=True))
    component_count = min(settings["n_components"], available - 1)
    if component_count < 2:
        raise ValueError("Too few good EEG channels to fit ICA after common-average reference.")
    ica = mne.preprocessing.ICA(n_components=component_count,
                                random_state=settings["random_state"], max_iter="auto")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica.fit(fit_raw)
    messages = list(dict.fromkeys(str(item.message) for item in caught))
    converged = not any("did not converge" in message.lower() for message in messages)
    eog_channels = [fit_raw.ch_names[i] for i in mne.pick_types(fit_raw.info, eeg=False, eog=True)]
    eeg_names = {name.casefold(): name for name in ica.ch_names}
    proxies = [eeg_names[name] for name in ("fp1", "fp2", "fpz") if name in eeg_names]
    recommendations = {}
    for channel in eog_channels or proxies:
        indices, scores = ica.find_bads_eog(fit_raw, ch_name=channel)
        recommendations[channel] = {
            "candidates": [int(index) for index in indices],
            "correlations": [float(value) if np.isfinite(value) else None for value in scores],
        }
    del fit_raw

    excluded = sorted({index for channel in eog_channels
                       for index in recommendations[channel]["candidates"]}) if converged else []
    reason = "MNE EOG correlation" if excluded else None
    if review is not None:
        excluded = sorted(set(review["exclude"]))
        if any(type(index) is not int or not 0 <= index < ica.n_components_ for index in excluded):
            raise ValueError("ICA review contains an invalid component index.")
        reason = review["reason"].strip()
        if not reason:
            raise ValueError("ICA review requires a reason, including when retaining all components.")
    elif not converged:
        status = "fit_did_not_converge"
    elif not eog_channels:
        status = "manual_review_required"
    else:
        status = "assessed_with_eog"
    if review is not None:
        status = "manually_reviewed"

    ica.exclude = excluded
    with tempfile.TemporaryDirectory(dir=ica_path.parent) as tmp:
        temp_path = Path(tmp) / ica_path.name
        ica.save(temp_path, overwrite=True)
        temp_path.replace(ica_path)
    if excluded:
        ica.apply(raw)
    return {
        "status": status, "fit_sfreq_hz": rate, "component_count": int(ica.n_components_),
        "iterations": int(ica.n_iter_), "converged": converged, "fit_warnings": messages,
        "eog_channels": eog_channels, "frontal_proxy_channels": proxies,
        "scores_by_channel": recommendations, "excluded": excluded, "reason": reason,
        "model_file": ica_path.name,
    }


def preprocess(subject, raw_dir, output_dir, qa_dir, config, review):
    settings = config["preprocessing"]
    codes = config["dataset"]["events"]
    if set(codes) != {"standard", "target"} or len(set(codes.values())) != 2:
        raise ValueError("Expected distinct standard and target event codes.")
    stem = raw_dir / subject / "eeg" / f"{subject}_task-{config['dataset']['task']}"
    raw = mne.io.read_raw_brainvision(f"{stem}_eeg.vhdr", preload=False)
    sfreq = float(raw.info["sfreq"])
    events, alignment = stimulus_events(raw, Path(f"{stem}_events.tsv"), codes)
    raw.pick(mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])).load_data()
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    bad_channels = list(raw.info["bads"])
    for name in eeg_names:
        signal = raw.get_data(picks=[name])[0]
        if not np.isfinite(signal).all() or np.ptp(signal) == 0:
            bad_channels.append(name)
    bad_channels = sorted(set(bad_channels))
    if len(set(eeg_names) - set(bad_channels)) < 2:
        raise ValueError("Fewer than two usable EEG channels.")
    if bad_channels:
        raw.drop_channels(bad_channels)
    raw.set_eeg_reference("average")

    low, high = settings["bandpass_hz"]
    filter_eeg(raw, low, high, settings["filter_order"])
    ica_info = clean_ica(raw, {**settings["ica"], "filter_order": settings["filter_order"]},
                         settings["ica"]["fit_sfreq_hz"], review, qa_dir / f"{subject}-ica.fif")

    rate = settings["resample_hz"]
    raw, resampled_events = raw.resample(rate, events=events, npad="auto")
    if len(np.unique(resampled_events[:, 0])) != len(events):
        raise ValueError("Two candidate stimuli share a sample after resampling.")
    bounds = {key: tuple(value / 1000 for value in settings[key])
              for key in ("epoch_ms", "baseline_ms", "model_input_ms")}
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    epochs = mne.Epochs(
        raw, resampled_events, event_id=codes, tmin=bounds["epoch_ms"][0],
        tmax=bounds["epoch_ms"][1], baseline=bounds["baseline_ms"],
        picks=eeg_picks, preload=True,
        reject={"eeg": settings["reject_peak_to_peak_uv"] * 1e-6},
        flat={"eeg": settings["flat_peak_to_peak_uv"] * 1e-6},
    )
    drop_reasons = {label: Counter() for label in codes}
    kept = Counter(epochs.events[:, 2])
    label_by_code = {code: label for label, code in codes.items()}
    for event, reasons in zip(resampled_events, epochs.drop_log):
        if reasons:
            label = label_by_code[event[2]]
            drop_reasons[label].update(reasons)
    summary = {
        "subject": subject, "raw_sfreq_hz": sfreq, "output_sfreq_hz": rate,
        "recorded_eeg_channels": len(eeg_names), "output_channels": epochs.ch_names,
        "bad_channels": bad_channels, "event_alignment": alignment, "ica": ica_info,
        "class_counts": {label: {"before": alignment[label]["count"],
                                 "kept": kept[code],
                                 "dropped": alignment[label]["count"] - kept[code],
                                 "drop_log_reasons": dict(drop_reasons[label])}
                         for label, code in codes.items()},
    }
    if any(not kept[code] for code in codes.values()):
        write_json(qa_dir / f"{subject}.json", summary)
        raise ValueError("At least one class has zero retained epochs. See the QC JSON.")
    summary["epoch_times_s"] = [float(epochs.times[0]), float(epochs.times[-1])]
    summary["requested_baseline_s"] = list(bounds["baseline_ms"])
    epochs.crop(*bounds["model_input_ms"])
    features = epochs.get_data().astype(np.float32)
    labels = (epochs.events[:, 2] == codes["target"]).astype(np.int8)
    summary["shape"] = list(features.shape)
    summary["data_units"] = "V"
    with atomic_output(output_dir / f"{subject}.npz") as output:
        np.savez(output, X=features, y=labels, channel_names=np.array(epochs.ch_names),
                 times_s=epochs.times, raw_event_samples=events[epochs.selection, 0],
                 subject=subject, sfreq_hz=rate)
    write_json(qa_dir / f"{subject}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "experiment.yaml")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, metavar="sub-NN")
    parser.add_argument("--ica-review", type=Path, help="JSON mapping subjects to exclude lists and reasons.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = config["paths"]
    raw_dir = REPO / paths["raw_dir"] / config["dataset"]["id"]
    output_dir = REPO / paths["processed_dir"]
    qa_dir = REPO / paths["results_dir"] / "quality" / "preprocessing"
    qa_dir.mkdir(parents=True, exist_ok=True)
    reviews = json.loads(args.ica_review.read_text(encoding="utf-8")) if args.ica_review else {}
    subjects = sorted(set(args.subjects or SUBJECTS))
    reports = {}
    errors = {}
    for subject in subjects:
        print(f"\nPreprocessing {subject}", flush=True)
        try:
            reports[subject] = preprocess(subject, raw_dir, output_dir, qa_dir, config, reviews.get(subject))
        except (OSError, ValueError, RuntimeError) as error:
            errors[subject] = str(error)
            print(f"{subject} failed: {error}", flush=True)
    scope = "all" if subjects == list(SUBJECTS) else "_".join(subjects)
    write_json(qa_dir / f"summary_{scope}.json", {"subjects": reports, "errors": errors})
    for subject, result in reports.items():
        counts = result["class_counts"]
        print(f"{subject}: standard {counts['standard']['kept']}/{counts['standard']['before']}, "
              f"target {counts['target']['kept']}/{counts['target']['before']}, "
              f"ICA: {result['ica']['status']}")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
