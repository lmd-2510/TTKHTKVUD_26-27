"""Generate EEG, classification and temporal occlusion figures for the report."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from download_data import REPO, atomic_output
from extract_features import load_split, window_features

COLORS = ["#687789", "#BA6438"]


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def save_json(path, data):
    with atomic_output(path) as stream:
        stream.write((json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=160)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def plot_metrics(root, output):
    models = []
    expected_test = None
    for name in ("majority_baseline", "shrinkage_lda", "rbf_svm", "compact_eeg_cnn", "eeg_inception"):
        family = "dl" if name in ("compact_eeg_cnn", "eeg_inception") else "ml"
        path = root / "results/metrics" / family / f"{name}.json"
        if not path.exists() and family == "ml":
            path = root / "results/channels16/metrics/ml" / f"{name}.json"
        result = read_json(path)
        matrix = np.array(result["confusion_matrix"])
        test = result["test_participants"]
        if expected_test is None:
            expected_test = test
        if test != expected_test or matrix.shape != (2, 2) or matrix.sum() != result["test_samples"]:
            raise ValueError(f"{name}: inconsistent test results.")
        models.append(result)
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        color="white" if matrix[row, col] > matrix.max()/2 else "#172A3D")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Target"],
               yticklabels=["Standard", "Target"], xlabel="Predicted label", ylabel="True label",
               title=f"{name}\nMacro-F1 = {result['macro_f1']:.4f}")
        save_figure(fig, output, f"confusion_{name}")
        if family == "dl":
            history = read_json(path.with_name(f"{name}_history.json"))
            best = max(history, key=lambda item: item["val_macro_f1"])
            if best["epoch"] != result["best_epoch"]:
                raise ValueError(f"{name}: selected epoch differs from validation history.")
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            axes[0].plot([x["epoch"] for x in history], [x["train_loss"] for x in history])
            axes[0].set(xlabel="Epoch", ylabel="Weighted training loss")
            axes[1].plot([x["epoch"] for x in history], [x["val_macro_f1"] for x in history])
            axes[1].axvline(result["best_epoch"], color=COLORS[1], linestyle="--",
                            label=f"Selected epoch {result['best_epoch']}")
            axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", ylim=(0, 1))
            axes[1].legend()
            fig.suptitle(name)
            save_figure(fig, output, f"learning_{name}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([m["model"] for m in models], [m["macro_f1"] for m in models], color="#12645C")
    ax.set(ylabel="Held-out test macro-F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    for i, result in enumerate(models):
        ax.text(i, result["macro_f1"] + .015, f"{result['macro_f1']:.4f}", ha="center")
    save_figure(fig, output, "model_comparison")
    save_json(output / "metrics_summary.json", models)


def load_selected(split, subjects):
    arrays, labels, groups = [], [], []
    names = times = None
    for subject in subjects:
        record = split["participants"][subject]
        with np.load(record["file"], allow_pickle=False) as data:
            X, y = data["X"], data["y"]
            if (X.ndim != 3 or y.shape != (len(X),) or not np.isfinite(X).all()
                    or str(data["subject"]) != subject or set(np.unique(y)) != {0, 1}):
                raise ValueError(f"{subject}: invalid EEG.")
            if any(int((y == code).sum()) != record[label]
                   for code, label in enumerate(("standard", "target"))):
                raise ValueError(f"{subject}: counts differ from split inventory.")
            if names is None:
                names, times = data["channel_names"].tolist(), data["times_s"].copy()
            if names != data["channel_names"].tolist() or not np.array_equal(times, data["times_s"]):
                raise ValueError(f"{subject}: differing channel order/time grid.")
            arrays.append(X.astype(np.float32) * 1e6)
            labels.append(y)
            groups.append(np.full(len(y), subject))
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(groups), names, times


def plot_eeg(split, output):
    X, y, groups, names, times = load_selected(split, split["train"])
    pz = names.index("Pz")
    fig, ax = plt.subplots(figsize=(8, 4))
    averages = {}
    for code, label in enumerate(("Standard", "Target")):
        per_person = np.stack([X[(groups == subject) & (y == code), pz].mean(0)
                               for subject in split["train"]])
        mean, sd = per_person.mean(0), per_person.std(0, ddof=1)
        ax.plot(times * 1000, mean, label=label, color=COLORS[code])
        ax.fill_between(times * 1000, mean - sd, mean + sd, color=COLORS[code], alpha=.17)
        averages[label] = {"mean_uv": mean.tolist(), "between_participant_sd_uv": sd.tolist()}
    ax.axhline(0, color="0.8", linewidth=.8)
    ax.set(xlabel="Time after stimulus (ms)", ylabel="Pz amplitude (uV)",
           title="Training participants | equal participant weighting; shaded ±1 SD")
    ax.legend()
    save_figure(fig, output, "erp_pz")
    features = window_features(X * 1e-6, times)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, index in zip(axes, (6, 8, 10)):
        for code, label in enumerate(("Standard", "Target")):
            ax.hist(features[y == code, pz * 20 + index], bins=40, density=True,
                    alpha=.5, label=label, color=COLORS[code])
        ax.set(title=f"Pz: {index*50}–{(index+1)*50} ms",
               xlabel="Window mean amplitude (uV)", ylabel="Density")
    axes[0].legend()
    save_figure(fig, output, "feature_distributions_pz")
    save_json(output / "erp_pz.json", {"participants": split["train"], "times_s": times.tolist(),
                                      "averages": averages, "weighting": "equal participant means"})


def infer(model, X, batch_size, device, mask=None):
    import torch
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start+batch_size]).to(device)
            if mask is not None:
                batch = batch.clone()
                batch[:, :, mask] = 0
            outputs.append(model(batch).softmax(1).cpu().numpy())
    return np.concatenate(outputs)


def occlusion(root, split_path, split, output, device, batch_size):
    import torch
    from train_dl import make_model
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Use --device cpu or check the server environment.")
    X, y, groups, names, times = load_selected(split, split["test"])
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    for name in ("compact_eeg_cnn", "eeg_inception"):
        checkpoint = torch.load(root / f"results/checkpoints/dl/{name}.pt",
                                map_location="cpu", weights_only=True)
        if (checkpoint["model"] != name or checkpoint["split_sha256"] != split_hash
                or checkpoint["channel_names"] != names
                or not np.array_equal(checkpoint["times_s"], times)
                or checkpoint["test_participants"] != split["test"]):
            raise ValueError(f"{name}: checkpoint metadata differs from test data.")
        mean = np.array(checkpoint["normalization_mean_uv"], dtype=np.float32)
        std = np.array(checkpoint["normalization_std_uv"], dtype=np.float32)
        if mean.shape != (1, len(names), 1) or std.shape != mean.shape or not (std > 0).all():
            raise ValueError("Invalid checkpoint normalization.")
        normalized = ((X - mean) / std).astype(np.float32)
        model = make_model(name, len(names), len(times)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        original = infer(model, normalized, batch_size, device)
        base_score = float(f1_score(y, original.argmax(1), labels=[0, 1], average="macro", zero_division=0))
        expected = read_json(root / f"results/metrics/dl/{name}.json")
        matrix = confusion_matrix(y, original.argmax(1), labels=[0, 1])
        if not np.array_equal(matrix, expected["confusion_matrix"]):
            raise ValueError(f"{name}: checkpoint predictions do not reproduce stored confusion matrix.")
        windows = []
        for index in range(10):
            low, high = index / 10, (index + 1) / 10
            mask = (times >= low - 1e-10) & (times < high - 1e-10)
            if index == 9:
                mask = (times >= low - 1e-10) & (times <= high + 1e-10)
            changed = infer(model, normalized, batch_size, device, mask)
            score = float(f1_score(y, changed.argmax(1), labels=[0, 1], average="macro", zero_division=0))
            windows.append({"start_ms": index*100, "end_ms": (index+1)*100,
                            "macro_f1": score, "macro_f1_drop": base_score-score,
                            "mean_true_class_probability_drop":
                                float((original[np.arange(len(y)), y] - changed[np.arange(len(y)), y]).mean())})
            print(f"{name}: masked {index*100}–{(index+1)*100} ms, macro-F1={score:.4f}", flush=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([f"{v['start_ms']}–{v['end_ms']}" for v in windows],
               [v["macro_f1_drop"] for v in windows], color="#12645C")
        ax.axhline(0, color="0.4", linewidth=.8)
        ax.set(xlabel="Masked interval (ms)", ylabel="Baseline macro-F1 minus masked macro-F1",
               title=f"{name} | all channels masked together | test participants")
        ax.tick_params(axis="x", rotation=35)
        save_figure(fig, output, f"occlusion_{name}")
        save_json(output / f"occlusion_{name}.json",
                  {"model": name, "baseline_macro_f1": base_score, "test_participants": split["test"],
                   "mask_value": "0 after normalization, equal to fitting-channel mean before normalization",
                   "interpretation": "Perturbation sensitivity; not a causal brain localization.",
                   "window_ms": 100, "windows": windows})
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "metrics", "eeg", "occlusion"], default="all")
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/figures")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "metrics"):
        plot_metrics(REPO, args.output_dir)
    if args.stage in ("all", "eeg", "occlusion"):
        split = load_split(args.splits)
        if args.stage in ("all", "eeg"):
            plot_eeg(split, args.output_dir)
        if args.stage in ("all", "occlusion"):
            occlusion(REPO, args.splits, split, args.output_dir, args.device, args.batch_size)
    print(f"Figures and numerical summaries saved to {args.output_dir}")


if __name__ == "__main__":
    main()
