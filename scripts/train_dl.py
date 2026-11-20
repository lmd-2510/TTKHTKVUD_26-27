"""Train Compact EEG CNN and Braindecode EEG-Inception with participant validation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import confusion_matrix, f1_score
from download_data import REPO, atomic_output
from extract_features import load_split


class CompactEEGCNN(nn.Module):
    """Small temporal, spatial and separable CNN for single-trial EEG."""
    def __init__(self, channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 33), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 17), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


def load_epochs(split):
    arrays, labels, groups = [], [], []
    names = times = None
    for subject in split["train"] + split["test"]:
        record = split["participants"][subject]
        with np.load(record["file"], allow_pickle=False) as data:
            X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
            if (X.ndim != 3 or y.shape != (len(X),) or not np.isfinite(X).all()
                    or str(data["subject"]) != subject or set(np.unique(y)) != {0, 1}):
                raise ValueError(f"{subject}: invalid EEG data.")
            if any(int((y == code).sum()) != record[label]
                   for code, label in enumerate(("standard", "target"))):
                raise ValueError(f"{subject}: class counts differ from split.")
            current_names, current_times = data["channel_names"].tolist(), data["times_s"]
            if names is None:
                names, times = current_names, current_times.copy()
            if current_names != names or not np.array_equal(times, current_times):
                raise ValueError(f"{subject}: different channels or time grid.")
            if not np.isclose(float(data["sfreq_hz"]), 128):
                raise ValueError(f"{subject}: expected sampling frequency 128 Hz.")
            arrays.append(X * 1e6)
            labels.append(y)
            groups.append(np.full(len(y), subject))
    if arrays[0].shape[1:] != (16, 129):
        raise ValueError("Expected 16 channels and 129 time points.")
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(groups), names, times


def make_model(name, channels, n_times):
    if name == "compact_eeg_cnn":
        return CompactEEGCNN(channels)
    from braindecode.models import EEGInceptionERP
    return EEGInceptionERP(n_chans=channels, n_outputs=2, n_times=n_times, sfreq=128)


def loader(X, y, batch, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch, shuffle=shuffle, generator=generator, num_workers=0)


def predict(model, batches, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for X, _ in batches:
            logits = model(X.to(device))
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError(f"Expected batch x 2 outputs, received {tuple(logits.shape)}")
            predictions.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def save_json(path, value):
    with atomic_output(path) as output:
        output.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=REPO / "configs/splits.json")
    parser.add_argument("--models", nargs="+", choices=["compact_eeg_cnn", "eeg_inception"],
                        default=["compact_eeg_cnn", "eeg_inception"])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/metrics/dl")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO / "results/checkpoints/dl")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, patience, batch size and learning rate must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this venv. Check PyTorch/GPU or use --device cpu.")
    split = load_split(args.splits)
    if len(split["train"]) != 34 or len(split["test"]) != 8:
        raise ValueError("Expected the established 34/8 participant split.")
    people = sorted(split["train"])
    random.Random(args.seed).shuffle(people)
    validation_people = sorted(people[:4])
    fit_people = sorted(people[4:])
    X, y, groups, names, times = load_epochs(split)
    fit = np.isin(groups, fit_people)
    validation = np.isin(groups, validation_people)
    test = np.isin(groups, split["test"])
    # Channel normalization uses fitting participants only, across trials and time.
    mean = X[fit].mean(axis=(0, 2), keepdims=True)
    std = X[fit].std(axis=(0, 2), keepdims=True)
    if (std <= 0).any():
        raise ValueError("A training channel has zero variance.")
    X = ((X - mean) / std).astype(np.float32)
    fit_loader = loader(X[fit], y[fit], args.batch_size, True, args.seed)
    val_loader = loader(X[validation], y[validation], args.batch_size, False, args.seed)
    test_loader = loader(X[test], y[test], args.batch_size, False, args.seed)
    counts = np.bincount(y[fit], minlength=2)
    weights = torch.tensor(len(y[fit]) / (2 * counts), dtype=torch.float32, device=args.device)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_hash = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    for name in dict.fromkeys(args.models):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Reset shuffle generator so both models receive the same seeded batch sequence.
        fit_loader.generator.manual_seed(args.seed)
        model = make_model(name, len(names), len(times)).to(device)
        model.eval()
        with torch.inference_mode():
            probe = model(torch.zeros(2, len(names), len(times), device=device))
        if probe.shape != (2, 2) or not torch.isfinite(probe).all():
            raise ValueError(f"{name}: invalid model output for input shape (2,16,129).")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights)
        best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
        history = []
        started = time.monotonic()
        print(f"{name}: {sum(p.numel() for p in model.parameters())} parameters; device={device}", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_total = 0.0
            for batch_X, batch_y in fit_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_X), batch_y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Training produced non-finite loss.")
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(batch_y)
            predicted = predict(model, val_loader, device)
            score = float(f1_score(y[validation], predicted, labels=[0, 1], average="macro", zero_division=0))
            history.append({"epoch": epoch, "train_loss": loss_total / int(fit.sum()), "val_macro_f1": score})
            if score > best_score:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                checkpoint = {"state_dict": best_state, "model": name, "channel_names": names,
                              "times_s": times.tolist(), "normalization_mean_uv": mean.tolist(),
                              "normalization_std_uv": std.tolist(), "epoch": epoch,
                              "validation_macro_f1": score, "split_sha256": split_hash,
                              "fit_participants": fit_people, "validation_participants": validation_people,
                              "test_participants": split["test"], "arguments": {key: str(value) if isinstance(value, Path) else value
                                                                             for key, value in vars(args).items()}}
                with atomic_output(args.checkpoint_dir / f"{name}.pt") as output:
                    torch.save(checkpoint, output)
            else:
                stale += 1
            save_json(args.output_dir / f"{name}_history.json", history)
            print(f"{name} epoch {epoch}: loss={history[-1]['train_loss']:.4f}, val macro-F1={score:.4f}", flush=True)
            if stale >= args.patience:
                break
        model.load_state_dict(best_state)
        predicted = predict(model, test_loader, device)
        result = {
            "model": name, "macro_f1": float(f1_score(y[test], predicted, labels=[0, 1], average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y[test], predicted, labels=[0, 1]).tolist(),
            "matrix_rows": "true", "matrix_columns": "predicted", "label_order": ["standard", "target"],
            "train_participants": split["train"], "fit_participants": fit_people,
            "validation_participants": validation_people, "test_participants": split["test"],
            "fit_samples": int(fit.sum()), "validation_samples": int(validation.sum()),
            "test_samples": int(test.sum()), "best_epoch": best_epoch, "val_macro_f1": best_score,
            "selection": "Best validation macro-F1; 30 fitting and 4 validation participants within original 34 train.",
            "class_weights": weights.cpu().tolist(), "parameters": sum(p.numel() for p in model.parameters()),
            "torch_version": str(torch.__version__), "device": str(device),
            "elapsed_seconds": time.monotonic() - started, "split_sha256": split_hash,
        }
        save_json(args.output_dir / f"{name}.json", result)
        print(f"{name}: TEST macro-F1={result['macro_f1']:.4f}\n{np.array(result['confusion_matrix'])}", flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
