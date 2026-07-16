from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.cache import (
    load_predictions,
    prediction_cache_metadata,
    predictions_path,
    save_predictions,
)
from utils.global_position_classifier import (
    build_global_predictions_dataframe,
    majority_class_baselines,
    split_dataframe,
)
from utils.metrics import compute_localization_metrics


class RawCsiWindowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        array_index = int(self.indices[item])
        sample = {
            band: torch.as_tensor(array[array_index], dtype=torch.float32)
            for band, array in self.arrays.items()
        }
        target = torch.as_tensor(self.targets[item], dtype=torch.long)
        return sample, target


class BandEncoder(nn.Module):
    def __init__(self, n_anchors: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_anchors, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _safe_branch_name(name: str) -> str:
    """Return a PyTorch-safe module key without changing the real band name."""
    return "b_" + name.replace(".", "_").replace(" ", "_").replace("-", "_")


class DualBandCNN(nn.Module):
    def __init__(self, branch_channels: dict[str, int], n_classes: int) -> None:
        super().__init__()
        self._order = list(branch_channels)
        self.branches = nn.ModuleDict(
            {
                _safe_branch_name(band): BandEncoder(n_anchors)
                for band, n_anchors in branch_channels.items()
            },
        )
        self.head = nn.Sequential(
            nn.Linear(128 * len(branch_channels), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        latents = [
            self.branches[_safe_branch_name(band)](inputs[band]) for band in self._order
        ]
        return self.head(torch.cat(latents, dim=1))


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeds: random={seed}, numpy={seed}, torch={seed}")


def print_torch_environment() -> torch.device:
    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch: {torch.__version__}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    return device


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def assert_window_identity(
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    left = meta[["group_id", "window_idx"]].reset_index(drop=True)
    right = feature_df[["group_id", "window_idx"]].reset_index(drop=True)
    if not left.equals(right):
        msg = f"WINDOW IDENTITY FAIL for {band}: raw arrays do not match ML dataframe order."
        raise AssertionError(msg)
    print(f"WINDOW IDENTITY PASS: {band} ({len(meta)} windows)")


def assert_block_split_identity(
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> None:
    meta_train, meta_test = split_dataframe(
        meta,
        test_size=test_size,
        random_state=random_state,
        split_mode="block",
        stratify_column="location",
        n_blocks=n_blocks,
    )
    rf_train, rf_test = split_dataframe(
        feature_df,
        test_size=test_size,
        random_state=random_state,
        split_mode="block",
        stratify_column="location",
        n_blocks=n_blocks,
    )
    train_ok = _window_pairs(meta_train).equals(_window_pairs(rf_train))
    test_ok = _window_pairs(meta_test).equals(_window_pairs(rf_test))
    if not train_ok or not test_ok:
        msg = f"BLOCK SPLIT IDENTITY FAIL for {band}: CNN split differs from RF split."
        raise AssertionError(msg)
    print(
        f"BLOCK SPLIT IDENTITY PASS: {band} "
        f"(train={len(meta_train)}, test={len(meta_test)})"
    )


def train_evaluate_cnn(  # noqa: PLR0913, PLR0915
    *,
    band: str,
    arrays: dict[str, np.ndarray],
    meta: pd.DataFrame,
    label_encoder: LabelEncoder,
    device: torch.device,
    results_dir: Path,
    plots_dir: Path,
    params: dict[str, Any],
    test_size: float,
    random_state: int,
    n_blocks: int,
    val_size: float,
    force_retrain: bool,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    expected_metadata = prediction_cache_metadata(
        model="CNN",
        band=band,
        split_mode="block",
        params=params,
    )
    if not force_retrain:
        cached = load_predictions(
            results_dir,
            "CNN",
            band,
            "block",
            expected_metadata=expected_metadata,
        )
        if cached is not None:
            metrics = compute_localization_metrics(cached)
            print(f"[CNN cache hit] {band}: loaded {len(cached)} predictions")
            return cached, metrics, pd.DataFrame()

    targets = label_encoder.transform(meta["location"].astype(str))
    train_df, test_df = split_dataframe(
        meta,
        test_size=test_size,
        random_state=random_state,
        split_mode="block",
        stratify_column="location",
        n_blocks=n_blocks,
    )
    train_df, val_df = split_dataframe(
        train_df,
        test_size=val_size,
        random_state=random_state + 1,
        split_mode="block",
        stratify_column="location",
        n_blocks=n_blocks,
    )
    train_idx = train_df.index.to_numpy(dtype=np.int64)
    val_idx = val_df.index.to_numpy(dtype=np.int64)
    test_idx = test_df.index.to_numpy(dtype=np.int64)

    test_loader = _loader(arrays, test_idx, targets, params, shuffle=False)

    branch_channels = {band_name: int(array.shape[1]) for band_name, array in arrays.items()}
    model = DualBandCNN(branch_channels, n_classes=len(label_encoder.classes_)).to(device)
    total_params = parameter_count(model)
    print(f"[CNN] {band}: parameters={total_params}")
    checkpoint_path = predictions_path(results_dir, "CNN", band, "block").with_suffix(".pt")

    history_rows: list[dict[str, float]] = []
    fit_seconds = 0.0
    best_val_acc = float("nan")

    if checkpoint_path.exists() and not force_retrain:
        best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        print(f"[CNN checkpoint hit] {band}: loaded {checkpoint_path}")
    else:
        train_loader = _loader(arrays, train_idx, targets, params, shuffle=True)
        val_loader = _loader(arrays, val_idx, targets, params, shuffle=False)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()
        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = -np.inf
        started_at = time.perf_counter()

        for epoch in range(1, int(params["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_loss, train_acc = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
            )
            val_loss, val_acc = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
            )
            epoch_seconds = time.perf_counter() - epoch_started
            history_rows.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seconds": float(epoch_seconds),
                },
            )
            print(
                f"[CNN] {band} epoch {epoch:02d}/{params['epochs']}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"seconds={epoch_seconds:.1f}",
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
            if epoch_seconds > float(params["max_epoch_seconds"]):
                msg = (
                    f"{band} epoch exceeded {params['max_epoch_seconds']} seconds "
                    f"({epoch_seconds:.1f}s). Stopping this run."
                )
                raise RuntimeError(msg)

        fit_seconds = time.perf_counter() - started_at
        model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        print(f"[CNN checkpoint] {band}: saved {checkpoint_path}")

    predict_started_at = time.perf_counter()
    pred_labels = _predict(model, test_loader, device)
    predict_seconds = time.perf_counter() - predict_started_at
    pred_positions = label_encoder.inverse_transform(pred_labels)
    predictions = build_global_predictions_dataframe(
        test_df,
        pred_positions,
        dataset_name=band,
        model_name="CNN",
        split_mode="block",
    )
    metrics = compute_localization_metrics(predictions)
    metrics.update(majority_class_baselines(train_df, test_df))
    metrics.update(
        {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "wall_seconds": float(fit_seconds + predict_seconds),
            "used_estimator": "DualBandCNN",
            "parameter_count": float(total_params),
            "best_val_accuracy": float(best_val_acc),
            "mean_seconds_per_epoch": (
                float(np.mean([r["seconds"] for r in history_rows]))
                if history_rows
                else float("nan")
            ),
        },
    )
    history = pd.DataFrame(history_rows)
    if not history.empty:
        _save_history_plot(
            history,
            plots_dir / f"cnn_training_curves_{_band_stem(band)}.pdf",
            band,
        )
    save_predictions(
        predictions,
        results_dir,
        "CNN",
        band,
        "block",
        metadata=expected_metadata,
    )
    return predictions, metrics, history


def append_cnn_summary_row(
    *,
    summary_path: Path,
    band: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> pd.DataFrame:
    row = {"dataset": band, "model": "CNN", "split": "block", **params, **metrics}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        keep = ~(
            (summary["dataset"] == band)
            & (summary["model"] == "CNN")
            & (summary["split"] == "block")
        )
        summary = pd.concat([summary.loc[keep], pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    summary.to_csv(summary_path, index=False)
    print(f"[summary] CNN row written to {summary_path}")
    return summary


def save_label_classes(classes: np.ndarray, results_dir: Path) -> None:
    path = results_dir / "predictions" / "cnn_label_classes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(map(str, classes)), indent=2), encoding="utf-8")
    print(f"[CNN] label classes saved to {path}")


def _loader(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    targets: np.ndarray,
    params: dict[str, Any],
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        RawCsiWindowDataset(arrays, indices, targets[indices]),
        batch_size=int(params["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, target in loader:
            inputs = {band: tensor.to(device) for band, tensor in inputs.items()}
            target = target.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == target).sum().item())
            total += batch_size
    return total_loss / total, correct / total


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    predictions: list[list[int]] = []
    with torch.no_grad():
        for batch_inputs, _ in loader:
            device_inputs = {
                band: tensor.to(device) for band, tensor in batch_inputs.items()
            }
            logits = model(device_inputs)
            predictions.append(logits.argmax(dim=1).cpu().tolist())
    flattened_list = [prediction for batch in predictions for prediction in batch]
    return np.asarray(flattened_list, dtype=np.int64)


def _save_history_plot(history: pd.DataFrame, save_path: Path, band: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title(f"{band} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title(f"{band} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[plots] saved {save_path}")


def _window_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["group_id", "window_idx"]].reset_index(drop=True)


def _band_stem(band: str) -> str:
    return band.lower().replace(".", "_").replace(" ", "")
