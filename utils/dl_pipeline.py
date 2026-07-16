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
    def __init__(
        self,
        n_anchors: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        params = params or {}
        conv1_filters = int(params.get("conv1_filters", 16))
        conv2_filters = int(params.get("conv2_filters", 32))
        latent_dim = int(params.get("latent_dim", 64))
        self.features = nn.Sequential(
            nn.Conv2d(n_anchors, conv1_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_filters),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(conv2_filters, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def _safe_branch_name(name: str) -> str:
    """Return a PyTorch-safe module key without changing the real band name."""
    return "b_" + name.replace(".", "_").replace(" ", "_").replace("-", "_")


class DualBandCNN(nn.Module):
    def __init__(
        self,
        branch_channels: dict[str, int],
        n_classes: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        params = params or {}
        latent_dim = int(params.get("latent_dim", 64))
        head_hidden = int(params.get("head_hidden", 128))
        dropout = float(params.get("dropout", 0.5))
        self._order = list(branch_channels)
        self.branches = nn.ModuleDict(
            {
                _safe_branch_name(band): BandEncoder(n_anchors, params)
                for band, n_anchors in branch_channels.items()
            },
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * len(branch_channels), head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
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
    model_label = str(params.get("model_label", "CNN_small"))
    expected_metadata = prediction_cache_metadata(
        model=model_label,
        band=band,
        split_mode="block",
        params=params,
    )
    if not force_retrain:
        cached = load_predictions(
            results_dir,
            model_label,
            band,
            "block",
            expected_metadata=expected_metadata,
        )
        if cached is not None:
            metrics = compute_localization_metrics(cached)
            branch_channels = {
                band_name: int(array.shape[1]) for band_name, array in arrays.items()
            }
            cached_model = DualBandCNN(
                branch_channels,
                n_classes=len(label_encoder.classes_),
                params=params,
            )
            total_params = parameter_count(cached_model)
            checkpoint_path = predictions_path(
                results_dir,
                model_label,
                band,
                "block",
            ).with_suffix(".pt")
            checkpoint_metadata = _load_checkpoint_metadata(
                checkpoint_path.with_suffix(".pt.metadata.json"),
            )
            best_val_acc = float(checkpoint_metadata.get("best_val_accuracy", np.nan))
            best_epoch = int(checkpoint_metadata.get("best_epoch", 0))
            stopped_epoch = int(checkpoint_metadata.get("stopped_epoch", 0))
            patience_triggered = bool(
                checkpoint_metadata.get("patience_triggered", False),
            )
            metrics.update(
                {
                    "parameter_count": float(total_params),
                    "best_val_accuracy": best_val_acc,
                    "best_epoch": float(best_epoch),
                    "stopped_epoch": float(stopped_epoch),
                    "patience_triggered": patience_triggered,
                },
            )
            print(f"[{model_label} cache hit] {band}: loaded {len(cached)} predictions")
            print(
                f"[{model_label} result] {band}: best_val_accuracy={best_val_acc:.4f} "
                f"best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
                f"patience_triggered={patience_triggered} "
                f"test_position_accuracy={metrics['position_accuracy']:.4f} "
                f"parameter_count={total_params} (cached)",
            )
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
    model = DualBandCNN(
        branch_channels,
        n_classes=len(label_encoder.classes_),
        params=params,
    ).to(device)
    total_params = parameter_count(model)
    print(f"[{model_label}] {band}: parameters={total_params}")
    checkpoint_path = predictions_path(
        results_dir,
        model_label,
        band,
        "block",
    ).with_suffix(".pt")
    checkpoint_metadata_path = checkpoint_path.with_suffix(".pt.metadata.json")

    history_rows: list[dict[str, float]] = []
    fit_seconds = 0.0
    best_val_acc = float("nan")
    best_epoch = 0
    stopped_epoch = 0
    patience_triggered = False

    if checkpoint_path.exists() and not force_retrain:
        best_state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state)
        checkpoint_metadata = _load_checkpoint_metadata(checkpoint_metadata_path)
        best_val_acc = float(checkpoint_metadata.get("best_val_accuracy", np.nan))
        best_epoch = int(checkpoint_metadata.get("best_epoch", 0))
        stopped_epoch = int(checkpoint_metadata.get("stopped_epoch", 0))
        patience_triggered = bool(checkpoint_metadata.get("patience_triggered", False))
        print(f"[{model_label} checkpoint hit] {band}: loaded {checkpoint_path}")
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
        patience = int(params.get("patience", 15))
        epochs_without_improvement = 0
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
                f"[{model_label}] {band} epoch {epoch:02d}/{params['epochs']}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"seconds={epoch_seconds:.1f}",
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch_seconds > float(params["max_epoch_seconds"]):
                msg = (
                    f"{band} epoch exceeded {params['max_epoch_seconds']} seconds "
                    f"({epoch_seconds:.1f}s). Stopping this run."
                )
                raise RuntimeError(msg)
            if epochs_without_improvement >= patience:
                patience_triggered = True
                stopped_epoch = epoch
                print(
                    f"[{model_label}] {band}: early stopping at epoch {epoch}; "
                    f"validation accuracy did not improve for {patience} epochs.",
                )
                break

        fit_seconds = time.perf_counter() - started_at
        if not patience_triggered:
            stopped_epoch = len(history_rows)
        model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        _save_checkpoint_metadata(
            checkpoint_metadata_path,
            {
                "best_val_accuracy": float(best_val_acc),
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "patience_triggered": patience_triggered,
            },
        )
        print(f"[{model_label} checkpoint] {band}: saved {checkpoint_path}")

    predict_started_at = time.perf_counter()
    pred_labels = _predict(model, test_loader, device)
    predict_seconds = time.perf_counter() - predict_started_at
    pred_positions = label_encoder.inverse_transform(pred_labels)
    predictions = build_global_predictions_dataframe(
        test_df,
        pred_positions,
        dataset_name=band,
        model_name=model_label,
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
            "best_epoch": float(best_epoch),
            "stopped_epoch": float(stopped_epoch),
            "patience_triggered": patience_triggered,
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
            plots_dir
            / f"{_band_stem(model_label)}_training_curves_{_band_stem(band)}.pdf",
            f"{model_label} - {band}",
        )
    save_predictions(
        predictions,
        results_dir,
        model_label,
        band,
        "block",
        metadata=expected_metadata,
    )
    print(
        f"[{model_label} result] {band}: best_val_accuracy={best_val_acc:.4f} "
        f"best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
        f"patience_triggered={patience_triggered} "
        f"test_position_accuracy={metrics['position_accuracy']:.4f} "
        f"parameter_count={total_params}",
    )
    return predictions, metrics, history


def append_cnn_summary_row(
    *,
    summary_path: Path,
    band: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> pd.DataFrame:
    model_label = str(params.get("model_label", "CNN_small"))
    row = {
        "dataset": band,
        "model": model_label,
        "split": "block",
        **params,
        **metrics,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        keep = ~(
            (summary["dataset"] == band)
            & (summary["model"] == model_label)
            & (summary["split"] == "block")
        )
        summary = pd.concat([summary.loc[keep], pd.DataFrame([row])], ignore_index=True)
    else:
        summary = pd.DataFrame([row])
    summary.to_csv(summary_path, index=False)
    print(f"[summary] {model_label} row written to {summary_path}")
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


def _save_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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
