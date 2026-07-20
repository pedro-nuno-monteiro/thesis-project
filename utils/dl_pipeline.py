from __future__ import annotations

import copy
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.cache import (
    RESULTS_ROOT,
    load_predictions,
    make_run_id,
    prediction_cache_metadata,
    save_predictions,
)
from utils.config import (
    CNN_CONV1_FILTERS,
    CNN_CONV2_FILTERS,
    CNN_DROPOUT,
    CNN_HEAD_HIDDEN,
    CNN_LATENT_DIM,
    CPU_BATCH_SIZE,
    CPU_NUM_WORKERS,
    CPU_PERSISTENT_WORKERS,
    CPU_PIN_MEMORY,
    CUDA_BATCH_SIZE,
    CUDA_NUM_WORKERS,
    CUDA_PERSISTENT_WORKERS,
    CUDA_PIN_MEMORY,
    CUDA_PREFETCH_FACTOR,
    DEFAULT_CNN_PARAMS,
    PLOT_DPI,
    PLOT_FORMAT,
    SEEDS,
)
from utils.global_position_classifier import (
    _window_identity_fingerprint,
    build_global_predictions_dataframe,
    majority_class_baselines,
    split_dataframe,
)
from utils.metrics import compute_localization_metrics
from utils.results import (
    build_run_row,
    checkpoint_path,
    derive_seed_summary,
    derive_table_from_runs,
    upsert_fold_rows,
    upsert_run,
    write_run_manifest,
)

SplitMode = Literal["block", "lovo", "cross_session"]


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
        conv1_filters = int(params.get("conv1_filters", CNN_CONV1_FILTERS))
        conv2_filters = int(params.get("conv2_filters", CNN_CONV2_FILTERS))
        latent_dim = int(params.get("latent_dim", CNN_LATENT_DIM))
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
        latent_dim = int(params.get("latent_dim", CNN_LATENT_DIM))
        head_hidden = int(params.get("head_hidden", CNN_HEAD_HIDDEN))
        dropout = float(params.get("dropout", CNN_DROPOUT))
        self._order = list(branch_channels)
        self.branches = nn.ModuleDict(
            {
                _safe_branch_name(band): BandEncoder(n_anchors, params)
                for band, n_anchors in branch_channels.items()
            }
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
    """Seed Python, NumPy, CPU torch, and every CUDA device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeds: random={seed}, numpy={seed}, torch={seed}")


def print_torch_environment(*, require_cuda: bool = False) -> torch.device:
    """Print/assert the GPU environment and execute a CUDA matmul smoke test."""
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("!!! CUDA IS NOT AVAILABLE; DL will use the configured CPU fallback. !!!")
        if require_cuda:
            raise AssertionError("CUDA is required for this run but torch.cuda.is_available() is false.")
        return torch.device("cpu")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"torch.cuda.get_device_name(0): {device_name}")
    print(f"torch.cuda.get_device_capability(0): {capability}")
    assert device_name, "CUDA device name must be non-empty."
    assert len(capability) == 2, "CUDA compute capability must be a (major, minor) pair."

    if capability >= (12, 0) and (
        _version_pair(torch.__version__) < (2, 7)
        or _version_pair(torch.version.cuda or "0") < (12, 8)
    ):
        print(
            "!!! BLACKWELL COMPATIBILITY WARNING: compute capability sm_120 requires "
            "CUDA 12.8+ and torch >= 2.7; kernels may fall back or fail. !!!"
        )

    left = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    right = torch.eye(4, dtype=torch.float32, device="cuda")
    smoke_result = left @ right
    torch.cuda.synchronize()
    assert smoke_result.is_cuda and torch.isfinite(smoke_result).all()
    print(f"CUDA matmul smoke test result: {smoke_result.cpu().tolist()} (PASS)")
    return torch.device("cuda")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def assert_window_identity(
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    left = _window_pairs(meta)
    right = _window_pairs(feature_df)
    if not left.equals(right):
        raise AssertionError(
            f"WINDOW IDENTITY FAIL for {band}: raw arrays do not match ML dataframe order."
        )
    print(f"WINDOW IDENTITY PASS: {band} ({len(meta)} windows)")


def assert_split_identity(  # noqa: PLR0913
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
    split_mode: SplitMode,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> None:
    """Assert and print that CNN and RF receive identical protocol windows."""
    meta_split = split_dataframe(
        meta,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    rf_split = split_dataframe(
        feature_df,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    meta_folds = meta_split if isinstance(meta_split, list) else [meta_split]
    rf_folds = rf_split if isinstance(rf_split, list) else [rf_split]
    if len(meta_folds) != len(rf_folds):
        raise AssertionError(f"SPLIT IDENTITY FAIL for {band}: fold counts differ.")
    for fold_index, ((meta_train, meta_test), (rf_train, rf_test)) in enumerate(
        zip(meta_folds, rf_folds), start=1
    ):
        if not _window_pairs(meta_train).equals(_window_pairs(rf_train)) or not _window_pairs(
            meta_test
        ).equals(_window_pairs(rf_test)):
            raise AssertionError(
                f"SPLIT IDENTITY FAIL for {band}/{split_mode}, fold {fold_index}: "
                "CNN split differs from RF split."
            )
        print(
            f"SPLIT IDENTITY PASS: {band}/{split_mode} fold={fold_index} "
            f"train={len(meta_train)} test={len(meta_test)}"
        )


def assert_block_split_identity(
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    n_blocks: int,
) -> None:
    """Backward-compatible alias for block split parity checks."""
    assert_split_identity(
        band=band,
        meta=meta,
        feature_df=feature_df,
        split_mode="block",
        test_size=test_size,
        random_state=random_state,
        n_blocks=n_blocks,
    )


def train_evaluate_cnn(  # noqa: PLR0913, PLR0915
    *,
    band: str,
    arrays: dict[str, np.ndarray],
    meta: pd.DataFrame,
    label_encoder: LabelEncoder,
    device: torch.device,
    results_dir: Path = RESULTS_ROOT,
    plots_dir: Path | None = None,
    params: dict[str, Any] | None = None,
    split_mode: SplitMode = "block",
    test_size: float = 0.3,
    random_state: int = 42,
    seed: int | None = None,
    n_blocks: int = 10,
    val_size: float = 0.15,
    force_retrain: bool = False,
    preproc_opts: dict[str, Any] | None = None,
    feat_opts: dict[str, Any] | None = None,
    run_id: str | None = None,
    feature_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Train one seeded CNN run for block, LOVO, or cross-session evaluation."""
    if split_mode not in {"block", "lovo", "cross_session"}:
        raise ValueError("split_mode must be block, lovo, or cross_session.")
    resolved_seed = int(random_state if seed is None else seed)
    set_reproducible_seeds(resolved_seed)
    resolved_params = {**DEFAULT_CNN_PARAMS, **(params or {})}
    resolved_params["seed"] = resolved_seed
    model_label = str(resolved_params.get("model_label", "CNN"))
    preproc = dict(preproc_opts or {})
    preproc.setdefault("normalization", resolved_params.get("normalization", "none"))
    if preproc["normalization"] == "empty_baseline":
        preproc.setdefault("baseline_scope", resolved_params.get("baseline_scope", "per_session"))
    features = dict(feat_opts or {})
    features.setdefault("window_size", int(resolved_params.get("window_size", 60)))
    features.setdefault("overlap_size", int(resolved_params.get("overlap_size", 30)))
    features.setdefault("require_all_esps", bool(resolved_params.get("require_all_esps", False)))
    split_params = {
        "test_size": test_size,
        "random_state": random_state,
        "n_blocks": n_blocks,
        "validation_size": val_size,
    }
    full_config = {
        "preproc_opts": preproc,
        "feat_opts": features,
        "model_params": resolved_params,
        "seed": resolved_seed,
        "split_params": split_params,
    }
    resolved_run_id = run_id or make_run_id(
        family="dl",
        model="cnn",
        band=band,
        split=split_mode,
        normalization=str(preproc["normalization"]),
        baseline_scope=preproc.get("baseline_scope"),
        seed=resolved_seed,
        config=full_config,
    )
    result_root = Path(results_dir)
    write_run_manifest(resolved_run_id, full_config, results_root=result_root)
    if feature_df is not None:
        assert_window_identity(band=band, meta=meta, feature_df=feature_df)
        assert_split_identity(
            band=band,
            meta=meta,
            feature_df=feature_df,
            split_mode=split_mode,
            test_size=test_size,
            random_state=random_state,
            n_blocks=n_blocks,
        )
    targets = label_encoder.transform(meta["location"].astype(str))
    split_result = split_dataframe(
        meta,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        stratify_column="location",
        n_blocks=n_blocks,
    )
    folds = split_result if isinstance(split_result, list) else [split_result]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    fold_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []
    total_started = time.perf_counter()
    total_fit = 0.0
    total_predict = 0.0
    run_parameter_count = float("nan")

    for fold_index, (protocol_train, test_df) in enumerate(folds):
        held_out_user = _single_user(test_df) if split_mode == "lovo" else None
        validation_user: object | None = None
        if split_mode == "lovo":
            training_users = sorted(protocol_train["user"].dropna().unique())
            validation_user = training_users[(fold_index - 1) % len(training_users)]
            validation_mask = protocol_train["user"] == validation_user
            train_df = protocol_train.loc[~validation_mask].copy()
            val_df = protocol_train.loc[validation_mask].copy()
            if train_df.empty or val_df.empty:
                raise ValueError(
                    f"LOVO validation-user split is empty for held_out_user={held_out_user}, "
                    f"validation_user={validation_user}."
                )
            print(
                f"[DL LOVO] held_out_user={held_out_user} "
                f"validation_user={validation_user} (entire user, deterministic rotation)"
            )
        else:
            validation_split = split_dataframe(
                protocol_train,
                test_size=val_size,
                random_state=random_state + 1,
                split_mode="block",
                stratify_column="location",
                n_blocks=n_blocks,
            )
            if isinstance(validation_split, list):
                raise RuntimeError("Validation split unexpectedly produced multiple folds.")
            train_df, val_df = validation_split

        fold_token = held_out_user if split_mode == "lovo" else None
        fold_pred, fold_metrics, history = _train_predict_fold(
            band=band,
            arrays=arrays,
            targets=targets,
            label_encoder=label_encoder,
            device=device,
            params=resolved_params,
            model_label=model_label,
            split_mode=split_mode,
            train_df=train_df,
            protocol_train_df=protocol_train,
            val_df=val_df,
            test_df=test_df,
            results_root=result_root,
            run_id=resolved_run_id,
            fold=fold_token,
            force_retrain=force_retrain,
        )
        if held_out_user is not None:
            fold_pred["held_out_user"] = held_out_user
        fold_predictions.append(fold_pred)
        total_fit += float(fold_metrics["fit_seconds"])
        total_predict += float(fold_metrics["predict_seconds"])
        run_parameter_count = float(fold_metrics["parameter_count"])
        if not history.empty:
            history.insert(0, "fold", str(held_out_user or "single"))
            histories.append(history)
            plot_root = (
                result_root / "plots" / resolved_run_id
                if plots_dir is None
                else Path(plots_dir) / resolved_run_id
            )
            fold_suffix = "" if held_out_user is None else f"__fold-{held_out_user}"
            _save_history_plot(
                history,
                plot_root / f"training_curves{fold_suffix}.{PLOT_FORMAT}",
                f"{model_label} - {band} - {split_mode}",
            )
        fold_rows.append(
            {
                "run_id": resolved_run_id,
                "fold": str(held_out_user if held_out_user is not None else "single"),
                "held_out_user": held_out_user,
                "validation_user": validation_user,
                "n_train_windows": int(len(protocol_train)),
                "n_test_windows": int(len(test_df)),
                "trials_used": _trials_used(protocol_train, test_df),
                **fold_metrics,
            }
        )
        del fold_pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(fold_predictions, ignore_index=True)
    history_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    if split_mode == "lovo":
        per_fold = pd.DataFrame(fold_rows)
        metrics = _aggregate_fold_metrics(per_fold, predictions)
        metrics.update(
            {
                "fit_seconds": total_fit,
                "predict_seconds": total_predict,
                "wall_seconds": time.perf_counter() - total_started,
                "parameter_count": run_parameter_count,
            }
        )
        pooled_metadata = prediction_cache_metadata(
            model=model_label,
            band=band,
            split_mode=split_mode,
            params=resolved_params,
            random_state=resolved_seed,
            train_fingerprint=_fold_fingerprint(folds, side="train"),
            test_fingerprint=_fold_fingerprint(folds, side="test"),
        )
        save_predictions(
            predictions,
            result_root,
            model_label,
            band,
            split_mode,
            metadata=pooled_metadata,
            run_id=resolved_run_id,
        )
        upsert_fold_rows(fold_rows, results_root=result_root)
        print(
            f"[DL LOVO] seed={resolved_seed} fold mean position_accuracy="
            f"{metrics['position_accuracy_mean']:.4f} +/- {metrics['position_accuracy_std']:.4f}; "
            f"pooled={metrics['position_accuracy_pooled']:.4f}"
        )
    else:
        metrics = dict(fold_rows[0])
        for key in (
            "run_id",
            "fold",
            "held_out_user",
            "validation_user",
            "n_train_windows",
            "n_test_windows",
            "trials_used",
        ):
            metrics.pop(key, None)
        if split_mode == "cross_session":
            user_rows = _cross_session_user_rows(
                resolved_run_id,
                folds[0][0],
                predictions,
                metrics,
            )
            upsert_fold_rows(user_rows, results_root=result_root)

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    metrics["peak_cuda_memory_bytes"] = peak_memory
    metrics["torch_version"] = str(torch.__version__)
    print(f"[DL GPU] run_id={resolved_run_id} peak_cuda_memory_bytes={peak_memory}")
    trials = sorted(
        {
            trial
            for protocol_train, test_df in folds
            for trial in _trial_values(protocol_train, test_df)
        }
    )
    n_train = sum(len(train_df) for train_df, _ in folds)
    n_test = sum(len(test_df) for _, test_df in folds)
    run_row = build_run_row(
        run_id=resolved_run_id,
        preproc_opts=preproc,
        feat_opts=features,
        hyperparameters=resolved_params,
        metrics=metrics,
        trials_used=trials,
        n_train=n_train,
        n_test=n_test,
        n_classes=len(label_encoder.classes_),
        device=str(device),
    )
    upsert_run(
        run_row,
        hyperparameter_columns=resolved_params.keys(),
        results_root=result_root,
    )
    derive_table_from_runs("global_summary", results_root=result_root)
    return predictions, metrics, history_df


def train_evaluate_cnn_multi_seed(
    *,
    seeds: tuple[int, ...] = SEEDS,
    **kwargs: Any,
) -> dict[int, tuple[pd.DataFrame, dict[str, float], pd.DataFrame]]:
    """Execute one independent DL run per configured seed and report seed spread."""
    if not seeds:
        raise ValueError("At least one DL seed is required.")
    outputs = {
        int(seed): train_evaluate_cnn(seed=int(seed), **kwargs) for seed in seeds
    }
    result_root = Path(kwargs.get("results_dir", RESULTS_ROOT))
    seed_summary = derive_seed_summary(results_root=result_root)
    if not seed_summary.empty:
        print(
            "[DL seeds] mean +/- std across seeds written to tables/seed_summary; "
            "LOVO uses each seed's fold mean and keeps within-seed fold std separate."
        )
    return outputs


def append_cnn_summary_row(
    *,
    summary_path: Path,
    band: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> pd.DataFrame:
    """Compatibility shim: regenerate the global view from ``runs.csv`` only."""
    del summary_path, band, params, metrics
    print("[summary] runs.csv is authoritative; regenerating tables/global_summary.*")
    return derive_table_from_runs("global_summary", results_root=RESULTS_ROOT)


def save_label_classes(classes: np.ndarray, results_dir: Path = RESULTS_ROOT) -> None:
    path = Path(results_dir) / "manifests" / "cnn_label_classes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(map(str, classes)), indent=2), encoding="utf-8")
    print(f"[CNN] label classes saved to {path}")


def resolved_dataloader_settings(
    device: torch.device,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Resolve every DataLoader option from config for CUDA or CPU."""
    if device.type == "cuda":
        workers = int(params.get("cuda_num_workers", CUDA_NUM_WORKERS))
        settings = {
            "batch_size": int(params.get("cuda_batch_size", CUDA_BATCH_SIZE)),
            "num_workers": workers,
            "pin_memory": bool(params.get("cuda_pin_memory", CUDA_PIN_MEMORY)),
            "persistent_workers": bool(
                params.get("cuda_persistent_workers", CUDA_PERSISTENT_WORKERS)
            )
            and workers > 0,
            "prefetch_factor": int(
                params.get("cuda_prefetch_factor", CUDA_PREFETCH_FACTOR)
            ),
        }
        if workers == 0:
            settings.pop("prefetch_factor")
    else:
        workers = int(params.get("cpu_num_workers", CPU_NUM_WORKERS))
        settings = {
            "batch_size": int(params.get("cpu_batch_size", CPU_BATCH_SIZE)),
            "num_workers": workers,
            "pin_memory": bool(params.get("cpu_pin_memory", CPU_PIN_MEMORY)),
            "persistent_workers": bool(
                params.get("cpu_persistent_workers", CPU_PERSISTENT_WORKERS)
            )
            and workers > 0,
        }
    print(f"[DataLoader] device={device.type} settings={settings}")
    return settings


def _train_predict_fold(  # noqa: PLR0913, PLR0915
    *,
    band: str,
    arrays: dict[str, np.ndarray],
    targets: np.ndarray,
    label_encoder: LabelEncoder,
    device: torch.device,
    params: dict[str, Any],
    model_label: str,
    split_mode: SplitMode,
    train_df: pd.DataFrame,
    protocol_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results_root: Path,
    run_id: str,
    fold: object | None,
    force_retrain: bool,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    expected_metadata = prediction_cache_metadata(
        model=model_label,
        band=band,
        split_mode=split_mode,
        params=params,
        fold=None if fold is None else f"user-{fold}",
        random_state=int(params.get("seed", params.get("random_state", 42))),
        train_fingerprint=_window_identity_fingerprint(protocol_train_df),
        test_fingerprint=_window_identity_fingerprint(test_df),
    )
    branch_channels = {
        band_name: int(array.shape[1]) for band_name, array in arrays.items()
    }
    model = DualBandCNN(
        branch_channels,
        n_classes=len(label_encoder.classes_),
        params=params,
    ).to(device)
    total_params = parameter_count(model)
    print(
        f"[{model_label}] {band}/{split_mode} fold={fold or 'single'} "
        f"parameters={total_params}"
    )
    checkpoint = checkpoint_path(run_id, fold=fold, results_root=results_root)
    checkpoint_metadata_path = checkpoint.with_suffix(".pt.metadata.json")
    cached = None
    if not force_retrain:
        cached = load_predictions(
            results_root,
            model_label,
            band,
            split_mode,
            fold=None if fold is None else f"user-{fold}",
            expected_metadata=expected_metadata,
            run_id=run_id,
        )

    history_rows: list[dict[str, float]] = []
    fit_seconds = 0.0
    best_val_acc = float("nan")
    best_epoch = 0
    stopped_epoch = 0
    patience_triggered = False
    if cached is None:
        train_loader = _loader(
            arrays,
            train_df.index.to_numpy(dtype=np.int64),
            targets,
            params,
            device=device,
            shuffle=True,
        )
        val_loader = _loader(
            arrays,
            val_df.index.to_numpy(dtype=np.int64),
            targets,
            params,
            device=device,
            shuffle=False,
        )
        test_loader = _loader(
            arrays,
            test_df.index.to_numpy(dtype=np.int64),
            targets,
            params,
            device=device,
            shuffle=False,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        criterion = nn.CrossEntropyLoss()
        best_state = copy.deepcopy(model.state_dict())
        best_val_acc = -np.inf
        patience = int(params.get("patience", 15))
        epochs_without_improvement = 0
        started_at = time.perf_counter()
        for epoch in range(1, int(params.get("epochs", 50)) + 1):
            epoch_started = time.perf_counter()
            train_loss, train_acc = _run_epoch(
                model, train_loader, criterion, device, optimizer=optimizer
            )
            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, device, optimizer=None
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
                }
            )
            print(
                f"[{model_label}] {band}/{split_mode} fold={fold or 'single'} "
                f"epoch {epoch:02d}/{params.get('epochs', 50)}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"seconds={epoch_seconds:.1f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epoch_seconds > float(params["max_epoch_seconds"]):
                raise RuntimeError(
                    f"{band} epoch exceeded max_epoch_seconds={params['max_epoch_seconds']} "
                    f"({epoch_seconds:.1f}s)."
                )
            if epochs_without_improvement >= patience:
                patience_triggered = True
                stopped_epoch = epoch
                print(
                    f"[{model_label}] early stopping at epoch {epoch}; "
                    f"no validation improvement for {patience} epochs."
                )
                break
        fit_seconds = time.perf_counter() - started_at
        if not patience_triggered:
            stopped_epoch = len(history_rows)
        model.load_state_dict(best_state)
        torch.save(best_state, checkpoint)
        _save_checkpoint_metadata(
            checkpoint_metadata_path,
            {
                "best_val_accuracy": float(best_val_acc),
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "patience_triggered": patience_triggered,
            },
        )
        predict_started = time.perf_counter()
        pred_labels = _predict(model, test_loader, device)
        predict_seconds = time.perf_counter() - predict_started
        pred_positions = label_encoder.inverse_transform(pred_labels)
        predictions = build_global_predictions_dataframe(
            test_df,
            pred_positions,
            dataset_name=band,
            model_name=model_label,
            split_mode=split_mode,
        )
        save_predictions(
            predictions,
            results_root,
            model_label,
            band,
            split_mode,
            fold=None if fold is None else f"user-{fold}",
            metadata=expected_metadata,
            run_id=run_id,
        )
    else:
        predictions = cached
        predict_seconds = 0.0
        checkpoint_metadata = _load_checkpoint_metadata(checkpoint_metadata_path)
        best_val_acc = float(checkpoint_metadata.get("best_val_accuracy", np.nan))
        best_epoch = int(checkpoint_metadata.get("best_epoch", 0))
        stopped_epoch = int(checkpoint_metadata.get("stopped_epoch", 0))
        patience_triggered = bool(checkpoint_metadata.get("patience_triggered", False))

    metrics = compute_localization_metrics(predictions)
    metrics.update(majority_class_baselines(protocol_train_df, test_df))
    metrics.update(
        {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "wall_seconds": float(fit_seconds + predict_seconds),
            "parameter_count": float(total_params),
            "best_val_accuracy": float(best_val_acc),
            "best_epoch": float(best_epoch),
            "stopped_epoch": float(stopped_epoch),
            "patience_triggered": patience_triggered,
            "mean_seconds_per_epoch": (
                float(np.mean([row["seconds"] for row in history_rows]))
                if history_rows
                else float("nan")
            ),
        }
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, metrics, pd.DataFrame(history_rows)


def _loader(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    targets: np.ndarray,
    params: dict[str, Any],
    *,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    settings = resolved_dataloader_settings(device, params)
    return DataLoader(
        RawCsiWindowDataset(arrays, indices, targets[indices]),
        shuffle=shuffle,
        **settings,
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
    non_blocking = device.type == "cuda"
    with context:
        for inputs, target in loader:
            inputs = {
                band: tensor.to(device, non_blocking=non_blocking)
                for band, tensor in inputs.items()
            }
            target = target.to(device, non_blocking=non_blocking)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
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
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch_inputs, _ in loader:
            device_inputs = {
                band: tensor.to(device, non_blocking=non_blocking)
                for band, tensor in batch_inputs.items()
            }
            logits = model(device_inputs)
            predictions.append(logits.argmax(dim=1).cpu().tolist())
    return np.asarray([value for batch in predictions for value in batch], dtype=np.int64)


def _aggregate_fold_metrics(
    per_fold: pd.DataFrame,
    pooled_predictions: pd.DataFrame,
) -> dict[str, float]:
    metric_names = [
        "position_accuracy",
        "macro_f1",
        "room_accuracy",
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
        "p90_distance_error",
        "majority_position_accuracy",
        "majority_room_accuracy",
        "best_val_accuracy",
        "best_epoch",
        "mean_seconds_per_epoch",
    ]
    aggregated: dict[str, float] = {}
    for metric_name in metric_names:
        values = pd.to_numeric(per_fold[metric_name], errors="coerce")
        aggregated[f"{metric_name}_mean"] = float(values.mean())
        aggregated[f"{metric_name}_std"] = float(values.std(ddof=1))
    pooled = compute_localization_metrics(pooled_predictions)
    aggregated["position_accuracy"] = aggregated["position_accuracy_mean"]
    aggregated["macro_f1"] = aggregated["macro_f1_mean"]
    aggregated["room_accuracy"] = aggregated["room_accuracy_mean"]
    for metric_name in (
        "mean_distance_error",
        "median_distance_error",
        "rmse_distance_error",
        "p90_distance_error",
        "majority_position_accuracy",
        "majority_room_accuracy",
        "best_val_accuracy",
        "best_epoch",
        "mean_seconds_per_epoch",
    ):
        aggregated[metric_name] = aggregated[f"{metric_name}_mean"]
    aggregated["position_accuracy_pooled"] = float(pooled["position_accuracy"])
    aggregated["n_folds"] = float(len(per_fold))
    return aggregated


def _cross_session_user_rows(
    run_id: str,
    train_df: pd.DataFrame,
    predictions: pd.DataFrame,
    run_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for user, user_predictions in predictions.groupby("user", sort=True):
        metrics = compute_localization_metrics(user_predictions)
        user_test_locations = user_predictions[["true_position"]].rename(
            columns={"true_position": "location"}
        )
        metrics.update(majority_class_baselines(train_df, user_test_locations))
        rows.append(
            {
                "run_id": run_id,
                "fold": str(user),
                "held_out_user": user,
                "validation_user": None,
                "n_train_windows": len(train_df),
                "n_test_windows": len(user_predictions),
                "trials_used": "01,02",
                **metrics,
                "fit_seconds": run_metrics.get("fit_seconds", np.nan),
                "predict_seconds": run_metrics.get("predict_seconds", np.nan),
                "wall_seconds": run_metrics.get("wall_seconds", np.nan),
            }
        )
    return rows


def _fold_fingerprint(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    *,
    side: Literal["train", "test"],
) -> str:
    frames = [train if side == "train" else test for train, test in folds]
    return _window_identity_fingerprint(pd.concat(frames, ignore_index=True))


def _trial_values(*frames: pd.DataFrame) -> set[str]:
    return {
        str(value).removeprefix("trial_").zfill(2)
        for frame in frames
        for value in frame["trial"].dropna().unique()
    }


def _trials_used(*frames: pd.DataFrame) -> str:
    return ",".join(sorted(_trial_values(*frames)))


def _single_user(df: pd.DataFrame) -> object:
    users = df["user"].dropna().unique()
    if len(users) != 1:
        raise ValueError(f"Expected one LOVO test user, observed {users!r}.")
    return users[0]


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


def _save_history_plot(history: pd.DataFrame, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title(f"{title} loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title(f"{title} accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.savefig(save_path, dpi=PLOT_DPI, format=PLOT_FORMAT)
    plt.close(fig)
    print(
        f"[plots] saved {save_path} at dpi={PLOT_DPI}. PNG is raster; set "
        "PLOT_FORMAT='pdf' for a LaTeX vector figure without other code changes."
    )


def _window_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["group_id", "window_idx"]].reset_index(drop=True)


def _version_pair(version: str) -> tuple[int, int]:
    numbers = []
    for part in version.split("+")[0].split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple((numbers + [0, 0])[:2])  # type: ignore[return-value]
