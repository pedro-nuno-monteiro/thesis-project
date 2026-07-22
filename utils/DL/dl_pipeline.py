from __future__ import annotations

import copy
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.cache import (
    RESULTS_ROOT,
    get_all_dataframes,
    get_cache_path,
    get_results_path,
    load_window_array_cache,
    load_window_arrays,
    load_predictions,
    make_run_id,
    open_window_array_writers,
    prediction_cache_metadata,
    save_predictions,
    save_window_array_metadata,
)
from utils.config import (
    ANCHOR_GROUPS,
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
    SEEDS,
)
from utils.csi_processing import process_magnitude_data
from utils.DL.models import DualBandCNN
from utils.feature_pipeline import (
    METADATA_COLUMNS,
    CsiMap,
    FeatureScenario,
    build_frequency_feature_dataframes,
    iter_window_groups,
)
from utils.import_data import get_csv_files, sort_meta_info
from utils.load_csi import process_csv_files
from utils.ML.ml_pipeline import (
    _window_identity_fingerprint,
    split_dataframe,
)
from utils.plots import save_training_curves
from utils.results import (
    build_global_predictions_dataframe,
    build_run_row,
    checkpoint_path,
    compute_localization_metrics,
    derive_seed_summary,
    derive_table_from_runs,
    majority_class_baselines,
    upsert_fold_rows,
    upsert_run,
    write_run_manifest,
)

SplitMode = Literal["block", "lovo", "cross_session"]


def prepare_dl_data(
    data_dir: Path,
    *,
    calibration_mode: str,
    csv_options: dict[str, Any],
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
) -> tuple[CsiMap, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Discover, load, preprocess, and cache the data used by the CNN pipeline."""
    data_files = get_csv_files(str(data_dir))
    scenarios, locations, users, esps, _ = sort_meta_info(str(data_dir))
    print(f"Scenarios present: {', '.join(scenarios) or 'none'}")
    print(f"Locations: {len(locations)} | Users: {len(users)} | ESPs: {len(esps)}")
    magnitude_data, csv_diagnostics = process_csv_files(
        data_files,
        return_diagnostics=True,
        calibration_mode=calibration_mode,
        **csv_options,
    )
    processed_magnitude_data, magnitude_summary = process_magnitude_data(
        magnitude_data,
        **preproc_opts,
    )
    feature_dataframes = get_all_dataframes(
        preproc_opts,
        feat_opts,
        lambda: build_frequency_feature_dataframes(processed_magnitude_data, **feat_opts),
    )
    return (
        processed_magnitude_data,
        feature_dataframes,
        csv_diagnostics,
        magnitude_summary,
    )


def create_position_label_encoder(
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    results_dir: Path,
    expected_classes: int | None = 52,
) -> LabelEncoder:
    """Fit the shared position encoder and store its class ordering."""
    all_locations = sorted(
        set().union(*(set(df["location"].astype(str)) for df in feature_dataframes.values()))
    )
    if expected_classes is not None and len(all_locations) != expected_classes:
        msg = (
            f"Expected {expected_classes} position classes, found {len(all_locations)}: "
            f"{all_locations}"
        )
        raise RuntimeError(msg)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_locations)
    save_label_classes(label_encoder.classes_, results_dir)
    return label_encoder


def run_dl_experiments(  # noqa: PLR0913, PLR0914
    processed_magnitude_data: CsiMap,
    feature_dataframes: dict[str, pd.DataFrame],
    *,
    bands: tuple[str, ...],
    split_modes: tuple[SplitMode, ...],
    label_encoder: LabelEncoder,
    device: torch.device,
    results_dir: Path,
    plots_dir: Path,
    params: dict[str, Any],
    preproc_opts: dict[str, Any],
    feat_opts: dict[str, Any],
    expected_subcarriers: dict[str, int],
    expected_anchors: dict[str, int],
    seeds: tuple[int, ...] = SEEDS,
    test_size: float = 0.30,
    random_state: int = 42,
    n_blocks: int = 10,
    val_size: float = 0.15,
    force_retrain: bool = False,
) -> dict[tuple[str, str, int], tuple[pd.DataFrame, dict[str, float], pd.DataFrame]]:
    """Run the selected CNN bands and data-splitting protocols."""
    cnn_runs: dict[
        tuple[str, str, int],
        tuple[pd.DataFrame, dict[str, float], pd.DataFrame],
    ] = {}
    # Build and validate one aligned raw-window representation per requested band.
    for band in bands:
        print(f"\n=== {band} ===")
        arrays, meta = build_frequency_window_arrays(
            processed_magnitude_data,
            band,
            window_size=int(feat_opts["window_size"]),
            overlap_size=int(feat_opts["overlap_size"]),
            require_all_esps=bool(feat_opts["require_all_esps"]),
            preproc_opts=preproc_opts,
        )
        for branch_band, array in arrays.items():
            observed_subcarriers = int(array.shape[2])
            observed_anchors = int(array.shape[1])
            print(
                f"{branch_band}: anchors={observed_anchors}, "
                f"subcarriers={observed_subcarriers}, windows={array.shape[0]}"
            )
            if observed_subcarriers != expected_subcarriers[branch_band]:
                raise RuntimeError(
                    f"{branch_band} subcarrier count changed: expected "
                    f"{expected_subcarriers[branch_band]}, observed {observed_subcarriers}"
                )
            if observed_anchors != expected_anchors[branch_band]:
                raise RuntimeError(
                    f"{branch_band} anchor count changed: expected "
                    f"{expected_anchors[branch_band]}, observed {observed_anchors}"
                )

        if band == "Fusion":
            n_24 = arrays["2.4 GHz"].shape[0]
            n_5 = arrays["5 GHz"].shape[0]
            if n_24 != n_5 or n_24 != len(feature_dataframes[band]):
                raise RuntimeError(
                    f"Fusion N mismatch: 2.4={n_24}, 5={n_5}, "
                    f"ML={len(feature_dataframes[band])}"
                )
            print(f"Fusion N PASS: {n_24} windows")

        # Prove the raw CNN arrays and ML DataFrame describe the same windows
        # before reusing the shared split logic.
        assert_window_identity(band=band, meta=meta, feature_df=feature_dataframes[band])
        for split_mode in split_modes:
            assert_split_identity(
                band=band,
                meta=meta,
                feature_df=feature_dataframes[band],
                split_mode=split_mode,
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
            )
            # Each seed is a distinct recorded run, while band arrays are reused.
            seed_outputs = train_evaluate_cnn_multi_seed(
                seeds=seeds,
                band=band,
                arrays=arrays,
                meta=meta,
                label_encoder=label_encoder,
                device=device,
                results_dir=results_dir,
                plots_dir=plots_dir,
                params=params,
                split_mode=split_mode,
                test_size=test_size,
                random_state=random_state,
                n_blocks=n_blocks,
                val_size=val_size,
                force_retrain=force_retrain,
                preproc_opts=preproc_opts,
                feat_opts=feat_opts,
            )
            for seed, output in seed_outputs.items():
                cnn_runs[(band, split_mode, seed)] = output
    return cnn_runs


def show_dl_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the DL run summary and the RF/CNN block-split comparison."""
    global_summary = pd.read_csv(results_dir / "runs.csv")
    cnn_summary = global_summary.loc[global_summary["family"].eq("dl")].copy()
    comparison = global_summary.loc[
        global_summary["model"].isin(["rf", "cnn"])
        & global_summary["split"].eq("block"),
        ["band", "model", "seed", "position_accuracy", "parameter_count"],
    ].sort_values(["band", "model", "seed"])
    return cnn_summary, comparison


class RawCsiWindowDataset(Dataset):
    """Expose cached multi-band CSI windows and encoded targets to PyTorch."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Store shared arrays and the row indices selected for this split."""
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        """Return the number of selected windows."""
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Return one band-to-tensor sample mapping and its class target."""
        array_index = int(self.indices[item])
        sample = {
            band: torch.as_tensor(array[array_index], dtype=torch.float32)
            for band, array in self.arrays.items()
        }
        target = torch.as_tensor(self.targets[item], dtype=torch.long)
        return sample, target


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
    """Return the total number of trainable and non-trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def assert_window_identity(
    *,
    band: str,
    meta: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    """Assert that raw window arrays follow the same row order as ML features."""
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
    # When ML features are available, prove both pipelines use identical windows
    # and splits before training the CNN.
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

    # Train each protocol fold independently; LOVO reserves one whole training
    # user for validation to keep early stopping user-independent.
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
        # The fold helper owns model construction, checkpointing, early stopping,
        # prediction caching, and fold-level metric calculation.
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
            save_training_curves(
                history,
                plot_root / f"training_curves{fold_suffix}",
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


def save_label_classes(classes: np.ndarray, results_dir: Path = RESULTS_ROOT) -> None:
    """Persist the ordered position labels used by CNN classification heads."""
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
    """Train or restore one CNN fold and return predictions, metrics, and history."""
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
    # Prediction metadata fingerprints both protocol sides, preventing reuse when
    # a split changes even if its human-readable run label stays the same.
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
        # DataLoaders share mmap-backed arrays and differ only in selected indices.
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
        # Keep the best validation state in memory and stop after the configured
        # number of epochs without improvement.
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
        # Restore the best epoch before checkpoint export and test prediction.
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
    """Create a DataLoader over selected indices in shared CSI window arrays."""
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
    """Run one training or validation epoch and return mean loss and accuracy."""
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
    """Predict encoded class labels for every sample in a DataLoader."""
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
    """Aggregate CNN LOVO metrics across folds and compute pooled accuracy."""
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
    """Build per-user result rows for a cross-session prediction table."""
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
    """Fingerprint all train or test window identities in a fold collection."""
    frames = [train if side == "train" else test for train, test in folds]
    return _window_identity_fingerprint(pd.concat(frames, ignore_index=True))


def _trial_values(*frames: pd.DataFrame) -> set[str]:
    """Return normalized trial identifiers found across DataFrames."""
    return {
        str(value).removeprefix("trial_").zfill(2)
        for frame in frames
        for value in frame["trial"].dropna().unique()
    }


def _trials_used(*frames: pd.DataFrame) -> str:
    """Return the normalized trial identifiers as a comma-separated value."""
    return ",".join(sorted(_trial_values(*frames)))


def _single_user(df: pd.DataFrame) -> object:
    """Return the single user represented by a LOVO test fold."""
    users = df["user"].dropna().unique()
    if len(users) != 1:
        raise ValueError(f"Expected one LOVO test user, observed {users!r}.")
    return users[0]


def _save_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Atomically persist early-stopping metadata beside a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load checkpoint metadata, returning an empty mapping when unavailable."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _window_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Return stable group/window identity columns for ordering comparisons."""
    return df[["group_id", "window_idx"]].reset_index(drop=True)


def _version_pair(version: str) -> tuple[int, int]:
    """Extract a major/minor integer pair from a version string."""
    numbers = []
    for part in version.split("+")[0].split("."):
        digits = "".join(character for character in part if character.isdigit())
        if not digits:
            break
        numbers.append(int(digits))
    return tuple((numbers + [0, 0])[:2])  # type: ignore[return-value]


# DL window-array preparation.

BandName = Literal["2.4 GHz", "5 GHz", "Fusion"]

DEFAULT_PREPROC_OPTS = {
    "normalization": "empty_baseline",
    "baseline_scope": "per_session",
}


def build_frequency_window_arrays(
    magnitude_data: CsiMap,
    frequency_scenario: BandName | FeatureScenario,
    *,
    window_size: int = 60,
    overlap_size: int = 30,
    require_all_esps: bool = True,
    preproc_opts: dict[str, Any] | None = None,
    force_rebuild: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build or load mmap-backed raw CSI amplitude windows for one band config."""
    band_name, scenario_key = _normalize_frequency_scenario(frequency_scenario)
    feat_opts = {
        "window_size": window_size,
        "overlap_size": overlap_size,
        "require_all_esps": require_all_esps,
    }
    feature_cache_dir = get_cache_path(preproc_opts or DEFAULT_PREPROC_OPTS, feat_opts)
    cache_dir = feature_cache_dir / "window_arrays" / _band_stem(band_name)
    print(f"[window arrays] cache path: {cache_dir.resolve()}")

    bands = ["2.4 GHz", "5 GHz"] if band_name == "Fusion" else [band_name]
    array_paths = {band: cache_dir / f"{_band_stem(band)}.npy" for band in bands}
    meta_path = cache_dir / "meta.parquet"
    manifest_path = cache_dir / "manifest.json"

    # Reuse mmap-backed arrays only when their metadata and manifest are complete.
    if not force_rebuild:
        cached = load_window_array_cache(array_paths, meta_path, manifest_path)
        if cached is not None:
            arrays, meta, manifest = cached
            _print_loaded_arrays(arrays, manifest, cache_dir)
            return arrays, meta

    # Discover aligned groups first so array shapes can be allocated once.
    groups = list(
        iter_window_groups(
            magnitude_data,
            scenario_key,
            window_size=window_size,
            overlap_size=overlap_size,
            require_all_esps=require_all_esps,
        )
    )
    total_windows = int(sum(group.min_windows for group in groups))
    if total_windows == 0:
        msg = f"No windows found for {band_name}."
        raise ValueError(msg)

    anchor_order = {band: list(ANCHOR_GROUPS[band]) for band in bands}
    subcarrier_counts = _subcarrier_counts(groups, anchor_order)
    shapes = {
        band: (
            total_windows,
            len(anchor_order[band]),
            subcarrier_counts[band],
            window_size,
        )
        for band in bands
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write each aligned window directly into its final band-specific mmap file.
    writers = open_window_array_writers(array_paths, shapes)
    rows: list[dict[str, object]] = []
    step = window_size - overlap_size
    row_idx = 0

    for group in groups:
        for window_idx in range(group.min_windows):
            start = window_idx * step
            for band in bands:
                for anchor_idx, esp_key in enumerate(anchor_order[band]):
                    magnitude = group.magnitudes_by_esp[esp_key]
                    window = magnitude[start : start + window_size]
                    writers[band][row_idx, anchor_idx] = window.T.astype(
                        np.float16,
                        copy=False,
                    )
            rows.append(
                {
                    "frequency_scenario": scenario_key,
                    "scenario": group.scenario_key.removeprefix("scenario_"),
                    "location": group.location_key.removeprefix("location_"),
                    "user": group.user_key.removeprefix("user_"),
                    "trial": group.trial_key.removeprefix("trial_"),
                    "group_id": group.group_id,
                    "window_idx": window_idx,
                    "label": group.label,
                }
            )
            row_idx += 1

    for writer in writers.values():
        writer.flush()
        del writer

    meta = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
    manifest = {
        "frequency_scenario": band_name,
        "feature_scenario": scenario_key,
        "window_size": window_size,
        "overlap_size": overlap_size,
        "require_all_esps": require_all_esps,
        "preprocessing": preproc_opts or DEFAULT_PREPROC_OPTS,
        "anchor_order": anchor_order,
        "subcarrier_counts": subcarrier_counts,
        "shapes": {band: list(shape) for band, shape in shapes.items()},
        "dtype": "float16",
        "meta_rows": int(len(meta)),
    }
    save_window_array_metadata(meta, meta_path, manifest, manifest_path)

    arrays = load_window_arrays(array_paths)
    _print_array_summary(arrays, anchor_order)
    return arrays, meta


def _normalize_frequency_scenario(
    frequency_scenario: BandName | FeatureScenario,
) -> tuple[BandName, FeatureScenario]:
    """Resolve accepted band aliases to display and feature-scenario names."""
    aliases: dict[str, tuple[BandName, FeatureScenario]] = {
        "2.4 ghz": ("2.4 GHz", "2.4 GHz"),
        "2.4ghz": ("2.4 GHz", "2.4 GHz"),
        "2_4ghz": ("2.4 GHz", "2.4 GHz"),
        "5 ghz": ("5 GHz", "5 GHz"),
        "5ghz": ("5 GHz", "5 GHz"),
        "fusion": ("Fusion", "Fusion"),
    }
    key = str(frequency_scenario).strip().lower()
    try:
        return aliases[key]
    except KeyError as exc:
        msg = f"Unknown frequency scenario {frequency_scenario!r}."
        raise ValueError(msg) from exc


def _subcarrier_counts(
    groups: list,
    anchor_order: dict[str, list[str]],
) -> dict[str, int]:
    """Infer one subcarrier width for each band from aligned window groups."""
    counts: dict[str, int] = {}
    for band, esp_keys in anchor_order.items():
        for esp_key in esp_keys:
            for group in groups:
                magnitude = group.magnitudes_by_esp.get(esp_key)
                if magnitude is not None:
                    counts[band] = int(magnitude.shape[1])
                    break
            if band in counts:
                break
        if band not in counts:
            msg = f"Could not infer subcarrier count for {band}."
            raise ValueError(msg)
    return counts


def _print_loaded_arrays(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    cache_dir: Path,
) -> None:
    """Print the location and shape summary of cached window arrays."""
    print(f"[window arrays cache hit] {cache_dir.resolve()}")
    _print_array_summary(arrays, manifest["anchor_order"])


def _print_array_summary(
    arrays: dict[str, np.ndarray],
    anchor_order: dict[str, list[str]],
) -> None:
    """Print shapes, dtypes, and anchor order for window arrays."""
    for band, array in arrays.items():
        print(f"[window arrays] {band}: shape={tuple(array.shape)}, dtype={array.dtype}")
        print(f"[window arrays] {band} anchors: {anchor_order[band]}")


def _band_stem(band: str) -> str:
    """Convert a display band name to its compact cache-path stem."""
    return band.lower().replace(".", "_").replace(" ", "")
