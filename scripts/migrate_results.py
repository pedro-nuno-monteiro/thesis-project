"""One-shot, non-destructive migration from nested legacy results to flat runs.

The script performs a complete preflight before copying anything.  Missing split
parameters, seeds, or train counts are reported as ambiguities; no destination is
written in that case.  This is deliberate: inventing values would undermine the
content-addressed run identifier.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import make_run_id  # noqa: E402
from utils.metrics import compute_localization_metrics  # noqa: E402
from utils.results import (  # noqa: E402
    build_run_row,
    checkpoint_path,
    ensure_results_layout,
    upsert_fold_rows,
    upsert_run,
    write_run_manifest,
)


@dataclass(frozen=True)
class MigrationItem:
    source: Path
    legacy_root: Path
    band: str
    model: str
    split: str
    fold: str | None
    family: str
    preproc_opts: dict[str, Any]
    feat_opts: dict[str, Any]
    model_params: dict[str, Any]
    split_params: dict[str, Any]
    seed: int | None
    run_id: str
    summary_row: dict[str, Any]


def migrate(results_root: Path, *, dry_run: bool = False) -> list[MigrationItem]:
    """Preflight and then copy legacy artifacts, leaving the old tree untouched."""
    legacy_roots = sorted(results_root.glob("preproc=*/feat=*"))
    if not legacy_roots:
        raise FileNotFoundError(f"No legacy results/preproc=*/feat=*/ tree under {results_root}.")

    items: list[MigrationItem] = []
    ambiguities: list[str] = []
    for legacy_root in legacy_roots:
        try:
            items.extend(_preflight_root(legacy_root))
        except ValueError as exc:
            ambiguities.append(f"{legacy_root}: {exc}")

    print("old_path -> new_run_id")
    for item in items:
        print(f"{item.source} -> {item.run_id}")
    for ambiguity in ambiguities:
        print(f"AMBIGUOUS: {ambiguity}")
    if ambiguities:
        raise RuntimeError(
            "Migration stopped before writing because existing parameters could not be "
            "reconstructed unambiguously:\n" + "\n".join(ambiguities)
        )
    if dry_run:
        print(f"Dry run complete: {len(items)} prediction artifact(s) are unambiguous.")
        return items

    ensure_results_layout(results_root)
    pooled_items = [item for item in items if item.fold is None]
    pooled_by_run = {item.run_id: item for item in pooled_items}
    for item in items:
        destination_name = (
            f"{item.run_id}.parquet"
            if item.fold is None
            else f"{item.run_id}__fold-{item.fold.removeprefix('user-')}.parquet"
        )
        destination = results_root / "predictions" / destination_name
        shutil.copy2(item.source, destination)

    for run_id, item in pooled_by_run.items():
        full_config = {
            "preproc_opts": item.preproc_opts,
            "feat_opts": item.feat_opts,
            "model_params": item.model_params,
            "seed": item.seed,
            "split_params": item.split_params,
        }
        predictions = pd.read_parquet(item.source)
        metrics = _metrics_from_summary_or_predictions(item.summary_row, predictions)
        trials = sorted(
            predictions["trial"].astype(str).str.removeprefix("trial_").str.zfill(2).unique()
        )
        n_train = int(item.summary_row["n_train"])
        row = build_run_row(
            run_id=run_id,
            preproc_opts=item.preproc_opts,
            feat_opts=item.feat_opts,
            hyperparameters=item.model_params,
            metrics=metrics,
            trials_used=trials,
            n_train=n_train,
            n_test=len(predictions),
            n_classes=int(predictions["true_position"].nunique()),
            device=str(item.summary_row.get("device", "legacy")),
            timestamp=item.summary_row.get("timestamp"),
        )
        upsert_run(
            row,
            hyperparameter_columns=item.model_params.keys(),
            results_root=results_root,
        )
        write_run_manifest(run_id, full_config, results_root=results_root)
        _copy_checkpoint(item, results_root)

    _migrate_fold_rows(items, results_root)
    _copy_plots(items, results_root)
    print(f"Migration complete: copied {len(items)} prediction artifact(s); old tree untouched.")
    return items


def _preflight_root(legacy_root: Path) -> list[MigrationItem]:
    preproc_opts, feat_opts, manifest = _load_legacy_options(legacy_root)
    summaries = _load_summary_rows(legacy_root)
    predictions_dir = legacy_root / "predictions"
    if not predictions_dir.exists():
        return []
    items: list[MigrationItem] = []
    for source in sorted(predictions_dir.glob("*.parquet")):
        band, model, split, fold = _parse_legacy_prediction_name(source.stem)
        family = "dl" if model.lower().startswith("cnn") else "ml"
        metadata = _read_json(source.with_suffix(source.suffix + ".metadata.json")) or {}
        model_params = _model_params(manifest, metadata, model=model, band=band)
        summary_row = _matching_summary(summaries, band=band, model=model, split=split)
        split_params = _split_params(manifest, metadata, summary_row)
        seed = _seed(family, model, metadata, summary_row)
        missing = [
            name
            for name, value in {
                "model parameters": model_params,
                "test_size": split_params.get("test_size"),
                "random_state": split_params.get("random_state"),
                "n_blocks": split_params.get("n_blocks") if split == "block" else True,
                "n_train": summary_row.get("n_train"),
                "stochastic seed": seed if _is_stochastic(family, model) else True,
            }.items()
            if value in (None, {}, "")
        ]
        if missing:
            raise ValueError(
                f"{source.name} lacks {', '.join(missing)} in its summary/manifest/metadata"
            )
        full_config = {
            "preproc_opts": preproc_opts,
            "feat_opts": feat_opts,
            "model_params": model_params,
            "seed": seed,
            "split_params": split_params,
        }
        run_id = make_run_id(
            family=family,
            model="cnn" if family == "dl" else model,
            band=band,
            split=split,
            normalization=str(preproc_opts.get("normalization", "none")),
            baseline_scope=preproc_opts.get("baseline_scope"),
            seed=seed,
            config=full_config,
        )
        items.append(
            MigrationItem(
                source=source,
                legacy_root=legacy_root,
                band=band,
                model=model,
                split=split,
                fold=fold,
                family=family,
                preproc_opts=preproc_opts,
                feat_opts=feat_opts,
                model_params=model_params,
                split_params=split_params,
                seed=seed,
                run_id=run_id,
                summary_row=summary_row,
            )
        )
    return items


def _load_legacy_options(
    legacy_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root_manifest = _read_json(legacy_root / "manifest.json")
    if root_manifest is not None:
        return (
            dict(root_manifest.get("preprocessing_options") or {}),
            dict(root_manifest.get("feature_extraction_options") or {}),
            root_manifest,
        )
    window_manifests = [
        _read_json(path) for path in sorted((legacy_root / "window_arrays").glob("*/manifest.json"))
    ]
    window_manifests = [manifest for manifest in window_manifests if manifest is not None]
    if not window_manifests:
        raise ValueError("no root or window-array manifest")
    first = window_manifests[0]
    preproc = dict(first.get("preprocessing") or {})
    feat = {
        "window_size": first.get("window_size"),
        "overlap_size": first.get("overlap_size"),
        "require_all_esps": first.get("require_all_esps"),
    }
    for other in window_manifests[1:]:
        other_feat = {
            "window_size": other.get("window_size"),
            "overlap_size": other.get("overlap_size"),
            "require_all_esps": other.get("require_all_esps"),
        }
        if dict(other.get("preprocessing") or {}) != preproc or other_feat != feat:
            raise ValueError("window-array manifests disagree on preprocessing/features")
    return preproc, feat, {}


def _load_summary_rows(legacy_root: Path) -> list[dict[str, Any]]:
    path = legacy_root / "summary" / "global_summary.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).where(pd.notna(pd.read_csv(path)), None).to_dict("records")


def _parse_legacy_prediction_name(stem: str) -> tuple[str, str, str, str | None]:
    parts = stem.split("__")
    if len(parts) not in {3, 4}:
        raise ValueError(f"unrecognized prediction filename {stem!r}")
    band_lookup = {"2_4ghz": "2.4 GHz", "5ghz": "5 GHz", "fusion": "Fusion"}
    if parts[0] not in band_lookup:
        raise ValueError(f"unrecognized band stem {parts[0]!r}")
    return band_lookup[parts[0]], parts[1], parts[2], parts[3] if len(parts) == 4 else None


def _model_params(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    *,
    model: str,
    band: str,
) -> dict[str, Any]:
    if metadata.get("params"):
        return dict(metadata["params"])
    classifiers = manifest.get("classifier_hyperparameters") or {}
    return dict(classifiers.get(f"{model.upper()}:{band}") or {})


def _matching_summary(
    summaries: list[dict[str, Any]],
    *,
    band: str,
    model: str,
    split: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in summaries
        if str(row.get("dataset", "")).lower() == band.lower()
        and str(row.get("model", "")).lower() == model.lower()
        and str(row.get("split", "")).lower() == split.lower()
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple summary rows match {band}/{model}/{split}")
    return dict(matches[0]) if matches else {}


def _split_params(
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "test_size": summary.get("test_size") or manifest.get("test_size"),
        "random_state": metadata.get("random_state") or summary.get("random_state"),
        "n_blocks": summary.get("n_blocks"),
        "validation_size": summary.get("validation_size"),
        "row_spacing": metadata.get("row_spacing"),
        "column_spacing": metadata.get("column_spacing"),
    }


def _seed(
    family: str,
    model: str,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> int | None:
    if not _is_stochastic(family, model):
        return None
    value = metadata.get("random_state") or summary.get("random_state")
    return int(value) if value not in (None, "") else None


def _is_stochastic(family: str, model: str) -> bool:
    return family == "dl" or model.lower() in {"rf", "svm"}


def _metrics_from_summary_or_predictions(
    summary: dict[str, Any],
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    metrics = compute_localization_metrics(predictions)
    for key, value in summary.items():
        if key not in {"dataset", "model", "split", "n_train"} and value is not None:
            metrics[key] = value
    return metrics


def _copy_checkpoint(item: MigrationItem, results_root: Path) -> None:
    old_checkpoint = item.source.with_suffix(".pt")
    if old_checkpoint.exists():
        shutil.copy2(
            old_checkpoint,
            checkpoint_path(item.run_id, fold=item.fold, results_root=results_root),
        )


def _migrate_fold_rows(items: list[MigrationItem], results_root: Path) -> None:
    rows = []
    for item in items:
        if item.fold is None:
            continue
        predictions = pd.read_parquet(item.source)
        metrics = compute_localization_metrics(predictions)
        rows.append(
            {
                "run_id": item.run_id,
                "fold": item.fold.removeprefix("user-"),
                "held_out_user": item.fold.removeprefix("user-"),
                "validation_user": None,
                "n_train_windows": None,
                "n_test_windows": len(predictions),
                "trials_used": ",".join(
                    sorted(predictions["trial"].astype(str).str.zfill(2).unique())
                ),
                **metrics,
            }
        )
    upsert_fold_rows(rows, results_root=results_root)


def _copy_plots(items: list[MigrationItem], results_root: Path) -> None:
    by_root: dict[Path, list[MigrationItem]] = {}
    for item in items:
        if item.fold is None:
            by_root.setdefault(item.legacy_root, []).append(item)
    for legacy_root, root_items in by_root.items():
        for plot in sorted((legacy_root / "plots").glob("*")):
            if not plot.is_file():
                continue
            matching = _plot_run_candidates(plot.name, root_items)
            if not matching:
                raise RuntimeError(f"Plot {plot} could not be associated with any run.")
            for item in matching:
                destination = results_root / "plots" / item.run_id / plot.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(plot, destination)


def _plot_run_candidates(name: str, items: list[MigrationItem]) -> list[MigrationItem]:
    normalized = name.lower().replace(".", "-").replace("_", "-")
    candidates = items
    model_tokens = [model for model in {item.model.lower() for item in items} if model in normalized]
    if model_tokens:
        candidates = [item for item in candidates if item.model.lower() in model_tokens]
    split_tokens = [split for split in {item.split for item in items} if split in normalized]
    if split_tokens:
        candidates = [item for item in candidates if item.split in split_tokens]
    band_tokens = {
        "2-4-ghz": "2.4 GHz",
        "5-ghz": "5 GHz",
        "fusion": "Fusion",
    }
    matched_bands = [band for token, band in band_tokens.items() if token in normalized]
    if matched_bands:
        candidates = [item for item in candidates if item.band in matched_bands]
    return candidates


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    migrate(args.results_root.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
