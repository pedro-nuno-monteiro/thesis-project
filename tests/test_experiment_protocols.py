from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.cache import (
    _prediction_metadata_matches,
    make_run_id,
    parse_run_id,
    prediction_cache_metadata,
)
from utils.global_position_classifier import (
    _window_identity_fingerprint,
    split_dataframe,
)
from utils.results import build_run_row, upsert_run


def _protocol_frame() -> pd.DataFrame:
    rows = []
    for trial in ("01", "02"):
        for user in ("01", "02"):
            for window_idx in range(20):
                rows.append(
                    {
                        "trial": trial,
                        "user": user,
                        "location": "A-1" if window_idx % 2 else "A-2",
                        "group_id": f"{trial}-{user}",
                        "window_idx": window_idx,
                        "label": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_non_cross_session_splits_filter_to_trial_01() -> None:
    frame = _protocol_frame()
    for split_mode in ("block", "group", "random"):
        train_df, test_df = split_dataframe(frame, split_mode=split_mode)
        assert set(train_df["trial"]) == {"01"}
        assert set(test_df["trial"]) == {"01"}
    folds = split_dataframe(frame, split_mode="lovo")
    assert isinstance(folds, list)
    assert all(set(train["trial"]) == {"01"} for train, _ in folds)
    assert all(set(test["trial"]) == {"01"} for _, test in folds)


def test_cross_session_does_not_apply_training_protocol_filter() -> None:
    train_df, test_df = split_dataframe(_protocol_frame(), split_mode="cross_session")
    assert set(train_df["trial"]) == {"01"}
    assert set(test_df["trial"]) == {"02"}


def test_trial_filter_empty_error_is_clear() -> None:
    frame = _protocol_frame().loc[lambda value: value["trial"] == "02"]
    try:
        split_dataframe(frame, split_mode="block")
    except ValueError as exc:
        assert "trial filter emptied the dataframe" in str(exc)
    else:
        raise AssertionError("Expected the empty trial filter to raise ValueError.")


def test_run_id_round_trip_and_config_hash_changes() -> None:
    config = {
        "preproc_opts": {
            "normalization": "empty_baseline",
            "baseline_scope": "per_session",
        },
        "feat_opts": {"window_size": 60, "overlap_size": 30},
        "model_params": {"n_estimators": 500},
        "seed": 42,
        "split_params": {"test_size": 0.3},
    }
    run_id = make_run_id(
        family="ml",
        model="rf",
        band="Fusion",
        split="block",
        normalization="empty_baseline",
        baseline_scope="per_session",
        seed=42,
        config=config,
    )
    parsed = parse_run_id(run_id)
    assert parsed["family"] == "ml"
    assert parsed["model"] == "rf"
    assert parsed["band"] == "fusion"
    assert parsed["baseline_scope"] == "session"
    assert parsed["seed"] == 42
    changed = {
        **config,
        "feat_opts": {**config["feat_opts"], "window_size": 61},
    }
    changed_run_id = make_run_id(
        family="ml",
        model="rf",
        band="Fusion",
        split="block",
        normalization="empty_baseline",
        baseline_scope="per_session",
        seed=42,
        config=changed,
    )
    assert changed_run_id != run_id


def test_prediction_cache_checks_train_and_test_fingerprints(
    tmp_path: Path,
    capsys,
) -> None:
    frame = _protocol_frame()
    train_df, test_df = split_dataframe(frame, split_mode="block")
    metadata = prediction_cache_metadata(
        model="RF",
        band="Fusion",
        split_mode="block",
        params={},
        train_fingerprint=_window_identity_fingerprint(train_df),
        test_fingerprint=_window_identity_fingerprint(test_df),
    )
    parquet_path = tmp_path / "predictions.parquet"
    parquet_path.with_suffix(".parquet.metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    assert _prediction_metadata_matches(parquet_path, metadata)
    changed = {**metadata, "train_fingerprint": "different"}
    assert not _prediction_metadata_matches(parquet_path, changed)
    assert "train fingerprint changed" in capsys.readouterr().out


def test_runs_csv_upserts_by_run_id(tmp_path: Path) -> None:
    config = {"model_params": {"n_estimators": 10}}
    run_id = make_run_id(
        family="ml",
        model="rf",
        band="Fusion",
        split="block",
        normalization="none",
        seed=42,
        config=config,
    )
    row = build_run_row(
        run_id=run_id,
        preproc_opts={"normalization": "none"},
        feat_opts={"window_size": 60, "overlap_size": 30},
        hyperparameters={"n_estimators": 10},
        metrics={"position_accuracy": 0.1},
        trials_used=["01"],
        n_train=10,
        n_test=2,
        n_classes=2,
    )
    assert len(upsert_run(row, results_root=tmp_path)) == 1
    row["position_accuracy"] = 0.2
    updated = upsert_run(row, results_root=tmp_path)
    assert len(updated) == 1
    assert updated.iloc[0]["position_accuracy"] == 0.2
