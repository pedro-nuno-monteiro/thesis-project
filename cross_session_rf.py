from __future__ import annotations

from pathlib import Path

from utils.cache import get_results_path
from utils.ml_pipeline import (
    load_feature_dataframes,
    load_params_lookup,
    load_raw_csi_data,
    run_global_baselines,
    validate_cross_session_inventory,
)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PREPROCESSING = {
    "normalization": "empty_baseline",
    "baseline_scope": "per_session",
    "epsilon": 1e-8,
}
FEATURES = {
    "window_size": 60,
    "overlap_size": 30,
    "require_all_esps": False,
}
BANDS = ("2.4 GHz", "5 GHz", "Fusion")
MODELS = ("RF",)


def _select_protocol_trials(magnitude_data: dict[str, object]) -> dict[str, object]:
    """Keep only the two sessions defined by the cross-session protocol."""
    selected: dict[str, object] = {}
    excluded = 0
    for scenario_key, locations in magnitude_data.items():
        selected_locations = {}
        for location_key, users in locations.items():
            selected_users = {}
            for user_key, esps in users.items():
                selected_esps = {}
                for esp_key, trials in esps.items():
                    selected_trials = {}
                    for trial_key, magnitude in trials.items():
                        trial = str(trial_key).removeprefix("trial_").zfill(2)
                        if trial in {"01", "02"}:
                            selected_trials[trial_key] = magnitude
                        else:
                            excluded += 1
                    if selected_trials:
                        selected_esps[esp_key] = selected_trials
                if selected_esps:
                    selected_users[user_key] = selected_esps
            if selected_users:
                selected_locations[location_key] = selected_users
        if selected_locations:
            selected[scenario_key] = selected_locations
    print(f"[cross_session] excluded {excluded} recording(s) outside trials 01/02")
    return selected


def main() -> None:
    """Run the frozen-train/new-session RF evaluation protocol."""
    magnitude_data, diagnostics = load_raw_csi_data(
        DATA_DIR,
        calibration_mode="rssi",
        csv_options={
            "max_workers": 8,
            "cache_dir": None,
            "use_cache": True,
            "force_reprocess": False,
            "min_rssi_dbm": -95.0,
            "calibration_eps": 1e-12,
        },
    )
    validate_cross_session_inventory(diagnostics)
    magnitude_data = _select_protocol_trials(magnitude_data)
    feature_dataframes = load_feature_dataframes(
        magnitude_data,
        preproc_opts=PREPROCESSING,
        feat_opts=FEATURES,
        bands_to_run=BANDS,
    )
    results_dir = get_results_path(PREPROCESSING, FEATURES)
    summary_dir = results_dir / "summary"
    params = load_params_lookup(
        results_dir / "tuning" / "tuned_summary.csv",
        models_to_run=MODELS,
        bands_to_run=BANDS,
        svm_kernel="rbf",
    )
    summary, _ = run_global_baselines(
        feature_dataframes,
        params_lookup=params,
        models_to_run=MODELS,
        bands_to_run=BANDS,
        split_modes=("cross_session",),
        test_size=0.3,
        random_state=42,
        n_blocks=10,
        n_jobs=-1,
        results_dir=results_dir,
        summary_dir=summary_dir,
        preproc_opts=PREPROCESSING,
        feat_opts=FEATURES,
        force_retrain=False,
        save_predictions=True,
        svm_fallback_seconds=30 * 60,
        row_spacing=1.0,
        column_spacing=1.0,
        frozen_block_results_dir=(PROJECT_ROOT / "results" / "preproc=agc-off" / "feat=win60-step30"),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
