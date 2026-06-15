#!/usr/bin/env python3
"""Cache and results management CLI for the CSI localization pipeline.

Usage:
    python scripts/manage_cache.py list
    python scripts/manage_cache.py clean [--all | --older-than DAYS]
    python scripts/manage_cache.py results clean [--all | --older-than DAYS]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataframes"
RESULTS_DIR = PROJECT_ROOT / "results"


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def _confirm(msg: str) -> bool:
    try:
        response = input(f"{msg} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return response in ("y", "yes")


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list(_args: argparse.Namespace) -> None:
    if not CACHE_DIR.exists():
        print("No cache directory found (.cache/dataframes/ does not exist).")
        return

    options_files = sorted(CACHE_DIR.rglob("options.json"))
    if not options_files:
        print("Cache is empty.")
        return

    for options_file in options_files:
        folder = options_file.parent
        size_mb = _dir_size_mb(folder)
        rel = folder.relative_to(PROJECT_ROOT)

        try:
            opts: dict = json.loads(options_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            opts = {}

        preproc = opts.get("preprocessing", {})
        feat = opts.get("feature_extraction", {})

        print(f"\n{rel}/  [{size_mb:.1f} MB]")

        agc = preproc.get("apply_agc_compensation", False)
        filt = preproc.get("filter_method", "none")
        norm = preproc.get("normalization", "none")
        print(f"  preprocessing: AGC={'on' if agc else 'off'}, filter={filt}, normalization={norm}")

        win = feat.get("window_size", "?")
        step = feat.get("overlap_size", "?")
        cal = feat.get("calibrate", False)
        all_esps = feat.get("require_all_esps", False)
        print(f"  feature extraction: window={win}, step={step}, calibrate={cal}, require_all_esps={all_esps}")

        parquet_files = sorted(folder.glob("*.parquet"))
        if parquet_files:
            names = ", ".join(f.stem for f in parquet_files)
            print(f"  cached bands: {names}")


# ── clean helpers ─────────────────────────────────────────────────────────────

def _collect_folders(root: Path, older_than_days: int | None) -> list[Path]:
    """Return leaf cache/results folders (those containing an options.json or manifest.json)."""
    marker_files = list(root.rglob("options.json")) + list(root.rglob("manifest.json"))
    folders = sorted({f.parent for f in marker_files})

    if older_than_days is not None:
        cutoff = time.time() - older_than_days * 86400
        folders = [d for d in folders if d.stat().st_mtime < cutoff]

    return folders


def _run_clean(root: Path, args: argparse.Namespace, label: str) -> None:
    if not root.exists():
        print(f"No {label} directory found.")
        return

    folders = _collect_folders(root, getattr(args, "older_than", None))
    if not folders:
        print(f"No matching {label} folders found.")
        return

    total_mb = sum(_dir_size_mb(d) for d in folders)
    print(f"Found {len(folders)} {label} folder(s) totalling {total_mb:.1f} MB:")
    for d in folders:
        print(f"  {d.relative_to(PROJECT_ROOT)}/")

    if not getattr(args, "all", False) and not _confirm(f"Delete these {label} folders?"):
        print("Aborted.")
        return

    for d in folders:
        shutil.rmtree(d)
        print(f"Deleted {d.relative_to(PROJECT_ROOT)}/")
    print("Done.")


def cmd_clean(args: argparse.Namespace) -> None:
    _run_clean(CACHE_DIR, args, "cache")


def cmd_results_clean(args: argparse.Namespace) -> None:
    _run_clean(RESULTS_DIR, args, "results")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_clean_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Remove all folders without prompting.")
    group.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Remove folders whose mtime is older than DAYS days.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage cached feature dataframes and results for the CSI localization pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("list", help="List cache folders with sizes and preprocessing options.")

    clean_p = sub.add_parser("clean", help="Remove cache folders.")
    _add_clean_args(clean_p)

    results_p = sub.add_parser("results", help="Manage the results/ directory.")
    results_sub = results_p.add_subparsers(dest="results_command", metavar="SUBCOMMAND")
    results_clean_p = results_sub.add_parser("clean", help="Remove results folders.")
    _add_clean_args(results_clean_p)

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "results":
        if getattr(args, "results_command", None) == "clean":
            cmd_results_clean(args)
        else:
            results_p.print_help()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
