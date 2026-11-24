"""Tests for the notebook PDF report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd

from utils.plots import FIGURE_HOOKS, plot_metric_by_esp
from utils.report import NotebookReport

if TYPE_CHECKING:
    from pathlib import Path


def _pilot_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "esp": [f"esp_{i % 3 + 11:02d}" for i in range(24)],
            "trial": ["01", "02"] * 12,
            "spike_packet_fraction": rng.random(24),
        }
    )


def test_report_captures_figures_replaces_reruns_and_writes_pdf(tmp_path: Path) -> None:
    report = NotebookReport("Test report", settings={"band": "Fusion"})
    report.section("1. Section", "Intro with **bold** and `code`.\n- a bullet")
    long_table = pd.DataFrame(np.arange(300.0).reshape(100, 3), columns=["a", "b", "c"])
    report.table(long_table, "Long table", max_rows=100)
    for title in ("first", "second"):  # a re-run of the same cell replaces the figure
        with report.capture("Pilot spikes"):
            plot_metric_by_esp(
                _pilot_table(), metric="spike_packet_fraction", hue="trial", title=title, show=False
            )
    with report.capture("Ideal view", when=False):
        plot_metric_by_esp(_pilot_table(), metric="spike_packet_fraction", show=False)

    section = next(iter(report._sections.values()))  # noqa: SLF001
    assert list(section.blocks) == ["__intro__", "table:Long table", "figure:Pilot spikes"]
    assert len(section.blocks["figure:Pilot spikes"]) == 1
    assert not FIGURE_HOOKS

    path = report.build(tmp_path / "report.pdf")
    assert path.read_bytes().startswith(b"%PDF")


def test_disabled_report_collects_and_writes_nothing(tmp_path: Path) -> None:
    report = NotebookReport("Off", enabled=False)
    report.section("1. Section")
    report.table(pd.DataFrame({"a": [1]}), "Table")
    with report.capture("Plot"):
        plot_metric_by_esp(_pilot_table(), metric="spike_packet_fraction", show=False)

    assert not report._sections  # noqa: SLF001
    assert report.build(tmp_path / "off.pdf") is None
    assert not (tmp_path / "off.pdf").exists()
