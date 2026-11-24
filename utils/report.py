"""Structured PDF report of one notebook run, built with matplotlib only.

A notebook creates one :class:`NotebookReport`, mirrors its own headings with
:meth:`NotebookReport.section` / :meth:`NotebookReport.subsection`, adds the
tables and figures worth keeping, and finally calls :meth:`NotebookReport.build`::

    report = NotebookReport("CSI Fingerprint Analysis", settings={"band": "Fusion"})
    report.section("3. Within-Session Discriminability")
    with report.capture("Position bubbles"):
        plot_fingerprint_embedding(...)  # any utils.plots function
    report.table(scales, "Noise versus neighbour separation")
    report.build(report_dir / "report.pdf")

Every block is keyed by its heading and title, so re-running a cell replaces its
entries in place instead of appending duplicates.
"""

from __future__ import annotations

import io
import math
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from matplotlib import image as mimage
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextToPath

from utils import plots
from utils.results import _git_commit

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

# A4 portrait, in points.
PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 48.0, 54.0, 50.0
CONTENT_W = PAGE_W - 2 * MARGIN_X
CONTENT_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM

BODY_SIZE = 9.0
CAPTION_SIZE = 9.5
HEADING_SIZES = {1: 16.0, 2: 12.0}
TABLE_SIZE = 7.5
TABLE_MIN_SIZE = 5.5
TABLE_MAX_COL_W = 170.0
TABLE_MAX_ROWS = 40
CAPTURE_DPI = 160
HEADING_ROOM = 70.0  # figures leave space for a section + subsection heading above them
TOC_ENTRIES_PER_PAGE = 42

INK = "#1a1a1a"
MUTED = "#666666"
RULE = "#b0b0b0"
HEADER_FILL = "#e8ecf1"
STRIPE_FILL = "#f6f7f9"
ACCENT = "#1f4e79"

_TEXT_TO_PATH = TextToPath()


@lru_cache(maxsize=20_000)
def _text_width(text: str, size: float, *, bold: bool = False) -> float:
    """Width in points of ``text`` rendered in the default font."""
    if not text:
        return 0.0
    props = FontProperties(size=size, weight="bold" if bold else "normal")
    width, _, _ = _TEXT_TO_PATH.get_text_width_height_descent(text, props, ismath=False)
    return float(width)


def _wrap(text: str, size: float, width: float, *, bold: bool = False) -> list[str]:
    """Greedy word wrap using measured glyph widths."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        bullet = paragraph.lstrip().startswith(("- ", "* "))
        indent = "    " if bullet else ""
        words = (("• " + paragraph.lstrip()[2:]) if bullet else paragraph).split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if current and _text_width(candidate, size, bold=bold) > width:
                lines.append(current)
                current = indent + word
            else:
                current = candidate
        lines.append(current)
    return lines


def _clean_markdown(text: str) -> str:
    """Strip the inline Markdown marks the notebook uses (bold, code)."""
    return textwrap.dedent(text).strip().replace("**", "").replace("`", "")


def _truncate(text: str, size: float, width: float, *, bold: bool = False) -> str:
    """Shorten ``text`` with an ellipsis until it fits ``width``."""
    if _text_width(text, size, bold=bold) <= width:
        return text
    while text and _text_width(text + "…", size, bold=bold) > width:
        text = text[:-1]
    return text + "…"


def _format_cell(value: object) -> str:  # noqa: PLR0911
    """Format one table value for display."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"  # noqa: RUF001
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value.is_integer() and abs(value) < 1e9:  # noqa: PLR2004
            return f"{int(value):,}"
        if abs(value) >= 1000:  # noqa: PLR2004
            return f"{value:,.0f}"
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)


# ── Flowables ────────────────────────────────────────────────────────────────


class _Page:
    """One A4 page; ``y`` is measured downward from the top edge, in points."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(PAGE_W / 72, PAGE_H / 72))

    def text(self, x: float, y: float, text: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Draw text whose top-left (by default) sits at ``(x, y)``."""
        kwargs.setdefault("va", "top")
        kwargs.setdefault("color", INK)
        self.figure.text(x / PAGE_W, 1 - y / PAGE_H, text, **kwargs)

    def rect(self, x: float, y: float, w: float, h: float, **kwargs: Any) -> None:  # noqa: ANN401
        """Draw a rectangle whose top-left corner is ``(x, y)``."""
        self.figure.patches.append(
            Rectangle(
                (x / PAGE_W, 1 - (y + h) / PAGE_H),
                w / PAGE_W,
                h / PAGE_H,
                transform=self.figure.transFigure,
                linewidth=0,
                **kwargs,
            )
        )

    def hline(self, x0: float, x1: float, y: float, *, color: str = RULE, lw: float = 0.6) -> None:
        """Draw a horizontal rule."""
        self.figure.add_artist(
            Line2D(
                [x0 / PAGE_W, x1 / PAGE_W],
                [1 - y / PAGE_H, 1 - y / PAGE_H],
                transform=self.figure.transFigure,
                color=color,
                linewidth=lw,
            )
        )


class _Flowable:
    """A block that can measure, draw and (optionally) split itself."""

    keep_with_next = False

    def height(self) -> float:
        raise NotImplementedError

    def draw(self, page: _Page, y: float) -> None:
        raise NotImplementedError

    def split(self, available: float) -> tuple[_Flowable, _Flowable] | None:
        """Return (head that fits, rest), or None when the block must move whole."""
        del available
        return None


@dataclass
class _Heading(_Flowable):
    text: str
    level: int
    keep_with_next = True

    def height(self) -> float:
        size = HEADING_SIZES[self.level]
        return size * 1.4 + (10 if self.level == 1 else 6) + 6

    def draw(self, page: _Page, y: float) -> None:
        size = HEADING_SIZES[self.level]
        top = y + (4 if self.level == 1 else 6)
        page.text(MARGIN_X, top, self.text, fontsize=size, weight="bold", color=ACCENT)
        if self.level == 1:
            page.hline(MARGIN_X, PAGE_W - MARGIN_X, top + size * 1.35, color=ACCENT, lw=0.9)


@dataclass
class _Paragraph(_Flowable):
    lines: list[str]
    size: float = BODY_SIZE
    color: str = INK
    style: str = "normal"
    space_after: float = 6.0

    @classmethod
    def of(cls, text: str, **kwargs: Any) -> _Paragraph:  # noqa: ANN401
        size = kwargs.get("size", BODY_SIZE)
        return cls(_wrap(_clean_markdown(text), size, CONTENT_W), **kwargs)

    @property
    def line_h(self) -> float:
        return self.size * 1.4

    def height(self) -> float:
        return len(self.lines) * self.line_h + self.space_after

    def draw(self, page: _Page, y: float) -> None:
        for index, line in enumerate(self.lines):
            page.text(
                MARGIN_X,
                y + index * self.line_h,
                line,
                fontsize=self.size,
                color=self.color,
                style=self.style,
            )

    def split(self, available: float) -> tuple[_Flowable, _Flowable] | None:
        fits = int(available // self.line_h)
        if fits < 2 or fits >= len(self.lines):  # noqa: PLR2004
            return None
        head = _Paragraph(self.lines[:fits], self.size, self.color, self.style, 0)
        rest = _Paragraph(self.lines[fits:], self.size, self.color, self.style, self.space_after)
        return head, rest


@dataclass
class _Image(_Flowable):
    title: str
    png: bytes
    note: str | None = None
    _pixels: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        array = mimage.imread(io.BytesIO(self.png))
        self._pixels = (array.shape[1], array.shape[0])
        self._title = _wrap(self.title, CAPTION_SIZE, CONTENT_W, bold=True)
        self._note = (
            _wrap(_clean_markdown(self.note), BODY_SIZE - 0.5, CONTENT_W) if self.note else []
        )

    def _image_size(self) -> tuple[float, float]:
        width_px, height_px = self._pixels
        width = min(CONTENT_W, width_px / CAPTURE_DPI * 72)
        height = width * height_px / width_px
        max_h = CONTENT_H - self._text_h() - 12 - HEADING_ROOM
        if height > max_h:
            width, height = width * max_h / height, max_h
        return width, height

    def _text_h(self) -> float:
        return len(self._title) * CAPTION_SIZE * 1.4 + len(self._note) * (BODY_SIZE - 0.5) * 1.4

    def height(self) -> float:
        return self._text_h() + self._image_size()[1] + 14

    def draw(self, page: _Page, y: float) -> None:
        for index, line in enumerate(self._title):
            page.text(
                MARGIN_X, y + index * CAPTION_SIZE * 1.4, line, fontsize=CAPTION_SIZE, weight="bold"
            )
        y += len(self._title) * CAPTION_SIZE * 1.4
        for index, line in enumerate(self._note):
            page.text(
                MARGIN_X,
                y + index * (BODY_SIZE - 0.5) * 1.4,
                line,
                fontsize=BODY_SIZE - 0.5,
                color=MUTED,
                style="italic",
            )
        y += len(self._note) * (BODY_SIZE - 0.5) * 1.4 + 4
        width, height = self._image_size()
        left = MARGIN_X + (CONTENT_W - width) / 2
        axes = page.figure.add_axes(
            (left / PAGE_W, 1 - (y + height) / PAGE_H, width / PAGE_W, height / PAGE_H)
        )
        axes.imshow(mimage.imread(io.BytesIO(self.png)), interpolation="none")
        axes.set_axis_off()


@dataclass
class _Table(_Flowable):
    title: str
    header: list[str]
    rows: list[list[str]]
    numeric: list[bool]
    widths: list[float]
    size: float
    note: str | None = None
    continued: bool = False

    @property
    def row_h(self) -> float:
        return self.size * 1.75

    def _title_h(self) -> float:
        return CAPTION_SIZE * 1.4 + 3

    def _note_h(self) -> float:
        return (BODY_SIZE - 1) * 1.4 + 2 if self.note else 0.0

    def height(self) -> float:
        return self._title_h() + (len(self.rows) + 1) * self.row_h + self._note_h() + 12

    def draw(self, page: _Page, y: float) -> None:
        title = self.title + (" (continued)" if self.continued else "")
        page.text(MARGIN_X, y, title, fontsize=CAPTION_SIZE, weight="bold")
        y += self._title_h()
        total_w = sum(self.widths)
        page.rect(MARGIN_X, y, total_w, self.row_h, facecolor=HEADER_FILL)
        self._draw_row(page, y, self.header, bold=True)
        y += self.row_h
        page.hline(MARGIN_X, MARGIN_X + total_w, y, color=INK, lw=0.6)
        for index, row in enumerate(self.rows):
            if index % 2:
                page.rect(MARGIN_X, y, total_w, self.row_h, facecolor=STRIPE_FILL)
            self._draw_row(page, y, row, bold=False)
            y += self.row_h
        page.hline(MARGIN_X, MARGIN_X + total_w, y, color=INK, lw=0.6)
        if self.note:
            page.text(
                MARGIN_X, y + 3, self.note, fontsize=BODY_SIZE - 1, color=MUTED, style="italic"
            )

    def _draw_row(self, page: _Page, y: float, cells: list[str], *, bold: bool) -> None:
        x = MARGIN_X
        middle = y + self.row_h / 2
        for cell, width, numeric in zip(cells, self.widths, self.numeric, strict=True):
            text = _truncate(cell, self.size, width - 6, bold=bold)
            if numeric and not bold:
                page.text(
                    x + width - 3,
                    middle,
                    text,
                    fontsize=self.size,
                    ha="right",
                    va="center_baseline",
                )
            else:
                page.text(
                    x + 3,
                    middle,
                    text,
                    fontsize=self.size,
                    va="center_baseline",
                    weight="bold" if bold else "normal",
                )
            x += width

    def split(self, available: float) -> tuple[_Flowable, _Flowable] | None:
        fits = int((available - self._title_h() - 12) // self.row_h) - 1
        if fits < 4 or fits >= len(self.rows):  # noqa: PLR2004
            return None
        common = (self.title, self.header)
        head = _Table(
            *common, self.rows[:fits], self.numeric, self.widths, self.size, None, self.continued
        )
        rest = _Table(
            *common,
            self.rows[fits:],
            self.numeric,
            self.widths,
            self.size,
            self.note,
            continued=True,
        )
        return head, rest


def _table_flowables(
    df: pd.DataFrame, title: str, *, note: str | None, max_rows: int
) -> list[_Flowable]:
    """Turn a DataFrame into one or more table blocks that fit the page width."""
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [" / ".join(map(str, col)).strip(" /") for col in frame.columns]
    has_index = not isinstance(frame.index, pd.RangeIndex) or frame.index.name is not None
    n_index = frame.index.nlevels if has_index else 0
    if has_index:
        frame = frame.reset_index()
    total_rows = len(frame)
    if total_rows > max_rows:
        frame = frame.head(max_rows)
        extra = f"Showing the first {max_rows} of {total_rows} rows."
        note = f"{note} {extra}" if note else extra

    header = [str(col) for col in frame.columns]
    numeric = [
        pd.api.types.is_numeric_dtype(frame[col]) and not pd.api.types.is_bool_dtype(frame[col])
        for col in frame.columns
    ]
    rows = [[_format_cell(value) for value in row] for row in frame.itertuples(index=False)]

    def widths_at(size: float) -> list[float]:
        return [
            min(
                TABLE_MAX_COL_W,
                max(
                    _text_width(header[i], size, bold=True),
                    *(_text_width(row[i], size) for row in rows),
                )
                + 8,
            )
            for i in range(len(header))
        ]

    size = TABLE_SIZE
    widths = widths_at(size)
    while sum(widths) > CONTENT_W and size > TABLE_MIN_SIZE:
        size = max(TABLE_MIN_SIZE, size - 0.25)
        widths = widths_at(size)

    # Still too wide: split the columns into groups, repeating the index columns.
    groups: list[list[int]] = []
    keys = list(range(n_index))
    current = list(keys)
    for col in range(n_index, len(header)):
        if current != keys and sum(widths[i] for i in [*current, col]) > CONTENT_W:
            groups.append(current)
            current = [*keys]
        current.append(col)
    groups.append(current)

    blocks: list[_Flowable] = []
    for number, cols in enumerate(groups, start=1):
        part = f" ({number}/{len(groups)})" if len(groups) > 1 else ""
        blocks.append(
            _Table(
                title + part,
                [header[i] for i in cols],
                [[row[i] for i in cols] for row in rows],
                [numeric[i] for i in cols],
                [widths[i] for i in cols],
                size,
                note if number == len(groups) else None,
            )
        )
    return blocks


class _PageBreak(_Flowable):
    def height(self) -> float:
        return 0.0


# ── Report ───────────────────────────────────────────────────────────────────


@dataclass
class _Section:
    title: str
    level: int
    blocks: dict[str, list[_Flowable]] = field(default_factory=dict)


class NotebookReport:
    """Collect headings, tables and figures during a notebook run, then write a PDF."""

    def __init__(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        settings: Mapping[str, object] | None = None,
        enabled: bool = True,
    ) -> None:
        """Start an empty report; ``settings`` are listed on the title page.

        With ``enabled=False`` every method is a no-op (nothing is collected or written).
        """
        self.enabled = enabled
        self.title = title
        self.subtitle = subtitle
        self.settings = dict(settings or {})
        self.started = datetime.now().astimezone()
        self._sections: dict[str, _Section] = {}
        self._current: _Section | None = None

    # Structure --------------------------------------------------------------

    def section(self, title: str, text: str | None = None) -> None:
        """Start (or return to) a top-level section, like a notebook ``##`` heading."""
        self._heading(title, 1, text)

    def subsection(self, title: str, text: str | None = None) -> None:
        """Start (or return to) a subsection, like a notebook ``###`` heading."""
        self._heading(title, 2, text)

    def _heading(self, title: str, level: int, text: str | None) -> None:
        if not self.enabled:
            return
        key = f"{level}:{title}"
        if key not in self._sections:
            self._sections[key] = _Section(title, level)
        self._current = self._sections[key]
        if text:
            self.text(text, key="__intro__")

    def _add(self, key: str, blocks: list[_Flowable]) -> None:
        if not self.enabled:
            return
        if self._current is None:
            self.section("Results")
        assert self._current is not None  # noqa: S101
        self._current.blocks[key] = blocks

    # Content ----------------------------------------------------------------

    def text(self, text: str, *, key: str | None = None) -> None:
        """Add a paragraph (simple ``- `` bullets, ``**`` and backticks are understood)."""
        self._add(key or f"text:{text[:80]}", [_Paragraph.of(text)])

    def table(
        self,
        df: pd.DataFrame,
        title: str,
        *,
        note: str | None = None,
        max_rows: int = TABLE_MAX_ROWS,
    ) -> None:
        """Add a DataFrame as a table; wide tables are split into column groups."""
        if not self.enabled:
            return
        self._add(f"table:{title}", _table_flowables(df, title, note=note, max_rows=max_rows))

    def figure(self, fig: Figure, title: str, *, note: str | None = None) -> None:
        """Add an existing matplotlib figure."""
        if not self.enabled:
            return
        self._add(f"figure:{title}", [_Image(title, _render_png(fig), note)])

    @contextmanager
    def capture(self, title: str, *, when: bool = True, note: str | None = None) -> Iterator[None]:
        """Collect every figure a ``utils.plots`` function finishes inside the block.

        ``when=False`` turns the capture off (e.g. for the Ideal view in a loop).
        Several figures in one block are numbered ``title (1/n)``.
        """
        if not (when and self.enabled):
            yield
            return
        captured: list[bytes] = []

        def hook(fig: Figure) -> None:
            captured.append(_render_png(fig))

        plots.FIGURE_HOOKS.append(hook)
        try:
            yield
        finally:
            plots.FIGURE_HOOKS.remove(hook)
        count = len(captured)
        self._add(
            f"figure:{title}",
            [
                _Image(
                    f"{title} ({index}/{count})" if count > 1 else title,
                    png,
                    note if index == count else None,
                )
                for index, png in enumerate(captured, start=1)
            ],
        )

    # Output -----------------------------------------------------------------

    def build(self, path: str | Path) -> Path | None:
        """Lay out and write the PDF; returns its path (None when the report is disabled)."""
        if not self.enabled:
            print("[report] not written: the report is disabled.")
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        front = self._front_matter()
        front_pages = _paginate(front)
        content: list[_Flowable] = []
        headings: list[tuple[_Heading, int]] = []
        for section in self._sections.values():
            if section.level == 1 and content:
                content.append(_PageBreak())
            heading = _Heading(section.title, section.level)
            content.append(heading)
            for blocks in section.blocks.values():
                content.extend(blocks)
        content_pages = _paginate(content)
        toc_pages = max(
            1, math.ceil(sum(1 for s in self._sections.values()) / TOC_ENTRIES_PER_PAGE)
        )
        offset = len(front_pages) + toc_pages
        for number, page in enumerate(content_pages, start=1):
            headings.extend(
                (item, number + offset) for item, _ in page if isinstance(item, _Heading)
            )
        total = offset + len(content_pages)

        with PdfPages(path, metadata={"Title": self.title, "Subject": self.subtitle or ""}) as pdf:
            page_number = 0
            for placed in front_pages:
                page_number += 1
                pdf.savefig(self._render(placed, page_number, total, header=False).figure)
            for chunk_start in range(0, max(1, len(headings)), TOC_ENTRIES_PER_PAGE):
                page_number += 1
                pdf.savefig(
                    self._toc_page(
                        headings[chunk_start : chunk_start + TOC_ENTRIES_PER_PAGE],
                        page_number,
                        total,
                        first=chunk_start == 0,
                    ).figure
                )
            for placed in content_pages:
                page_number += 1
                pdf.savefig(self._render(placed, page_number, total).figure)
        print(f"[report] wrote {path} ({total} pages)")
        return path

    def _front_matter(self) -> list[_Flowable]:
        blocks: list[_Flowable] = [
            _Paragraph(_wrap(self.title, 22, CONTENT_W, bold=True), 22, ACCENT, space_after=6),
        ]
        if self.subtitle:
            blocks.append(_Paragraph.of(self.subtitle, size=12, color=MUTED, space_after=14))
        commit = _git_commit() or "unknown"
        blocks.append(
            _Paragraph.of(
                f"Run started {self.started:%Y-%m-%d %H:%M %Z} · "
                f"report built {datetime.now().astimezone():%Y-%m-%d %H:%M} · "
                f"git commit {commit}",
                size=BODY_SIZE,
                color=MUTED,
                space_after=18,
            )
        )
        if self.settings:
            settings = pd.DataFrame(
                {"setting": list(self.settings), "value": [str(v) for v in self.settings.values()]}
            )
            blocks.extend(
                _table_flowables(settings, "Run settings", note=None, max_rows=len(settings))
            )
        return blocks

    def _render(
        self, placed: list[tuple[_Flowable, float]], number: int, total: int, *, header: bool = True
    ) -> _Page:
        page = _Page()
        if header:
            page.text(MARGIN_X, 26, self.title, fontsize=7.5, color=MUTED)
            page.text(
                PAGE_W - MARGIN_X,
                26,
                f"{self.started:%Y-%m-%d %H:%M}",
                fontsize=7.5,
                color=MUTED,
                ha="right",
            )
            page.hline(MARGIN_X, PAGE_W - MARGIN_X, 38, lw=0.4)
        page.text(
            PAGE_W / 2, PAGE_H - 28, f"{number} / {total}", fontsize=7.5, color=MUTED, ha="center"
        )
        for item, y in placed:
            item.draw(page, y)
        return page

    def _toc_page(
        self, headings: list[tuple[_Heading, int]], number: int, total: int, *, first: bool
    ) -> _Page:
        y = MARGIN_TOP
        items: list[tuple[_Flowable, float]] = []
        if first:
            items.append((_Heading("Contents", 1), y))
            y += _Heading("Contents", 1).height() + 4
        page = self._render(items, number, total)
        for heading, page_no in headings:
            indent = 0 if heading.level == 1 else 16
            size = 10 if heading.level == 1 else 9
            weight = "bold" if heading.level == 1 else "normal"
            y += 6 if heading.level == 1 else 0
            label = _truncate(heading.text, size, CONTENT_W - indent - 40)
            page.text(MARGIN_X + indent, y, label, fontsize=size, weight=weight)
            page.text(PAGE_W - MARGIN_X, y, str(page_no), fontsize=size, ha="right")
            y += size * 1.65
        return page


def _render_png(fig: Figure) -> bytes:
    """Rasterize a figure for embedding in the report."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=CAPTURE_DPI, bbox_inches="tight", facecolor="white")
    return buffer.getvalue()


def _paginate(flowables: list[_Flowable]) -> list[list[tuple[_Flowable, float]]]:
    """Place flowables top-to-bottom on pages, splitting tables/paragraphs if needed."""
    pages: list[list[tuple[_Flowable, float]]] = [[]]
    y = MARGIN_TOP
    queue = list(flowables)
    while queue:
        item = queue.pop(0)
        if isinstance(item, _PageBreak):
            if pages[-1]:
                pages.append([])
                y = MARGIN_TOP
            continue
        remaining = PAGE_H - MARGIN_BOTTOM - y
        needed = item.height()
        if item.keep_with_next and queue and not isinstance(queue[0], _PageBreak):
            # A heading needs its next block (or a useful start of it) on the same page.
            follower = queue[0]
            splittable = isinstance(follower, (_Paragraph, _Table))
            needed += min(follower.height(), 120.0 if splittable else CONTENT_H - needed)
        if needed <= remaining:
            pages[-1].append((item, y))
            y += item.height()
            continue
        parts = item.split(remaining) if not item.keep_with_next else None
        if parts is not None:
            head, rest = parts
            pages[-1].append((head, y))
            queue.insert(0, rest)
        else:
            queue.insert(0, item)
        if not pages[-1] and parts is None:
            # Taller than a whole page and unsplittable: place it anyway.
            pages[-1].append((queue.pop(0), y))
        pages.append([])
        y = MARGIN_TOP
    return [page for page in pages if page]
