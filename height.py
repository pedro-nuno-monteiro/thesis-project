"""Volunteer height and weight scatter for the thesis.

Regenerates the participant characterisation figure with circular markers,
collision-safe labels, a tight vertical range, and vector PDF output.

Run:
    python plot_volunteer_anthropometrics.py
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Data: (volunteer id, height in m, weight in kg)
# NOTE: volunteer 2 was hidden behind volunteer 5 in the original chart, so
# its values below are a guess. Replace them with the recorded ones.
# --------------------------------------------------------------------------
VOLUNTEERS = [
    (1, 1.72, 72),
    (2, 1.85, 84),
    (3, 1.65, 66),
    (4, 1.90, 79),
    (5, 1.85, 84),
    (6, 1.80, 75),
]

OUTPUT = Path("volunteer_anthropometrics.pdf")

MARKER_COLOUR = "#3B6EA5"
TEXT_COLOUR = "#333333"
GRID_COLOUR = "#D9D9D9"


def label_offsets(volunteers):
    """Return a label offset per volunteer, spreading coincident points."""
    groups = defaultdict(list)
    for vid, height, weight in volunteers:
        groups[(round(height, 3), round(weight, 1))].append(vid)

    spread = [(-20, 14), (20, 14), (-20, -18), (20, -18)]
    offsets = {}
    for ids in groups.values():
        if len(ids) == 1:
            offsets[ids[0]] = (0, 12)
        else:
            for position, vid in enumerate(sorted(ids)):
                offsets[vid] = spread[position % len(spread)]
    return offsets


def main():
    heights = [h for _, h, _ in VOLUNTEERS]
    weights = [w for _, _, w in VOLUNTEERS]
    offsets = label_offsets(VOLUNTEERS)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": TEXT_COLOUR,
        "axes.labelcolor": TEXT_COLOUR,
        "text.color": TEXT_COLOUR,
        "xtick.color": TEXT_COLOUR,
        "ytick.color": TEXT_COLOUR,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)

    ax.scatter(
        heights,
        weights,
        s=70,
        color=MARKER_COLOUR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for vid, height, weight in VOLUNTEERS:
        dx, dy = offsets[vid]
        ax.annotate(
            str(vid),
            xy=(height, weight),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color=MARKER_COLOUR,
            zorder=4,
            arrowprops=dict(
                arrowstyle="-",
                color=MARKER_COLOUR,
                linewidth=0.6,
                shrinkA=0,
                shrinkB=5,
            ) if (dx, dy) != (0, 12) else None,
        )

    height_pad = 0.03
    weight_pad = 5
    ax.set_xlim(min(heights) - height_pad, max(heights) + height_pad)
    ax.set_ylim(min(weights) - weight_pad, max(weights) + weight_pad)

    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Weight (kg)")

    ax.grid(True, color=GRID_COLOUR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)

    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"written: {OUTPUT} and {OUTPUT.with_suffix('.png')}")


if __name__ == "__main__":
    main()