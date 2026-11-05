"""Render the representation bake-off results as a figure.

Reads ``results.csv`` (and the mined HUIM patterns for the caption) and writes a
light and a dark PNG into ``figures/``:

    uv run python exp/Tab_exp/representation_bakeoff/make_figures.py

Paths resolve relative to this file, so it can be launched from anywhere.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import BoxStyle, FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.csv"
DRIVERS = HERE / "artifacts" / "huim_drivers.csv"
OUTDIR = HERE / "figures"

# --- theme -------------------------------------------------------------------
# Palette slots + chrome; the dark column is stepped for the dark surface,
# not an automatic flip of the light one.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        baseline="#c3c2b7",
        accent="#2a78d6",  # slot 1 — the winner
        second="#eb6834",  # slot 2 — the HUIM hybrid
        negative="#e34948",
        deemph="#a8a7a0",
        band="#ecebe6",
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        baseline="#383835",
        accent="#3987e5",
        second="#d95926",
        negative="#e66767",
        deemph="#5c5b56",
        band="#262624",
    ),
}

# Display names / dims for the rows in results.csv, best-first.
LABELS = {
    "skrub": ("skrub TableVectorizer", "B"),
    "D_skrub_plus_huim": ("skrub + HUIM patterns", "D"),
    "sentence_transformer": ("SentenceTransformer (MiniLM)", "A"),
}


def rounded_barh(ax, x0, x1, y, height, color, radius_in=0.035, zorder=3):
    """Horizontal bar with a rounded data-end and a square baseline end."""
    fig = ax.figure
    box = ax.get_window_extent()
    w_in, h_in = box.width / fig.dpi, box.height / fig.dpi
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    dx, dy = x_hi - x_lo, y_hi - y_lo

    r = min(radius_in * dx / w_in, abs(x1 - x0) * 0.5)
    aspect = (dy * w_in) / (dx * h_in)
    left, width = (x0, x1 - x0) if x1 >= x0 else (x1, x0 - x1)

    ax.add_patch(
        FancyBboxPatch(
            (left, y - height / 2),
            width,
            height,
            boxstyle=BoxStyle("Round", pad=0, rounding_size=r),
            mutation_aspect=aspect,
            linewidth=0,
            facecolor=color,
            zorder=zorder,
        )
    )
    # Square off the end that sits on the baseline.
    cap_x = x0 - r if x1 < x0 else x0
    ax.add_patch(
        Rectangle(
            (cap_x, y - height / 2),
            r,
            height,
            linewidth=0,
            facecolor=color,
            zorder=zorder + 0.1,
        )
    )


def row_label(ax, y, name, sub, t, bold=False):
    """Two-line y-axis label: entity in ink on top, qualifier in muted below.

    The y-axis is inverted (row 0 on top), so a *smaller* y sits higher.
    """
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        -0.015,
        y - 0.13,
        name,
        transform=tr,
        ha="right",
        va="center",
        fontsize=10,
        color=t["ink"],
        fontweight="semibold" if bold else "normal",
    )
    ax.text(
        -0.015,
        y + 0.18,
        sub,
        transform=tr,
        ha="right",
        va="center",
        fontsize=8.5,
        color=t["muted"],
    )


def panel_auc(ax, df, t):
    """AUC per representation — emphasis form, bars grown from chance (0.50)."""
    rows = [(k,) + LABELS[k] for k in ("skrub", "D_skrub_plus_huim", "sentence_transformer")]
    n = len(rows) + 1  # + the un-evaluated TabPFN row
    chance = 0.50

    ax.set_xlim(chance, 0.615)
    ax.set_ylim(n - 0.5, -0.5)  # first row on top
    ax.set_facecolor(t["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
    ax.set_xticklabels(["0.50", "0.52", "0.54", "0.56", "0.58", "0.60"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8.5, length=0, pad=6)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(chance, color=t["baseline"], linewidth=1.0, zorder=1)

    for i, (key, name, tag) in enumerate(rows):
        r = df.loc[key]
        color = t["accent"] if key == "skrub" else (t["second"] if key.startswith("D_") else t["deemph"])
        rounded_barh(ax, chance, r.auc_mean, i, 0.24, color)
        ax.errorbar(
            r.auc_mean,
            i,
            xerr=r.auc_std,
            color=t["ink2"],
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.2,
            zorder=5,
        )
        ax.text(
            r.auc_mean + r.auc_std + 0.0035,
            i,
            f"{r.auc_mean:.4f}  ± {r.auc_std:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color=t["ink"],
            fontweight="semibold" if key == "skrub" else "normal",
        )
        row_label(ax, i, f"{tag}  {name}", f"{int(r['dim'])}-d", t, bold=(key == "skrub"))

    # TabPFN: present in the design, absent from the evidence.
    i = len(rows)
    row_label(ax, i, "C  TabPFN (frozen)", "—", t)
    ax.text(
        chance + 0.003,
        i,
        "not evaluated — model weights are license-gated",
        va="center",
        ha="left",
        fontsize=9,
        color=t["muted"],
        style="italic",
    )

    ax.set_xlabel(
        "AUC-ROC on the held-out test split  ·  bars grow from chance (0.50)  ·  whiskers = ±1σ over 3 seeds",
        fontsize=9,
        color=t["ink2"],
        labelpad=8,
    )


def panel_delta(ax, df, t):
    """What HUIM patterns did to the winner, per metric, against seed noise."""
    b, d = df.loc["skrub"], df.loc["D_skrub_plus_huim"]
    metrics = [
        ("AUC-ROC", b.auc_mean, d.auc_mean, b.auc_std, d.auc_std),
        ("F1-macro", b.f1_mean, d.f1_mean, b.f1_std, d.f1_std),
        ("Accuracy", b.acc_mean, d.acc_mean, b.acc_std, d.acc_std),
    ]

    ax.set_xlim(-0.0125, 0.0125)
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    ax.set_facecolor(t["surface"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([-0.010, -0.005, 0.0, 0.005, 0.010])
    ax.set_xticklabels(["−0.010", "−0.005", "0", "+0.005", "+0.010"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8.5, length=0, pad=6)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    for i, (name, mb, md, sb, sd) in enumerate(metrics):
        delta = md - mb
        noise = math.sqrt(sb**2 + sd**2)  # 1σ of the seed-to-seed difference
        ax.add_patch(
            Rectangle(
                (-noise, i - 0.26),
                2 * noise,
                0.52,
                linewidth=0,
                facecolor=t["band"],
                zorder=1,
            )
        )
        rounded_barh(ax, 0.0, delta, i, 0.22, t["negative"], zorder=3)
        ax.text(
            delta - 0.0006,
            i,
            f"−{abs(delta):.4f}",
            va="center",
            ha="right",
            fontsize=9.5,
            color=t["ink"],
        )
        ratio = abs(delta) / noise
        verdict = "clear of seed noise" if ratio >= 2 else ("marginal" if ratio >= 1 else "within seed noise")
        ax.text(
            0.0125,
            i,
            f"{ratio:.1f}σ — {verdict}",
            va="center",
            ha="right",
            fontsize=8.5,
            color=t["ink2"] if ratio >= 2 else t["muted"],
        )
        row_label(ax, i, name, f"±{noise:.4f} noise", t)

    ax.axvline(0.0, color=t["baseline"], linewidth=1.0, zorder=2)
    ax.set_xlabel(
        "change from adding 40 HUIM pattern columns to skrub  (D − B); shaded = ±1σ seed noise",
        fontsize=9,
        color=t["ink2"],
        labelpad=8,
    )


def panel_table(ax, df, t):
    """Table view — every plotted value reachable without reading a color."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cols = [0.0, 0.33, 0.42, 0.64, 0.80]
    heads = ["Representation", "dim", "AUC (mean ± std)", "F1-macro", "Accuracy"]
    aligns = ["left", "right", "left", "left", "left"]

    rows = []
    for key in ("skrub", "D_skrub_plus_huim", "sentence_transformer"):
        r = df.loc[key]
        name, tag = LABELS[key]
        rows.append(
            [
                f"{tag}  {name}",
                f"{int(r['dim'])}",
                f"{r.auc_mean:.4f} ± {r.auc_std:.4f}",
                f"{r.f1_mean:.3f} ± {r.f1_std:.3f}",
                f"{r.acc_mean:.3f} ± {r.acc_std:.3f}",
            ]
        )
    rows.append(["C  TabPFN (frozen)", "—", None, None, None])

    y = 0.78
    step = 0.215
    for x, h, a in zip(cols, heads, aligns):
        ax.text(x, y + 0.20, h, fontsize=8.5, color=t["muted"], ha=a, va="center")
    ax.plot([0, 1], [y + 0.11, y + 0.11], color=t["baseline"], linewidth=0.9)

    for j, row in enumerate(rows):
        yy = y - j * step
        win = j == 0
        for x, cell, a in zip(cols, row, aligns):
            if cell is None:  # the un-evaluated row carries one note instead
                continue
            ax.text(
                x,
                yy,
                cell,
                fontsize=9,
                ha=a,
                va="center",
                color=t["ink"],
                fontweight="semibold" if win else "normal",
            )
        if row[2] is None:
            ax.text(
                cols[2],
                yy,
                "not evaluated — model weights are license-gated",
                fontsize=9,
                ha="left",
                va="center",
                color=t["muted"],
                style="italic",
            )
        if j < len(rows) - 1:
            ax.plot([0, 1], [yy - step / 2, yy - step / 2], color=t["grid"], linewidth=0.8)


def build(theme_name: str, df: pd.DataFrame, top_driver: str) -> Path:
    t = THEMES[theme_name]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
        }
    )

    fig = plt.figure(figsize=(10.5, 9.4), dpi=200)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[3.0, 2.2, 1.6],
        left=0.255,
        right=0.965,
        top=0.828,
        bottom=0.105,
        hspace=0.58,
    )

    fig.text(
        0.035,
        0.968,
        "Which tabular representation feeds the delay-prediction head best?",
        fontsize=15.5,
        color=t["ink"],
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.035,
        0.928,
        "One fixed MLP head ([256,128], dropout 0.1, Adam 1e-3) over 50k flights sampled across 2016–2024. "
        "Temporal FL_DAY split\n(train 14,690 / val 4,985 / test 30,325), target |ARR_DELAY| > 15 min "
        "(40.8% positive). Mean ± std over seeds 42/43/44 — only the input embedding varies.",
        fontsize=9,
        color=t["ink2"],
        va="top",
        linespacing=1.5,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    # The table carries its own row labels, so it spans the full card width
    # instead of sitting inside the plots' left gutter.
    p3 = ax3.get_position()
    ax3.set_position([0.035, p3.y0, 0.93, p3.height])

    fig.canvas.draw()  # bar geometry needs final axes extents
    panel_auc(ax1, df, t)
    panel_delta(ax2, df, t)
    panel_table(ax3, df, t)

    # Panel titles ride the figure's left margin, not the plot's, so the card
    # reads as one column of headings.
    for ax, title in (
        (ax1, "skrub wins by a clear, low-variance margin"),
        (ax2, "HUIM patterns do not help the winner"),
        (ax3, "All values"),
    ):
        fig.text(
            0.035,
            ax.get_position().y1 + 0.016,
            title,
            fontsize=11.5,
            color=t["ink"],
            fontweight="semibold",
            va="bottom",
        )

    fig.text(
        0.035,
        0.038,
        f"Top mined delay driver: {top_driver}.  Utility synthesised per flight — delay cost "
        "|ARR_DELAY| on delayed rows, on-time margin (15 − |ARR_DELAY|) otherwise;\nPAMI EFIM on train rows, "
        "minUtil p98, ≤3 items. The ~0.60 AUC ceiling matches the repo's other tabular baselines: "
        "these ~19 features are only\nmoderately predictive of a near-balanced target. Negative HUIM "
        "result reported as-is — no tuning-to-positive.",
        fontsize=8,
        color=t["muted"],
        va="bottom",
        linespacing=1.6,
    )

    OUTDIR.mkdir(exist_ok=True)
    suffix = "" if theme_name == "light" else "_dark"
    out = OUTDIR / f"bakeoff_results{suffix}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    df = pd.read_csv(RESULTS).set_index("Representation")
    top_driver = "—"
    if DRIVERS.exists():
        top_driver = pd.read_csv(DRIVERS).iloc[0]["Pattern"]
    for theme in ("light", "dark"):
        print(f"wrote {build(theme, df, top_driver)}")


if __name__ == "__main__":
    main()
