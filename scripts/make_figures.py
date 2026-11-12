"""Data figures for the paper, drawn only from experiments/results/final. Saved to figures/ as PDF and PNG."""
import json

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS, OUT = Path("experiments/results/final"), Path("figures")
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": SOFT, "axes.labelcolor": INK, "text.color": INK, "xtick.color": SOFT, "ytick.color": SOFT,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 600})
SIZES = [str(s) for s in (0, 10, 20, 40, 80, 160, 320)]


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf8"))["results"]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def curve(ax, row: dict, label: str, color: str, marker: str) -> None:
    x = [int(s) for s in SIZES]
    y = [row["pod"][s] for s in SIZES]
    ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.6, label=label, markeredgecolor="white", markeredgewidth=0.5)
    if "ci" in row:
        ax.fill_between(x, [row["ci"][s][0] for s in SIZES], [row["ci"][s][1] for s in SIZES], color=color, alpha=0.13, linewidth=0)


def size_axis(ax) -> None:
    ax.set_xscale("symlog", base=2, linthresh=10, linscale=0.5)
    ax.set_xticks([0, 10, 20, 40, 80, 160, 320])
    ax.set_xticklabels(["0", "10", "20", "40", "80", "160", "320"])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.axhline(0.8, color=SOFT, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.set_xlabel("Injected outbreak size (cases)")


# Figures 2 to 4: monochrome journal style. Black lines told apart by dash pattern and open markers, boxed axes, serif text.
MONO = {"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8.5,
        "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black", "ytick.color": "black",
        "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1.0}
DASHES = ["-", (0, (6, 2)), (0, (1, 1.4)), (0, (6, 2, 1, 2)), (0, (6, 2, 1, 2, 1, 2)), (0, (11, 3))]


def mono_line(ax, row: dict, label: str, style, width: float = 1.1) -> None:
    ax.plot([int(s) for s in SIZES], [row["pod"][s] for s in SIZES], color="black", linestyle=style, linewidth=width, label=label)


def mono_axis(ax) -> None:
    ax.set_xscale("symlog", base=2, linthresh=10, linscale=0.5)
    ax.set_xticks([0, 10, 20, 40, 80, 160, 320])
    ax.set_xticklabels(["0", "10", "20", "40", "80", "160", "320"])
    ax.minorticks_off()
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0.8, color="0.6", linewidth=0.6, zorder=0)
    ax.set_xlabel("Injected outbreak size (cases)")


MARKERS = ["o", "s", "^", "D", "v", "P"]


def draws() -> dict:
    return json.loads((RESULTS / "threshold_draws.json").read_text(encoding="utf8"))["results"]


def mono_band(ax, row: dict, label: str, style, marker: str, width: float = 1.1) -> None:
    """Mean over the threshold draws as a black line with open markers, and the range over draws as a light grey band."""
    x = [int(s) for s in SIZES]
    ax.fill_between(x, [row["pod_min"][s] for s in SIZES], [row["pod_max"][s] for s in SIZES], color="0.85", linewidth=0, zorder=0)
    ax.plot(x, [row["pod_mean"][s] for s in SIZES], color="black", linestyle=style, linewidth=width, marker=marker, markersize=3.6,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.8, label=label)


def fig_budget_designs() -> None:
    """Detection by budget design: mean and range over 15 threshold draws."""
    rows = draws()["designs"]
    designs = [("record_distributed", "Whole stream, per record"), ("person_distributed", "Whole stream, per person"),
               ("person_per_institution", "Per person, noise at each institution"), ("per_window_distributed", "Per window"),
               ("per_window_per_institution", "Per window, noise at each institution")]
    with plt.rc_context(MONO):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
        for ax, key, title in zip(axes, ("eps1", "eps2", "eps4"), ("(a) $\\varepsilon = 1$", "(b) $\\varepsilon = 2$", "(c) $\\varepsilon = 4$")):
            for (name, label), style, marker in zip(designs, DASHES, MARKERS):
                mono_band(ax, rows[f"{key}/{name}"], label, style, marker)
            mono_axis(ax)
            ax.set_title(title, fontsize=9)
        axes[0].set_ylabel("Probability of detection")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5, handlelength=3.2)
        fig.tight_layout()
        save(fig, "fig_budget_designs")


def fig_federation() -> None:
    """All institutions against each one alone at epsilon 2 per record: mean and range over 15 threshold draws."""
    rows = draws()["federation"]
    with plt.rc_context(MONO):
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        mono_band(ax, rows["all"], "All five institutions", "-", "o", 1.8)
        singles = sorted((k for k in rows if k != "all"), key=lambda k: -rows[k]["pod_mean"]["320"])
        for key, style, marker in zip(singles, DASHES[1:], MARKERS[1:]):
            mono_band(ax, rows[key], key.replace("IMSS_BIENESTAR_OPD", "IMSS-Bienestar"), style, marker)
        mono_axis(ax)
        ax.set_ylabel("Probability of detection")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=3.2)
        save(fig, "fig_federation")


def fig_calls() -> None:
    """Detection against LLM calls at epsilon 2: mean over 15 threshold draws, with the range as error bars."""
    rows = draws()["calls"]
    arms = [("naeg", "NAEG", "o", "black"), ("every_alert_cell_no_merging", "Per-alert baseline (no merging)", "P", "black"),
            ("resetting_cusum", "Resetting CUSUM", "D", "white"), ("running_cusum", "Running CUSUM", "v", "white"),
            ("periodic_same_calls", "Fixed schedule", "s", "white"), ("random_same_calls", "Random times", "^", "white")]
    with plt.rc_context(MONO):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, sharex=True)
        for ax, size in zip(axes, ("80", "160")):
            for name, label, marker, face in arms:
                calls, pod = np.array(rows[name]["calls_by_draw"]), np.array(rows[name][f"pod_{size}"])
                ax.errorbar(calls.mean(), pod.mean(), xerr=[[calls.mean() - calls.min()], [calls.max() - calls.mean()]],
                            yerr=[[pod.mean() - pod.min()], [pod.max() - pod.mean()]], color="black", linewidth=0.7, capsize=2,
                            marker=marker, markersize=5, markerfacecolor=face, markeredgecolor="black", markeredgewidth=0.8, label=label, linestyle="none")
            ax.axvline(55, color="0.6", linewidth=0.6, linestyle=(0, (1, 2)))
            ax.axhline(0.8, color="0.6", linewidth=0.6)
            ax.set_title(f"({'a' if size == '80' else 'b'}) {size}-case outbreak, $\\varepsilon = 2$", fontsize=9)
            ax.set_xlabel("LLM calls in the 55-window test period")
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 30, 55])
            ax.set_xticklabels(["5", "10", "20", "30", "55"])
            ax.minorticks_off()
            ax.set_ylim(0, 1.05)
        axes[0].text(52, 0.04, "One call per window", color="0.4", fontsize=7.5, ha="right")
        axes[0].set_ylabel("Probability of detection")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5)
        fig.tight_layout()
        save(fig, "fig_llm_calls")


def fig_spread() -> None:
    """Post-hoc check: detection at 80 cases across repeated threshold draws."""
    path = RESULTS / "calibration_spread.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf8"))["results"]
    arms = [("naeg", "NAEG", BLUE, "o"), ("c2_no_noise_floor", "C2", ORANGE, "s"), ("minus_floor", "NAEG, no noise floor", YELLOW, "D"), ("resetting_cusum", "Resetting CUSUM", AQUA, "^")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    for ax, epsilon in zip(axes, (1, 2, 4)):
        for k, (name, label, color, marker) in enumerate(arms):
            pod = rows[f"record_eps{epsilon}_{name}"]["pod"]["80"]
            ax.errorbar(k, pod["mean"], yerr=[[pod["mean"] - pod["min"]], [pod["max"] - pod["mean"]]], color=color, marker=marker, markersize=6, linewidth=1.6, label=label,
                        markeredgecolor="white", markeredgewidth=0.5)
        ax.axhline(0.8, color=SOFT, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_xticks([])
        ax.set_xlim(-0.6, 3.6)
        ax.set_title(f"ε = {epsilon}", fontsize=9, loc="left")
    axes[0].set_ylabel("Detection at 80 cases\n(mean and range over 15 thresholds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_threshold_spread")


for draw in (fig_budget_designs, fig_federation, fig_calls):
    draw()
print(sorted(p.name for p in OUT.glob("*.png")))
