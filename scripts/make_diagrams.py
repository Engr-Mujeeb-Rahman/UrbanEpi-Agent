"""Architecture and worked-example figures. The worked-example values come from experiments/results/phase4_demo_real_llm.txt and final/e6.json."""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("figures")
INK, SOFT, BLUE = "#0b0b0b", "#52514e", "#0b0b0b"
TINT = {"inst": "white", "agg": "white", "llm": "white", "code": "white"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 600})


def box(ax, x, y, w, h, text, kind, size=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", facecolor=TINT[kind], edgecolor=SOFT, linewidth=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)


def arrow(ax, a, b, text=None, at=None, color=SOFT):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=color, shrinkA=0, shrinkB=0))
    if text:
        ax.text(*at, text, fontsize=6.3, color=SOFT, ha="center", va="center", linespacing=1.2)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.add_patch(Rectangle((0.05, 0.3), 3.35, 6.0, facecolor="none", edgecolor=BLUE, linewidth=0.9, linestyle=(0, (4, 3))))
    ax.text(0.2, 5.92, "Each of K = 5 institutions\n(raw records stay here)", fontsize=6.6, color=BLUE, weight="bold", va="center", linespacing=1.15)
    box(ax, 0.3, 4.45, 2.85, 0.95, "Knowledge graph (Kùzu)\nFixed query templates only", "inst")
    box(ax, 0.3, 3.2, 2.85, 0.95, "Budget ledger\nReserves ρ before each\nfresh release", "inst")
    box(ax, 0.3, 1.95, 2.85, 0.95, "Count every cell\nAdd noise N(0, σ²/K)", "inst")
    box(ax, 0.3, 0.65, 2.85, 1.0, "Add pairwise masks\n(simulated secure\naggregation)", "inst")
    for y in (4.45, 3.2, 1.95):
        arrow(ax, (1.72, y), (1.72, y - 0.3))
    box(ax, 4.3, 2.45, 2.2, 1.7, "Aggregator\nSums shares;\nmasks cancel\n\nReleased table R\nNoisy cells, no raw data", "agg", 6.1)
    arrow(ax, (3.15, 1.5), (4.3, 2.75), "Masked\nshares", (3.82, 1.75))
    box(ax, 7.2, 4.9, 2.1, 1.1, "NAEG gate\nReads R only\nNo LLM", "code")
    box(ax, 9.75, 4.9, 2.1, 1.1, "LLM triage, briefing\nNumbers checked\nafter output", "llm")
    arrow(ax, (6.5, 3.85), (7.2, 5.1), "Monitoring", (6.4, 4.65))
    arrow(ax, (9.3, 5.45), (9.75, 5.45))
    ax.text(9.52, 6.17, "Called only on an alert", fontsize=6.3, color=SOFT, ha="center")
    box(ax, 9.75, 3.45, 2.1, 0.85, "Official's question", "agg")
    box(ax, 9.75, 1.95, 2.1, 1.0, "LLM parser\nQuestion to\ntyped plan", "llm", 6.6)
    box(ax, 7.2, 1.95, 2.1, 1.0, "Validator\nRefuses or asks to\nclarify if needed", "code")
    box(ax, 4.3, 0.45, 2.2, 1.2, "PB-QD\nReuse, allocate noise,\ncoarsen", "code")
    box(ax, 7.2, 0.5, 4.65, 0.95, "Answer with interval and privacy cost\nWritten by code, not the LLM", "agg", 6.8)
    arrow(ax, (10.8, 3.45), (10.8, 2.95))
    arrow(ax, (9.75, 2.45), (9.3, 2.45))
    arrow(ax, (7.2, 2.2), (6.5, 1.4))
    arrow(ax, (5.4, 2.45), (5.4, 1.65), "Reuse:\nno cost", (5.95, 2.05))
    arrow(ax, (6.5, 0.97), (7.2, 0.97))
    arrow(ax, (4.3, 0.75), (3.4, 0.75), "Fresh\nsub-queries", (3.87, 0.2), color=BLUE)
    fig.savefig(OUT / "fig_architecture.pdf")
    fig.savefig(OUT / "fig_architecture.png")
    plt.close(fig)


def worked_example() -> None:
    rows = json.loads(Path("experiments/results/final/e6.json").read_text(encoding="utf8"))["rows"]
    alert = next(r for r in rows if r["group"] == "real_unlabeled" and r["window_start"] == "2025-01-07")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    panels = [
        ("(a) Question, validation dates, ε = 2 per record", [
            ("Question", "Did respiratory cases in Iztapalapa rise in July 2024 compared with the period before?", "agg"),
            ("Typed plan from the LLM", "intent: compare; syndrome: RESP; district: 007; period: July 2024; comparison: previous period", "llm"),
            ("PB-QD", "Dates snap to the window grid. Both cells are already in R: 2 reused, 0 fresh.", "code"),
            ("Answer, written by code", "2024-06-29 to 07-31: 278.3 (95% CI 255.4 to 301.1). 2024-05-27 to 06-28: 98.9 (76.1 to 121.7). Change: 179.4 (147.1 to 211.6). Privacy cost: ρ = 0.", "agg"),
        ]),
        ("(b) Real alert, test period, ε = 2 per record", [
            ("Gate", f"Window starting {alert['window_start']}: score passes the threshold in {alert['boroughs']} borough; one event goes to the LLM.", "code"),
            ("Evidence to the LLM", "Released cells, baselines, z-scores, and citywide FLU and COVID values only.", "agg"),
            ("LLM verdict, a priority label", alert["llm"].capitalize() + ". The verdict never hides an alert from officials.", "llm"),
            ("Briefing, numbers checked", alert["briefing"].replace("Suspended", "Suspended [sic]"), "llm"),
        ]),
    ]
    for ax, (title, steps) in zip(axes, panels):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=8.2, loc="left", color=INK)
        y = 10
        for (name, text, kind), h in zip(steps, [1.9, 1.7, 1.7, 4.4]):
            y -= h
            ax.add_patch(FancyBboxPatch((0, y), 10, h - 0.12, boxstyle="round,pad=0,rounding_size=0.12", facecolor=TINT[kind], edgecolor=SOFT, linewidth=0.7))
            ax.text(0.22, y + h - 0.42, name, fontsize=7, weight="bold", va="center", color=INK)
            ax.text(0.22, y + h - 0.72, "\n".join(textwrap.wrap(text, 50)), fontsize=6.7, va="top", color=INK, linespacing=1.22)
            y -= 0.1
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(OUT / "fig_worked_example.pdf")
    fig.savefig(OUT / "fig_worked_example.png")
    plt.close(fig)


architecture()
worked_example()
