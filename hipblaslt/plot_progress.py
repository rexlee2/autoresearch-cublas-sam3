#!/usr/bin/env python3
"""
Plot autoresearch progress — Karpathy-style progress chart.

Reads iteration_log.jsonl and produces progress.png showing:
  - All experiments (red = discarded, green = kept)
  - Running best staircase line
  - Clean annotations for kept improvements
  - Summary stats box

Usage:
    python3 plot_progress.py                    # default output: progress.png
    python3 plot_progress.py -o my_plot.png     # custom output path
    python3 plot_progress.py --log custom.jsonl  # custom log file
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
except ImportError:
    print("ERROR: matplotlib not installed.  Run: pip install matplotlib")
    sys.exit(1)


def load_iterations(log_path: str) -> list[dict]:
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def shorten_description(desc: str) -> str:
    """Extract a clean, human-readable label that says what actually helped."""
    desc = desc.strip()

    # Strip everything after the em-dash explanation
    for sep in (" \u2014 ", " — "):
        if sep in desc:
            desc = desc.split(sep)[0]

    d = desc.lower()

    # --- Match known patterns and produce clean labels ---

    # Workspace
    if "workspace" in d:
        m = re.search(r"(\d+)\s*mb", d)
        if m:
            return f"workspace {m.group(1)}MB"

    # Transpose B
    if "transpose b" in d or "transpose" in d.lower():
        # Extract shape label (e.g. "trk_xattn_qk") or shape tuple
        m = re.search(r"for\s+(\w+)", desc)
        if m:
            label = m.group(1)
            return f"NT layout: {label}"
        return "NT layout"

    # Split-K
    if "split-k" in d:
        sk = re.search(r"split-k=(\d+)", d, re.IGNORECASE)
        label = re.search(r"for\s+(\w+)", desc)
        if sk and label:
            return f"split-K={sk.group(1)}: {label.group(1)}"
        if sk:
            return f"split-K={sk.group(1)}"

    # Dtype
    if "float16" in d and "bfloat" not in d:
        return "FP16"
    if "bfloat16" in d:
        return "BF16"

    # Batch windows / batch size
    if "window batch size" in d:
        m = re.search(r"size\s+(\d+)", d)
        if m:
            return f"batch={m.group(1)} windows"
    if "batch" in d and "window" in d:
        return "batch windows"

    # Precision
    if "precision" in d:
        m = re.search(r'precision[=\s]+"?(\w+)"?', d)
        if m:
            return f"precision={m.group(1)}"

    # BLAS library
    if "cublas" in d and "cublaslt" not in d:
        # Check if combined with something
        if "transpose" in d:
            m = re.search(r"for\s+(\w+)", desc)
            label = m.group(1) if m else ""
            return f"cuBLAS+NT: {label}"
        return "cuBLAS (legacy)"

    # Combos — extract the key changes
    if "combo" in d or "+" in desc:
        parts = []
        if "bfloat16" in d:
            parts.append("BF16")
        elif "float16" in d:
            parts.append("FP16")
        sk = re.search(r"split-k=(\d+)", d, re.IGNORECASE)
        if sk:
            parts.append(f"split-K={sk.group(1)}")
        ws = re.search(r"workspace[= ]*(\d+)", d)
        if ws:
            parts.append(f"ws={ws.group(1)}MB")
        if parts:
            return " + ".join(parts)

    # Padding
    if "pad" in d:
        m = re.search(r"(\d+)", d)
        if m:
            return f"pad to {m.group(1)}"

    # TF32
    if "tf32" in d:
        if "disable" in d or "no" in d:
            return "no TF32"
        return "TF32 on"

    # Fallback: truncate
    if len(desc) > 30:
        return desc[:27] + "..."
    return desc


def plot_progress(
    log_entries: list[dict],
    output_path: str = "progress.png",
    title_prefix: str = "autoresearch-cublas-sam3",
):
    """Generate Karpathy-style progress plot."""

    if not log_entries:
        print("No iterations found in log.")
        return

    # Parse data
    iterations = []
    scores = []
    kept_mask = []
    descriptions = []

    for entry in log_entries:
        iterations.append(entry["iteration"])
        scores.append(float(entry["score"]))
        kept_mask.append(entry.get("kept") in ("true", True))
        descriptions.append(entry.get("description", ""))

    # Running best (staircase)
    running_best = []
    current_best = float("inf")
    for score, kept in zip(scores, kept_mask):
        if kept and score < current_best:
            current_best = score
        running_best.append(current_best if current_best < float("inf") else score)

    total = len(iterations)
    num_kept = sum(kept_mask)

    # Display values: negate so higher = better (TFLOP/s)
    display_scores = [-s for s in scores]
    display_best = [-b for b in running_best]

    # ---- Style ----
    BG = "#1a1a2e"
    PANEL = "#16213e"
    GRID = "#2a2a4a"
    TEXT = "#e0e0e0"
    TEXT_DIM = "#8888aa"
    GREEN = "#00e676"
    GREEN_DIM = "#00c853"
    RED = "#ff5252"
    RED_DIM = "#b71c1c"
    ACCENT = "#64ffda"

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Grid
    ax.grid(True, alpha=0.15, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    # --- Discarded points (red) ---
    disc_x = [it for it, k in zip(iterations, kept_mask) if not k]
    disc_y = [sc for sc, k in zip(display_scores, kept_mask) if not k]
    ax.scatter(disc_x, disc_y, c=RED, s=50, zorder=2, alpha=0.45,
               edgecolors=RED_DIM, linewidth=0.5, label="Discarded")

    # --- Kept points (green, larger) ---
    kept_x = [it for it, k in zip(iterations, kept_mask) if k]
    kept_y = [sc for sc, k in zip(display_scores, kept_mask) if k]
    ax.scatter(kept_x, kept_y, c=GREEN, s=120, zorder=5, alpha=0.95,
               edgecolors="white", linewidth=1.0, label="Kept")

    # --- Running best staircase ---
    ax.step(iterations, display_best, where="post", c=GREEN_DIM,
            linewidth=2.5, zorder=4, alpha=0.8)
    # Fill under the staircase
    ax.fill_between(iterations, display_best,
                    min(display_scores) - 1,
                    step="post", alpha=0.06, color=GREEN)

    # --- Annotations for kept improvements ---
    # Collect kept points with their score improvement
    kept_points = []
    prev_best = None
    for i, (it, sc, kept, desc) in enumerate(
        zip(iterations, display_scores, kept_mask, descriptions)
    ):
        if kept:
            delta = sc - prev_best if prev_best is not None else 0
            kept_points.append((it, sc, desc, delta))
            prev_best = sc

    max_iter = max(iterations) if iterations else 1
    y_range = max(display_scores) - min(display_scores) if display_scores else 1

    # Only annotate top-N most impactful kept points to avoid clutter
    # Always annotate baseline and the final best
    non_baseline = [(it, sc, desc, delta, idx)
                    for idx, (it, sc, desc, delta) in enumerate(kept_points)
                    if shorten_description(desc) not in ("baseline", "")]
    # Sort by delta descending, pick top ones
    max_annotations = min(8, len(non_baseline))
    top_by_delta = sorted(non_baseline, key=lambda x: x[3], reverse=True)[:max_annotations]
    annotate_iters = {p[0] for p in top_by_delta}  # set of iteration numbers

    label_idx = 0
    for idx, (it, sc, desc, delta) in enumerate(kept_points):
        label = shorten_description(desc)
        if not label or label == "baseline":
            ax.annotate(
                f"baseline: {sc:.1f} TFLOP/s",
                xy=(it, sc), xytext=(15, -25),
                textcoords="offset points",
                fontsize=10, color=ACCENT, fontweight="bold",
                ha="left",
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, alpha=0.6, lw=1.2),
                path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
            )
            continue

        # Skip non-top annotations
        if it not in annotate_iters:
            continue

        # Add improvement delta to label
        if delta > 0:
            label = f"{label}  (+{delta:.2f})"

        # Stagger labels using modular positions
        positions = [
            (15, 30, "left", "bottom"),
            (15, -30, "left", "top"),
            (-15, 35, "right", "bottom"),
            (-15, -35, "right", "top"),
        ]
        ox, oy, ha, va = positions[label_idx % len(positions)]

        # Adjust for right edge
        if it > max_iter * 0.75:
            ox = -abs(ox) - 5
            ha = "right"

        ax.annotate(
            label,
            xy=(it, sc),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=8, color=TEXT,
            ha=ha, va=va,
            arrowprops=dict(arrowstyle="-|>", color=TEXT_DIM, alpha=0.4, lw=0.7),
            path_effects=[pe.withStroke(linewidth=2.5, foreground=PANEL)],
        )
        label_idx += 1

    # --- Mark the best point ---
    best_tflops = max(display_best)
    best_idx = display_best.index(best_tflops)
    best_iter = iterations[best_idx]
    ax.scatter([best_iter], [best_tflops], c=ACCENT, s=200, zorder=6,
               marker="*", edgecolors="white", linewidth=0.8)

    # Annotate the best point explicitly
    ax.annotate(
        f"best: {best_tflops:.2f} TFLOP/s",
        xy=(best_iter, best_tflops),
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=9, color=ACCENT, fontweight="bold",
        ha="center",
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)],
    )

    # --- Labels ---
    ax.set_xlabel("Experiment #", fontsize=12, color=TEXT, labelpad=10)
    ax.set_ylabel("Weighted Average TFLOP/s", fontsize=12, color=TEXT, labelpad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    # Integer x-ticks
    ax.set_xticks(range(0, max(iterations) + 1,
                        max(1, (max(iterations) + 1) // 20)))

    # Y range — extra headroom so stats box sits above the data
    ymin = min(display_scores) - 0.5
    ymax = 70
    ax.set_ylim(ymin, ymax)

    # --- Title ---
    baseline_tflops = display_best[0]
    improvement = (best_tflops - baseline_tflops) / baseline_tflops * 100
    ax.set_title(
        f"{title_prefix}",
        fontsize=16, fontweight="bold", color=TEXT, pad=15,
    )

    # --- Stats box (top-left) ---
    stats_text = (
        f"Experiments: {total}\n"
        f"Kept: {num_kept}\n"
        f"Baseline: {baseline_tflops:.2f} TFLOP/s\n"
        f"Best: {best_tflops:.2f} TFLOP/s\n"
        f"Gain: {improvement:+.2f}%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor=BG, edgecolor=GRID, alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, color=TEXT, verticalalignment="top",
            fontfamily="monospace", bbox=props)

    # --- Legend ---
    legend = ax.legend(loc="lower right", fontsize=9,
                       facecolor=PANEL, edgecolor=GRID,
                       labelcolor=TEXT_DIM, framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Plot saved to {output_path}")
    print(f"  Experiments: {total}  |  Kept: {num_kept}")
    print(f"  Baseline: {baseline_tflops:.2f} TFLOP/s")
    print(f"  Best:     {best_tflops:.2f} TFLOP/s  ({improvement:+.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch progress")
    parser.add_argument("-o", "--output", default="progress.png", help="Output image path")
    parser.add_argument("--log", default="iteration_log.jsonl", help="Iteration log file")
    parser.add_argument("--title", default="autoresearch-cublas-sam3", help="Plot title")
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"No log file found at {args.log}")
        print("Run agent_loop.py first to generate data.")
        sys.exit(1)

    entries = load_iterations(args.log)
    plot_progress(entries, args.output, args.title)


if __name__ == "__main__":
    main()
