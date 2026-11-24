#!/usr/bin/env python3
"""Plots for the additional SBRP studies (eviction policy, mechanism
breakdown, GPU size, PM latency, persistence domain).

Each study reads <root>/<study>_results/<model>_<benchmark>.out, computes the
speedup of every series over its own epoch baseline (higher is better), and
writes <root>/outputs/<study>_graph.pdf and <root>/outputs/<study>_output.txt.

usage: python3 thesis_plots.py <study> [<study> ...] [--root ..]
       python3 thesis_plots.py all
"""
import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARKS = ["KVS", "HM", "SRAD", "Red", "MQ", "Scan"]
LABELS = ["gpKVS", "HM", "SRAD", "Red", "MQ", "Scan", "Gmean"]

# Categorical (identity) and ordinal (magnitude) palettes; both validated for
# colour-vision deficiency and contrast on white. Hatches keep categorical
# series distinguishable in grayscale print.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
HATCHES = ["", "///", "..."]
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# study -> (series kind, [(model, baseline, legend label), ...])
STUDIES = {
    "eviction": ("categorical", [
        ("sbrp_near_eager", "epoch_near", "Eager"),
        ("sbrp_near_lazy", "epoch_near", "Lazy"),
        ("sbrp_near", "epoch_near", "Window (default)"),
    ]),
    "mechanism": ("categorical", [
        ("sbrp_near", "epoch_near", "SBRP"),
        ("sbrp_near_nodeprio", "epoch_near", "No warp deprioritization"),
    ]),
    "smscale": ("ordinal", [
        ("sbrp_near_sm15", "epoch_near_sm15", "15 SMs"),
        ("sbrp_near", "epoch_near", "30 SMs (default)"),
        ("sbrp_near_sm60", "epoch_near_sm60", "60 SMs"),
        ("sbrp_near_sm80", "epoch_near_sm80", "80 SMs"),
    ]),
    "latency": ("ordinal", [
        ("sbrp_near_lat150", "epoch_near_lat150", "150 cycles"),
        ("sbrp_near", "epoch_near", "300 cycles (default)"),
        ("sbrp_near_lat600", "epoch_near_lat600", "600 cycles"),
        ("sbrp_near_lat900", "epoch_near_lat900", "900 cycles"),
    ]),
    "noadr": ("categorical", [
        ("sbrp_near", "epoch_near", "ADR (default)"),
        ("sbrp_near_noadr", "epoch_near_noadr", "No ADR"),
        ("sbrp_near_noadr_win24", "epoch_near_noadr", "No ADR, window of 24"),
    ]),
}


def cycles(results, model, bench):
    """Total simulated cycles of a finished, validated run, else None."""
    path = os.path.join(results, f"{model}_{bench}.out")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    if "exit detected" not in text or "DON'T Match" in text:
        return None
    found = re.findall(r"gpu_tot_sim_cycle = (\d+)", text)
    return int(found[-1]) if found else None


def speedups(results, series):
    table = []
    for model, base, _ in series:
        row = []
        for b in BENCHMARKS:
            m, e = cycles(results, model, b), cycles(results, base, b)
            row.append(e / m if m and e else None)
        valid = [x for x in row if x]
        row.append(math.exp(sum(map(math.log, valid)) / len(valid)) if valid else None)
        table.append(row)
    return table


def plot(study, root):
    kind, series = STUDIES[study]
    results = os.path.join(root, f"{study}_results")
    table = speedups(results, series)
    os.makedirs(os.path.join(root, "outputs"), exist_ok=True)

    with open(os.path.join(root, "outputs", f"{study}_output.txt"), "w") as out:
        out.write("series, " + ", ".join(LABELS) + "\n")
        for (_, _, label), row in zip(series, table):
            out.write(label + ", " + ", ".join("n/a" if v is None else f"{v:.3f}" for v in row) + "\n")

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    n = len(series)
    width = 0.8 / n
    for i, ((_, _, label), row) in enumerate(zip(series, table)):
        color = CATEGORICAL[i] if kind == "categorical" else ORDINAL[i]
        hatch = HATCHES[i] if kind == "categorical" else ""
        labelled = False
        for g, v in enumerate(row):
            x = g + (i - (n - 1) / 2) * width
            if v is None:
                ax.text(x, 0.04, "n/a", rotation=90, ha="center", va="bottom", fontsize=7, color="#555555")
                continue
            ax.bar(x, v, width * 0.92, color=color, hatch=hatch, edgecolor="black", linewidth=0.5,
                   label=None if labelled else label)
            labelled = True
    ax.axhline(1.0, color="#444444", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(len(BENCHMARKS) - 0.5, color="#bbbbbb", linewidth=0.6, zorder=0)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Speedup over epoch")
    ax.set_ylim(0, max(v for row in table for v in row if v) * 1.15)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=n, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), fontsize=8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "outputs", f"{study}_graph.pdf"), bbox_inches="tight")
    plt.close(fig)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("studies", nargs="+", help=" ".join(STUDIES) + " or all")
    ap.add_argument("--root", default="..", help="directory holding <study>_results/ (default: ..)")
    args = ap.parse_args()
    for study in (list(STUDIES) if args.studies == ["all"] else args.studies):
        table = plot(study, args.root)
        print(study)
        for (_, _, label), row in zip(STUDIES[study][1], table):
            print(f"  {label:<28}" + " ".join("  n/a" if v is None else f"{v:5.2f}" for v in row))
