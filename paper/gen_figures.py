"""Generate paper figures from benchmark results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# ---------- Load data ----------
with open(os.path.join(PROJ, "benchmark_isolated_results.json")) as f:
    local_data = json.load(f)
with open(os.path.join(PROJ, "benchmark_4090_results.json")) as f:
    gpu_data = json.load(f)
with open(os.path.join(PROJ, "overlap_sweep_results.json")) as f:
    overlap_data = json.load(f)
with open(os.path.join(PROJ, "hotpotqa_results.json")) as f:
    qa_data = json.load(f)

# Common style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
})

COLORS = {
    "no_cache": "#888888",
    "apc_random": "#c0c0c0",
    "apc_sorted": "#5b9bd5",
    "apc_retrieval": "#ed7d31",
    "apc_optimized": "#2ca02c",
}
LABELS = {
    "no_cache": "No Cache",
    "apc_random": "APC + Random",
    "apc_sorted": "APC + Sorted",
    "apc_retrieval": "APC + Retrieval",
    "apc_optimized": "APC + Optimized\n(Ours)",
}

# ============================================================
# Figure 1: TTFT comparison bar chart (both GPUs)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=False)

for ax, (data, title, model) in zip(axes, [
    (local_data, "RTX 4060 Ti (Qwen2.5-1.5B)", "1.5B"),
    (gpu_data, "RTX 4090 (Qwen2.5-7B)", "7B"),
]):
    strategies = ["no_cache", "apc_sorted", "apc_retrieval", "apc_optimized"]
    vals = [data["results"][s]["ttft_p50_ms"] for s in strategies]
    colors = [COLORS[s] for s in strategies]
    labels = [LABELS[s].replace("\n", " ") for s in strategies]

    bars = ax.bar(range(len(strategies)), vals, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title(title, fontsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, max(vals) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ttft_comparison.pdf"), bbox_inches="tight")
plt.close()
print("Generated: ttft_comparison.pdf")

# ============================================================
# Figure 2: Overlap sensitivity
# ============================================================
fig, ax = plt.subplots(figsize=(3.2, 2.4))

overlaps = []
improvements = []
retrieval_ttfts = []
optimized_ttfts = []

for key in sorted(overlap_data.keys()):
    d = overlap_data[key]
    overlaps.append(d["overlap_fraction"])
    improvements.append(d["improvement_pct"])
    retrieval_ttfts.append(d["retrieval_ttft_p50"])
    optimized_ttfts.append(d["optimized_ttft_p50"])

ax.plot(overlaps, improvements, "o-", color=COLORS["apc_optimized"],
        linewidth=1.5, markersize=5, label="Improvement %")
ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
ax.set_xlabel("Document Overlap Fraction")
ax.set_ylabel("TTFT Improvement (%)")
ax.set_title("Overlap Sensitivity", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annotate peak
peak_idx = improvements.index(max(improvements))
ax.annotate(f"Peak: {improvements[peak_idx]:.0f}%",
            xy=(overlaps[peak_idx], improvements[peak_idx]),
            xytext=(overlaps[peak_idx] + 0.15, improvements[peak_idx] - 5),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "overlap_sensitivity.pdf"), bbox_inches="tight")
plt.close()
print("Generated: overlap_sensitivity.pdf")

# ============================================================
# Figure 3: Prefix hit rate vs TTFT scatter (shows tree ordering
#            achieves lower TTFT despite lower hit rate than sorted)
# ============================================================
fig, ax = plt.subplots(figsize=(3.2, 2.4))

for data, marker, gpu_label in [
    (local_data, "o", "4060 Ti"),
    (gpu_data, "s", "4090"),
]:
    for s in ["apc_sorted", "apc_retrieval", "apc_optimized"]:
        r = data["results"][s]
        ax.scatter(r["prefix_hit_rate"] * 100, r["ttft_p50_ms"],
                   color=COLORS[s], marker=marker, s=50, edgecolors="black",
                   linewidths=0.5, zorder=5)

# Legend entries
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["apc_sorted"],
           markeredgecolor="black", markersize=6, label="APC + Sorted"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["apc_retrieval"],
           markeredgecolor="black", markersize=6, label="APC + Retrieval"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["apc_optimized"],
           markeredgecolor="black", markersize=6, label="APC + Optimized"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=6, label="4060 Ti"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=6, label="4090"),
]
ax.legend(handles=legend_elements, fontsize=6, loc="upper left")
ax.set_xlabel("Prefix Hit Rate (%)")
ax.set_ylabel("TTFT p50 (ms)")
ax.set_title("Hit Rate vs. Latency", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "hitrate_vs_ttft.pdf"), bbox_inches="tight")
plt.close()
print("Generated: hitrate_vs_ttft.pdf")

print("\nAll figures generated in", FIG_DIR)
