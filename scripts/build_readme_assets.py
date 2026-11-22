from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectra_probe.metrics import bucket_mass, global_buckets
from spectra_probe.toy import POPULARITY_ORDER, toy_example

COLORS = {"truth": "#20364e", "dg": "#8a9bb3", "spectra": "#149d81", "tail": "#7253c8", "grid": "#e6ebf2"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9c3d0")
    ax.tick_params(colors="#44546a", labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def build(output_dir: Path) -> dict:
    example = toy_example()
    order = [example.categories.index(c) for c in POPULARITY_ORDER]
    labels = [example.categories[i] for i in order]
    truth = example.truth[order]
    spectra = example.likelihood.distribution[order]
    decoded = example.direct_generation(3)[order]
    tail_idx = global_buckets(example.popularity[None, :])["tail"]
    tail_start = min(order.index(i) for i in tail_idx)
    ks = list(range(1, len(labels) + 1))
    js_dg = [example.js(example.direct_generation(k)) for k in ks]
    js_spectra = example.js(example.likelihood.distribution)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.75, 1]})
    x = np.arange(len(labels))
    left.axvspan(tail_start - 0.5, len(labels) - 0.5, color=COLORS["tail"], alpha=0.06, lw=0)
    left.bar(x - 0.2, decoded, 0.38, color=COLORS["dg"], label="Direct generation, top-3 list")
    left.bar(x + 0.2, spectra, 0.38, color=COLORS["spectra"], label="SPECTRA readout")
    left.scatter(x, truth, s=34, color=COLORS["truth"], zorder=3, label="User's true distribution")
    left.set_xticks(x, labels, rotation=42, ha="right")
    left.set_ylabel("Probability", color="#44546a")
    left.set_title("A decoded list drops the long tail", loc="left", fontsize=12, color=COLORS["truth"], weight="bold")
    left.text(len(labels) - 0.6, 0.325, "long-tail genres", ha="right", va="top", color=COLORS["tail"], fontsize=9.5)
    left.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.42, 1.0))
    style(left)

    right.plot(ks, js_dg, color=COLORS["dg"], lw=2, marker="o", ms=4.5)
    right.axhline(js_spectra, color=COLORS["spectra"], lw=2)
    right.text(19, js_spectra + 0.022, f"SPECTRA  {js_spectra:.3f}", color=COLORS["spectra"], ha="right", fontsize=10, weight="bold")
    right.text(3.4, js_dg[2] + 0.02, f"top-3 list  {js_dg[2]:.3f}", color="#5f7391", fontsize=10)
    right.set_xlabel("Length of the decoded list", color="#44546a")
    right.set_ylabel("JS divergence to truth (bits)", color="#44546a")
    right.set_title("Longer lists do not close the gap", loc="left", fontsize=12, color=COLORS["truth"], weight="bold")
    right.set_ylim(0, max(js_dg) * 1.12)
    right.set_xticks([1, 5, 10, 15, 19])
    style(right)

    fig.text(0.008, 0.012, "Synthetic example produced by the released code. Not a result from the paper.", fontsize=8.5, color="#7a8798")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "toy_demo.png", dpi=150, facecolor="white")
    fig.savefig(output_dir / "toy_demo.svg", facecolor="white")
    plt.close(fig)

    values = {
        "categories": labels,
        "truth": [round(float(v), 6) for v in truth],
        "spectra": [round(float(v), 6) for v in spectra],
        "direct_generation_top3": [round(float(v), 6) for v in decoded],
        "js_direct_generation_by_list_length": [round(float(v), 6) for v in js_dg],
        "js_spectra": round(float(js_spectra), 6),
        "true_tail_mass": round(bucket_mass(example.truth, tail_idx), 6),
        "spectra_tail_mass": round(bucket_mass(example.likelihood.distribution, tail_idx), 6),
    }
    (output_dir / "toy_demo.json").write_text(json.dumps(values, indent=2) + "\n")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    result = build(args.output_dir)
    print(f"JS: top-3 list {result['js_direct_generation_by_list_length'][2]:.3f}, SPECTRA {result['js_spectra']:.3f}")
