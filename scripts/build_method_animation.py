from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from spectra_probe.metrics import bucket_mass, global_buckets
from spectra_probe.toy import POPULARITY_ORDER, toy_example

COLORS = {
    "ink": "#20364e",
    "muted": "#66788f",
    "dg": "#8a9bb3",
    "spectra": "#149d81",
    "probe": "#f0a53a",
    "tail": "#7253c8",
    "grid": "#e6ebf2",
}
PHASES = [
    ("Direct generation decodes a short list", "Every category outside the list gets zero probability."),
    ("SPECTRA asks the LLM about every category", "One yes/no probe per category. The score is logit(Yes) minus logit(No)."),
    ("A softmax over the scores gives the full distribution", "Long-tail interests keep their probability."),
]
LIST_LENGTH = 5


def example_data():
    example = toy_example()
    order = [example.categories.index(c) for c in POPULARITY_ORDER]
    tail_idx = global_buckets(example.popularity[None, :])["tail"]
    decoded_names = example.decoded_list(LIST_LENGTH)
    decoded = example.direct_generation(LIST_LENGTH)
    return {
        "labels": [example.categories[i] for i in order],
        "truth": example.truth[order],
        "spectra": example.likelihood.distribution[order],
        "scores": example.likelihood.scores[order],
        "decoded": decoded[order],
        "decoded_positions": [POPULARITY_ORDER.index(c) for c in decoded_names],
        "tail_start": min(order.index(i) for i in tail_idx),
        "js_dg": example.js(decoded),
        "js_spectra": example.js(example.likelihood.distribution),
        "true_tail": bucket_mass(example.truth, tail_idx),
        "dg_tail": bucket_mass(decoded, tail_idx),
        "spectra_tail": bucket_mass(example.likelihood.distribution, tail_idx),
    }


def draw(data, phase, shown, probing=None, final=False):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0.075, 0.2, 0.9, 0.56])
    n = len(data["labels"])
    x = np.arange(n)
    ax.axvspan(data["tail_start"] - 0.5, n - 0.5, color=COLORS["tail"], alpha=0.06, lw=0)
    ax.text(n - 0.6, 0.238, "long-tail genres", ha="right", va="top", color=COLORS["tail"], fontsize=9.5)
    dg_heights = np.zeros(n)
    if phase == 0:
        for position in data["decoded_positions"][:shown]:
            dg_heights[position] = 1.0 / max(shown, 1)
    else:
        dg_heights = data["decoded"].copy()
    dg_alpha = 1.0 if phase == 0 else 0.45
    ax.bar(x - 0.2, np.minimum(dg_heights, 0.25), 0.38, color=COLORS["dg"], alpha=dg_alpha)
    for position in np.nonzero(dg_heights > 0.25)[0]:
        ax.text(position - 0.2, 0.243, f"{dg_heights[position]:.2f}", ha="center", va="top", color="white", fontsize=7.5, rotation=90)
    if phase >= 1:
        heights = np.where(x < shown, data["spectra"], 0.0)
        ax.bar(x + 0.2, np.minimum(heights, 0.25), 0.38, color=COLORS["spectra"])
        for position in np.nonzero(heights > 0.25)[0]:
            ax.text(position + 0.2, 0.243, f"{heights[position]:.2f}", ha="center", va="top", color="white", fontsize=7.5, rotation=90)
    if probing is not None:
        ax.axvspan(probing - 0.5, probing + 0.5, color=COLORS["probe"], alpha=0.22, lw=0)
        score = data["scores"][probing]
        ax.annotate(
            f"Does the viewer enjoy {data['labels'][probing]} films?\nlogit(Yes) - logit(No) = {score:+.2f}",
            xy=(probing, 0.205),
            xytext=(min(max(probing, 3.2), n - 4.2), 0.228 if probing < data["tail_start"] - 3 else 0.192),
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["ink"],
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff6e5", ec=COLORS["probe"], lw=1),
        )
    ax.scatter(x, data["truth"], s=30, color=COLORS["ink"], zorder=3)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 0.25)
    ax.set_xticks(x, data["labels"], rotation=42, ha="right")
    ax.set_ylabel("Probability", color=COLORS["muted"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b9c3d0")
    ax.tick_params(colors="#44546a", labelsize=9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    title, subtitle = PHASES[phase]
    fig.text(0.075, 0.925, title, fontsize=16, weight="bold", color=COLORS["ink"])
    fig.text(0.075, 0.87, subtitle, fontsize=11, color=COLORS["muted"])
    for i in range(len(PHASES)):
        fig.text(0.9 + 0.025 * i, 0.93, "●", fontsize=11, color=COLORS["spectra"] if i == phase else "#cfd7e2")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS["ink"], ms=6, label="User's true distribution"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["dg"], label=f"Direct generation, top-{LIST_LENGTH} list"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["spectra"], label="SPECTRA readout"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.83), ncol=3, frameon=False, fontsize=9.5)
    if final:
        ax.text(
            11.2,
            0.168,
            f"JS divergence to truth:  list {data['js_dg']:.3f}   SPECTRA {data['js_spectra']:.3f}\n"
            f"Tail mass:  true {data['true_tail']:.2f}   list {data['dg_tail']:.2f}   SPECTRA {data['spectra_tail']:.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=COLORS["ink"],
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.55", fc="#eef8f5", ec=COLORS["spectra"], lw=1),
        )
    fig.text(0.075, 0.02, "Synthetic example produced by the released code. Not a result from the paper.", fontsize=8.5, color="#7a8798")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build(output_dir: Path) -> dict:
    data = example_data()
    n = len(data["labels"])
    frames, durations = [], []

    def add(image, ms):
        frames.append(image)
        durations.append(ms)

    add(draw(data, 0, 0), 900)
    for shown in range(1, LIST_LENGTH + 1):
        add(draw(data, 0, shown), 520 if shown < LIST_LENGTH else 1900)
    add(draw(data, 1, 0), 900)
    for j in range(n):
        add(draw(data, 1, j + 1, probing=j), 300)
    add(draw(data, 2, n), 700)
    poster = draw(data, 2, n, final=True)
    add(poster, 4200)

    output_dir.mkdir(parents=True, exist_ok=True)
    quantized = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
    quantized[0].save(
        output_dir / "method_animation.gif",
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    poster.save(output_dir / "method_animation_poster.png")
    values = {
        "list_length": LIST_LENGTH,
        "decoded_list": [data["labels"][p] for p in data["decoded_positions"]],
        "js_direct_generation": round(float(data["js_dg"]), 6),
        "js_spectra": round(float(data["js_spectra"]), 6),
        "true_tail_mass": round(float(data["true_tail"]), 6),
        "direct_generation_tail_mass": round(float(data["dg_tail"]), 6),
        "spectra_tail_mass": round(float(data["spectra_tail"]), 6),
        "frames": len(frames),
    }
    (output_dir / "method_animation.json").write_text(json.dumps(values, indent=2) + "\n")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir)))
