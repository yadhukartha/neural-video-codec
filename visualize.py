"""RD curve plotting and visual reconstruction comparisons.

Usage:
    python visualize.py --results results/results_Beauty.json --output plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_rd_curves(results: dict, output_dir: Path, metric: str = "psnr"):
    """Plot rate-distortion curves for learned vs classical codecs."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {"h264": "#e41a1c", "h265": "#377eb8", "av1": "#4daf4a"}
    markers = {"h264": "s", "h265": "^", "av1": "D"}

    # Classical codecs
    for codec, data in results.get("classical", {}).items():
        if not data:
            continue
        rates = [d["bpp"] for d in data]
        values = [d[metric] for d in data]
        ax.plot(rates, values, f"-{markers.get(codec, 'o')}",
                color=colors.get(codec, "gray"),
                label=codec.upper(), markersize=6, linewidth=1.5)

    # Learned codec (single or multiple points)
    learned = results.get("learned", {})
    if isinstance(learned, dict) and "bpp" in learned:
        ax.plot(learned["bpp"], learned[metric], "r*",
                markersize=15, label="Ours (learned)", zorder=5)
    elif isinstance(learned, list):
        rates = [d["bpp"] for d in learned]
        values = [d[metric] for d in learned]
        ax.plot(rates, values, "-*", color="red",
                label="Ours (learned)", markersize=10, linewidth=2, zorder=5)

    ylabel = "PSNR (dB)" if metric == "psnr" else "MS-SSIM"
    ax.set_xlabel("Bitrate (bpp)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"Rate-Distortion: {results.get('sequence', 'Unknown')}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"rd_curve_{metric}_{results.get('sequence', 'seq')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved RD curve to {path}")


def plot_visual_comparison(original_path: str, reconstructed_paths: dict,
                           output_path: str):
    """Side-by-side visual comparison of reconstructions at similar bitrates."""
    from PIL import Image

    orig = np.array(Image.open(original_path))
    n_codecs = len(reconstructed_paths) + 1
    fig, axes = plt.subplots(1, n_codecs, figsize=(5 * n_codecs, 5))

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for i, (name, path) in enumerate(reconstructed_paths.items(), 1):
        recon = np.array(Image.open(path))
        axes[i].imshow(recon)
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results JSON")
    parser.add_argument("--output", type=str, default="plots")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    output_dir = Path(args.output)

    plot_rd_curves(results, output_dir, metric="psnr")
    plot_rd_curves(results, output_dir, metric="ms_ssim")

    print("Done!")


if __name__ == "__main__":
    main()
