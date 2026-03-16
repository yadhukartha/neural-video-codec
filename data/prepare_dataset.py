"""Download UVG dataset sequences and extract frames as PNG.

The UVG (Ultra Video Group) dataset is a standard benchmark for video codec
evaluation. This script downloads a subset of sequences and extracts
individual frames using ffmpeg.
"""

import argparse
import os
import subprocess
from pathlib import Path

# UVG dataset URLs (1080p YUV420 → we'll use the MP4 versions from the media server)
# These are the commonly used test sequences
UVG_SEQUENCES = {
    "Beauty": "https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
    "Bosphorus": "https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv",
    "HoneyBee": "https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
}

# Frame dimensions for raw YUV reading
WIDTH, HEIGHT = 1920, 1080


def download_sequence(name: str, url: str, raw_dir: Path) -> Path:
    """Download a raw YUV sequence if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{name}.yuv"

    if out_path.exists():
        print(f"  [skip] {name} already downloaded")
        return out_path

    print(f"  Downloading {name}...")
    subprocess.run(["curl", "-L", "-o", str(out_path), url], check=True)
    return out_path


def extract_frames(name: str, yuv_path: Path, frame_dir: Path,
                   num_frames: int = 120) -> Path:
    """Extract PNG frames from raw YUV using ffmpeg."""
    out_dir = frame_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    first_frame = out_dir / "frame_0001.png"
    if first_frame.exists():
        print(f"  [skip] {name} frames already extracted")
        return out_dir

    print(f"  Extracting {num_frames} frames from {name}...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-pixel_format", "yuv420p",
        "-i", str(yuv_path),
        "-frames:v", str(num_frames),
        "-q:v", "1",
        str(out_dir / "frame_%04d.png"),
    ], check=True, capture_output=True)

    count = len(list(out_dir.glob("*.png")))
    print(f"  Extracted {count} frames to {out_dir}")
    return out_dir


def create_dummy_dataset(frame_dir: Path, num_frames: int = 20):
    """Create small dummy PNG frames for testing without downloading UVG.

    Generates random-ish images with gradients and noise so the model
    has something meaningful to train on during development.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("Need numpy and Pillow for dummy data. pip install numpy Pillow")
        return

    print("Creating dummy dataset for development/testing...")
    for name in ["Beauty", "Bosphorus", "HoneyBee"]:
        out_dir = frame_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, num_frames + 1):
            if (out_dir / f"frame_{i:04d}.png").exists():
                continue
            # Generate a gradient + noise image (more interesting than pure random)
            h, w = 256, 256
            y_grad = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
            x_grad = np.ones((h, 1)) * np.linspace(0, 1, w)[None, :]
            base = np.stack([
                y_grad * 0.5 + 0.2,
                x_grad * 0.4 + 0.3,
                (y_grad + x_grad) * 0.3 + 0.1 + i * 0.01,
            ], axis=-1)
            noise = np.random.RandomState(i + hash(name) % 10000).rand(h, w, 3) * 0.15
            img = np.clip((base + noise) * 255, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"frame_{i:04d}.png")

        print(f"  Created {num_frames} dummy frames in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare UVG dataset")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--sequences", nargs="+",
                        default=["Beauty", "Bosphorus", "HoneyBee"])
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--dummy", action="store_true",
                        help="Create small dummy dataset for testing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frame_dir = data_dir / "frames"
    raw_dir = data_dir / "raw"

    if args.dummy:
        create_dummy_dataset(frame_dir, num_frames=20)
        return

    print(f"Preparing UVG dataset in {data_dir}")
    for name in args.sequences:
        if name not in UVG_SEQUENCES:
            print(f"  [warn] Unknown sequence: {name}, skipping")
            continue
        print(f"\nProcessing {name}:")
        yuv_path = download_sequence(name, UVG_SEQUENCES[name], raw_dir)
        extract_frames(name, yuv_path, frame_dir, args.num_frames)

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
