"""Evaluation pipeline: benchmark learned codec against H.264/H.265/AV1.

Computes BD-Rate, RD curves, PSNR, MS-SSIM, and VMAF metrics.

Usage:
    python evaluate.py --config configs/base.yaml --checkpoint checkpoints/intra_lmbda0.013/best.pt
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.codec import IntraFrameCodec


def encode_with_ffmpeg(input_pattern: str, output_path: str, codec: str,
                       qp: int, width: int = 1920, height: int = 1080) -> float:
    """Encode frames with a classical codec via ffmpeg, return file size in bytes."""
    codec_args = {
        "h264": ["-c:v", "libx264", "-crf", str(qp), "-preset", "medium"],
        "h265": ["-c:v", "libx265", "-crf", str(qp), "-preset", "medium"],
        "av1": ["-c:v", "libaom-av1", "-crf", str(qp), "-cpu-used", "4",
                "-row-mt", "1"],
    }
    if codec not in codec_args:
        raise ValueError(f"Unknown codec: {codec}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "30",
        "-i", input_pattern,
        *codec_args[codec],
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return os.path.getsize(output_path)


def decode_with_ffmpeg(input_path: str, output_dir: str) -> list[str]:
    """Decode a video file back to PNG frames."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-q:v", "1",
        os.path.join(output_dir, "frame_%04d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(Path(output_dir).glob("*.png"))


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PSNR between two images (uint8 arrays)."""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_vmaf(ref_path: str, dist_path: str) -> float:
    """Compute VMAF score using ffmpeg's libvmaf filter."""
    cmd = [
        "ffmpeg", "-y",
        "-i", dist_path,
        "-i", ref_path,
        "-lavfi", "libvmaf=log_fmt=json:log_path=/dev/stdout",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        vmaf_data = json.loads(result.stdout)
        return vmaf_data["pooled_metrics"]["vmaf"]["mean"]
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        return -1.0  # VMAF not available


def compute_ms_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute MS-SSIM using torchmetrics."""
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    orig_t = torch.from_numpy(original).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    recon_t = torch.from_numpy(reconstructed).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return ms_ssim(recon_t, orig_t).item()


def bd_rate(rate1, psnr1, rate2, psnr2):
    """Compute Bjontegaard delta rate (BD-Rate).

    Negative values mean rate2 is better (uses fewer bits at same quality).
    rate/psnr arrays should have ≥4 points.
    """
    from scipy.interpolate import interp1d

    # Sort by PSNR
    idx1 = np.argsort(psnr1)
    idx2 = np.argsort(psnr2)
    rate1, psnr1 = np.array(rate1)[idx1], np.array(psnr1)[idx1]
    rate2, psnr2 = np.array(rate2)[idx2], np.array(psnr2)[idx2]

    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)

    # Overlap range
    min_psnr = max(psnr1[0], psnr2[0])
    max_psnr = min(psnr1[-1], psnr2[-1])

    if min_psnr >= max_psnr:
        return float("nan")

    # Fit cubic splines
    f1 = interp1d(psnr1, log_rate1, kind="cubic", fill_value="extrapolate")
    f2 = interp1d(psnr2, log_rate2, kind="cubic", fill_value="extrapolate")

    # Integrate
    from scipy.integrate import quad
    int1, _ = quad(f1, min_psnr, max_psnr)
    int2, _ = quad(f2, min_psnr, max_psnr)

    avg1 = int1 / (max_psnr - min_psnr)
    avg2 = int2 / (max_psnr - min_psnr)

    return (np.exp(avg2 - avg1) - 1) * 100  # Percentage


@torch.no_grad()
def evaluate_learned_codec(model, frame_dir: str, device: torch.device):
    """Evaluate learned intra-frame codec on a sequence of frames.

    Returns:
        bpp: Average bits per pixel.
        psnr: Average PSNR in dB.
        ms_ssim: Average MS-SSIM.
    """
    model.eval()
    to_tensor = transforms.ToTensor()

    frame_paths = sorted(Path(frame_dir).glob("*.png"))
    total_bpp = 0.0
    total_psnr = 0.0
    total_ms_ssim = 0.0

    for fpath in frame_paths:
        img = Image.open(fpath).convert("RGB")
        w, h = img.size
        # Pad to multiple of 64 for the 16× downsampling
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64

        x = to_tensor(img).unsqueeze(0).to(device)
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x_hat, likelihoods = model(x)

        # Remove padding
        if pad_h or pad_w:
            x_hat = x_hat[:, :, :h, :w]
            for k in likelihoods:
                pass  # likelihoods are at lower resolution, keep them as is

        # Rate
        num_pixels = h * w
        bpp = sum(
            -torch.log2(lk.clamp(min=1e-9)).sum().item()
            for lk in likelihoods.values()
        ) / num_pixels
        total_bpp += bpp

        # Distortion
        x_orig = to_tensor(img).unsqueeze(0).to(device)
        x_hat_clamped = x_hat.clamp(0, 1)
        mse = torch.mean((x_orig - x_hat_clamped) ** 2).item()
        psnr = -10.0 * np.log10(mse + 1e-10)
        total_psnr += psnr

        # MS-SSIM
        orig_np = np.array(img)
        recon_np = (x_hat_clamped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        ms_ssim_val = compute_ms_ssim(orig_np, recon_np)
        total_ms_ssim += ms_ssim_val

    n = len(frame_paths)
    return {
        "bpp": total_bpp / n,
        "psnr": total_psnr / n,
        "ms_ssim": total_ms_ssim / n,
    }


def evaluate_classical_codec(frame_dir: str, codec: str, qp_values: list[int]):
    """Evaluate a classical codec at multiple QP values.

    Returns:
        List of dicts with bpp, psnr, ms_ssim for each QP.
    """
    frame_paths = sorted(Path(frame_dir).glob("*.png"))
    if not frame_paths:
        return []

    # Read first frame to get dimensions
    first_img = np.array(Image.open(frame_paths[0]))
    h, w = first_img.shape[:2]
    num_pixels = h * w
    num_frames = len(frame_paths)

    results = []
    for qp in qp_values:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Encode
            encoded_path = os.path.join(tmpdir, f"encoded.mp4")
            input_pattern = str(Path(frame_dir) / "frame_%04d.png")
            file_size = encode_with_ffmpeg(input_pattern, encoded_path, codec, qp, w, h)

            # Decode
            decoded_dir = os.path.join(tmpdir, "decoded")
            decoded_frames = decode_with_ffmpeg(encoded_path, decoded_dir)

            # Compute metrics
            bpp = (file_size * 8) / (num_pixels * num_frames)
            total_psnr = 0.0
            total_ms_ssim = 0.0

            for orig_path, dec_path in zip(frame_paths, decoded_frames):
                orig = np.array(Image.open(orig_path).convert("RGB"))
                dec = np.array(Image.open(dec_path).convert("RGB"))
                # Handle dimension mismatch from ffmpeg
                min_h = min(orig.shape[0], dec.shape[0])
                min_w = min(orig.shape[1], dec.shape[1])
                orig = orig[:min_h, :min_w]
                dec = dec[:min_h, :min_w]

                total_psnr += compute_psnr(orig, dec)
                total_ms_ssim += compute_ms_ssim(orig, dec)

            n = min(len(frame_paths), len(decoded_frames))
            results.append({
                "qp": qp,
                "bpp": bpp,
                "psnr": total_psnr / n,
                "ms_ssim": total_ms_ssim / n,
            })
            print(f"  {codec} QP={qp}: bpp={bpp:.4f} PSNR={total_psnr/n:.2f}dB "
                  f"MS-SSIM={total_ms_ssim/n:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--sequence", type=str, default="Beauty",
                        help="UVG sequence to evaluate")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load model
    model_cfg = cfg["model"]["intra"]
    model = IntraFrameCodec(
        in_channels=3,
        latent_channels=model_cfg["channels"],
        hyper_channels=model_cfg["hyper_channels"],
        num_res_blocks=model_cfg["num_res_blocks"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    lmbda = ckpt.get("lmbda", "unknown")
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, λ={lmbda}")

    frame_dir = os.path.join(cfg["data"]["frame_dir"], args.sequence)

    # Evaluate learned codec
    print(f"\nEvaluating learned codec on {args.sequence}:")
    learned_results = evaluate_learned_codec(model, frame_dir, device)
    print(f"  Learned: bpp={learned_results['bpp']:.4f} "
          f"PSNR={learned_results['psnr']:.2f}dB "
          f"MS-SSIM={learned_results['ms_ssim']:.4f}")

    # Evaluate classical codecs
    classical_results = {}
    for codec in cfg["evaluation"]["codecs"]:
        print(f"\nEvaluating {codec}:")
        classical_results[codec] = evaluate_classical_codec(
            frame_dir, codec, cfg["evaluation"]["qp_values"]
        )

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "sequence": args.sequence,
        "learned": learned_results,
        "classical": classical_results,
        "lmbda": lmbda,
    }
    results_path = out_dir / f"results_{args.sequence}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # BD-Rate vs H.264
    if "h264" in classical_results and len(classical_results["h264"]) >= 4:
        h264_rates = [r["bpp"] for r in classical_results["h264"]]
        h264_psnrs = [r["psnr"] for r in classical_results["h264"]]
        # For BD-rate we need multiple points from the learned codec too
        # With a single checkpoint we have one point; report it for reference
        print(f"\nNote: BD-Rate requires multiple RD points from the learned codec.")
        print(f"Train with multiple λ values and re-run to compute BD-Rate.")


if __name__ == "__main__":
    main()
