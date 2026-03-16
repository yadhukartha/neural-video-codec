"""Training loop for the intra-frame codec with multi-rate λ support.

Usage:
    python train.py --config configs/base.yaml --lmbda 0.013
    python train.py --config configs/base.yaml --lmbda 0.013 --dummy  # quick test
"""

import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml
from PIL import Image

from models.codec import IntraFrameCodec
from losses.rd_loss import RateDistortionLoss


class FrameDataset(Dataset):
    """Dataset of individual video frames for intra-frame training."""

    def __init__(self, frame_dir: str, patch_size: int = 256,
                 sequences: list[str] | None = None):
        self.patch_size = patch_size
        self.frame_paths = []

        frame_dir = Path(frame_dir)
        if sequences:
            dirs = [frame_dir / seq for seq in sequences]
        else:
            dirs = [d for d in frame_dir.iterdir() if d.is_dir()]

        for d in dirs:
            self.frame_paths.extend(sorted(d.glob("*.png")))

        if not self.frame_paths:
            raise FileNotFoundError(
                f"No PNG frames found in {frame_dir}. "
                "Run: python data/prepare_dataset.py --dummy"
            )

        self.transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return self.transform(img)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device,
                    grad_clip):
    model.train()
    total_loss = 0.0
    total_rate = 0.0
    total_dist = 0.0
    total_psnr = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            x_hat, likelihoods = model(batch)
            losses = criterion(batch, x_hat, likelihoods)

        loss = losses["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_rate += losses["rate"].item() if torch.is_tensor(losses["rate"]) else losses["rate"]
        total_dist += losses["distortion"].item()
        total_psnr += losses["psnr"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "rate": total_rate / n_batches,
        "distortion": total_dist / n_batches,
        "psnr": total_psnr / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rate = 0.0
    total_dist = 0.0
    total_psnr = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            x_hat, likelihoods = model(batch)
            losses = criterion(batch, x_hat, likelihoods)

        total_loss += losses["loss"].item()
        total_rate += losses["rate"].item() if torch.is_tensor(losses["rate"]) else losses["rate"]
        total_dist += losses["distortion"].item()
        total_psnr += losses["psnr"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "rate": total_rate / n_batches,
        "distortion": total_dist / n_batches,
        "psnr": total_psnr / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--lmbda", type=float, default=0.013,
                        help="Rate-distortion tradeoff λ")
    parser.add_argument("--distortion", type=str, default="mse",
                        choices=["mse", "ms-ssim", "lpips"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy data for quick testing")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seed
    seed = cfg["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data
    if args.dummy:
        from data.prepare_dataset import create_dummy_dataset
        create_dummy_dataset(Path(cfg["data"]["frame_dir"]), num_frames=20)

    dataset = FrameDataset(
        cfg["data"]["frame_dir"],
        patch_size=cfg["data"]["patch_size"],
        sequences=cfg["data"]["sequences"],
    )
    # Use 90/10 train/val split
    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"Train: {len(train_set)} frames, Val: {len(val_set)} frames")

    # Model
    model_cfg = cfg["model"]["intra"]
    model = IntraFrameCodec(
        in_channels=3,
        latent_channels=model_cfg["channels"],
        hyper_channels=model_cfg["hyper_channels"],
        num_res_blocks=model_cfg["num_res_blocks"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")

    # Loss
    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion=args.distortion)
    if args.distortion == "lpips":
        criterion = criterion.to(device)

    # Optimizer + scheduler
    epochs = args.epochs or cfg["training"]["epochs"]
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg["training"]["lr_min"]
    )

    # Mixed precision
    use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.GradScaler() if use_amp else None

    # Wandb
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg["wandb"]["project"],
                config={**cfg, "lmbda": args.lmbda, "distortion": args.distortion},
                name=f"intra_lmbda{args.lmbda}_{args.distortion}",
            )
        except Exception as e:
            print(f"wandb init failed: {e}, continuing without logging")
            args.no_wandb = True

    # Checkpointing
    ckpt_dir = Path(args.checkpoint_dir) / f"intra_lmbda{args.lmbda}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            cfg["training"]["grad_clip"],
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss={train_metrics['loss']:.4f} rate={train_metrics['rate']:.4f}bpp "
            f"PSNR={train_metrics['psnr']:.2f}dB | "
            f"Val loss={val_metrics['loss']:.4f} rate={val_metrics['rate']:.4f}bpp "
            f"PSNR={val_metrics['psnr']:.2f}dB | "
            f"lr={lr:.2e}"
        )

        if not args.no_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/rate": train_metrics["rate"],
                "train/distortion": train_metrics["distortion"],
                "train/psnr": train_metrics["psnr"],
                "val/loss": val_metrics["loss"],
                "val/rate": val_metrics["rate"],
                "val/distortion": val_metrics["distortion"],
                "val/psnr": val_metrics["psnr"],
                "lr": lr,
            })

        # Save best + periodic checkpoints
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "lmbda": args.lmbda,
            }, ckpt_dir / "best.pt")

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "lmbda": args.lmbda,
            }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")

    if not args.no_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
