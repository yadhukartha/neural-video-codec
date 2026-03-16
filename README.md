# Neural Video Codec

End-to-end learned video compression with hyperprior entropy model, benchmarked against classical codecs (H.264, H.265, AV1).

## Architecture

- **Intra-frame codec**: CNN encoder/decoder with GDN activations + scale hyperprior entropy model (Ballé et al. 2018)
- **Inter-frame codec**: SpyNet optical flow estimation → motion compensation → residual coding
- **Entropy model**: Learned hyperprior with Gaussian conditional, using CompressAI's entropy bottleneck
- **Loss**: Rate-distortion L = R + λ·D with MSE, MS-SSIM, or LPIPS distortion

## Setup

```bash
conda activate neural-video-codec
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Create dummy data for testing
python data/prepare_dataset.py --dummy

# 2. Train intra-frame codec (quick test)
python train.py --lmbda 0.013 --epochs 5 --dummy --no-wandb

# 3. Evaluate against classical codecs
python evaluate.py --checkpoint checkpoints/intra_lmbda0.013/best.pt --sequence Beauty

# 4. Plot RD curves
python visualize.py --results results/results_Beauty.json
```

## Training multiple rate points

```bash
for lmbda in 0.0018 0.0035 0.0067 0.013 0.025; do
    python train.py --lmbda $lmbda --no-wandb
done
```

## Full UVG dataset

```bash
python data/prepare_dataset.py --sequences Beauty Bosphorus HoneyBee
```

## Key references

1. Ballé et al. 2018 — "Variational image compression with a scale hyperprior"
2. Agustsson et al. 2020 — "Scale-space flow for end-to-end optimized video compression"
3. Li et al. 2021 — "Deep Contextual Video Compression" (DCVC)
