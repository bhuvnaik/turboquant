import argparse
import os
import numpy as np   # ← add this
import torch

from core.prod_quantizer import TurboQuantProd
from scripts.generate_codebook import generate_codebook


def normalize(x):
    return x / torch.norm(x, dim=1, keepdim=True)

def mse(x, x_hat):
    return torch.mean((x - x_hat) ** 2).item()

def inner_product_error(x, x_hat, y):
    true   = torch.sum(x * y, dim=1)
    approx = torch.sum(x_hat * y, dim=1)
    return torch.mean((true - approx) ** 2).item()

def snr_db(x, x_hat):
    signal_power = torch.mean(x ** 2).item()
    noise_power  = torch.mean((x - x_hat) ** 2).item()
    return 10 * np.log10(signal_power / max(noise_power, 1e-12))

def cosine_similarity(x, x_hat):
    num = torch.sum(x * x_hat, dim=1)
    den = x.norm(dim=1) * x_hat.norm(dim=1)
    return torch.mean(num / den.clamp(min=1e-8)).item()

def uniform_quantize(x, b):
    """Per-coordinate uniform scalar quantization — no rotation baseline."""
    n_levels = 2 ** b
    x_min = x.min(dim=0, keepdim=True).values   # ← per-coordinate, not global
    x_max = x.max(dim=0, keepdim=True).values
    step  = (x_max - x_min) / n_levels
    step  = step.clamp(min=1e-8)
    indices = torch.clamp(((x - x_min) / step).long(), 0, n_levels - 1)
    return x_min + (indices.float() + 0.5) * step

def resolve_codebook(d, b):
    path = f"config/codebooks/codebook_d{d}_b{b}.npy"
    if not os.path.exists(path):
        print(f"[INFO] Codebook not found → generating for d={d}, b={b}")
        generate_codebook(d, b)
    return path


def run(d, b, batch_size=1024, device="cpu", normalize_input=True):
    print(f"\nRunning TurboQuant → d={d}, bits={b}")

    codebook_path = resolve_codebook(d, b)
    quantizer     = TurboQuantProd(d, codebook_path, device=device)

    x = torch.randn(batch_size, d, device=device)
    y = torch.randn(batch_size, d, device=device)

    if normalize_input:
        x = normalize(x)
        y = normalize(y)   # ← normalize y too

    packed = quantizer.quantize(x)
    x_hat  = quantizer.dequantize(packed)

    x_hat_uniform = uniform_quantize(x, b)   # ← baseline

    print("\n===== TurboQuant =====")
    print(f"  MSE:              {mse(x, x_hat):.6f}")
    print(f"  SNR (dB):         {snr_db(x, x_hat):.2f}")
    print(f"  Inner Prod Error: {inner_product_error(x, x_hat, y):.6f}")
    print(f"  Cosine Sim:       {cosine_similarity(x, x_hat):.6f}")

    print("\n===== Uniform (baseline) =====")
    print(f"  MSE:              {mse(x, x_hat_uniform):.6f}")
    print(f"  SNR (dB):         {snr_db(x, x_hat_uniform):.2f}")
    print(f"  Inner Prod Error: {inner_product_error(x, x_hat_uniform, y):.6f}")
    print(f"  Cosine Sim:       {cosine_similarity(x, x_hat_uniform):.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",          type=int, required=True)
    parser.add_argument("--bits",         type=int, required=True)
    parser.add_argument("--batch",        type=int, default=1024)
    parser.add_argument("--device",       type=str, default="cpu")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    run(d=args.dim, b=args.bits, batch_size=args.batch,
        device=args.device, normalize_input=not args.no_normalize)


if __name__ == "__main__":
    main()