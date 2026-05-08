

import os
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from core.prod_quantizer import TurboQuantProd
from scripts.generate_codebook import generate_codebook

# ── config ────────────────────────────────────────────────────────────────────
BITS       = [1, 2, 3, 4, 5, 6]
DIM_MAIN   = 128
DIMS_EXTRA = [64, 256, 512]
BATCH      = 1024
SEED       = 42
OUTDIR     = "results"

# ── quantizers ────────────────────────────────────────────────────────────────

def resolve_codebook(d, b):
    path = f"config/codebooks/codebook_d{d}_b{b}.npy"
    if not os.path.exists(path):
        print(f"  [codebook] generating d={d} b={b} ...")
        generate_codebook(d, b)
    return path

def uniform_quantize(x, b):
    """Per-coordinate uniform scalar quantization — no rotation. Baseline."""
    n  = 2 ** b
    lo = x.min(dim=0, keepdim=True).values
    hi = x.max(dim=0, keepdim=True).values
    step = (hi - lo) / n
    step = step.clamp(min=1e-8)
    idx  = ((x - lo) / step).long().clamp(0, n - 1)
    return lo + (idx.float() + 0.5) * step

def turbo_mse_only(x, d, b, device):
    """Rotation + Lloyd-Max only — no QJL residual."""
    cb  = resolve_codebook(d, b)
    tq  = TurboQuantProd(d, cb, device=device)
    idx = tq.quantize_mse_only(x)
    return tq.dequantize_mse_only(idx)

def turbo_full(x, d, b, device):
    """Full pipeline: rotation + Lloyd-Max + QJL residual correction."""
    cb = resolve_codebook(d, b)
    tq = TurboQuantProd(d, cb, device=device)
    pk = tq.quantize(x)
    return tq.dequantize(pk)

# ── metrics ───────────────────────────────────────────────────────────────────

def mse(x, xh):
    return torch.mean((x - xh) ** 2).item()

def snr_db(x, xh):
    sig   = torch.mean(x ** 2).item()
    noise = torch.mean((x - xh) ** 2).item()
    return 10.0 * np.log10(sig / max(noise, 1e-12))

def ip_error(x, xh, y):
    return torch.mean(((x * y).sum(1) - (xh * y).sum(1)) ** 2).item()

def cosine_sim(x, xh):
    num = (x * xh).sum(1)
    den = (x.norm(dim=1) * xh.norm(dim=1)).clamp(min=1e-8)
    return torch.mean(num / den).item()

METRICS = ["mse", "snr", "ip_err", "cos"]
TAGS    = ["uniform", "turbo_mse", "turbo_full"]

# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(d, bits_list, batch, device, seed):
    torch.manual_seed(seed)
    x = torch.randn(batch, d, device=device)
    x = x / x.norm(dim=1, keepdim=True)
    y = torch.randn(batch, d, device=device)
    y = y / y.norm(dim=1, keepdim=True)

    res = {tag: {m: [] for m in METRICS} for tag in TAGS}
    res["oracle"] = {"mse": 0.0, "snr": float("inf"),
                     "ip_err": ip_error(x, x, y), "cos": 1.0}

    for b in bits_list:
        print(f"  d={d:4d}  bits={b}", end="  ", flush=True)

        xh_u   = uniform_quantize(x, b)
        xh_mse = turbo_mse_only(x, d, b, device)
        xh_qjl = turbo_full(x, d, b, device)

        for tag, xh in [("uniform",    xh_u),
                        ("turbo_mse",  xh_mse),
                        ("turbo_full", xh_qjl)]:
            res[tag]["mse"].append(mse(x, xh))
            res[tag]["snr"].append(snr_db(x, xh))
            res[tag]["ip_err"].append(ip_error(x, xh, y))
            res[tag]["cos"].append(cosine_sim(x, xh))

        print(f"MSE  uniform={res['uniform']['mse'][-1]:.2e}"
              f"  mse_only={res['turbo_mse']['mse'][-1]:.2e}"
              f"  +QJL={res['turbo_full']['mse'][-1]:.2e}")
    return res

# ── figure ────────────────────────────────────────────────────────────────────

STYLE = {
    "uniform":    dict(color="#d62728", ls="-",  lw=2.0, marker="s", ms=5,
                       label="Uniform (no rotation)"),
    "turbo_mse":  dict(color="#2ca02c", ls="-",  lw=2.0, marker="^", ms=5,
                       label="TurboQuant MSE-only (no QJL)"),
    "turbo_full": dict(color="#1f77b4", ls="-",  lw=2.0, marker="o", ms=5,
                       label="TurboQuant + QJL residual"),
    "oracle":     dict(color="#444",    ls="--", lw=1.4),
}

def make_figure(main_res, dims_ip, bits_list, d_main):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle(
        f"TurboQuant: Uniform vs MSE-only vs Full (+QJL)\n"
        f"Synthetic Gaussian · d={d_main} · batch={BATCH} · unit-norm inputs · seed={SEED}",
        fontsize=11, y=1.01
    )

    panels = [
        (axes[0, 0], "mse",    "Reconstruction MSE",  True,  False),
        (axes[0, 1], "snr",    "SNR (dB)",             False, False),
        (axes[1, 0], "ip_err", "Inner-product Error",  True,  True),
        (axes[1, 1], "cos",    "Cosine Similarity",    False, False),
    ]

    for ax, metric, ylabel, log_y, multi_dim in panels:
        ax.set_xlabel("Bits per coordinate", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")

        if multi_dim:
            extra_colors = plt.cm.Greens(np.linspace(0.3, 0.7, len(dims_ip)))
            for ci, (d_ex, res_ex) in enumerate(dims_ip):
                if d_ex == d_main:
                    continue
                ax.plot(bits_list, res_ex["turbo_mse"][metric],
                        color=extra_colors[ci], lw=1.0, ls=":",
                        alpha=0.55, label=f"MSE-only d={d_ex}")

        oracle_val = main_res["oracle"][metric]
        if np.isfinite(oracle_val) and metric in ("cos", "ip_err"):
            ax.axhline(oracle_val, label="No quant (oracle)", **STYLE["oracle"])

        for tag in TAGS:
            ax.plot(bits_list, main_res[tag][metric], **STYLE[tag])

        ax.legend(fontsize=7, loc="best")

    # theoretical −6 dB/bit slope on MSE panel
    ax_mse = axes[0, 0]
    ref    = main_res["turbo_mse"]["mse"][bits_list.index(2)]
    theory = [ref * (4.0 ** (2 - b)) for b in bits_list]
    ax_mse.plot(bits_list, theory, color="#888", ls="-.", lw=1.2,
                label="Theoretical −6 dB/bit")
    ax_mse.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    return fig

# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(res, bits_list, d, path):
    rows = []
    for i, b in enumerate(bits_list):
        row = {"dim": d, "bits": b}
        for tag in TAGS:
            for m in METRICS:
                row[f"{tag}_{m}"] = f"{res[tag][m][i]:.8f}"
        rows.append(row)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {path}")

# ── summary table ─────────────────────────────────────────────────────────────

def print_table(res, bits_list, metric="mse"):
    label = {"mse":"MSE","snr":"SNR(dB)","ip_err":"IP-err","cos":"cosine"}[metric]
    lower_better = metric in ("mse", "ip_err")
    print(f"\n=== {label} ({'lower' if lower_better else 'higher'} is better) ===")
    print(f"{'bits':>4}  {'uniform':>12}  {'turbo_mse':>12}  {'turbo_full':>12}  "
          f"{'rot helps':>9}  {'qjl helps':>9}")
    print("-" * 72)
    for i, b in enumerate(bits_list):
        u  = res["uniform"][metric][i]
        tm = res["turbo_mse"][metric][i]
        tf = res["turbo_full"][metric][i]
        rot_helps = (tm < u)  if lower_better else (tm > u)
        qjl_helps = (tf < tm) if lower_better else (tf > tm)
        print(f"{b:>4}  {u:>12.6f}  {tm:>12.6f}  {tf:>12.6f}  "
              f"{'YES':>9}" + ("" if rot_helps  else " NO") +
              f"  {'YES':>9}" + ("" if qjl_helps else " NO"))

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch",  type=int, default=BATCH)
    parser.add_argument("--seed",   type=int, default=SEED)
    args = parser.parse_args()

    print(f"Device: {args.device}  |  batch={args.batch}  |  seed={args.seed}\n")
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"=== Primary sweep  d={DIM_MAIN} ===")
    main_res = sweep(DIM_MAIN, BITS, args.batch, args.device, args.seed)

    dims_ip = [(DIM_MAIN, main_res)]
    for d_ex in DIMS_EXTRA:
        print(f"\n=== Extra dim sweep  d={d_ex} ===")
        dims_ip.append((d_ex, sweep(d_ex, BITS, args.batch, args.device, args.seed)))

    fig = make_figure(main_res, dims_ip, BITS, DIM_MAIN)
    fig_path = os.path.join(OUTDIR, "turboquant_results.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n[saved] {fig_path}")

    save_csv(main_res, BITS, DIM_MAIN, os.path.join(OUTDIR, "metrics.csv"))

    for metric in METRICS:
        print_table(main_res, BITS, metric)


if __name__ == "__main__":
    main()