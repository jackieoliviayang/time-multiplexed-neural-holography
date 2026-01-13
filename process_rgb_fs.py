#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import imageio.v2 as imageio

def _squeeze_stack(x):
    """
    Accept [D,H,W], [1,D,H,W], [B,1,D,H,W], etc -> return [D,H,W]
    """
    if isinstance(x, dict):
        raise ValueError("Expected tensor, got dict")
    x = x.detach().cpu().float()
    x = x.squeeze()
    # after squeeze, could be [D,H,W] (good) or [H,W] (single slice)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError(f"Unexpected stack shape after squeeze: {tuple(x.shape)}")
    return x

def _to_uint8_rgb(img_rgb, norm):
    # img_rgb: [H,W,3] float
    x = (img_rgb / norm * 255.0).clip(0, 255)
    return x.round().astype(np.uint8)

def write_rgb_video(stack_rgb, out_mp4, fps=12, norm=None):
    """
    stack_rgb: [D,H,W,3] float
    """
    D = stack_rgb.shape[0]
    if norm is None:
        # robust-ish normalization across all depths/channels
        flat = stack_rgb.reshape(-1, 3)
        norm = np.quantile(flat, 0.999) + 1e-12

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    for d in range(D):
        frame = _to_uint8_rgb(stack_rgb[d], norm)
        writer.append_data(frame)
    writer.close()
    return norm

def write_rgb_pngs(stack_rgb, out_dir, prefix, norm):
    os.makedirs(out_dir, exist_ok=True)
    D = stack_rgb.shape[0]
    for d in range(D):
        frame = _to_uint8_rgb(stack_rgb[d], norm)
        imageio.imwrite(os.path.join(out_dir, f"{prefix}_z{d:03d}.png"), frame)

def find_amp_pt(run_dir, channel):
    # looks for focalstack_amp_ch{channel}.pt anywhere under run_dir
    pat = os.path.join(run_dir, "**", f"focalstack_amp_ch{channel}.pt")
    hits = glob.glob(pat, recursive=True)
    if not hits:
        raise FileNotFoundError(f"No match for {pat}")
    # choose newest
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def main(red_dir, green_dir, blue_dir, out_dir):
    # load per-channel stacks
    r_path = find_amp_pt(red_dir,   0)
    g_path = find_amp_pt(green_dir, 1)
    b_path = find_amp_pt(blue_dir,  2)

    r = torch.load(r_path, map_location="cpu")
    g = torch.load(g_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")

    r_recon = _squeeze_stack(r["recon_amp"])
    g_recon = _squeeze_stack(g["recon_amp"])
    b_recon = _squeeze_stack(b["recon_amp"])

    r_tgt = _squeeze_stack(r["target_amp"])
    g_tgt = _squeeze_stack(g["target_amp"])
    b_tgt = _squeeze_stack(b["target_amp"])

    # sanity: match shapes
    if not (r_recon.shape == g_recon.shape == b_recon.shape):
        raise ValueError(f"Recon shapes mismatch: R{tuple(r_recon.shape)} G{tuple(g_recon.shape)} B{tuple(b_recon.shape)}")
    if not (r_tgt.shape == g_tgt.shape == b_tgt.shape):
        raise ValueError(f"Target shapes mismatch: R{tuple(r_tgt.shape)} G{tuple(g_tgt.shape)} B{tuple(b_tgt.shape)}")

    # stack into RGB: [D,H,W,3]
    recon_rgb = torch.stack([r_recon, g_recon, b_recon], dim=-1).numpy()
    tgt_rgb   = torch.stack([r_tgt,   g_tgt,   b_tgt],   dim=-1).numpy()

    os.makedirs(out_dir, exist_ok=True)

    # use one shared norm for both target & recon so colors match visually
    combo = np.concatenate([recon_rgb.reshape(-1,3), tgt_rgb.reshape(-1,3)], axis=0)
    norm = np.quantile(combo, 0.999) + 1e-12

    recon_mp4 = os.path.join(out_dir, "recon_focalstack_rgb.mp4")
    tgt_mp4   = os.path.join(out_dir, "target_focalstack_rgb.mp4")

    write_rgb_video(tgt_rgb,   tgt_mp4,   fps=12, norm=norm)
    write_rgb_video(recon_rgb, recon_mp4, fps=12, norm=norm)

    write_rgb_pngs(tgt_rgb,   out_dir, "target_rgb", norm)
    write_rgb_pngs(recon_rgb, out_dir, "recon_rgb",  norm)

    print("[done]")
    print("  target:", tgt_mp4)
    print("  recon :", recon_mp4)
    print("  pngs  :", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--red_dir", required=True)
    ap.add_argument("--green_dir", required=True)
    ap.add_argument("--blue_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.red_dir, args.green_dir, args.blue_dir, args.out_dir)
