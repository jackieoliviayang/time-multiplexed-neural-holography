#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import imageio.v2 as imageio

def _to_uint8_rgb(img_rgb, norm):
    x = (img_rgb / norm * 255.0).clip(0, 255)
    return x.round().astype(np.uint8)

def _read_video_gray(path):
    """
    Read an mp4 into float32 grayscale frames in [0,1]-ish scale (we keep raw 0-255 as float).
    Returns: frames [D,H,W] float32
    """
    rdr = imageio.get_reader(path)
    frames = []
    for f in rdr:
        # f could be [H,W] or [H,W,3] depending on decoder
        if f.ndim == 3:
            # convert to gray by taking first channel (they should all be identical)
            f = f[..., 0]
        frames.append(f.astype(np.float32))
    rdr.close()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)  # [D,H,W]

def write_rgb_video(stack_rgb, out_mp4, fps=12, norm=None):
    """
    stack_rgb: [D,H,W,3] float32 (in same units as input; we normalize by norm)
    """
    D = stack_rgb.shape[0]
    if norm is None:
        flat = stack_rgb.reshape(-1, 3)
        norm = float(np.quantile(flat, 0.999) + 1e-12)

    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8)
    for d in range(D):
        writer.append_data(_to_uint8_rgb(stack_rgb[d], norm))
    writer.close()
    return norm

def write_rgb_pngs(stack_rgb, out_dir, prefix, norm):
    os.makedirs(out_dir, exist_ok=True)
    D = stack_rgb.shape[0]
    for d in range(D):
        frame = _to_uint8_rgb(stack_rgb[d], norm)
        imageio.imwrite(os.path.join(out_dir, f"{prefix}_v{d:03d}.png"), frame)

def find_latest_file(run_dir, filename):
    pat = os.path.join(run_dir, "**", filename)
    hits = glob.glob(pat, recursive=True)
    if not hits:
        raise FileNotFoundError(f"No match for {pat}")
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def _infer_fps(video_path, default=12):
    try:
        rdr = imageio.get_reader(video_path)
        meta = rdr.get_meta_data()
        rdr.close()
        fps = meta.get("fps", None)
        if fps is None:
            return default
        return float(fps)
    except Exception:
        return default

def main(red_dir, green_dir, blue_dir, out_dir, fps=None, save_pngs=False):
    # locate per-color LF videos
    r_tgt = find_latest_file(red_dir,   "target_lfparallax.mp4")
    g_tgt = find_latest_file(green_dir, "target_lfparallax.mp4")
    b_tgt = find_latest_file(blue_dir,  "target_lfparallax.mp4")

    r_rec = find_latest_file(red_dir,   "recon_lfparallax.mp4")
    g_rec = find_latest_file(green_dir, "recon_lfparallax.mp4")
    b_rec = find_latest_file(blue_dir,  "recon_lfparallax.mp4")

    # read grayscale frames (as float32 in 0..255)
    r_tgt_f = _read_video_gray(r_tgt)
    g_tgt_f = _read_video_gray(g_tgt)
    b_tgt_f = _read_video_gray(b_tgt)

    r_rec_f = _read_video_gray(r_rec)
    g_rec_f = _read_video_gray(g_rec)
    b_rec_f = _read_video_gray(b_rec)

    if not (r_tgt_f.shape == g_tgt_f.shape == b_tgt_f.shape):
        raise ValueError(f"Target video shapes mismatch: R{r_tgt_f.shape} G{g_tgt_f.shape} B{b_tgt_f.shape}")
    if not (r_rec_f.shape == g_rec_f.shape == b_rec_f.shape):
        raise ValueError(f"Recon video shapes mismatch: R{r_rec_f.shape} G{g_rec_f.shape} B{b_rec_f.shape}")

    # stack into RGB: [D,H,W,3]
    tgt_rgb = np.stack([r_tgt_f, g_tgt_f, b_tgt_f], axis=-1).astype(np.float32)
    rec_rgb = np.stack([r_rec_f, g_rec_f, b_rec_f], axis=-1).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)

    # choose fps (prefer reading from one of the videos)
    if fps is None:
        fps = _infer_fps(r_tgt, default=12)

    # shared visualization norm across target+recon (quantile like your FS script)
    combo = np.concatenate([tgt_rgb.reshape(-1, 3), rec_rgb.reshape(-1, 3)], axis=0)
    norm = float(np.quantile(combo, 0.999) + 1e-12)

    # save .pt payload (full stacks)
    rgb_pt_path = os.path.join(out_dir, "lfparallax_rgb.pt")
    torch.save(
        {
            "target_lfparallax_rgb": torch.from_numpy(tgt_rgb).contiguous(),  # [D,H,W,3]
            "recon_lfparallax_rgb":  torch.from_numpy(rec_rgb).contiguous(),  # [D,H,W,3]
            "norm_vis_0p999": norm,
            "fps": fps,
            "src": {
                "target": {"red": r_tgt, "green": g_tgt, "blue": b_tgt},
                "recon":  {"red": r_rec, "green": g_rec, "blue": b_rec},
            },
        },
        rgb_pt_path
    )
    print(f"[saved] {rgb_pt_path}")

    # write rgb videos
    tgt_mp4 = os.path.join(out_dir, "target_lfparallax_rgb.mp4")
    rec_mp4 = os.path.join(out_dir, "recon_lfparallax_rgb.mp4")

    write_rgb_video(tgt_rgb, tgt_mp4, fps=fps, norm=norm)
    write_rgb_video(rec_rgb, rec_mp4, fps=fps, norm=norm)

    # optional per-view PNGs
    if save_pngs:
        write_rgb_pngs(tgt_rgb, out_dir, "target_lf_rgb", norm)
        write_rgb_pngs(rec_rgb, out_dir, "recon_lf_rgb", norm)

    print("[done]")
    print("  target:", tgt_mp4)
    print("  recon :", rec_mp4)
    print("  pt    :", rgb_pt_path)
    if save_pngs:
        print("  pngs  :", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--red_dir", required=True)
    ap.add_argument("--green_dir", required=True)
    ap.add_argument("--blue_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fps", type=float, default=None, help="Override FPS (default: infer from video, else 12).")
    ap.add_argument("--save_pngs", action="store_true", help="Also dump per-view RGB PNGs.")
    args = ap.parse_args()
    main(args.red_dir, args.green_dir, args.blue_dir, args.out_dir, fps=args.fps, save_pngs=args.save_pngs)