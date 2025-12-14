#!/usr/bin/env python3
"""
gen_target_lf_clean.py

Build a light-field *amplitude* target directly from clean GT LF views
stored as PNGs in /miele/jackie/lightfield_all_scenes/<scene>/*.png.

Output:
  - A .pt file with shape [1, N_views, H, W]
    (same convention as target_ampLF_Tref{mi_T}.pt from gen_target_lf.py)

This is compatible with ComplexFramesToFocalStackTarget + main.py in TMNH,
which expects [B, 1, N_views, H, W] after the dataset adds a channel dim.
"""

import os
import glob
import argparse
import numpy as np
import torch
import imageio.v2 as imageio


def load_gt_views(gt_root: str, scene: str):
    """
    Load GT LF RGB views from:
      gt_root/scene/*.png

    Returns:
      views: [N, H, W, 3] in [0, 1]
    """
    pattern = os.path.join(gt_root, scene, "*.png")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No PNGs found at {pattern}")

    paths.sort()  # assumes naming like 0_0.png, 0_1.png, etc.
    print(f"[load_gt_views] Found {len(paths)} views for scene '{scene}'")

    imgs = [imageio.imread(p) for p in paths]
    views = np.stack(imgs, axis=0).astype(np.float32) / 255.0  # [N,H,W,3] or [N,H,W]
    return views


def rgb_to_amp(views: np.ndarray) -> np.ndarray:
    """
    Convert RGB views in [0,1] to a single-channel amplitude.

    views: [N,H,W,3] or [N,H,W]
    returns: [N,H,W] in [0,1]
    """
    if views.ndim == 4 and views.shape[-1] == 3:
        # Standard luminance-like grayscale
        r = views[..., 0]
        g = views[..., 1]
        b = views[..., 2]
        amp = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif views.ndim == 3:
        # Already single-channel
        amp = views
    else:
        raise ValueError(f"Unexpected GT views shape {views.shape}; "
                         f"expected [N,H,W,3] or [N,H,W].")

    # Ensure non-negative + float32
    amp = np.clip(amp, 0.0, 1.0).astype(np.float32)
    return amp


def main():
    p = argparse.ArgumentParser("Build LF amplitude target from clean 9x9 GT views.")
    p.add_argument("--scene", type=str, default="hotdog",
                   help="Scene name (subfolder of gt_root).")
    p.add_argument("--gt_root", type=str,
                   default="/miele/jackie/lightfield_all_scenes",
                   help="Root folder containing per-scene GT LF PNGs.")
    p.add_argument("--mi_T", type=int, default=24,
                   help="Only used in filename for consistency (Tref).")
    p.add_argument("--out_dir", type=str, default="outputs_lfopt_shared",
                   help="Where to save the .pt file.")
    p.add_argument("--outfile", type=str, default=None,
                   help="Optional explicit filename for the .pt file. "
                        "If not set, a default name is used.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load GT views and convert to amplitude
    # ------------------------------------------------------------------
    views = load_gt_views(args.gt_root, args.scene)   # [N,H,W,3] in [0,1]
    amp = rgb_to_amp(views)                           # [N,H,W]

    N, H, W = amp.shape
    print(f"[clean LF] amp shape (N,H,W) = {amp.shape}")

    # Sanity: check N is a perfect square (e.g., 9x9 -> 81 views)
    root = int(round(N ** 0.5))
    if root * root != N:
        print(f"[warning] N_views={N} is not a perfect square; "
              f"sqrt(N)≈{N**0.5:.3f}. main.py will expect U*V=N with U=V=int(sqrt(N)).")

    # ------------------------------------------------------------------
    # 2) Pack to [1, N_views, H, W] to match gen_target_lf.py convention
    # ------------------------------------------------------------------
    target_amp = torch.from_numpy(amp)          # [N,H,W]
    target_amp = target_amp.unsqueeze(0)        # [1,N,H,W]

    print(f"[clean LF] target_amp tensor shape (saved) = {tuple(target_amp.shape)}, "
          f"dtype={target_amp.dtype}")

    # ------------------------------------------------------------------
    # 3) Choose filename and save
    # ------------------------------------------------------------------
    if args.outfile is not None:
        cache_path = os.path.join(args.out_dir, args.outfile)
    else:
        # Default name – easy to distinguish from the GWS-based one
        cache_path = os.path.join(
            args.out_dir,
            f"target_ampLF_clean9x9_{args.scene}_Tref{args.mi_T}.pt"
        )

    torch.save(target_amp.cpu(), cache_path)
    print(f"[clean LF] Saved: {cache_path}")


if __name__ == "__main__":
    main()
