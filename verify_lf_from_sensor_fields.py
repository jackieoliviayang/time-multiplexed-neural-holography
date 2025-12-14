#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify that sensor-plane complex fields saved by TMNH-LF reconstruct the same
LF amplitude as used during optimization.

Usage example:

    python verify_lf_from_sensor_fields.py \
        --pred_dir outputs_lfopt_24/predicted_T8_sensor \
        --channel 1 \
        --num_frames 8 \
        --n_fft 96 \
        --hop_len 12 \
        --win_len 96 \
        --time_joint \
        --out_dir outputs_lfopt_24/lf_verify

Make sure n_fft / hop_len / win_len match the ones used in your run (opt.n_fft, opt.hop_len, opt.win_len).
"""

import os
import argparse
import numpy as np
import torch
import imageio.v2 as imageio

from holo2lf import holo2lf  # same import as in algorithms.py / main.py


def _to8(x):
    x = np.nan_to_num(x)
    x = np.clip(x, 0, None)
    m = float(x.max()) if x.size and np.isfinite(x.max()) and x.max() > 0 else 1.0
    return (x / m * 255.0).round().astype(np.uint8)


def save_stack_video_with_norm(stack, path, norm, fps=12):
    """
    stack: [D,H,W] or [1,D,H,W]
    norm: scalar used for global normalization
    """
    arr = np.asarray(stack)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None]
    frames = [(arr[d] / norm * 255).clip(0, 255).astype(np.uint8)
              for d in range(arr.shape[0])]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=str, required=True,
                   help="Directory with U_pred_sensor_ch*_t**.pt files "
                        "(e.g. outputs_lfopt_24/predicted_T8_sensor)")
    p.add_argument("--channel", type=int, default=0,
                   help="Channel index used in filenames (chX)")
    p.add_argument("--num_frames", type=int, required=True,
                   help="Number of time-multiplexed frames T (e.g. 8)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to use (cuda or cpu)")
    p.add_argument("--n_fft", type=int, required=True)
    p.add_argument("--hop_len", type=int, required=True)
    p.add_argument("--win_len", type=int, required=True)
    p.add_argument("--time_joint", action="store_true",
                   help="Apply time-multiplexed energy pooling over frames")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to write verification outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ch = args.channel

    # -------------------------------------------------------------------------
    # 1. Load sensor-plane complex fields: U_pred_sensor_ch{ch}_t{t:02d}.pt
    # -------------------------------------------------------------------------
    fields = []
    for t in range(args.num_frames):
        fname = os.path.join(
            args.pred_dir,
            f"U_pred_sensor_ch{ch}_t{t:02d}.pt"
        )
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Missing field file: {fname}")
        U = torch.load(fname, map_location=dev)  # expect [H,W] complex
        if U.ndim != 2:
            raise ValueError(f"Expected [H,W] complex, got {U.shape} in {fname}")
        if not torch.is_complex(U):
            raise ValueError(f"Saved tensor is not complex in {fname}")
        fields.append(U)

    U_stack = torch.stack(fields, dim=0)  # [T,H,W]
    T, H, W = U_stack.shape
    print(f"[load] U_stack shape: {U_stack.shape} (T,H,W) on {U_stack.device}")

    # -------------------------------------------------------------------------
    # 2. Recompute LF amplitude using holo2lf (match training call)
    # -------------------------------------------------------------------------
    print("[holo2lf] Running STFT-based light field reconstruction...")

    # holo2lf expects [batch, 1, H, W]
    U_in = U_stack.unsqueeze(1)  # [T,1,H,W]
    print("[debug] U_in shape for holo2lf:", U_in.shape)

    recon_amp_t = holo2lf(
        U_in,
        n_fft=(args.n_fft, args.n_fft),
        hop_length=(args.hop_len, args.hop_len),
        win_length=(args.win_len, args.win_len),
        device=dev,
        impl='torch'
    ).sqrt()  # [T,1,Hy,Hx,U,V]
    print(f"[holo2lf] recon_amp_t shape: {recon_amp_t.shape}")

    if args.time_joint:
        # time-multiplexed pooling: (sum of intensities over T)^(1/2)
        recon_amp = (recon_amp_t ** 2).mean(dim=0, keepdims=True).sqrt()  # [1,1,Hy,Hx,U,V]
        print(f"[time_joint] recon_amp shape: {recon_amp.shape}")
    else:
        recon_amp = recon_amp_t  # [T,1,Hy,Hx,U,V]

    recon_np = recon_amp.detach().float().cpu().numpy()
    print("[stats] recon_amp: min={:.6f}, max={:.6f}, mean={:.6f}".format(
        recon_np.min(), recon_np.max(), recon_np.mean()
    ))

    # -------------------------------------------------------------------------
    # 3. Horizontal parallax scan & center view (match recon saving)
    # -------------------------------------------------------------------------
    arr = np.squeeze(recon_np)
    print(f"[recon_np] squeezed shape: {arr.shape}")

    if arr.ndim != 4:
        raise ValueError(
            f"Expected LF amplitude with shape [Hy,Hx,U,V], got {arr.shape}. "
            "Check n_fft/hop_len/win_len and whether this is really an LF run."
        )

    Hy, Hx, U, V = arr.shape
    print(f"[LF] Hy={Hy}, Hx={Hx}, U={U}, V={V}")

    ky = U // 2  # vertical angular index (middle)
    frames = []
    for kx in range(V):
        frames.append(arr[:, :, ky, kx])  # [Hy,Hx]
    parallax_stack = np.stack(frames, axis=0)  # [V,Hy,Hx]

    global_max = float(parallax_stack.max() + 1e-12)
    print("[LF] parallax_stack shape:", parallax_stack.shape)
    print("[LF] parallax global max:", global_max)

    lf_vid_path = os.path.join(args.out_dir, "recon_lfparallax_from_sensor.mp4")
    save_stack_video_with_norm(parallax_stack, lf_vid_path, global_max, fps=12)
    print(f"[saved] {lf_vid_path}")

    # Center view should use the *same* global_max
    kx_center = V // 2
    center_view = arr[:, :, ky, kx_center]  # [Hy,Hx]
    center_png_path = os.path.join(args.out_dir, "recon_lf_center_from_sensor.png")
    center_uint8 = (center_view / global_max * 255.0).clip(0, 255).astype(np.uint8)
    imageio.imwrite(center_png_path, center_uint8)
    print(f"[saved] {center_png_path}")

    print("\nDone. Compare these to your existing recon_lf outputs "
          "(parallax video + center view). They should match up to a "
          "global intensity scale if everything is wired correctly.")


if __name__ == "__main__":
    main()