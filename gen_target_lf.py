#!/usr/bin/env python3
"""
gen_target_lf.py

Precompute and cache a light-field *amplitude* target from a stack of complex
GWS frames using the same holo2lf STFT used in TMNH.

Backwards compatible with the old behavior (blender/hotdog loader by default),
but can optionally:
  1) load MipNeRF-360 wavefronts via load_scanlines_mipnerf.py
  2) load a precomputed complex target via --target_complex_path

The --target_complex_path path is useful for benchmark runs where we first
downsample the complex target to 256x256 and then build all method-specific
targets from that same shared field.
"""

import os
import argparse
import torch
import numpy as np

from holo2lf import holo2lf
from complex_dataset import _import_loaders_from_mutual_intensity


def _looks_like_mipnerf_scene(s: str | None) -> bool:
    if not s:
        return False
    s = s.lower()
    return any(k in s for k in ["kitchen", "garden", "bicycle", "_exact_awb"])


def parse_args():
    p = argparse.ArgumentParser("Precompute and cache LF amplitude target via STFT from GWS frames.")

    # Metadata / consistency
    p.add_argument(
        "--mi_T",
        type=int,
        default=24,
        help="Number of reference frames to use.",
    )
    p.add_argument(
        "--color",
        type=str,
        default="green",
        choices=["red", "green", "blue"],
        help="Which color channel this LF target corresponds to.",
    )

    # Benchmark path: use already-downsampled complex target
    p.add_argument(
        "--target_complex_path",
        type=str,
        default=None,
        help=(
            "Optional path to a precomputed complex field target. "
            "Expected tensor/dict with U of shape [T,H,W] or [T,1,H,W]. "
            "If provided, dataset loaders are bypassed."
        ),
    )

    # Dataset selection
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=[None, "blender", "mipnerf"],
        help=(
            "Which dataset loader to use. If omitted, auto-detects "
            "mipnerf if --scene looks like a mipnerf scene."
        ),
    )

    # MipNeRF args
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help="MipNeRF scene name, e.g. kitchen, garden, bicycle.",
    )
    p.add_argument(
        "--mipnerf_root",
        type=str,
        default="/miele/brian/GWS_RP_final_mipnerf360/24_frames",
        help="Root directory for mipnerf wavefronts.",
    )
    p.add_argument(
        "--crop_hw",
        type=int,
        nargs=2,
        default=(1024, 1024),
        metavar=("CROP_H", "CROP_W"),
        help="MipNeRF centered crop size, h w.",
    )
    p.add_argument(
        "--crop_shift",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("DY", "DX"),
        help="MipNeRF crop shift, dy dx.",
    )

    # Optical parameters
    p.add_argument("--wavelength", type=float, default=5.32e-7, help="Wavelength in meters.")
    p.add_argument("--asm_dx", type=float, default=8e-6, help="Pixel pitch in x, meters.")
    p.add_argument("--asm_dy", type=float, default=8e-6, help="Pixel pitch in y, meters.")

    # STFT / LF parameters
    p.add_argument(
        "--n_fft",
        type=int,
        default=7,
        help="STFT FFT size per dimension for holo2lf.",
    )
    p.add_argument(
        "--hop_len",
        type=int,
        default=128,
        help="STFT hop length in pixels for holo2lf, used for both dims.",
    )
    p.add_argument(
        "--win_len",
        type=int,
        default=7,
        help="STFT window length in pixels for holo2lf, used for both dims.",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs_lfopt_shared",
        help="Directory to save the cached LF target.",
    )
    p.add_argument(
        "--time_joint",
        action="store_true",
        help="If set, time-average over T like TMNH does. Recommended.",
    )
    p.add_argument(
        "--preview_dir",
        type=str,
        default=None,
        help="Optional directory to dump a few LF views as PNGs for sanity checks.",
    )

    p.add_argument(
        "--save_5d",
        action="store_true",
        help="If set, save cache as [1,1,N_views,H_lf,W_lf] instead of [1,N_views,H_lf,W_lf].",
    )

    return p.parse_args()


def maybe_preview_lf_amp(target_amp, preview_dir):
    """
    Optionally dump a few LF views as PNG for sanity.
    target_amp: [1,N_views,H_lf,W_lf] or [1,1,N_views,H_lf,W_lf]
    """
    import imageio

    os.makedirs(preview_dir, exist_ok=True)

    with torch.no_grad():
        arr = target_amp.detach().cpu().float()

    if arr.ndim == 5:
        arr = arr.squeeze(0).squeeze(0)
    elif arr.ndim == 4:
        arr = arr.squeeze(0)
    else:
        print(f"[preview] Unexpected target_amp shape {tuple(target_amp.shape)}; skipping preview.")
        return

    if arr.ndim != 3:
        print(f"[preview] Unexpected squeezed shape {tuple(arr.shape)}; skipping preview.")
        return

    n_views = arr.shape[0]
    idxs = [0, n_views // 2, n_views - 1] if n_views >= 3 else list(range(n_views))

    for i in idxs:
        img = arr[i]
        img = img / (img.max() + 1e-12)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        fname = os.path.join(preview_dir, f"lf_view_{i:02d}.png")
        imageio.imwrite(fname, img)
        print(f"[preview] wrote {fname}")


def _load_frames_from_target_complex(path: str, mi_T: int) -> torch.Tensor:
    """
    Load a precomputed complex target from a .pt file.

    Accepts:
      - tensor [T,H,W]
      - tensor [T,1,H,W]
      - dict with key "U", "target", "U_tgt", "field", or "complex"

    Returns:
      frames [mi_T,H,W] complex64 on CPU.
    """
    print(f"[gen_target_lf] Loading precomputed complex target: {path}")

    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        found = False
        for k in ["U", "target", "U_tgt", "field", "complex", "target_field", "target_cf"]:
            if k in payload:
                payload = payload[k]
                found = True
                break

        if not found:
            raise KeyError(
                f"Could not find complex field in dict at {path}. "
                f"Available keys: {list(payload.keys())}"
            )

    frames = payload

    if not torch.is_tensor(frames):
        raise TypeError(f"Loaded object from {path} is not a tensor: {type(frames)}")

    if not torch.is_complex(frames):
        if frames.ndim >= 1 and frames.shape[-1] == 2:
            frames = torch.complex(frames[..., 0], frames[..., 1])
        else:
            raise ValueError(
                f"Loaded tensor is not complex and does not have last-dim=2. "
                f"Shape: {tuple(frames.shape)}"
            )

    if frames.ndim == 4 and frames.shape[1] == 1:
        frames = frames[:, 0]
    elif frames.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected frames shape [T,H,W] or [T,1,H,W], got {tuple(frames.shape)}")

    if frames.shape[0] < mi_T:
        raise ValueError(f"Requested mi_T={mi_T}, but target only has T={frames.shape[0]} frames.")

    frames = frames[:mi_T].contiguous().to(torch.complex64).cpu()

    print(f"[gen_target_lf] Loaded precomputed frames: {tuple(frames.shape)} (T,H,W)")
    return frames


def _load_frames_from_dataset(args) -> torch.Tensor:
    device = torch.device("cpu")

    mode = args.dataset
    if mode is None:
        mode = "mipnerf" if _looks_like_mipnerf_scene(args.scene) else "blender"

    load_blender, load_mip = _import_loaders_from_mutual_intensity()

    if mode == "mipnerf":
        frames, _ = load_mip(
            T_desired=args.mi_T,
            output_dir=args.preview_dir,
            color=args.color,
            scene=args.scene if args.scene is not None else "bicycle",
            crop_hw=tuple(args.crop_hw),
            crop_shift=tuple(args.crop_shift),
            save_debug_first=True,
        )

        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        frames = frames.to(torch.complex64).to(device)
        print(f"[gen_target_lf] Loaded MipNeRF frames: {tuple(frames.shape)} (T,H,W)")
        return frames

    # Blender / hotdog
    load_wf = load_blender

    try:
        frames, _ = load_wf(
            T_desired=args.mi_T,
            output_dir=args.preview_dir,
            color=args.color,
        )
    except TypeError:
        channel = {"red": 0, "green": 1, "blue": 2}[args.color]
        frames, _ = load_wf(
            T_desired=args.mi_T,
            output_dir=args.preview_dir,
            channel=channel,
        )

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)

    frames = frames.to(torch.complex64).to(device)
    print(f"[gen_target_lf] Loaded Blender/Hotdog frames: {tuple(frames.shape)} (T,H,W)")
    return frames


def _load_frames(args) -> torch.Tensor:
    if args.target_complex_path is not None:
        return _load_frames_from_target_complex(args.target_complex_path, args.mi_T)
    return _load_frames_from_dataset(args)


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cache_path = os.path.join(
        args.out_dir,
        f"target_ampLF_{args.color}_Tref{args.mi_T}.pt",
    )

    torch.set_grad_enabled(False)

    print("[gen_target_lf] Building LF target on CPU from frames...")
    print(f"  dataset             : {args.dataset or 'auto'} (scene={args.scene})")
    print(f"  target_complex_path : {args.target_complex_path}")
    print(f"  mi_T                : {args.mi_T}")
    print(f"  color               : {args.color}")
    print(f"  wavelength          : {args.wavelength}")
    print(f"  asm_dx, asm_dy      : {args.asm_dx}, {args.asm_dy}")
    print(f"  n_fft               : {args.n_fft}")
    print(f"  hop_len             : {args.hop_len}")
    print(f"  win_len             : {args.win_len}")
    print(f"  cache_path          : {cache_path}")
    print(f"  time_joint          : {args.time_joint}")
    print(f"  save_5d             : {args.save_5d}")

    # 1) Load frames
    frames = _load_frames(args)  # [T,H,W] complex64 on CPU
    T, H, W = frames.shape
    print(f"  input frames shape  : T={T}, H={H}, W={W}")

    # 2) Reshape to match holo2lf expected input: [T,1,H,W]
    input_field = frames.unsqueeze(1).contiguous()

    # 3) Call holo2lf
    lf_full = holo2lf(
        input_field,
        n_fft=(args.n_fft, args.n_fft),
        hop_length=(args.hop_len, args.hop_len),
        win_length=(args.win_len, args.win_len),
        device=torch.device("cpu"),
        impl="torch",
    )

    # lf_full: [T,1,H_lf,W_lf,U,V], intensity
    print(f"  holo2lf output shape: {tuple(lf_full.shape)}, dtype={lf_full.dtype}")

    # 4) Convert intensity -> amplitude
    lf_amp_t = lf_full.clamp_min(0).sqrt()  # [T,1,H_lf,W_lf,U,V]

    # 5) Flatten angular dims into N_views
    lf_amp_t = lf_amp_t.squeeze(1)  # [T,H_lf,W_lf,U,V]

    if lf_amp_t.ndim != 5:
        raise ValueError(f"Expected LF amplitude shape [T,H_lf,W_lf,U,V], got {tuple(lf_amp_t.shape)}")

    T2, H_lf, W_lf, Uang, Vang = lf_amp_t.shape
    if T2 != T:
        raise ValueError(f"LF T mismatch: input T={T}, LF T={T2}")

    n_views = Uang * Vang

    lf_amp_t = lf_amp_t.permute(0, 3, 4, 1, 2).contiguous()
    lf_amp_t = lf_amp_t.view(T, n_views, H_lf, W_lf)

    print(
        f"  lf_amp_t reshaped to: {tuple(lf_amp_t.shape)} "
        f"(T, N_views={n_views}, H_lf, W_lf)"
    )

    # 6) Time-joint target
    if args.time_joint:
        target_amp = (lf_amp_t ** 2).mean(dim=0, keepdims=True).sqrt()
    else:
        target_amp = lf_amp_t

    # 7) Optional 5D cache format
    if args.save_5d:
        if target_amp.ndim != 4:
            raise ValueError(
                f"--save_5d expects target_amp to be 4D [1,N,H,W] or [T,N,H,W], "
                f"got {tuple(target_amp.shape)}"
            )

        if target_amp.shape[0] != 1:
            raise ValueError(
                f"--save_5d is intended for time_joint target with shape [1,N,H,W]. "
                f"Got {tuple(target_amp.shape)}. Pass --time_joint."
            )

        target_amp = target_amp.unsqueeze(1)  # [1,1,N,H,W]

    print(f"  target_amp shape saved: {tuple(target_amp.shape)}, dtype={target_amp.dtype}")

    # 8) Optional preview
    if args.preview_dir is not None:
        maybe_preview_lf_amp(target_amp, args.preview_dir)

    # 9) Save
    torch.save(target_amp.detach().cpu(), cache_path)
    print(f"[gen_target_lf] Saved: {cache_path}")


if __name__ == "__main__":
    main()