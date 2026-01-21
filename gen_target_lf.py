# #!/usr/bin/env python3
# """
# gen_target_lf.py

# Precompute and cache a light-field *amplitude* target from a stack of complex
# GWS frames using the same holo2lf STFT used in TMNH.

# This is analogous to gen_target_fstack.py, but:
#   - Uses holo2lf instead of ASM propagation over z.
#   - Operates directly on the 1024x1024 complex fields loaded from mutual-intensity.

# Output:
#   - target_ampLF_Tref{mi_T}.pt with shape:
#       [1, N_views, H_lf, W_lf]
#     where N_views should be n_fft_y * n_fft_x (e.g. 7*7) after flattening.
# """

# import os
# import argparse
# import torch
# import numpy as np

# from holo2lf import holo2lf
# from complex_dataset import _import_loader_from_mutual_intensity


# def parse_args():
#     p = argparse.ArgumentParser("Precompute and cache LF amplitude target via STFT from GWS frames.")

#     # Metadata / consistency
#     p.add_argument("--mi_T", type=int, default=24,
#                    help="Number of reference frames (for naming / sanity checks).")
#     p.add_argument("--color", type=str, default="green",
#                choices=["red", "green", "blue"],
#                help="Which color channel this LF target corresponds to (affects cache name).")


#     # Optical parameters (kept for consistency; not heavily used by holo2lf)
#     p.add_argument("--wavelength", type=float, default=5.32e-7,
#                    help="Wavelength in meters.")
#     p.add_argument("--asm_dx", type=float, default=8e-6,
#                    help="Pixel pitch in x (meters).")
#     p.add_argument("--asm_dy", type=float, default=8e-6,
#                    help="Pixel pitch in y (meters).")

#     # STFT / LF parameters – MUST match what you'll use in TMNH
#     p.add_argument("--n_fft", type=int, default=7,
#                    help="STFT FFT size (per dimension) for holo2lf. "
#                         "Angular grid will be n_fft x n_fft.")
#     p.add_argument("--hop_len", type=int, default=128,
#                    help="STFT hop length (pixels) for holo2lf (used for both dims).")
#     p.add_argument("--win_len", type=int, default=7,
#                    help="STFT window length (pixels) for holo2lf (used for both dims).")

#     p.add_argument("--out_dir", type=str, default="outputs_lfopt_shared",
#                    help="Directory to save the cached LF target.")
#     p.add_argument("--time_joint", action="store_true",
#                    help="If set, time-average over T like TMNH does (recommended).")

#     p.add_argument("--preview_dir", type=str, default=None,
#                    help="Optional: directory to dump a few LF views as PNGs for sanity checks.")

#     return p.parse_args()


# def maybe_preview_lf_amp(target_amp, preview_dir):
#     """
#     Optionally dump a few LF views as PNG for sanity.
#     target_amp: [1, N_views, H_lf, W_lf]
#     """
#     import imageio
#     os.makedirs(preview_dir, exist_ok=True)

#     with torch.no_grad():
#         arr = target_amp.detach().cpu().float()  # [1, N, H, W]
#     arr = arr.squeeze(0)  # -> [N, H, W]

#     if arr.ndim != 3:
#         print(f"[preview] Unexpected target_amp shape {tuple(target_amp.shape)}; "
#               f"skipping preview.")
#         return

#     N = arr.shape[0]
#     idxs = [0, N // 2, N - 1] if N >= 3 else list(range(N))

#     for i in idxs:
#         img = arr[i]
#         img = img / (img.max() + 1e-12)
#         img = (img * 255.0).clamp(0, 255).byte().numpy()
#         fname = os.path.join(preview_dir, f"lf_view_{i:02d}.png")
#         imageio.imwrite(fname, img)
#         print(f"[preview] wrote {fname}")


# def main():
#     args = parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)
#     cache_path = os.path.join(
#         args.out_dir,
#         f"target_ampLF_{args.color}_Tref{args.mi_T}.pt"
#     )

#     device = torch.device("cpu")
#     torch.set_grad_enabled(False)

#     print("[gen_target_lf] Building LF target on CPU from GWS frames...")
#     print(f"  mi_T       : {args.mi_T}")
#     print(f"  wavelength : {args.wavelength}")
#     print(f"  dx, dy     : {args.asm_dx}, {args.asm_dy}")
#     print(f"  n_fft      : {args.n_fft}")
#     print(f"  hop_len    : {args.hop_len}")
#     print(f"  win_len    : {args.win_len}")
#     print(f"  cache_path : {cache_path}")
#     print(f"  time_joint : {args.time_joint}")

#     # 1) Load GWS complex frames using the same loader as ComplexFramesToFocalStackTarget
#     load_wf = _import_loader_from_mutual_intensity()
#     try:
#         frames, _ = load_wf(T_desired=args.mi_T, output_dir=args.preview_dir, color=args.color)
#     except TypeError:
#         CHANNEL = {"red": 0, "green": 1, "blue": 2}[args.color]
#         frames, _ = load_wf(T_desired=args.mi_T, output_dir=args.preview_dir, channel=CHANNEL)

#     # frames: [T, H, W] complex (np or torch)
#     if isinstance(frames, np.ndarray):
#         frames = torch.from_numpy(frames)
#     frames = frames.to(torch.complex64).to(device)  # [T, H, W]
#     print(f"Loaded {frames.shape[0]} scanlines of shape {tuple(frames.shape[1:])}")
#     print(f"  Loaded GWS complex frames: shape={tuple(frames.shape)}, dtype={frames.dtype}")

#     T, H, W = frames.shape

#     # 2) Reshape to match holo2lf's expected input: (N, 1, H, W)
#     input_field = frames.unsqueeze(1)  # [T, 1, H, W]

#     # 3) Call holo2lf
#     #    NOTE: holo2lf expects tuples for n_fft / hop_length / win_length.
#     lf_full = holo2lf(
#         input_field,
#         n_fft=(args.n_fft, args.n_fft),
#         hop_length=(args.hop_len, args.hop_len),
#         win_length=(args.win_len, args.win_len),
#         device=device,
#         impl="torch",
#     )
#     # lf_full: [T, 1, H_lf, W_lf, U, V]
#     print(f"  holo2lf output shape: {tuple(lf_full.shape)}, dtype={lf_full.dtype}")

#     # 4) Convert intensity -> amplitude
#     lf_amp_t = lf_full.sqrt()  # [T, 1, H_lf, W_lf, U, V]

#     # 5) Flatten angular dims into N_views
#     lf_amp_t = lf_amp_t.squeeze(1)  # [T, H_lf, W_lf, U, V]
#     T2, H_lf, W_lf, U, V = lf_amp_t.shape
#     assert T2 == T
#     N_views = U * V

#     # move (U,V) in front and flatten to N_views
#     lf_amp_t = lf_amp_t.permute(0, 3, 4, 1, 2).contiguous()  # [T, U, V, H_lf, W_lf]
#     lf_amp_t = lf_amp_t.view(T, N_views, H_lf, W_lf)         # [T, N_views, H_lf, W_lf]
#     print(f"  lf_amp_t reshaped to: {tuple(lf_amp_t.shape)} "
#           f"(T, N_views={N_views}, H_lf, W_lf)")

#     # 6) Time-joint (recommended, like TMNH)
#     if args.time_joint:
#         # Mimic TMNH's time-averaging:
#         #   recon_amp = (recon_amp_t**2).mean(dim=0, keepdims=True).sqrt()
#         target_amp = (lf_amp_t ** 2).mean(dim=0, keepdims=True).sqrt()
#         # shape: [1, N_views, H_lf, W_lf]
#     else:
#         target_amp = lf_amp_t  # [T, N_views, H_lf, W_lf]

#     print(f"  target_amp shape (saved) = {tuple(target_amp.shape)}, "
#           f"dtype={target_amp.dtype}")

#     # 7) Optional preview
#     if args.preview_dir is not None:
#         maybe_preview_lf_amp(target_amp, args.preview_dir)

#     # 8) Save
#     torch.save(target_amp.detach().cpu(), cache_path)
#     print(f"[gen_target_lf] Saved: {cache_path}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
gen_target_lf.py

Precompute and cache a light-field *amplitude* target from a stack of complex
GWS frames using the same holo2lf STFT used in TMNH.

Backwards compatible with the old behavior (blender/hotdog loader by default),
but can optionally load MipNeRF-360 wavefronts via load_scanlines_mipnerf.py.
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
    # common mipnerf scenes you use
    s = s.lower()
    return any(k in s for k in ["kitchen", "garden", "bicycle", "_exact_awb"])


def parse_args():
    p = argparse.ArgumentParser("Precompute and cache LF amplitude target via STFT from GWS frames.")

    # Metadata / consistency
    p.add_argument("--mi_T", type=int, default=24,
                   help="Number of reference frames (for naming / sanity checks).")
    p.add_argument("--color", type=str, default="green",
                   choices=["red", "green", "blue"],
                   help="Which color channel this LF target corresponds to (affects cache name).")

    # NEW: dataset selection (optional; auto if omitted)
    p.add_argument("--dataset", type=str, default=None,
                   choices=[None, "blender", "mipnerf"],
                   help="Which dataset loader to use. If omitted, auto-detect "
                        "(uses mipnerf if --scene looks like a mipnerf scene).")

    # NEW: mipnerf args (only used if dataset=mipnerf or auto-detected)
    p.add_argument("--scene", type=str, default=None,
                   help="MipNeRF scene name (e.g., kitchen, garden, bicycle).")
    p.add_argument("--mipnerf_root", type=str,
                   default="/miele/brian/GWS_RP_final_mipnerf360/24_frames",
                   help="Root directory for mipnerf wavefronts.")
    p.add_argument("--crop_hw", type=int, nargs=2, default=(1024, 1024),
                   metavar=("CROP_H", "CROP_W"),
                   help="MipNeRF centered crop size (h w).")
    p.add_argument("--crop_shift", type=int, nargs=2, default=(0, 0),
                   metavar=("DY", "DX"),
                   help="MipNeRF crop shift (dy dx) to nudge crop.")

    # Optical parameters (kept for consistency; not heavily used by holo2lf)
    p.add_argument("--wavelength", type=float, default=5.32e-7,
                   help="Wavelength in meters.")
    p.add_argument("--asm_dx", type=float, default=8e-6,
                   help="Pixel pitch in x (meters).")
    p.add_argument("--asm_dy", type=float, default=8e-6,
                   help="Pixel pitch in y (meters).")

    # STFT / LF parameters – MUST match what you'll use in TMNH
    p.add_argument("--n_fft", type=int, default=7,
                   help="STFT FFT size (per dimension) for holo2lf. Angular grid will be n_fft x n_fft.")
    p.add_argument("--hop_len", type=int, default=128,
                   help="STFT hop length (pixels) for holo2lf (used for both dims).")
    p.add_argument("--win_len", type=int, default=7,
                   help="STFT window length (pixels) for holo2lf (used for both dims).")

    p.add_argument("--out_dir", type=str, default="outputs_lfopt_shared",
                   help="Directory to save the cached LF target.")
    p.add_argument("--time_joint", action="store_true",
                   help="If set, time-average over T like TMNH does (recommended).")

    p.add_argument("--preview_dir", type=str, default=None,
                   help="Optional: directory to dump a few LF views as PNGs for sanity checks "
                        "(also used as output_dir for mipnerf loader debug image).")

    # OPTIONAL: if you want to save a 5D cache compatible with your main.py reshaper
    p.add_argument("--save_5d", action="store_true",
                   help="If set, save cache as [1,1,N_views,H_lf,W_lf] instead of [1,N_views,H_lf,W_lf].")

    return p.parse_args()


def maybe_preview_lf_amp(target_amp, preview_dir):
    """
    Optionally dump a few LF views as PNG for sanity.
    target_amp: [1, N_views, H_lf, W_lf] OR [1,1,N_views,H_lf,W_lf]
    """
    import imageio
    os.makedirs(preview_dir, exist_ok=True)

    with torch.no_grad():
        arr = target_amp.detach().cpu().float()

    # normalize to [N,H,W]
    if arr.ndim == 5:  # [1,1,N,H,W]
        arr = arr.squeeze(0).squeeze(0)
    elif arr.ndim == 4:  # [1,N,H,W]
        arr = arr.squeeze(0)
    else:
        print(f"[preview] Unexpected target_amp shape {tuple(target_amp.shape)}; skipping preview.")
        return

    if arr.ndim != 3:
        print(f"[preview] Unexpected squeezed shape {tuple(arr.shape)}; skipping preview.")
        return

    N = arr.shape[0]
    idxs = [0, N // 2, N - 1] if N >= 3 else list(range(N))

    for i in idxs:
        img = arr[i]
        img = img / (img.max() + 1e-12)
        img = (img * 255.0).clamp(0, 255).byte().numpy()
        fname = os.path.join(preview_dir, f"lf_view_{i:02d}.png")
        imageio.imwrite(fname, img)
        print(f"[preview] wrote {fname}")


def _load_frames(args) -> torch.Tensor:
    device = torch.device("cpu")

    mode = args.dataset
    if mode is None:
        mode = "mipnerf" if _looks_like_mipnerf_scene(args.scene) else "blender"

    # Import loaders *after* mutual-intensity is added to sys.path
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

    # blender/hotdog
    load_wf = load_blender
    try:
        frames, _ = load_wf(T_desired=args.mi_T, output_dir=args.preview_dir, color=args.color)
    except TypeError:
        CHANNEL = {"red": 0, "green": 1, "blue": 2}[args.color]
        frames, _ = load_wf(T_desired=args.mi_T, output_dir=args.preview_dir, channel=CHANNEL)

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    frames = frames.to(torch.complex64).to(device)
    print(f"[gen_target_lf] Loaded Blender/Hotdog frames: {tuple(frames.shape)} (T,H,W)")
    return frames


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(
        args.out_dir,
        f"target_ampLF_{args.color}_Tref{args.mi_T}.pt"
    )

    torch.set_grad_enabled(False)

    print("[gen_target_lf] Building LF target on CPU from frames...")
    print(f"  dataset    : {args.dataset or 'auto'} (scene={args.scene})")
    print(f"  mi_T       : {args.mi_T}")
    print(f"  color      : {args.color}")
    print(f"  n_fft      : {args.n_fft}")
    print(f"  hop_len    : {args.hop_len}")
    print(f"  win_len    : {args.win_len}")
    print(f"  cache_path : {cache_path}")
    print(f"  time_joint : {args.time_joint}")

    # 1) Load frames
    frames = _load_frames(args)  # [T,H,W] complex64 on CPU
    T, H, W = frames.shape

    # 2) Reshape to match holo2lf's expected input: (N, 1, H, W)
    input_field = frames.unsqueeze(1)  # [T, 1, H, W]

    # 3) Call holo2lf
    lf_full = holo2lf(
        input_field,
        n_fft=(args.n_fft, args.n_fft),
        hop_length=(args.hop_len, args.hop_len),
        win_length=(args.win_len, args.win_len),
        device=torch.device("cpu"),
        impl="torch",
    )
    # lf_full: [T, 1, H_lf, W_lf, U, V]
    print(f"  holo2lf output shape: {tuple(lf_full.shape)}, dtype={lf_full.dtype}")

    # 4) Convert intensity -> amplitude
    lf_amp_t = lf_full.sqrt()  # [T, 1, H_lf, W_lf, U, V]

    # 5) Flatten angular dims into N_views
    lf_amp_t = lf_amp_t.squeeze(1)  # [T, H_lf, W_lf, U, V]
    T2, H_lf, W_lf, U, V = lf_amp_t.shape
    assert T2 == T
    N_views = U * V

    lf_amp_t = lf_amp_t.permute(0, 3, 4, 1, 2).contiguous()  # [T, U, V, H_lf, W_lf]
    lf_amp_t = lf_amp_t.view(T, N_views, H_lf, W_lf)         # [T, N_views, H_lf, W_lf]
    print(f"  lf_amp_t reshaped to: {tuple(lf_amp_t.shape)} (T, N_views={N_views}, H_lf, W_lf)")

    # 6) Time-joint (recommended, like TMNH)
    if args.time_joint:
        target_amp = (lf_amp_t ** 2).mean(dim=0, keepdims=True).sqrt()  # [1,N,H,W]
    else:
        target_amp = lf_amp_t  # [T,N,H,W]

    # OPTIONAL: save in 5D cache format [1,1,N,H,W]
    if args.save_5d:
        if target_amp.ndim != 4:
            raise ValueError(f"--save_5d expects target_amp to be 4D [1,N,H,W], got {tuple(target_amp.shape)}")
        target_amp = target_amp.unsqueeze(1)  # [1,1,N,H,W]

    print(f"  target_amp shape (saved) = {tuple(target_amp.shape)}, dtype={target_amp.dtype}")

    # 7) Optional preview
    if args.preview_dir is not None:
        maybe_preview_lf_amp(target_amp, args.preview_dir)

    # 8) Save
    torch.save(target_amp.detach().cpu(), cache_path)
    print(f"[gen_target_lf] Saved: {cache_path}")


if __name__ == "__main__":
    main()
