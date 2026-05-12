#!/usr/bin/env python3
import os
import argparse
import torch
from complex_dataset import ComplexFramesToFocalStackTarget


def parse_args():
    p = argparse.ArgumentParser("Precompute and cache focal-stack target (amplitude).")

    p.add_argument("--mi_T", type=int, default=96)
    p.add_argument(
        "--z_list_m",
        type=str,
        default=None,
        help="Comma-separated depths with unit given by --z_unit.",
    )
    p.add_argument(
        "--z_unit",
        type=str,
        default="m",
        choices=["m", "cm", "mm"],
        help="Unit for z values.",
    )
    p.add_argument("--wavelength", type=float, default=5.32e-7, help="meters")
    p.add_argument("--asm_dx", type=float, default=8e-6, help="meters")
    p.add_argument("--asm_dy", type=float, default=8e-6, help="meters")
    p.add_argument("--out_dir", type=str, default="outputs_fsopt")
    p.add_argument("--preview_dir", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--loader", type=str, default="auto", choices=["auto", "blender", "mipnerf"])
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--z_min", type=float, default=None)
    p.add_argument("--z_max", type=float, default=None)
    p.add_argument("--z_num", type=int, default=None)
    p.add_argument("--color", type=str, default="red", choices=["red", "green", "blue"])

    # Benchmark path: pre-downsampled complex field [T,H,W] or [T,1,H,W]
    p.add_argument("--target_complex_path", type=str, default=None)

    return p.parse_args()


def load_complex_target(path: str, mi_T: int) -> torch.Tensor:
    print(f"[gen_target_fstack] Loading precomputed complex target: {path}", flush=True)

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

    U = payload

    if not torch.is_tensor(U):
        raise TypeError(f"Loaded object is not a tensor: {type(U)}")

    if not torch.is_complex(U):
        if U.ndim >= 1 and U.shape[-1] == 2:
            U = torch.complex(U[..., 0], U[..., 1])
        else:
            raise ValueError(
                f"Loaded tensor is not complex and has no last-dim=2. "
                f"Shape: {tuple(U.shape)}"
            )

    if U.ndim == 4 and U.shape[1] == 1:
        U = U[:, 0]
    elif U.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected U shape [T,H,W] or [T,1,H,W], got {tuple(U.shape)}")

    if U.shape[0] < mi_T:
        raise ValueError(f"Requested mi_T={mi_T}, but target only has T={U.shape[0]} frames.")

    U = U[:mi_T].contiguous().to(torch.complex64).cpu()

    print(f"[gen_target_fstack] Loaded precomputed target shape={tuple(U.shape)}", flush=True)
    return U


def asm_prop_stack(U: torch.Tensor, z: float, wavelength: float, dx: float, dy: float) -> torch.Tensor:
    """
    U: [T,H,W] complex64 CPU
    returns Uz: [T,H,W] complex64 CPU
    """
    T, H, W = U.shape

    k = 2.0 * torch.pi / float(wavelength)

    fx = torch.fft.fftfreq(W, d=float(dx))
    fy = torch.fft.fftfreq(H, d=float(dy))
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    omega2 = (2.0 * torch.pi * FX) ** 2 + (2.0 * torch.pi * FY) ** 2

    pos = (k**2 - omega2).clamp(min=0).sqrt()
    neg = (omega2 - k**2).clamp(min=0).sqrt()

    H_z = torch.exp(1j * float(z) * pos) * torch.exp(-float(z) * neg)

    return torch.fft.ifft2(torch.fft.fft2(U) * H_z)


def build_focal_stack_from_complex(U: torch.Tensor, z_list, wavelength: float, dx: float, dy: float):
    """
    U: [T,H,W] complex target frames.
    Returns target_amp: [D,H,W] float32.
    """
    stack = []

    for zi, z in enumerate(z_list):
        print(f"[gen_target_fstack] ASM plane {zi+1}/{len(z_list)} z={z:.6e} m", flush=True)
        Uz = asm_prop_stack(U, z, wavelength, dx, dy)

        # amplitude of time-averaged intensity
        amp = torch.sqrt((Uz.abs() ** 2).mean(dim=0).clamp_min(0))
        stack.append(amp.float().cpu())

    return torch.stack(stack, dim=0).contiguous()


def main():
    args = parse_args()

    scale = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}[args.z_unit]

    if args.z_list_m is not None:
        z_list = [float(z.strip()) * scale for z in args.z_list_m.split(",") if z.strip()]
    else:
        if None in (args.z_min, args.z_max, args.z_num):
            raise SystemExit("Provide --z_list_m OR all of --z_min --z_max --z_num.")
        import numpy as np
        z = np.linspace(args.z_min, args.z_max, args.z_num)
        z_list = [float(v) * scale for v in z]

    os.makedirs(args.out_dir, exist_ok=True)

    cache_path = os.path.join(
        args.out_dir,
        f"target_amp_{args.color}_Tref{args.mi_T}_D{len(z_list)}.pt",
    )

    torch.set_grad_enabled(False)

    data_root = args.data_root if args.data_root is not None else args.preview_dir

    print("[gen_target_fstack] Building target on CPU...", flush=True)
    print(f"  mi_T                : {args.mi_T}", flush=True)
    print(f"  D (planes)          : {len(z_list)}", flush=True)
    print(f"  z_unit              : {args.z_unit} (converted to meters)", flush=True)
    print(f"  wavelength          : {args.wavelength}", flush=True)
    print(f"  dx, dy              : {args.asm_dx}, {args.asm_dy}", flush=True)
    print(f"  data_root           : {data_root}", flush=True)
    print(f"  loader              : {args.loader}", flush=True)
    print(f"  scene               : {args.scene}", flush=True)
    print(f"  target_complex_path : {args.target_complex_path}", flush=True)
    print(f"  cache_path          : {cache_path}", flush=True)

    if args.target_complex_path is not None:
        U = load_complex_target(args.target_complex_path, args.mi_T)
        target_amp = build_focal_stack_from_complex(
            U,
            z_list=z_list,
            wavelength=args.wavelength,
            dx=args.asm_dx,
            dy=args.asm_dy,
        )
    else:
        if data_root is None and args.loader != "mipnerf" and args.scene is None:
            raise SystemExit("Need --data_root or --preview_dir for blender data unless using --target_complex_path.")

        device = torch.device("cpu")

        ds = ComplexFramesToFocalStackTarget(
            T_ref=args.mi_T,
            z_list=z_list,
            wavelength=args.wavelength,
            dx=args.asm_dx,
            dy=args.asm_dy,
            device=device,
            cache_path=None,
            data_root=data_root,
            color=args.color,
            loader=args.loader,
            scene=args.scene,
        )

        target_amp = ds.target_amp.detach().cpu()

    torch.save(target_amp, cache_path)

    print(
        f"[gen_target_fstack] Saved: {cache_path} "
        f"shape={tuple(target_amp.shape)} dtype={target_amp.dtype}",
        flush=True,
    )


if __name__ == "__main__":
    main()