# gen_target_fstack.py
#!/usr/bin/env python3
import os, argparse, torch
from complex_dataset import ComplexFramesToFocalStackTarget

def parse_args():
    p = argparse.ArgumentParser("Precompute and cache 96f focal-stack target (amplitude).")
    p.add_argument("--mi_T", type=int, default=96)
    p.add_argument("--z_list_m", type=str, default=None,   # <— was required=True
                   help="Comma-separated depths with unit given by --z_unit.")
    p.add_argument("--z_unit", type=str, default="m", choices=["m","cm","mm"],
                   help="Unit for z values (default: meters).")
    p.add_argument("--wavelength", type=float, default=5.32e-7, help="meters")
    p.add_argument("--asm_dx", type=float, default=8e-6, help="meters")
    p.add_argument("--asm_dy", type=float, default=8e-6, help="meters")
    p.add_argument("--out_dir", type=str, default="outputs_fsopt")
    p.add_argument("--preview_dir", type=str, default=None)
    p.add_argument("--z_min", type=float, default=None, help="Depth start (in --z_unit)")
    p.add_argument("--z_max", type=float, default=None, help="Depth end (in --z_unit)")
    p.add_argument("--z_num", type=int,   default=None, help="Number of planes")
    return p.parse_args()

def main():
    args = parse_args()

    scale = {"m":1.0, "cm":1e-2, "mm":1e-3}[args.z_unit]

    # Prefer explicit list if provided; else use range params
    if args.z_list_m is not None:
        z_list = [float(z.strip())*scale for z in args.z_list_m.split(",") if z.strip()]
    else:
        if None in (args.z_min, args.z_max, args.z_num):
            raise SystemExit("Provide --z_list_m OR all of --z_min --z_max --z_num.")
        import numpy as np
        z = np.linspace(args.z_min, args.z_max, args.z_num)
        z_list = [float(v)*scale for v in z]

    cache_path = os.path.join(args.out_dir, f"target_amp_Tref{args.mi_T}_D{len(z_list)}.pt")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    print(f"[gen_target_fstack] Building target on CPU...")
    print(f"  mi_T       : {args.mi_T}")
    print(f"  D (planes) : {len(z_list)}")
    print(f"  z_unit     : {args.z_unit} (converted to meters)")
    print(f"  wavelength : {args.wavelength}")
    print(f"  dx, dy     : {args.asm_dx}, {args.asm_dy}")
    print(f"  cache_path : {cache_path}")

    ds = ComplexFramesToFocalStackTarget(
        T_ref=args.mi_T,
        z_list=z_list,                  # meters
        wavelength=args.wavelength,
        dx=args.asm_dx, dy=args.asm_dy,
        device=device,
        cache_path=None,
        preview_dir=args.preview_dir
    )
    target_amp = ds.target_amp.detach().cpu()
    torch.save(target_amp, cache_path)
    print(f"[gen_target_fstack] Saved: {cache_path}  shape={tuple(target_amp.shape)} dtype={target_amp.dtype}")

if __name__ == "__main__":
    main()
