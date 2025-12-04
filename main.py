"""
Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Time-multiplexed Neural Holography:
A Flexible Framework for Holographic Near-eye Displays with Fast Heavily-quantized Spatial Light Modulators
S. Choi*, M. Gopakumar*, Y. Peng, J. Kim, Matthew O'Toole, G. Wetzstein.
SIGGRAPH 2022
-----

$ python main.py --lr=0.01 --num_iters=10000 --num_frames=8 --quan_method=gumbel-softmax

"""
import os
import json
import torch
# import imageio
import imageio.v2 as imageio
import configargparse
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import utils
import params
import algorithms as algs
import quantization as q
import numpy as np
import image_loader as loaders
from torch.utils.data import DataLoader
import props.prop_model as prop_model
import props.prop_physical as prop_physical
from hw.phase_encodings import phase_encoding
from torchvision.utils import save_image

from pprint import pprint
from torch.utils.data import DataLoader
from complex_dataset import (
    ComplexFieldFocalStack,
    save_complex_stack,
    ComplexFramesToFocalStackTarget,   # <-- add this
)

import shutil, sys
from pathlib import Path

#import wx
#wx.DisableAsserts()
    
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## ADDED FOR SAVING
def save_stack_video(stack, path, fps=12):
    """stack: torch or np, shape [D,H,W] or [1,D,H,W]"""
    if hasattr(stack, "detach"):
        stack = stack.detach().float().cpu().numpy()
    stack = np.squeeze(stack)  # -> [D,H,W] or [H,W]
    if stack.ndim == 2:
        stack = stack[None]     # make it [1,H,W]
    # normalize per-frame for visibility; switch to global if you prefer
    m = stack.max() + 1e-12
    frames = [(stack[d] / m * 255).astype(np.uint8) for d in range(stack.shape[0])]
    # write mp4; requires imageio-ffmpeg installed
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

def save_stack_video_with_norm(stack, path, norm):
    if hasattr(stack, "detach"):
        stack = stack.detach().cpu().numpy()
    stack = np.squeeze(stack)
    frames = [(stack[d] / norm * 255).clip(0, 255).astype(np.uint8)
              for d in range(stack.shape[0])]
    writer = imageio.get_writer(path, fps=12, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()

def _to8(x):
    x = np.nan_to_num(x)
    x = np.clip(x, 0, None)
    m = float(x.max()) if x.size and np.isfinite(x.max()) and x.max() > 0 else 1.0
    return (x / m * 255.0).round().astype(np.uint8)

## ADDED FOR SAVING
def save_focal_result(arr, base_path):
    """
    Saves a focal result that might be [H,W], [D,H,W], [1,H,W], or [1,D,H,W].
    - If it's a stack, writes one PNG per depth: base_path -> base_path.replace('.png', f'_z{d:03d}.png')
    - If it's channels-first RGB(A), converts to HWC.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr)  # drop singleton dims

    if arr.ndim == 2:
        imageio.imwrite(base_path, _to8(arr))
        return

    if arr.ndim == 3:
        # Case A: [D,H,W] focal stack -> save each slice
        if arr.shape[0] not in (3, 4) and arr.shape[-1] not in (3, 4):
            for d in range(arr.shape[0]):
                imageio.imwrite(base_path.replace(".png", f"_z{d:03d}.png"),
                                _to8(arr[d]))
            return
        # Case B: channels-first -> move to HWC
        if arr.shape[0] in (3, 4) and arr.shape[1] != 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] in (3, 4):
            imageio.imwrite(base_path, _to8(arr))
            return

    raise ValueError(f"Unsupported shape for saving: {arr.shape}")

def ensure_shared_target_symlink(opt, D):
    """
    Make sure opt.out_dir has a symlink (or file) to a shared target cache:
      <shared_dir>/target_amp_Tref{mi_T}_D{D}.pt
    Creates the shared file if missing by computing on CPU once.
    Returns the absolute path that main should pass to ComplexFramesToFocalStackTarget as cache_path.
    """
    # Where to keep the single canonical copy (configurable via env var)
    shared_dir = Path(os.environ.get("TMNH_TARGET_CACHE_DIR", "outputs_fsopt_shared"))
    shared_dir.mkdir(parents=True, exist_ok=True)

    cache_basename = f"target_amp_Tref{getattr(opt,'mi_T',96)}_D{D}.pt"
    shared_path   = shared_dir / cache_basename
    out_dir       = Path(opt.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    local_path    = out_dir / cache_basename

    # 1) If the shared file doesn't exist, build it once on CPU (no GPU RAM).
    if not shared_path.exists():
        print(f"[cache] Missing shared cache -> building once on CPU: {shared_path}")
        from complex_dataset import ComplexFramesToFocalStackTarget
        z_list = [float(z) for z in opt.z_list]
        ds = ComplexFramesToFocalStackTarget(
            T_ref=getattr(opt, "mi_T", 96),
            z_list=z_list,
            wavelength=float(opt.wavelength),
            dx=float(opt.asm_dx), dy=float(opt.asm_dy),
            device="cpu",
            cache_path=None,
            preview_dir=getattr(opt, "mi_dbg_dir", None)
        )
        target_amp = ds.target_amp.detach().cpu()
        torch.save(target_amp, shared_path)
        print(f"[cache] Wrote shared CPU cache: {shared_path}  shape={tuple(target_amp.shape)}")

    # 2) Ensure the current run folder points to that shared file
    if local_path.exists():
        # already present (file or symlink) -> nothing to do
        return str(local_path)

    try:
        # Prefer a relative symlink so folders can be moved together
        rel_target = os.path.relpath(shared_path, start=out_dir)
        os.symlink(rel_target, local_path)
        print(f"[cache] Linked: {local_path} -> {rel_target}")
    except (OSError, NotImplementedError):
        # Windows without dev mode/admin: fall back to hardlink or copy
        try:
            os.link(shared_path, local_path)  # hardlink
            print(f"[cache] Hardlinked: {local_path} -> {shared_path}")
        except OSError:
            shutil.copy2(shared_path, local_path)
            print(f"[cache] Copied: {local_path} <- {shared_path}")

    return str(local_path)

def save_predicted_complex_fields(field_or_phase, z_list, wavelength, dx, dy,
                                  out_dir, channel=0):
    import os, torch
    os.makedirs(out_dir, exist_ok=True)

    fp = field_or_phase

    # ---- Normalize shape ----
    # Expect [T,H,W], [1,T,H,W], [T,1,H,W], [1,1,H,W], or [H,W]
    if fp.ndim == 2:
        fp = fp.unsqueeze(0)            # [H,W] -> [1,H,W]
    elif fp.ndim == 4:
        if fp.shape[0] == 1:            # [1,T,H,W] -> [T,H,W]
            fp = fp[0]
        elif fp.shape[1] == 1:          # [T,1,H,W] -> [T,H,W]
            fp = fp[:, 0]
        else:
            raise ValueError(
                f"Unsupported shape {tuple(fp.shape)}; "
                "expected batch or channel dim = 1."
            )
    elif fp.ndim != 3:
        raise ValueError(
            f"Unsupported shape {tuple(fp.shape)}; need 2D, 3D, or 4D."
        )

    fp = fp.contiguous()
    T, H, W = fp.shape

    device = fp.device
    wl = float(wavelength)
    dx = float(dx); dy = float(dy)

    # Build U0 from either phase or complex field
    if torch.is_complex(fp):
        U0 = fp
    else:
        U0 = torch.exp(1j * fp)

    def asm_prop(u0, z):
        k = 2.0 * torch.pi / wl
        fx = torch.fft.fftfreq(W, d=dx).to(device)
        fy = torch.fft.fftfreq(H, d=dy).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")
        FX, FY = FX.T, FY.T  # -> [H,W]
        omega2 = (2*torch.pi*FX)**2 + (2*torch.pi*FY)**2
        pos = (k**2 - omega2).clamp(min=0).sqrt()
        neg = (omega2 - k**2).clamp(min=0).sqrt()
        H_z = torch.exp(1j * (float(z) * pos)) * torch.exp(-float(z) * neg)
        return torch.fft.ifft2(torch.fft.fft2(u0) * H_z)

    # Save SLM-plane fields
    for t in range(T):
        torch.save(
            U0[t].detach().cpu(),
            os.path.join(out_dir, f"U_pred_slm_ch{channel}_t{t:02d}.pt")
        )

    # Propagate each frame to each z and save
    for di, z in enumerate(z_list):
        for t in range(T):
            Uz = asm_prop(U0[t], z)
            torch.save(
                Uz.detach().cpu(),
                os.path.join(
                    out_dir,
                    f"U_pred_ch{channel}_t{t:02d}_z{di:02d}_z{float(z):.6f}m.pt"
                )
            )



def main():
    # Command line argument processing / Parameters
    torch.set_default_dtype(torch.float32)
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False,
          is_config_file=True, help='Path to config file.')
    params.add_parameters(p, 'eval')
    opt = params.set_configs(p.parse_args())
    params.add_lf_params(opt)

    ## ADDED
    args = p.parse_args()
    opt = params.set_configs(args)
    params.add_lf_params(opt)
    
    # If user explicitly passed --wavelength on CLI, force it (and per-channel)
    if any(a.startswith("--wavelength") or a == "--wavelength" for a in os.sys.argv):
        user_wl = float(args.wavelength)
        opt.wavelength = user_wl
        if opt.channel is not None:
            opt.wavelengths = list(opt.wavelengths)
            opt.wavelengths[opt.channel] = user_wl

    # ---- Use 1024×1024 everywhere  ----
    if opt.complex_input:
        opt.citl = False
        opt.slm_type = 'holoeye'
        opt.use_lut = False
        opt.slm_res   = (1024, 1024)
        opt.image_res = opt.slm_res
        opt.roi_res   = opt.slm_res
        opt.full_roi  = True

    # ---- Make the forward model use ALL depths from --z_list_m ----
    # Target has D=len(opt.z_list); make sim model match it.
    opt.prop_dists_from_wrp = [float(z) for z in opt.z_list]
    opt.num_planes = len(opt.prop_dists_from_wrp)

    # Train/eval plane selections must be valid for the new length
    opt.training_plane_idxs = list(range(opt.num_planes))
    opt.heldout_plane_idxs = []
    opt.eval_plane_idx = None

    # Avoid shape/length assumptions in energy compensation
    opt.energy_compensation = False

    # >>> NEW: ensure a shared cache and a local symlink in opt.out_dir
    # cache_path = ensure_shared_target_symlink(opt, D=len(opt.z_list))
    cache_path = (opt.target_cache_path
              if getattr(opt, 'target_cache_path', None)
              else os.path.join(opt.out_dir, f"target_amp_Tref{getattr(opt,'mi_T',96)}_D{len(opt.z_list)}.pt"))


    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"[target-cache] Missing cache at {cache_path}\n"
            f"Hint: symlink or pass --target_cache_path to the shared file, "
            f"and ensure --mi_T and --z_list_m exactly match the cache."
        )
    print(f"[target-cache] Will load cached target: {cache_path}")

    dev = torch.device('cuda')

    run_id = params.run_id(opt)
    # path to save out optimized phases
    out_path = os.path.join(opt.out_path, run_id)
    print(f'  - out_path: {out_path}')

    # Tensorboard
    summaries_dir = os.path.join(out_path, 'summaries')
    utils.cond_mkdir(summaries_dir)
    writer = SummaryWriter(summaries_dir)

    # Write opt to experiment folder
    utils.write_opt(vars(p.parse_args()), out_path)

    # Propagations
    camera_prop = None
    if opt.citl:
        camera_prop = prop_physical.PhysicalProp(*(params.hw_params(opt)), shutter_speed=opt.shutter_speed).to(dev)
        camera_prop.calibrate_total_laser_energy() # important!

    # check wavelength
    print("Wavelength check:", "scalar =", opt.wavelength, "per-channel =", opt.wavelengths[opt.channel])

    sim_prop = prop_model.model(opt)
    sim_prop.eval()

    # Look-up table of SLM
    if opt.use_lut:
        lut = q.load_lut(sim_prop, opt)
    else:
        lut = None
    quantization = q.quantization(opt, lut)

    # Algorithm
    algorithm = algs.load_alg(opt.method, mem_eff=opt.mem_eff)

    # Loader
    if opt.complex_input:
        ds = ComplexFramesToFocalStackTarget(
        T_ref=getattr(opt, "mi_T", 96),
            z_list=opt.z_list,
            wavelength=opt.wavelength,
            dx=opt.asm_dx, dy=opt.asm_dy,
            device="cpu",               # keep target on CPU here
            cache_path=cache_path,      # MUST hit the early-return path
            preview_dir=getattr(opt, "mi_dbg_dir", None)
        )
        img_loader = DataLoader(ds, batch_size=1, shuffle=False)
    else:
        if ',' in opt.data_path:
            opt.data_path = opt.data_path.split(',')
        img_loader = loaders.TargetLoader(shuffle=opt.random_gen,
                                        vertical_flips=opt.random_gen,
                                        horizontal_flips=opt.random_gen,
                                        scale_vd_range=False, **opt)

    for i, batch in enumerate(img_loader):
        if opt.complex_input:
            # batch is a dict from ComplexFieldFocalStack
            target_amp = batch["target"].to(dev).detach()    # [D,H,W] amplitude
            target_mask = None
            target_idx = torch.tensor(0)                     # dummy index
            # If you ever need Uz stacks here:
            # U_stack = batch["U_complex"]  # [D,H,W] complex64 (on device already)
        else:
            target_amp, target_mask, target_idx = batch
            target_amp = target_amp.to(dev).detach()

        if target_mask is not None:
            target_mask = target_mask.to(dev).detach()
        if len(target_amp.shape) < 4:
            target_amp = target_amp.unsqueeze(0)

        print(f'  - run phase optimization for {target_idx}th image ...')

        if opt.random_gen:  # random parameters for dataset generation
            img_files = os.listdir(out_path)
            img_files = [f for f in img_files if f.endswith('.png')]
            if len(img_files) > opt.num_data: # generate enough data
                break
            print("Num images: ", len(img_files), " (max: ", opt.num_data)
            opt.num_frames, opt.num_iters, opt.init_phase_range, \
            target_range, opt.lr, opt.eval_plane_idx, \
            opt.quan_method, opt.reg_lf_var = utils.random_gen(**opt)
            sim_prop = prop_model.model(opt)
            quantization = q.quantization(opt, lut)
            target_amp *= target_range
            if opt.reg_lf_var > 0.0 and isinstance(sim_prop, prop_model.CNNpropCNN):
                opt.num_frames = min(opt.num_frames, 4)

        out_path_idx = f'{opt.out_path}_{target_idx}'

        # initial slm phase
        init_phase = utils.init_phase(opt.init_phase_type, target_amp, dev, opt)        
        
        # run algorithm
        results = algorithm(init_phase, target_amp, target_mask, target_idx,
                            forward_prop=sim_prop, camera_prop=camera_prop,
                            writer=writer, quantization=quantization,
                            out_path_idx=out_path_idx, **opt)
                            
        # optimized slm phase
        final_phase = results['final_phase']
        final_field = results['final_field']
        recon_amp = results['recon_amp']
        target_amp = results['target_amp']

        if getattr(opt, 'save_complex', False) and getattr(opt, 'complex_input', False):
            # Prefer final_field if available (complex mode), else fall back to phase
            field_or_phase = results.get('final_field', final_phase)

            save_predicted_complex_fields(
                field_or_phase,
                z_list=opt.z_list,
                wavelength=opt.wavelength,
                dx=opt.asm_dx, dy=opt.asm_dy,
                out_dir=os.path.join(opt.out_dir, f"predicted_T{field_or_phase.shape[0]}"),
                channel=opt.channel if opt.channel is not None else 0
            )


        # encoding for SLM & save it out
        # if opt.random_gen:
        #     # decompose it into several 1-bit phases
        #     for k, final_phase_1bit in enumerate(final_phase):
        #         phase_out = phase_encoding(final_phase_1bit.unsqueeze(0), opt.slm_type)
        #         phase_out_path = os.path.join(out_path, f'{target_idx}_{opt.num_iters}{k}.png')
        #         imageio.imwrite(phase_out_path, phase_out)
        # else:
        #     phase_out = phase_encoding(final_phase, opt.slm_type)
        #     recon_amp, target_amp = recon_amp.squeeze().detach().cpu().numpy(), target_amp.squeeze().detach().cpu().numpy()

        #     # save final phase and intermediate phases
        #     if phase_out is not None:
        #         phase_out_path = os.path.join(out_path, f'{target_idx}_phase.png')
        #         imageio.imwrite(phase_out_path, phase_out)

        #     if opt.save_images:
        #         recon_out_path = os.path.join(out_path, f'{target_idx}_recon.png')
        #         target_out_path = os.path.join(out_path, f'{target_idx}_target.png')
                
        #         if opt.channel is None:
        #             recon_amp = recon_amp.transpose(1, 2, 0)
        #             target_amp = target_amp.transpose(1, 2, 0)

        #         # recon_out = utils.srgb_lin2gamma(np.clip(recon_amp**2, 0, 1)) # linearize and gamma
        #         # target_out = utils.srgb_lin2gamma(np.clip(target_amp**2, 0, 1)) # linearize and gamma

        #         target_out = results['target_amp'].detach().float().cpu().numpy()
        #         recon_out = results['recon_amp'].detach().float().cpu().numpy()

        #         # Paths
        #         target_vid = os.path.join(out_path, "target_focalstack.mp4")
        #         recon_vid  = os.path.join(out_path, "recon_focalstack.mp4")

        #         # Save videos
        #         save_stack_video(results['target_amp'], target_vid)                 # your GT stack
        #         save_stack_video(results['recon_amp'], recon_vid)                   # optimized stack
        #         print(f"[saved] {target_vid}")
        #         print(f"[saved] {recon_vid}")
                
        #         save_focal_result(target_out, target_out_path)
        #         save_focal_result(recon_out, recon_out_path)  # e.g., ".../recon.png"

        #         # imageio.imwrite(recon_out_path, (recon_out * 255).astype(np.uint8))
        #         # imageio.imwrite(target_out_path, (target_out * 255).astype(np.uint8))
                # encoding for SLM & save it out
        if opt.random_gen:
            # decompose it into several 1-bit phases (original behavior)
            for k, final_phase_1bit in enumerate(final_phase):
                phase_out = phase_encoding(final_phase_1bit.unsqueeze(0), opt.slm_type)
                phase_out_path = os.path.join(out_path, f'{target_idx}_{opt.num_iters}{k}.png')
                imageio.imwrite(phase_out_path, phase_out)
        else:
            # ----- COMPLEX MULTI-FRAME CASE: average across frames -----
            if getattr(opt, 'complex_input', False) and getattr(opt, 'optimize_complex', False):
                phase_tensor = results['final_phase']  # could be [T,1,H,W], [1,1,H,W], [T,H,W], [H,W], etc.

                pt = phase_tensor

                # --- Normalize to at least [T, C, H, W] style ---

                if pt.dim() == 4:
                    # already [T, C, H, W]
                    pass
                elif pt.dim() == 3:
                    # assume [T, H, W] -> add channel dim
                    pt = pt.unsqueeze(1)          # [T,1,H,W]
                elif pt.dim() == 2:
                    # single [H,W] -> [1,1,H,W]
                    pt = pt.unsqueeze(0).unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected final_phase shape {pt.shape}")

                # Now pt is [T, C, H, W]; we only care about phase in the first channel
                # -> [T, H, W]
                phase_T_HW = pt[:, 0, :, :]

                # Circular average over time dimension T
                # This works even if T == 1.
                avg_complex = torch.exp(1j * phase_T_HW).mean(dim=0)  # [H,W] complex
                avg_phase   = torch.angle(avg_complex)                # [H,W] real

                # phase_encoding expects [B, C, H, W]
                avg_phase_in = avg_phase.unsqueeze(0).unsqueeze(0)    # [1,1,H,W]

                phase_out = phase_encoding(avg_phase_in, opt.slm_type)

                # Make sure imageio sees a 2D image
                phase_out = np.asarray(phase_out)
                phase_out = np.squeeze(phase_out)  # -> [H,W] for single-channel

                if phase_out.ndim != 2:
                    raise ValueError(f"phase_out still not 2D, got {phase_out.shape}")

                phase_out_path = os.path.join(out_path, f'{target_idx}_phase_avg.png')
                imageio.imwrite(phase_out_path, phase_out)

            else:
                # Original single-frame / phase-only behavior
                phase_out = phase_encoding(final_phase, opt.slm_type)
                phase_out = np.asarray(phase_out)
                phase_out = np.squeeze(phase_out)
                if phase_out.ndim > 2:
                    # if it somehow has an extra channel/batch dimension, take the first
                    phase_out = phase_out[0]
                if phase_out.ndim != 2:
                    raise ValueError(f"phase_out not 2D in phase-only branch, got {phase_out.shape}")
                phase_out_path = os.path.join(out_path, f'{target_idx}_phase.png')
                imageio.imwrite(phase_out_path, phase_out)

            # --- the rest (recon/target saving) stays the same ---
            recon_amp, target_amp = (
                recon_amp.squeeze().detach().cpu().numpy(),
                target_amp.squeeze().detach().cpu().numpy(),
            )

            target_out = results['target_amp'].detach().float().cpu().numpy()
            recon_out  = results['recon_amp'].detach().float().cpu().numpy()

            # Paths
            target_vid = os.path.join(out_path, "target_focalstack.mp4")
            recon_vid  = os.path.join(out_path, "recon_focalstack.mp4")

            # Global normalization for videos
            global_max = float((recon_out.max() if recon_out.mean() >= target_out.mean() else target_out.max()) + 1e-12)
            print("recon mean: ", recon_out.mean())
            print("target mean:", target_out.mean())

            save_stack_video_with_norm(target_out, target_vid, global_max)
            save_stack_video_with_norm(recon_out,  recon_vid,  global_max)
            print(f"[saved] {target_vid}")
            print(f"[saved] {recon_vid}")

            target_out_path = os.path.join(out_path, f'{target_idx}_target.png')
            recon_out_path  = os.path.join(out_path, f'{target_idx}_recon.png')
            save_focal_result(target_out, target_out_path)
            save_focal_result(recon_out,  recon_out_path)


    if camera_prop is not None:
        camera_prop.disconnect()

if __name__ == "__main__":
    main()
