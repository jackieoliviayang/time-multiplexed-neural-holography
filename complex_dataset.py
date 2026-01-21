# complex_dataset.py
from torch.utils.data import Dataset
from pathlib import Path
import sys, os, numpy as np, torch

# ---------------------------------------------------------------------
# Loader plumbing (Blender + MipNeRF)
# ---------------------------------------------------------------------

def _add_mutual_intensity_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    mi_dir = os.path.abspath(os.path.join(here, "..", "mutual-intensity"))
    if mi_dir not in sys.path:
        sys.path.insert(0, mi_dir)
    return mi_dir

def _import_loaders_from_mutual_intensity():
    _add_mutual_intensity_to_path()
    from load_scanlines import load_wavefront_scanlines
    from load_scanlines_mipnerf import load_wavefront_scanlines_mipnerf
    return load_wavefront_scanlines, load_wavefront_scanlines_mipnerf

def _auto_choose_loader(data_root: str, scene: str | None):
    """
    Heuristic:
      - if scene is provided -> mipnerf
      - else if data_root contains '*_exact_awb' dirs -> mipnerf
      - else -> blender
    """
    if scene is not None:
        return "mipnerf"
    try:
        p = Path(data_root)
        if any(p.glob("*_exact_awb")):
            return "mipnerf"
    except Exception:
        pass
    return "blender"

def _load_complex_frames(
    T_ref: int,
    data_root: str,
    color: str,
    loader: str = "auto",
    scene: str | None = None,
):
    load_blender, load_mip = _import_loaders_from_mutual_intensity()

    if loader == "auto":
        loader = _auto_choose_loader(data_root, scene)

    if loader == "mipnerf":
        # mipnerf loader has its own internal root (hotdog convention)
        # output_dir is only used for optional debug image saving
        frames, _ = load_mip(T_desired=T_ref, output_dir=None, color=color, scene=scene)
    elif loader == "blender":
        frames, _ = load_blender(T_desired=T_ref, output_dir=data_root, color=color)
    else:
        raise ValueError(f"Unknown loader='{loader}'. Use 'auto'|'blender'|'mipnerf'.")

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    return frames


# ---------------------------------------------------------------------
# Focal-stack target from complex frames
# ---------------------------------------------------------------------

class ComplexFramesToFocalStackTarget(Dataset):
    """
    Builds the focal-stack target by averaging intensity over T_ref complex frames:
        I_tgt(z) = mean_t | ASM(U_t, z) |^2

    Returns:
      - 'target': [D,H,W] float32 amplitude = sqrt(I_tgt(z))

    Supports both blender and MipNeRF via loader='auto'|'blender'|'mipnerf' and optional scene.
    """
    def __init__(
        self,
        T_ref,
        z_list,
        wavelength=532e-9,
        dx=8e-6,
        dy=8e-6,
        device="cpu",
        cache_path=None,
        data_root=None,      # NEW
        preview_dir=None,    # legacy alias
        color="red",
        loader="auto",       # 'auto'|'blender'|'mipnerf'
        scene=None,          # for mipnerf
    ):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.z_list = [float(z) for z in z_list]
        self.wavelength, self.dx, self.dy = float(wavelength), float(dx), float(dy)

        # backward compat
        if data_root is None:
            data_root = preview_dir
        if data_root is None and loader != "mipnerf" and scene is None:
            raise ValueError("You must pass data_root (or legacy preview_dir) for blender data.")


        # cache?
        if cache_path is not None and os.path.isfile(cache_path):
            self.target_amp = torch.load(cache_path, map_location=self.device)
            return

        frames = _load_complex_frames(
            T_ref=T_ref,
            data_root=data_root,
            color=color,
            loader=loader,
            scene=scene,
        )
        frames = frames.to(torch.complex64).to(self.device)  # [T_ref,H,W]

        D = len(self.z_list)
        H, W = frames.shape[-2], frames.shape[-1]
        I_accum = torch.zeros((D, H, W), dtype=torch.float32, device=self.device)

        # Precompute ASM frequency terms
        k = 2.0 * torch.pi / self.wavelength
        fx = torch.fft.fftfreq(W, d=self.dx).to(self.device)
        fy = torch.fft.fftfreq(H, d=self.dy).to(self.device)
        FX, FY = torch.meshgrid(fx, fy, indexing="xy")
        FX, FY = FX.T, FY.T
        omega2 = (2*torch.pi*FX)**2 + (2*torch.pi*FY)**2
        pos = (k**2 - omega2).clamp(min=0).sqrt()
        neg = (omega2 - k**2).clamp(min=0).sqrt()

        # Build averaged intensity stack
        for t in range(frames.shape[0]):
            u0 = frames[t]
            U0 = torch.fft.fft2(u0)
            for di, z in enumerate(self.z_list):
                H_z = torch.exp(1j * (float(z) * pos)) * torch.exp(-float(z) * neg)
                Uz = torch.fft.ifft2(U0 * H_z)
                I_accum[di] += (Uz.abs()**2).real.to(torch.float32)

        I_mean = I_accum / float(frames.shape[0])
        self.target_amp = torch.sqrt(torch.clamp(I_mean, min=0.0))  # [D,H,W]

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.target_amp.detach().cpu(), cache_path)

    def __len__(self): 
        return 1

    def __getitem__(self, idx):
        return {"target": self.target_amp}


# ---------------------------------------------------------------------
# Utilities (unchanged from your version)
# ---------------------------------------------------------------------

def load_blender_complex_frames(T_ref, device="cpu", data_root=None, preview_dir=None, color="red", loader="auto", scene=None):
    device = device if isinstance(device, torch.device) else torch.device(device)
    if data_root is None:
        data_root = preview_dir
    frames = _load_complex_frames(T_ref=T_ref, data_root=data_root, color=color, loader=loader, scene=scene)
    frames = frames.to(torch.complex64).to(device)
    return frames

def angular_spectrum_propagate(u0, z, wavelength, dx, dy):
    H, W = u0.shape
    device = u0.device
    k = 2.0 * np.pi / float(wavelength)

    fx = torch.fft.fftfreq(W, d=float(dx)).to(device)
    fy = torch.fft.fftfreq(H, d=float(dy)).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    FX = FX.T; FY = FY.T

    omega2 = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
    pos = (k**2 - omega2).clamp(min=0).sqrt()
    neg = (omega2 - k**2).clamp(min=0).sqrt()
    H_z = torch.exp(1j * (z * pos)) * torch.exp(-z * neg)

    U0 = torch.fft.fft2(u0)
    Uz = torch.fft.ifft2(U0 * H_z)
    return Uz

def complex_to_stacks(u0, z_list, wavelength, dx, dy):
    U_list, A_list = [], []
    for z in z_list:
        Uz = angular_spectrum_propagate(u0, float(z), wavelength, dx, dy)
        U_list.append(Uz.to(torch.complex64))
        A_list.append(Uz.abs().to(torch.float32))
    U_stack = torch.stack(U_list, dim=0)
    A_stack = torch.stack(A_list, dim=0)
    return U_stack, A_stack

def _load_complex(path):
    if path.endswith(".npy"):
        arr = np.load(path)
        if np.iscomplexobj(arr):
            u0 = torch.from_numpy(arr.astype(np.complex64))
        else:
            u0 = torch.from_numpy(arr[...,0] + 1j*arr[...,1]).to(torch.complex64)
    else:
        u0 = torch.load(path)
        if u0.dtype not in (torch.complex64, torch.complex128):
            u0 = u0[...,0] + 1j*u0[...,1]
            u0 = u0.to(torch.complex64)
    return u0

class ComplexFieldFocalStack(Dataset):
    def __init__(self, path, channel=0, z_list=None, wavelength=532e-9, dx=8e-6, dy=8e-6, device="cpu"):
        if isinstance(path, (list, tuple)):
            path = path[channel]
        u0 = _load_complex(path).to(torch.complex64)
        self.u0 = u0.to(device)
        self.z_list = [float(z) for z in z_list]
        self.wavelength, self.dx, self.dy = float(wavelength), float(dx), float(dy)
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.U_stack, self.A_stack = complex_to_stacks(
            self.u0, self.z_list, self.wavelength, self.dx, self.dy
        )

    def __len__(self): 
        return 1

    def __getitem__(self, idx):
        return {"target": self.A_stack, "U_complex": self.U_stack}

def save_complex_stack(out_dir, U_stack, z_list, channel=0, fmt="pt"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    D = U_stack.shape[0]
    for di in range(D):
        z = z_list[di]
        if fmt == "npy":
            np.save(out / f"U_complex_ch{channel}_z{di:02d}_z{z:.6f}m.npy",
                    U_stack[di].detach().cpu().numpy())
        else:
            torch.save(U_stack[di].detach().cpu(),
                       out / f"U_complex_ch{channel}_z{di:02d}_z{z:.6f}m.pt")
