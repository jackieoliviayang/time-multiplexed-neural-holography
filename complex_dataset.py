# complex_dataset.py
from torch.utils.data import Dataset
from pathlib import Path
import sys, os, numpy as np, torch

# complex_dataset.py
def _import_loader_from_mutual_intensity():
    here = os.path.dirname(os.path.abspath(__file__))
    # go to the parent of TMNH, then into mutual-intensity
    mi_dir = os.path.abspath(os.path.join(here, "..", "mutual-intensity"))
    if mi_dir not in sys.path:
        sys.path.insert(0, mi_dir)
    from load_scanlines import load_wavefront_scanlines
    return load_wavefront_scanlines

class ComplexFramesToFocalStackTarget(Dataset):
    """
    Builds the focal-stack target by averaging intensity over T_ref complex frames:
        I_tgt(z) = mean_t | ASM(U_t, z) |^2
    Returns:
      - 'target': [D,H,W] float32 amplitude = sqrt(I_tgt(z))
    Optionally caches the target tensor to disk to reuse across runs.
    """
    def __init__(self, T_ref, z_list, wavelength=532e-9, dx=8e-6, dy=8e-6,
                 device="cpu", cache_path=None, preview_dir=None):
        super().__init__()
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.z_list = [float(z) for z in z_list]
        self.wavelength, self.dx, self.dy = float(wavelength), float(dx), float(dy)

        # cache?
        if cache_path is not None and os.path.isfile(cache_path):
            self.target_amp = torch.load(cache_path, map_location=self.device)
            return

        load_wf = _import_loader_from_mutual_intensity()
        # Your loader returns (frames: [T_ref,H,W] complex, (H,W))
        frames, _ = load_wf(T_desired=T_ref, output_dir=preview_dir)
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        frames = frames.to(torch.complex64).to(self.device)  # [T_ref,H,W]

        D = len(self.z_list)
        H, W = frames.shape[-2], frames.shape[-1]
        I_accum = torch.zeros((D, H, W), dtype=torch.float32, device=self.device)

        # Build averaged intensity stack
        for t in range(frames.shape[0]):
            u0 = frames[t]
            # per-depth propagation
            U0 = torch.fft.fft2(u0)
            k = 2.0 * torch.pi / self.wavelength
            fx = torch.fft.fftfreq(W, d=self.dx).to(self.device)
            fy = torch.fft.fftfreq(H, d=self.dy).to(self.device)
            FX, FY = torch.meshgrid(fx, fy, indexing="xy")
            FX, FY = FX.T, FY.T
            omega2 = (2*torch.pi*FX)**2 + (2*torch.pi*FY)**2
            pos = (k**2 - omega2).clamp(min=0).sqrt()
            neg = (omega2 - k**2).clamp(min=0).sqrt()
            for di, z in enumerate(self.z_list):
                H_z = torch.exp(1j * (float(z) * pos)) * torch.exp(-float(z) * neg)
                Uz = torch.fft.ifft2(U0 * H_z)
                I_accum[di] += (Uz.abs()**2).real.to(torch.float32)

        I_mean = I_accum / float(frames.shape[0])
        self.target_amp = torch.sqrt(torch.clamp(I_mean, min=0.0))  # [D,H,W] amplitude

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.target_amp.detach().cpu(), cache_path)

    def __len__(self): return 1
    def __getitem__(self, idx):
        return {"target": self.target_amp}


def angular_spectrum_propagate(u0, z, wavelength, dx, dy):
    H, W = u0.shape
    device = u0.device
    k = 2.0 * np.pi / float(wavelength)

    fx = torch.fft.fftfreq(W, d=float(dx)).to(device)
    fy = torch.fft.fftfreq(H, d=float(dy)).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing="xy")
    FX = FX.T; FY = FY.T  # -> [H,W]

    # band-limited ASM transfer (handles evanescent with imaginary exponent)
    omega2 = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
    # split sqrt to avoid nan for negative radicand
    pos = (k**2 - omega2).clamp(min=0).sqrt()
    neg = (omega2 - k**2).clamp(min=0).sqrt()
    H_z = torch.exp(1j * (z * pos)) * torch.exp(-z * neg)  # standard decay for evanescent

    U0 = torch.fft.fft2(u0)
    Uz = torch.fft.ifft2(U0 * H_z)
    return Uz

def complex_to_stacks(u0, z_list, wavelength, dx, dy):
    """Returns (U_stack [D,H,W] complex64, A_stack [D,H,W] float32 amplitude)."""
    U_list, A_list = [], []
    for z in z_list:
        Uz = angular_spectrum_propagate(u0, float(z), wavelength, dx, dy)
        U_list.append(Uz.to(torch.complex64))
        A_list.append(Uz.abs().to(torch.float32))  # <-- amplitude
    U_stack = torch.stack(U_list, dim=0)
    A_stack = torch.stack(A_list, dim=0)
    return U_stack, A_stack

def _load_complex(path):
    if path.endswith(".npy"):
        arr = np.load(path)
        if np.iscomplexobj(arr):
            u0 = torch.from_numpy(arr.astype(np.complex64))
        else:
            # assume [...,2] = (real, imag)
            u0 = torch.from_numpy(arr[...,0] + 1j*arr[...,1]).to(torch.complex64)
    else:
        u0 = torch.load(path)  # torch tensor (complex or stacked real/imag)
        if u0.dtype not in (torch.complex64, torch.complex128):
            u0 = u0[...,0] + 1j*u0[...,1]
            u0 = u0.to(torch.complex64)
    return u0

class ComplexFieldFocalStack(Dataset):
    """
    Produces:
      - "target": intensity focal stack [D,H,W] float32
      - "U_complex": complex field stack [D,H,W] complex64 (optional consumer)
    """
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

    def __len__(self): return 1
    def __getitem__(self, idx):
        return {"target": self.A_stack, "U_complex": self.U_stack}

def save_complex_stack(out_dir, U_stack, z_list, channel=0, fmt="pt"):
    """
    Save each depth's complex field as .pt (torch.save) or .npy (complex64).
    U_stack: [D,H,W] complex64
    """
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
