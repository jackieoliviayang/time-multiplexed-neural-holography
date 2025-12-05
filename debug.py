import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Fix path expansion + missing slash
path = "~/time-multiplexed-neural-holography/outputs_lfopt_24/predicted_T8/U_pred_ch1_t00_z00_z0.000000m.pt"
path = os.path.expanduser(path)

# Load the complex tensor
U = torch.load(path, map_location="cpu")

# Convert to numpy
U_np = U.detach().cpu().numpy()

# Compute amplitude
if np.iscomplexobj(U_np):
    amplitude = np.abs(U_np)
else:
    real = U_np[..., 0]
    imag = U_np[..., 1]
    amplitude = np.sqrt(real**2 + imag**2)

# Normalize
amp_norm = amplitude / amplitude.max()

# Save amplitude image
plt.figure(figsize=(6, 6))
plt.imshow(amp_norm, cmap='gray', origin='lower')
plt.axis('off')
plt.tight_layout()
plt.savefig("amplitude.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved amplitude image as amplitude.png")
