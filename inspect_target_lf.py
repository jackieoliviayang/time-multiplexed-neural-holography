import torch

path = "outputs_lfopt_shared/target_ampLF_clean9x9_hotdog_Tref24.pt"

x = torch.load(path, map_location="cpu")

print("Type:", type(x))
print("Shape:", x.shape)

# If it's a tensor, show basic stats:
if isinstance(x, torch.Tensor):
    print("Min:", float(x.min()))
    print("Max:", float(x.max()))
    print("Dtype:", x.dtype)
