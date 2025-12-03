export CUDA_VISIBLE_DEVICES=8

python main.py \
  --complex_input=true \
  --wavelength=5.32e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --target_cache_path outputs_fsopt_shared/target_amp_Tref96_D60.pt \
  --z_list_m="$(python - <<'PY'
import numpy as np
z = np.linspace(-0.0025, 0.0025, 60)  # meters (±0.25 cm)
print(",".join(f"{v:.6f}" for v in z))
PY
)" \
  --channel=1 \
  --num_frames=8 \
  --num_iters=1000 \
  --lr=1e-4 \
  --save_images \
  --save_complex=true \
  --mem_eff true \
  --out_dir outputs_fsopt_T8
