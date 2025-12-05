export CUDA_VISIBLE_DEVICES=9

python main.py \
  --complex_input=true \
  --optimize_complex \
  --wavelength=5.32e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m="$(python - <<'PY'
import numpy as np
z = np.linspace(-0.0025, 0.0025, 7)  # meters (±0.25 cm)
print(",".join(f"{v:.6f}" for v in z))
PY
)" \
  --channel=1 \
  --num_frames=8 \
  --num_iters=5000 \
  --lr=0.2 \
  --save_images \
  --save_complex=true \
  --target_cache_path outputs_fsopt_shared/target_amp_Tref24_D7.pt \
  --out_dir outputs_fsopt_24
