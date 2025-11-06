export CUDA_VISIBLE_DEVICES=6

python main.py \
  --complex_input=true \
  --wavelength=5.32e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m="$(python - <<'PY'
import numpy as np
z = np.linspace(-0.25, 0.25, 60)
print(",".join(f"{v:.6f}" for v in z))
PY
)" \
  --channel=1 \
  --num_frames=1 \
  --num_iters=10000 \
  --lr=0.01 \
  --save_images \
  --save_complex=true \
  --out_dir outputs_fsopt
