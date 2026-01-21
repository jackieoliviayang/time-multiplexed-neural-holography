#!/bin/bash
set -euo pipefail

z_list="$(python - <<'PY'
import numpy as np
z = np.linspace(-0.0025, 0.0025, 7)
print(",".join(f"{v:.6f}" for v in z))
PY
)"
echo "z_list = '$z_list'"

# # (Optional) make sure caches exist first (CPU anyway, no GPU needed)
mkdir -p outputs_fsopt_shared/bicycle
mkdir -p logs 

# TARGETS ##
# RED cache
python gen_target_fs.py \
  --mi_T 24 \
  --color red \
  --wavelength 6.38e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --z_list_m="$z_list" \
  --loader mipnerf --scene bicycle \
  --out_dir outputs_fsopt_shared/bicycle

# GREEN cache
python gen_target_fs.py \
  --mi_T 24 \
  --color green \
  --wavelength 5.20e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --z_list_m="$z_list" \
  --loader mipnerf --scene bicycle \
  --out_dir outputs_fsopt_shared/bicycle

# BLUE cache
python gen_target_fs.py \
  --mi_T 24 \
  --color blue \
  --wavelength 4.88e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --z_list_m="$z_list" \
  --loader mipnerf --scene bicycle \
  --out_dir outputs_fsopt_shared/bicycle 

# Launch 3 runs on 3 different GPUs (example GPUs 7,8,9)
CUDA_VISIBLE_DEVICES=3 python main.py \
  --complex_input=true --optimize_complex \
  --wavelength=6.38e-7 --asm_dx=8e-6 --asm_dy=8e-6 --z_list_m="$z_list" \
  --channel=0 --num_frames=8 --num_iters=5000 --lr=0.2 \
  --save_images --save_complex=true --mi_T=24 \
  --target_cache_path outputs_fsopt_shared/bicycle/target_amp_red_Tref24_D7.pt \
  --out_dir outputs_fsopt_24_red/bicycle \
  > logs/fsopt_red.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python main.py \
  --complex_input=true --optimize_complex \
  --wavelength=5.20e-7 --asm_dx=8e-6 --asm_dy=8e-6 --z_list_m="$z_list" \
  --channel=1 --num_frames=8 --num_iters=5000 --lr=0.2 \
  --save_images --save_complex=true --mi_T=24 \
  --target_cache_path outputs_fsopt_shared/bicycle/target_amp_green_Tref24_D7.pt \
  --out_dir outputs_fsopt_24_green/bicycle \
  > logs/fsopt_green.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python main.py \
  --complex_input=true --optimize_complex \
  --wavelength=4.88e-7 --asm_dx=8e-6 --asm_dy=8e-6 --z_list_m="$z_list" \
  --channel=2 --num_frames=8 --num_iters=5000 --lr=0.2 \
  --save_images --save_complex=true --mi_T=24 \
  --target_cache_path outputs_fsopt_shared/bicycle/target_amp_blue_Tref24_D7.pt \
  --out_dir outputs_fsopt_24_blue/bicycle \
  > logs/fsopt_blue.log 2>&1 &


wait
echo "All three finished."
