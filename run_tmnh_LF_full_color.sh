#!/bin/bash
set -euo pipefail

mkdir -p outputs_lfopt_shared/bicycle
mkdir -p outputs_lfopt_shared/bicycle/previews
mkdir -p logs

MI_T=24

# -------------------------
# 1) Build LF targets (CPU)
# -------------------------

python gen_target_lf.py \
  --mi_T ${MI_T} \
  --color red \
  --wavelength 6.38e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --n_fft 7 \
  --hop_len 1 \
  --win_len 7 \
  --time_joint \
  --out_dir outputs_lfopt_shared/bicycle \
  --dataset mipnerf --scene bicycle \
  --preview_dir outputs_lfopt_shared/bicycle/previews/red

python gen_target_lf.py \
  --mi_T ${MI_T} \
  --color green \
  --wavelength 5.20e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --n_fft 7 \
  --hop_len 1 \
  --win_len 7 \
  --time_joint \
  --out_dir outputs_lfopt_shared/bicycle \
  --dataset mipnerf --scene bicycle \
  --preview_dir outputs_lfopt_shared/bicycle/previews/green

python gen_target_lf.py \
  --mi_T ${MI_T} \
  --color blue \
  --wavelength 4.88e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --n_fft 7 \
  --hop_len 1 \
  --win_len 7 \
  --time_joint \
  --out_dir outputs_lfopt_shared/bicycle \
  --dataset mipnerf --scene bicycle \
  --preview_dir outputs_lfopt_shared/bicycle/previews/blue

# ---------------------------------
# 2) Run LF optimization (3 GPUs)
# ---------------------------------

CUDA_VISIBLE_DEVICES=3 python main.py \
  --complex_input=true \
  --optimize_complex \
  --target=lf \
  --wavelength=6.38e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m=0.0 \
  --channel=0 \
  --num_frames=8 \
  --num_iters=10000 \
  --lr=0.6 \
  --save_images \
  --save_complex=true \
  --mi_T=${MI_T} \
  --target_cache_path outputs_lfopt_shared/bicycle/target_ampLF_red_Tref${MI_T}.pt \
  --out_dir outputs_lfopt_24_red/bicycle \
  > logs/lfopt_red.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python main.py \
  --complex_input=true \
  --optimize_complex \
  --target=lf \
  --wavelength=5.20e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m=0.0 \
  --channel=1 \
  --num_frames=8 \
  --num_iters=10000 \
  --lr=0.6 \
  --save_images \
  --save_complex=true \
  --mi_T=${MI_T} \
  --target_cache_path outputs_lfopt_shared/bicycle/target_ampLF_green_Tref${MI_T}.pt \
  --out_dir outputs_lfopt_24_green/bicycle \
  > logs/lfopt_green.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python main.py \
  --complex_input=true \
  --optimize_complex \
  --target=lf \
  --wavelength=4.88e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m=0.0 \
  --channel=2 \
  --num_frames=8 \
  --num_iters=10000 \
  --lr=0.6 \
  --save_images \
  --save_complex=true \
  --mi_T=${MI_T} \
  --target_cache_path outputs_lfopt_shared/bicycle/target_ampLF_blue_Tref${MI_T}.pt \
  --out_dir outputs_lfopt_24_blue/bicycle \
  > logs/lfopt_blue.log 2>&1 &

wait
echo "All three LF runs finished."
