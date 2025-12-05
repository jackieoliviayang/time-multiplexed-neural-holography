#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=8

python main.py \
  --complex_input=true \
  --optimize_complex \
  --target=lf \
  --wavelength=5.32e-7 \
  --asm_dx=8e-6 --asm_dy=8e-6 \
  --z_list_m=0.0 \
  --channel=1 \
  --num_frames=8 \
  --num_iters=5000 \
  --lr=0.2 \
  --save_images \
  --save_complex=true \
  --target_cache_path outputs_lfopt_shared/target_ampLF_Tref24.pt \
  --out_dir outputs_lfopt_24