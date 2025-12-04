#!/usr/bin/env bash

# Generate LF target (7x7 angular grid) from GWS random-phase frames on CPU.

python gen_target_lf.py \
  --mi_T 24 \
  --wavelength 5.32e-7 \
  --asm_dx 8e-6 --asm_dy 8e-6 \
  --n_fft 7 \
  --hop_len 1 \
  --win_len 7 \
  --time_joint \
  --out_dir outputs_lfopt_shared \
  --preview_dir outputs_lfopt_shared/previews
