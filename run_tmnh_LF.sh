export CUDA_VISIBLE_DEVICES=7

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
  --lr=0.6 \
  --save_images \
  --save_complex=true \
  --target_cache_path outputs_lfopt_shared/target_ampLF_Tref24.pt \
  --out_dir outputs_lfopt_24

  # works with 5000 iters at lr=0.5 (but not as high contrast as target)
  # currently trying 5k iters at lr=0.4 (to check contrast) --> this was worse, background a bit grey
  # currently trying 10k iters at lr=0.5 to see if it fixes contrast (this is great!)
  # trying 5k iters at lr=0.6 (this is great!)