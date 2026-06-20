# ntc1 Neural Texture Compression

This repository trains compact neural material textures and exports feature grids plus MLP weights for use in the DDGI Vulkan renderer. A material is represented by a small feature-grid pyramid and a shared decoder MLP. The training input is the highest-resolution material texture set; the dataset loader builds a mipmap chain internally and can supervise the network from either discrete mip levels or trilinear continuous LOD samples.

## Main Scripts

- `prepare_sponza4k_material.py`: extracts one Intel Sponza 4K glTF material into canonical texture files.
- `train.py`: trains one material NTC model from a texture folder.
- `export.py`: exports trained feature grids and MLP weights for runtime decoding.
- `inference.py`: reconstructs textures from a trained checkpoint for visual inspection.
- `eval_grid_level.py`: forces one feature grid level during reconstruction and reports per-LOD PSNR.
- `batch_train_sponza4k.py`: prepares, trains, and optionally exports many Sponza 4K materials.
- `batch_reconstruct_sponza4k.py`: reconstructs many trained Sponza 4K materials for the DDGI validation scene.

## Training Data

`TextureDataset` loads the maximum-resolution textures in a material folder, aligns them to one resolution, converts color textures to linear space, and concatenates available maps into the canonical 11-channel layout:

```text
diffuse.rgb
normal.rgb
roughness
occlusion
metallic
specular
displacement
```

It then generates a full mipmap chain with antialiased bicubic downsampling. Training can sample this chain in two modes:

- `--mip_target_mode discrete`: choose an integer mip level and train against one texel from that mip. This is the default and preserves the previous behavior.
- `--mip_target_mode trilinear`: choose an integer mip bucket, add a random fractional offset, bilinearly sample both adjacent mip levels, then linearly blend between them. The model input receives the same continuous normalized LOD value.

Internally, `TextureDataset.sample_discrete_lod()` implements the integer mip path and `TextureDataset.sample_trilinear_lod()` implements the continuous LOD target path.

LOD buckets are selected with `--lod_sampling`:

- `exp`: exponential distribution biased toward high-resolution levels.
- `uniform`: uniform over all mip levels.
- `fixed0`: only train the base level.

## Train One Material

```powershell
python train.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --output_dir ".\runs\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --batch_size 65536 `
  --max_iter 40000 `
  --lod_sampling uniform `
  --mip_target_mode trilinear `
  --eval_interval 1000 `
  --save_interval 5000 `
  --device cuda
```

Use `--mip_target_mode discrete` when you want the original integer-mip reconstruction target. Use `trilinear` when the runtime path needs smoother LOD-conditioned decoding.

To reduce visible jumps when switching between feature grid levels, add a small boundary continuity loss:

```powershell
python train.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --output_dir ".\runs\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --lod_sampling uniform `
  --mip_target_mode trilinear `
  --boundary_continuity_weight 0.03 `
  --boundary_loss_preset normal_roughness `
  --device cuda
```

The boundary term samples the mip boundaries from `grid_config.json` and forces adjacent grid levels to produce similar decoder outputs at those LODs. The `normal_roughness` preset emphasizes channels that most often turn small discontinuities into visible shimmer. Keep the global weight small at first, such as `0.01` to `0.05`, then check both single-level PSNR and `eval_grid_level.py --boundary`. For targeted experiments, pass custom JSON weights, for example `--boundary_loss_weights '{"normal":2.0,"roughness":6.0,"diffuse":0.25}'`.

## Batch Train Sponza 4K

```powershell
python batch_train_sponza4k.py `
  --gltf "<asset_root>\main1_sponza\NewSponza_Main_glTF_003.gltf" `
  --work_dir ".\runs_sponza4k_batch" `
  --resolution 4096 `
  --export `
  --lod_sampling uniform `
  --mip_target_mode trilinear `
  --boundary_continuity_weight 0.03 `
  --boundary_loss_preset normal_roughness `
  --device cuda
```

The batch script creates:

```text
datasets_4096/
runs_4096/
exported_4096/
logs/
```

`datasets_4096` contains prepared maximum-resolution material inputs. `runs_4096` contains checkpoints. `exported_4096` contains runtime data such as `grid_*_hi.png`, `grid_*_lo.png`, `mlp_*.bin`, and `metadata.json`.

## Reconstruct and Inspect

```powershell
python inference.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_4096\model_best.pth" `
  --output_dir ".\reconstructed\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --device cuda
```

For a full Sponza batch:

```powershell
python batch_reconstruct_sponza4k.py `
  --batch_dir ".\runs_sponza4k_batch" `
  --resolution 4096 `
  --device cuda
```

The DDGI renderer can use reconstructed material textures for validation before switching to true online NTC decoding.

## Evaluate One Grid Level

To diagnose whether a nonzero feature grid level is trained well, force that level during reconstruction. By default, the script evaluates the mip range assigned to the selected level in `grid_config.json`.

```powershell
python eval_grid_level.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_4096\model_best.pth" `
  --output_dir ".\eval_grid1\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --grid_level 1 `
  --device cuda
```

For a 4096 model, `grid_level 1` corresponds to LODs `[5, 7)`, so it evaluates `lod 5` and `lod 6`. The script writes reconstructed PNGs and `metrics.csv`.

To check whether adjacent grid levels are continuous at their boundaries, use boundary mode:

```powershell
python eval_grid_level.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_4096\model_best.pth" `
  --output_dir ".\eval_boundaries\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --boundary `
  --device cuda
```

For a 4096 model this compares `level0/level1` at `lod 5`, `level1/level2` at `lod 7`, and `level2/level3` at `lod 10`. It writes `boundary_metrics.csv` plus left/right/difference images.

## Quick Boundary Experiment

Use `quick_boundary_experiment.py` to train a small subset and immediately evaluate boundary continuity. This is useful for tuning `--boundary_continuity_weight` or boundary channel weights without retraining every Sponza material.

```bash
python quick_boundary_experiment.py \
  --gltf "main1_sponza/main_sponza/NewSponza_Main_glTF_003.gltf" \
  --work_dir "./quick_boundary_w005" \
  --material-ids "0" \
  --resolution 1024 \
  --max_iter 1200 \
  --batch_size 32768 \
  --lod_sampling uniform \
  --mip_target_mode trilinear \
  --boundary_continuity_weight 0.05 \
  --boundary_loss_preset normal_roughness \
  --device cuda
```

The script runs `batch_train_sponza4k.py`, then `eval_grid_level.py --boundary`, and writes `boundary_summary.csv` under the experiment directory. Use `--material-ids "0,3,8"` for a slightly broader sample or `--resolution 4096 --max_iter 3000` for a closer but heavier test.

## Export Runtime Data

```powershell
python export.py `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_4096\model_best.pth" `
  --output_dir ".\exported\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --device cuda
```

The exported feature grids follow the configured feature-grid pyramid in `grid_config.json`. The decoder MLP is shared across grid levels, and the normalized LOD is part of the network input.

## Notes for the DDGI Project

The DDGI Vulkan project has two useful validation paths:

- reconstructed-image path: use reconstructed PNGs to check material mapping and lighting correctness.
- online-decoder path: upload exported feature grids and MLP weights, then decode in `gbuffer.frag`.

The trilinear target mode is a training-side step toward smoother runtime LOD behavior. It does not by itself guarantee continuity between different feature-grid levels; that still depends on runtime filtering, temporal accumulation, and any future grid-level boundary regularization.
