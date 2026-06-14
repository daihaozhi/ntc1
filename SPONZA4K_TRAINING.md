# Sponza4K Single-Material NTC Training

This document describes the first integration step for using `ntc1` with the Vulkan DDGI renderer: train one Intel Sponza 4K glTF material as an NTC texture set, reconstruct it for quality checks, and export feature grids/MLP weights for a future Vulkan runtime decoder.

## 1. Environment

Use a CUDA-capable machine for training because the model uses `tiny-cuda-nn`.

```powershell
git clone https://github.com/daihaozhi/ntc1.git
cd ntc1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The Intel Sponza 4K package should be extracted so the glTF path looks like:

```text
<asset_root>\main1_sponza\NewSponza_Main_glTF_003.gltf
```

## 2. Inspect Materials

List material ids and names:

```powershell
python prepare_sponza4k_material.py `
  --gltf "<asset_root>\main1_sponza\NewSponza_Main_glTF_003.gltf" `
  --list-materials
```

Good first test materials are:

- `0 arch_stone_wall_01`
- `3 brickwall_02`
- `8 stones_01_tile`
- `9 column_1stfloor`

Avoid materials with no texture references for the first test because they only exercise fallback constants.

## 3. Prepare One Material Dataset

Prepare a 1024 version first for a quick smoke test:

```powershell
python prepare_sponza4k_material.py `
  --gltf "<asset_root>\main1_sponza\NewSponza_Main_glTF_003.gltf" `
  --material-id 0 `
  --resolution 1024 `
  --output_dir ".\datasets\sponza4k_arch_stone_wall_01_1024" `
  --overwrite
```

Prepare a full 4096 dataset when the pipeline is validated:

```powershell
python prepare_sponza4k_material.py `
  --gltf "<asset_root>\main1_sponza\NewSponza_Main_glTF_003.gltf" `
  --material-id 0 `
  --resolution 4096 `
  --output_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --overwrite
```

The output directory contains:

```text
diffuse.png
normal.png
roughness.png
metallic.png
occlusion.png
material.json
```

`prepare_sponza4k_material.py` splits glTF metallic-roughness maps into roughness from the G channel and metallic from the B channel. Missing maps are generated as constants so every material can be trained.

## 4. Train

Quick 1024 smoke test:

```powershell
python train.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_1024" `
  --output_dir ".\runs\sponza4k_arch_stone_wall_01_1024" `
  --texture_resolution 1024 `
  --batch_size 65536 `
  --max_iter 20000 `
  --eval_interval 500 `
  --save_interval 2000 `
  --device cuda
```

Full 4096 run:

```powershell
python train.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_4096" `
  --output_dir ".\runs\sponza4k_arch_stone_wall_01_4096" `
  --texture_resolution 4096 `
  --batch_size 65536 `
  --max_iter 40000 `
  --eval_interval 1000 `
  --save_interval 5000 `
  --device cuda
```

If VRAM is tight, reduce `--batch_size` first. If training is too slow, prepare a `2048` dataset and use `--texture_resolution 2048`.

## 5. Reconstruct for Quality Checks

```powershell
python inference.py `
  --data_dir ".\datasets\sponza4k_arch_stone_wall_01_1024" `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_1024\model_best.pth" `
  --output_dir ".\reconstructed\sponza4k_arch_stone_wall_01_1024" `
  --texture_resolution 1024 `
  --device cuda
```

Inspect `diffuse.png`, `normal.png`, `roughness.png`, `metallic.png`, and `occlusion.png`, then compare PSNR printed by the script.

## 6. Export Runtime Data

```powershell
python export.py `
  --checkpoint ".\runs\sponza4k_arch_stone_wall_01_1024\model_best.pth" `
  --output_dir ".\exported\sponza4k_arch_stone_wall_01_1024" `
  --texture_resolution 1024 `
  --device cuda
```

The export contains:

- `metadata.json`
- `grid_*_hi.png`
- `grid_*_lo.png`
- `mlp_*.bin`

This is not yet directly consumed by the DDGI renderer. The next project step is to add a Vulkan GLSL decoder that reads the exported feature grids and MLP weights in `gbuffer.frag`.

## 7. DDGI Integration Notes

The DDGI renderer currently samples large atlases in the G-buffer shader:

- base color
- normal
- metallic-roughness
- occlusion

For the first Vulkan integration pass, replace only one material with NTC and keep all other materials on the existing atlas path. This keeps debugging focused:

1. Add a per-material NTC metadata table to the renderer.
2. Add a GUI switch: `Atlas / NTC single material`.
3. In `gbuffer.frag`, branch by material id.
4. Decode the exported material only for the selected Sponza4K material.
5. Compare tracked GPU memory, G-buffer time, and visual error.

