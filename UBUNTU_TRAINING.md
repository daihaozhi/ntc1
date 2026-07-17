# Ubuntu Training: `main1_sponza`

This guide trains the shared-material NTC model from the `main1_sponza` asset and evaluates each reconstructed material with MSE/PSNR.

## Requirements

- Ubuntu 22.04 or newer
- NVIDIA GPU with a CUDA toolkit/driver compatible with the installed PyTorch
- Python 3.10+
- Enough disk space for prepared datasets, checkpoints, reconstructions, and exported grids

The Python dependencies are listed in `requirements.txt`:

```text
torch
torchvision
numpy
Pillow
tinycudann
torchtyping
```

## Installation

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential

git clone https://github.com/daihaozhi/ntc1.git
cd ntc1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build appropriate for the machine. For example, for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify the environment:

```bash
python - <<'PY'
import torch
import tinycudann
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## Expected asset layout

Point `--data-root` to the directory containing the Sponza glTF asset. The launcher searches recursively for `NewSponza_Main_glTF_003.gltf` (or the first `.gltf` file):

```text
main1_sponza/
  NewSponza_Main_glTF_003.gltf
  ... referenced images and buffers ...
```

The batch preparation step extracts each material into canonical texture files before training.

## Train every material

```bash
source .venv/bin/activate
python train_main1_sponza.py \
  --data-root /datasets/main1_sponza \
  --mode all \
  --work-dir /runs/main1_sponza_ntc \
  --resolution 4096 \
  --device cuda \
  --max-iter 40000
```

This performs preparation, training, export, reconstruction, and evaluation. The default training configuration follows the NTC document:

```text
12 feature channels per large/small grid
16-bit scalar grid quantization
8 random 256x256 crops per batch
Adam + cosine annealing
73 -> 64 -> 64 -> 8 MLP
```

## Quick validation: one random material

```bash
python train_main1_sponza.py \
  --data-root /datasets/main1_sponza \
  --mode random \
  --seed 1234 \
  --work-dir /runs/main1_sponza_quick \
  --resolution 4096 \
  --device cuda \
  --max-iter 2000
```

The seed makes the selected material reproducible. Use a small `--max-iter` for a smoke test, then increase it for quality.

## Evaluation output

The launcher runs `batch_reconstruct_sponza4k.py` after training. Each material's reconstruction log contains per-texture metrics:

```text
diffuse: MSE=..., PSNR=... dB
normal: MSE=..., PSNR=... dB
metallic: MSE=..., PSNR=... dB
roughness: MSE=..., PSNR=... dB
```

A machine-readable aggregate is written to:

```text
<work-dir>/evaluation_summary.json
```

Reconstructed images are stored under:

```text
<work-dir>/reconstructed_<resolution>/
```

## Useful options

Use these options to control runtime and memory:

```bash
--device cuda              # or cpu for functional debugging
--max-iter 2000            # quick validation
--batch-size 65536         # reconstruction/fallback sample batch
--crop-size 256            # crop side length
--crops-per-batch 8        # set 0 to use legacy random texels
--skip-train               # only reconstruct/evaluate existing checkpoints
--skip-eval                # train/export without reconstruction
--overwrite-dataset        # regenerate prepared material folders
```

If GPU memory is insufficient, reduce `--crop-size`, `--crops-per-batch`, or `--batch-size` for a smoke test. Keep the default 8×256² crop configuration for the paper-style training run when memory permits.
