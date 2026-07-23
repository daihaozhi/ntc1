# ntc1

Neural texture compression baseline for diffuse, normal, roughness, AO, and displacement maps.

## Baseline

The repository contains one training configuration:

```text
configs/baseline.yaml
```

It uses:

- 1024 texture resolution and 4-bit feature grids;
- 8 random `128x128` crops per step (`131072` samples);
- `lr=0.005` and `network_lr=0.0025`;
- maximum schedule length `100000` steps;
- screening run length `10000` steps;
- Cutlass MLP without an output activation;
- historical weighted MSE loss.

Run training with:

```powershell
python scripts/train.py --config configs/baseline.yaml --data_dir data --output_dir runs/baseline
```

## Reconstruction and export

```powershell
python scripts/inference.py `
  --data_dir data `
  --checkpoint runs/baseline/model_final.pth `
  --output_dir runs/baseline/reconstruction `
  --texture_resolution 1024 `
  --pe_type tcnn_triangle `
  --mlp_type tcnn_cutlass `
  --output_activation none
```

The runtime export entry point is `scripts/export.py`.

## Project structure

```text
configs/
└── baseline.yaml

engine/
├── dataset.py
├── evaluator.py
├── exporter.py
└── trainer.py

models/
├── base.py
├── learnable_grid_network.py
└── components/

scripts/
├── train.py
├── inference.py
└── export.py
```
