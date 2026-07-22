# ntc1 — Neural Texture Compression

Train compact neural material textures and export feature grids plus MLP weights for real-time GPU decoding.

## Architecture

```
UV + LOD (3D)
    │
    ├── Tiled TriangleWave positional encoding (12D)
    ├── Feature grid pyramid (4 levels, 60D active)
    │     ├── Large grid: 4-corner nearest sampling (48D)
    │     └── Small grid: bilinear interpolation (12D)
    └── Normalized LOD (1D)
    │
    ▼
MLP: 73 → 64 → 64 → 8
    │
    ▼
basecolor.rgb(3) + metalness(1) + normal.rgb(3) + roughness(1)
```

## Project Structure

```
ntc1/
├── models/                 # Model implementations
│   ├── base.py             # Abstract NTCModel interface
│   ├── learnable_grid_network.py  # PyTorch PE + MLP + tcnn Grid (main)
│   └── tcnn_model.py       # Full tcnn (requires DDGI externals)
├── engine/                 # Training & evaluation library
│   ├── dataset.py          # TextureDataset with mipmap chain
│   ├── trainer.py          # Unified training loop
│   ├── evaluator.py        # PSNR metrics, reconstruction
│   └── exporter.py         # Grid + MLP export for runtime
├── scripts/                # CLI entry points
│   ├── train.py            # Unified training (--config or CLI args)
│   ├── inference.py        # Texture reconstruction
│   ├── export.py           # Export runtime assets
│   ├── eval_grid_level.py  # Per-level PSNR evaluation
│   ├── eval_mip_transition.py  # Mip transition quality
│   ├── analyze_grid.py     # Grid texture statistics
│   └── batch_*.py          # Sponza4K batch processing
├── tests/
│   └── benchmark/
│       ├── bench_forward.py    # Forward pass timing
│       └── bench_training.py   # Training step throughput
├── configs/
│   ├── baseline_learnable_4096.yaml  # Baseline config
│   ├── baseline_tcnn_4096.yaml       # (future) TCNN config
│   └── grid/
│       ├── grid_1024.json
│       ├── grid_2048.json
│       └── grid_4096.json
└── experiments/            # Experiment logs (git-ignored)
```

## Quick Start

### 1. Baseline Training

```powershell
python scripts/train.py --config configs/baseline_learnable_4096.yaml --data_dir data/my_material
```

Or with CLI overrides:

```powershell
python scripts/train.py `
  --model learnable_grid `
  --data_dir data/my_material `
  --texture_resolution 4096 `
  --max_iter 40000 `
  --batch_size 65536 `
  --lod_sampling exp `
  --mip_target_mode trilinear `
  --output_dir runs/my_material
```

### 2. Run Benchmarks

```powershell
# Forward pass timing
python -m tests.benchmark.bench_forward --texture_resolution 4096

# Training throughput
python -m tests.benchmark.bench_training --texture_resolution 4096 --max_iter 200
```

### 3. Batch Train Sponza 4K

```powershell
python scripts/batch_train_sponza4k.py `
  --gltf "<path>/NewSponza_Main_glTF_003.gltf" `
  --work_dir runs_sponza4k `
  --resolution 4096 `
  --export `
  --device cuda
```

### 4. Reconstruct & Evaluate

```powershell
python scripts/inference.py `
  --data_dir data/my_material `
  --checkpoint runs/my_material/model_best.pth `
  --output_dir reconstructed/my_material `
  --texture_resolution 4096
```

## Optimization Workflow

1. **Establish baseline**: `python -m tests.benchmark.bench_training --output baseline.json`
2. **Create experiment branch**: `git checkout -b opt/my-optimization`
3. **Make changes**, run benchmark: `python -m tests.benchmark.bench_training --output opt.json`
4. **Compare**: `python -m tests.benchmark.compare baseline.json opt.json`
5. **Record**: add row to `experiments/results.csv`

### Key Metrics

| Metric | Tool |
|--------|------|
| Training throughput (samples/sec) | `bench_training.py` |
| Forward pass latency (ms) | `bench_forward.py` |
| Peak VRAM (MB) | `bench_training.py` |
| Final PSNR (dB) | `train.py` output |
| Model parameters | `bench_forward.py` |

## Dependencies

- PyTorch >= 2.0
- tiny-cuda-nn >= 1.7
- NumPy, Pillow, PyYAML

## License

Internal research project.
