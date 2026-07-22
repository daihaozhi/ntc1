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

## Pluggable Components

The model is built from three swappable components, selectable via YAML config:

| Component | Options | Config Key |
|-----------|---------|------------|
| **Positional Encoding** | `torch_triangle`, `tcnn_triangle` | `pe.type` |
| **MLP Decoder** | `torch_linear`, `tcnn_cutlass` | `mlp.type` |
| **Grid Sampler** | `corner_four`, `bilinear`, `fused_corner_four` | `grid_sampler.high_res` |

```yaml
# Example: all-PyTorch baseline
pe:
  type: torch_triangle
  n_frequencies: 5
  tiled: true
  tile_size: 8

mlp:
  type: torch_linear
  hidden_dim: 64
  num_hidden_layers: 2
  output_dim: 8

grid_sampler:
  high_res: corner_four
  low_res: bilinear
```

## Project Structure

```
ntc1/
├── models/
│   ├── base.py             # Abstract NTCModel interface
│   ├── learnable_grid_network.py  # Main model (uses pluggable components)
│   ├── tcnn_model.py       # Legacy full-tcnn model (needs DDGI externals)
│   └── components/         # Pluggable PE, MLP, Grid Sampler
│       ├── pe.py           # TorchTriangleWavePE, TcnnTriangleWavePE
│       ├── mlp.py          # TorchMLP, TcnnCutlassMLP
│       └── grid_sampler.py # CornerFour, Bilinear, FusedCornerFour
├── engine/                 # Training & evaluation library
│   ├── dataset.py          # TextureDataset with mipmap chain
│   ├── trainer.py          # Unified training loop (model-agnostic)
│   ├── evaluator.py        # PSNR metrics, reconstruction
│   └── exporter.py         # Grid + MLP export for runtime
├── scripts/                # CLI entry points
│   ├── train.py            # Unified training (--config or CLI)
│   ├── inference.py        # Texture reconstruction
│   ├── export.py           # Export runtime assets
│   └── ...
├── tests/benchmark/
│   ├── bench_forward.py    # Forward pass timing
│   └── bench_training.py   # Training throughput
├── configs/
│   ├── baseline_learnable_4096.yaml
│   ├── opt_tcnn_pe_4096.yaml
│   ├── opt_tcnn_mlp_4096.yaml
│   ├── opt_fused_all_4096.yaml
│   └── grid/{1024,2048,4096}.json
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

## Optimization Workflow (no branching needed)

All optimizations are selectable via YAML config. No git branching required — just switch the config file and compare.

### Step 1: Establish baseline

```bash
python -m tests.benchmark.bench_training \
    --texture_resolution 4096 --max_iter 200 \
    --output baseline_training.json
```

### Step 2: Test an optimization

```bash
# Test tcnn TriangleWave PE
python scripts/train.py --config configs/opt_tcnn_pe_4096.yaml --data_dir data/my_material

# Or test tcnn CutlassMLP
python scripts/train.py --config configs/opt_tcnn_mlp_4096.yaml --data_dir data/my_material

# Or test everything fused
python scripts/train.py --config configs/opt_fused_all_4096.yaml --data_dir data/my_material
```

### Step 3: Compare results

All configs produce a summary dict at the end. Record in CSV:

```csv
experiment,pe_type,mlp_type,samples_per_sec,best_psnr,time_min
baseline,torch_triangle,torch_linear,1250000,38.2,45.3
opt_pe,tcnn_triangle,torch_linear,1900000,38.0,32.1
opt_all,tcnn_triangle,tcnn_cutlass,3400000,37.8,18.7
```

### Adding a new component

1. Implement in `models/components/` following the abstract interface
2. Register in the factory function (e.g., `build_pe`)
3. Add a YAML config with `type: your_new_type`
4. Run benchmark and record results

No merge conflicts — all code lives on `main`.

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
