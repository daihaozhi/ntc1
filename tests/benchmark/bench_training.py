"""Benchmark full training step performance.

Usage:
    python -m tests.benchmark.bench_training \
        --model learnable_grid \
        --texture_resolution 4096 \
        --max_iter 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.dataset import TextureDataset
from engine.trainer import Trainer
from models.learnable_grid_network import LearnableGridNetwork


def build_model_and_trainer(
    model_type: str,
    texture_resolution: int,
    grid_config_path: str | Path,
    data_dir: str | None = None,
    device: torch.device = None,
    pe_cfg: dict | None = None,
    mlp_cfg: dict | None = None,
    grid_sampler_cfg: dict | None = None,
) -> tuple:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_config_path = Path(grid_config_path)

    # Create minimal synthetic dataset if no real data
    if data_dir and Path(data_dir).exists():
        dataset = TextureDataset(data_dir=data_dir, device=device)
    else:
        # Synthetic dataset: random texture 256x256 for quick benchmark
        import numpy as np
        from PIL import Image
        import tempfile
        import os
        tmpdir = tempfile.mkdtemp()
        size = 256
        for name, mode, ch in [("diffuse.png", "RGB", 3), ("normal.png", "RGB", 3),
                                ("roughness.png", "L", 1), ("metallic.png", "L", 1)]:
            arr = (np.random.rand(size, size, ch) * 255).astype(np.uint8)
            if ch == 1:
                Image.fromarray(arr.reshape(size, size), mode=mode).save(os.path.join(tmpdir, name))
            else:
                Image.fromarray(arr, mode=mode).save(os.path.join(tmpdir, name))
        dataset = TextureDataset(data_dir=tmpdir, device=device)
    dataset.eval()

    if model_type == "learnable_grid":
        pe_cfg = pe_cfg or {"type": "torch_triangle", "n_frequencies": 5, "tiled": True, "tile_size": 8}
        mlp_cfg = mlp_cfg or {"type": "torch_linear", "hidden_dim": 64, "num_hidden_layers": 2, "output_dim": 8}
        grid_sampler_cfg = grid_sampler_cfg or {"high_res": "corner_four", "low_res": "bilinear"}

        model = LearnableGridNetwork(
            grid_config_path=str(grid_config_path),
            texture_resolution=texture_resolution,
            pe_cfg=pe_cfg,
            mlp_cfg=mlp_cfg,
            grid_sampler_cfg=grid_sampler_cfg,
            output_dim=8,
            default_save_bits=48,
            default_quantize_bits=4,
            max_iter=10000,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return model, dataset


def main():
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument("--model", default="learnable_grid", choices=["learnable_grid", "tcnn"])
    parser.add_argument("--texture_resolution", type=int, default=4096)
    parser.add_argument("--grid_config", default="configs/grid/grid_4096.json")
    parser.add_argument("--data_dir", default=None, help="Real dataset dir (optional)")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--crops_per_batch", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=100, help="Number of steps to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--cuda-graph", action="store_true", help="Enable CUDA graph capture for training step")
    # Component overrides
    parser.add_argument("--pe_type", default=None, choices=["torch_triangle", "tcnn_triangle"])
    parser.add_argument("--mlp_type", default=None, choices=["torch_linear", "tcnn_cutlass"])
    parser.add_argument("--grid_sampler_type", default=None, choices=["corner_four", "bilinear", "fused_corner_four", "custom_cuda"])
    parser.add_argument("--config", default=None, help="YAML config file (for component settings)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Setting up benchmark...")
    # Parse component configs
    pe_cfg = {"type": args.pe_type} if args.pe_type else None
    mlp_cfg = {"type": args.mlp_type} if args.mlp_type else None
    grid_sampler_cfg = {"high_res": args.grid_sampler_type, "low_res": "bilinear"} if args.grid_sampler_type else None

    # Override from YAML config if provided
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "pe" in cfg:
            pe_cfg = cfg["pe"]
        if "mlp" in cfg:
            mlp_cfg = cfg["mlp"]
        if "grid_sampler" in cfg:
            grid_sampler_cfg = cfg["grid_sampler"]

    model, dataset = build_model_and_trainer(
        model_type=args.model,
        texture_resolution=args.texture_resolution,
        grid_config_path=args.grid_config,
        data_dir=args.data_dir,
        device=device,
        pe_cfg=pe_cfg,
        mlp_cfg=mlp_cfg,
        grid_sampler_cfg=grid_sampler_cfg,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
    print(f"Resolution: {dataset.texture_width}x{dataset.texture_height}")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        crops_per_batch=args.crops_per_batch,
        max_iter=args.max_iter + args.warmup,
        lod_sampling="exp",
        mip_target_mode="discrete",
        eval_interval=0,
        save_interval=0,
        device=device,
        use_cuda_graph=args.cuda_graph,
    )

    # Warmup
    print(f"Warming up ({args.warmup} steps)...")
    for _ in range(args.warmup):
        trainer.train_step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed
    print(f"Benchmarking ({args.max_iter} steps)...")
    t0 = time.perf_counter()
    for _ in range(args.max_iter):
        trainer.train_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg_step_ms = elapsed / args.max_iter * 1000
    steps_per_sec = args.max_iter / elapsed
    samples_per_sec = args.batch_size * steps_per_sec

    mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0

    print(f"\n─── Training Benchmark Results ───")
    print(f"  Steps:            {args.max_iter}")
    print(f"  Total time:       {elapsed:.2f} s")
    print(f"  Avg step time:    {avg_step_ms:.2f} ms")
    print(f"  Steps/sec:        {steps_per_sec:.2f}")
    print(f"  Samples/sec:      {samples_per_sec:,.0f}")
    print(f"  Peak VRAM:        {mem_mb:.1f} MB")
    print(f"  GPU:              {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    if args.output:
        summary = {
            "model": args.model,
            "pe_type": pe_cfg.get("type", "torch_triangle") if pe_cfg else "torch_triangle",
            "mlp_type": mlp_cfg.get("type", "torch_linear") if mlp_cfg else "torch_linear",
            "sampler_type": grid_sampler_cfg.get("high_res", "corner_four") if grid_sampler_cfg else "corner_four",
            "params": total_params,
            "resolution": f"{dataset.texture_width}x{dataset.texture_height}",
            "batch_size": args.batch_size,
            "steps": args.max_iter,
            "total_time_s": round(elapsed, 3),
            "avg_step_ms": round(avg_step_ms, 3),
            "steps_per_sec": round(steps_per_sec, 2),
            "samples_per_sec": round(samples_per_sec),
            "peak_memory_mb": round(mem_mb, 1),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
