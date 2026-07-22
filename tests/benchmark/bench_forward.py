"""Benchmark model forward pass performance.

Usage:
    python -m tests.benchmark.bench_forward \
        --model learnable_grid \
        --texture_resolution 4096 \
        --grid_config configs/grid/grid_4096.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Allow running from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.learnable_grid_network import LearnableGridNetwork


def build_model(
    model_type: str,
    texture_resolution: int,
    grid_config_path: str | Path,
    hidden_dim: int = 64,
    num_hidden_layers: int = 2,
    n_frequencies: int = 5,
    device: torch.device = None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_config_path = Path(grid_config_path)

    if model_type == "learnable_grid":
        model = LearnableGridNetwork(
            grid_config_path=str(grid_config_path),
            texture_resolution=texture_resolution,
            output_dim=8,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            n_frequencies=n_frequencies,
            use_tiled_encoding=True,
            default_save_bits=48,
            default_quantize_bits=4,
            max_iter=1000,
        ).to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


@torch.no_grad()
def benchmark_forward(
    model: torch.nn.Module,
    batch_sizes: list[int] = None,
    warmup: int = 10,
    repeat: int = 50,
    device: torch.device = None,
) -> list[dict]:
    """Benchmark forward pass for different batch sizes."""
    device = device or next(model.parameters()).device
    batch_sizes = batch_sizes or [4096, 16384, 65536, 262144, 524288]

    results = []
    for bs in batch_sizes:
        # Generate random UV+LOD input
        x = torch.rand(bs, 3, device=device)
        x[:, 2] = torch.rand(bs, device=device)  # LOD in [0, 1]

        # Warmup
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        for _ in range(repeat):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_per_call = elapsed / repeat * 1000
        samples_per_sec = bs * repeat / elapsed

        results.append({
            "batch_size": bs,
            "ms_per_forward": round(ms_per_call, 4),
            "samples_per_sec": round(samples_per_sec),
        })
        print(f"  batch={bs:6d}  {ms_per_call:8.3f} ms  {samples_per_sec:12,} samples/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark model forward pass")
    parser.add_argument("--model", default="learnable_grid", choices=["learnable_grid", "tcnn"])
    parser.add_argument("--texture_resolution", type=int, default=4096)
    parser.add_argument("--grid_config", default="configs/grid/grid_4096.json")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--n_frequencies", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Building model: {args.model}, resolution={args.texture_resolution}")
    model = build_model(
        model_type=args.model,
        texture_resolution=args.texture_resolution,
        grid_config_path=args.grid_config,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        n_frequencies=args.n_frequencies,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print()

    results = benchmark_forward(
        model,
        warmup=args.warmup,
        repeat=args.repeat,
        device=device,
    )

    mem_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    print(f"\nPeak memory: {mem_bytes / 1024**2:.1f} MB")

    if args.output:
        summary = {
            "model": args.model,
            "params": total_params,
            "resolution": args.texture_resolution,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "peak_memory_mb": round(mem_bytes / 1024**2, 1),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
