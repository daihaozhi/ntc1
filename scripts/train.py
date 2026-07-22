"""Unified NTC training entry point.

Supports both LearnableGridNetwork and TCNNModel backends via --model flag.

Usage:
    python scripts/train.py --config configs/baseline_learnable_4096.yaml
    python scripts/train.py --model learnable_grid --data_dir datasets/xxx --texture_resolution 4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import yaml

from engine.dataset import TextureDataset
from engine.trainer import Trainer
from models.learnable_grid_network import LearnableGridNetwork


def load_config(config_path: str | Path) -> dict:
    """Load config from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if config_path.suffix in (".yaml", ".yml"):
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def resolve_grid_config(cfg: dict) -> str:
    """Resolve grid_config path relative to repo root."""
    grid_config = cfg.get("grid_config", "configs/grid/grid_4096.json")
    if not os.path.isabs(grid_config):
        grid_config = str(_REPO_ROOT / grid_config)
    return grid_config


def main():
    parser = argparse.ArgumentParser(description="Train NTC model")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON config file (overrides CLI args)")
    parser.add_argument("--model", type=str, default="learnable_grid",
                        choices=["learnable_grid", "tcnn"],
                        help="Model architecture")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing texture images")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for checkpoints")
    parser.add_argument("--texture_resolution", type=int, default=4096)
    parser.add_argument("--grid_config", type=str, default=None,
                        help="Path to grid_config.json")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--crops_per_batch", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--network_lr", type=float, default=0.005)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--n_frequencies", type=int, default=5)
    parser.add_argument("--use_tiled_encoding", action="store_true", default=True)
    parser.add_argument("--lod_sampling", default="exp",
                        choices=["uniform", "exp", "fixed0"])
    parser.add_argument("--mip_target_mode", default="discrete",
                        choices=["discrete", "trilinear"])
    parser.add_argument("--boundary_continuity_weight", type=float, default=0.0)
    parser.add_argument("--boundary_band_width", type=float, default=0.0)
    parser.add_argument("--boundary_loss_preset", default="normal_roughness")
    parser.add_argument("--transition_delta_weight", type=float, default=0.0)
    parser.add_argument("--transition_delta_band_width", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load config file if provided
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # CLI args override config file
    def _get(key: str, default=None):
        cli_val = getattr(args, key, None)
        if cli_val is not None and cli_val != parser.get_default(key):
            return cli_val
        return cfg.get(key, default if default is not None else cli_val)

    model_type = _get("model", "learnable_grid")
    data_dir = _get("data_dir")
    output_dir = _get("output_dir", "./output")
    texture_resolution = int(_get("texture_resolution", 4096))
    grid_config = _get("grid_config", f"configs/grid/grid_{texture_resolution}.json")
    grid_config = resolve_grid_config({"grid_config": grid_config})

    if not data_dir:
        parser.error("--data_dir or --config with data_dir is required")

    device = torch.device(_get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = TextureDataset(data_dir=data_dir, device=device)
    dataset.eval()
    print(f"Dataset: {dataset.texture_width}x{dataset.texture_height}, "
          f"channels={dataset.num_channels}, LODs={dataset.num_lods}")

    # ── Model ────────────────────────────────────────────────────────
    if model_type == "learnable_grid":
        # New component-based API: pass cfg dicts directly
        pe_cfg = cfg.get("pe", {})
        mlp_cfg = cfg.get("mlp", {})
        grid_sampler_cfg = cfg.get("grid_sampler", {})

        # Fallback: build component cfgs from flat params if not in YAML
        if not pe_cfg:
            pe_cfg = {
                "type": "torch_triangle",
                "n_frequencies": int(_get("n_frequencies", 5)),
                "tiled": bool(_get("use_tiled_encoding", True)),
                "tile_size": 8,
            }
        if not mlp_cfg:
            mlp_cfg = {
                "type": "torch_linear",
                "hidden_dim": int(_get("hidden_dim", 64)),
                "num_hidden_layers": int(_get("num_hidden_layers", 2)),
                "output_dim": 8,
            }
        if not grid_sampler_cfg:
            grid_sampler_cfg = {"high_res": "corner_four", "low_res": "bilinear"}

        model = LearnableGridNetwork(
            grid_config_path=grid_config,
            texture_resolution=texture_resolution,
            pe_cfg=pe_cfg,
            mlp_cfg=mlp_cfg,
            grid_sampler_cfg=grid_sampler_cfg,
            output_dim=8,
            default_save_bits=48 if texture_resolution == 4096 else 192,
            default_quantize_bits=4 if texture_resolution == 4096 else 16,
            max_iter=int(_get("max_iter", 40000)),
        ).to(device)
    elif model_type == "tcnn":
        raise NotImplementedError(
            "TCNNModel requires external configs.py and utils.py from the DDGI project. "
            "Use 'learnable_grid' for now."
        )
    else:
        raise ValueError(f"Unknown model: {model_type}")

    print(f"Model: {model_type}, params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  PE: {model.pe_cfg['type']}, MLP: {model.mlp_cfg['type']}, "
          f"Sampler: {model.grid_sampler_cfg['high_res']}/{model.grid_sampler_cfg['low_res']}")

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        dataset=dataset,
        lr=float(_get("lr", 0.01)),
        network_lr=float(_get("network_lr", 0.005)),
        max_iter=int(_get("max_iter", 40000)),
        batch_size=int(_get("batch_size", 65536)),
        crop_size=int(_get("crop_size", 256)),
        crops_per_batch=int(_get("crops_per_batch", 8)),
        lod_sampling=_get("lod_sampling", "exp"),
        mip_target_mode=_get("mip_target_mode", "discrete"),
        boundary_continuity_weight=float(_get("boundary_continuity_weight", 0.0)),
        boundary_band_width=float(_get("boundary_band_width", 0.0)),
        boundary_loss_preset=_get("boundary_loss_preset", "normal_roughness"),
        transition_delta_weight=float(_get("transition_delta_weight", 0.0)),
        transition_delta_band_width=float(_get("transition_delta_band_width", 1.0)),
        eval_interval=int(_get("eval_interval", 1000)),
        save_interval=int(_get("save_interval", 5000)),
        output_dir=output_dir,
        device=device,
    )

    summary = trainer.run()
    print(f"\nDone. Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
