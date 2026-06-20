import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import CANONICAL_CHANNEL_SLICES, CANONICAL_NUM_CHANNELS, TextureDataset
from learnable_grid_network import LearnableGridNetwork


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / mse) if mse > 1e-10 else float("inf")


def make_forced_level_input(model: LearnableGridNetwork, uv: torch.Tensor, lod_norm: torch.Tensor, level: int) -> torch.Tensor:
    pos_encoding = model._compute_positional_encoding(uv)
    features = model.sample_features(uv, level=level).to(pos_encoding.dtype)
    return torch.cat([pos_encoding, features, lod_norm], dim=1)


@torch.no_grad()
def evaluate_lod(
    model: LearnableGridNetwork,
    dataset: TextureDataset,
    level: int,
    lod_value: float,
    output_dir: Path,
    batch_size: int,
    save_images: bool,
) -> list[dict[str, str]]:
    lod_int = int(round(lod_value))
    if abs(lod_value - lod_int) > 1e-6:
        raise ValueError("eval_grid_level.py currently evaluates discrete integer mip levels")
    if lod_int < 0 or lod_int >= dataset.num_lods:
        raise ValueError(f"LOD {lod_int} out of range [0, {dataset.num_lods - 1}]")

    device = dataset.device
    lod_h = max(dataset.texture_height >> lod_int, 1)
    lod_w = max(dataset.texture_width >> lod_int, 1)
    num_pixels = lod_h * lod_w

    ys = torch.arange(lod_h, device=device).view(-1, 1).expand(lod_h, lod_w).reshape(-1)
    xs = torch.arange(lod_w, device=device).view(1, -1).expand(lod_h, lod_w).reshape(-1)
    lod_norm_value = lod_value / float(model.num_mip_levels - 1)

    pred = torch.zeros((num_pixels, CANONICAL_NUM_CHANNELS), device=device)
    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)
        idx = slice(start, end)
        uv = torch.stack(
            [
                (xs[idx].float() + 0.5) / float(lod_w),
                (ys[idx].float() + 0.5) / float(lod_h),
            ],
            dim=1,
        )
        lod_norm = torch.full((end - start, 1), lod_norm_value, device=device)
        pred[idx] = model.network(make_forced_level_input(model, uv, lod_norm, level))

    pred = torch.nan_to_num(pred.reshape(lod_h, lod_w, CANONICAL_NUM_CHANNELS).cpu(), nan=0.0, posinf=1.0, neginf=0.0)

    ref_loaded = dataset.lod_cache[lod_int, :lod_h, :lod_w, :].reshape(-1, dataset.num_channels)
    ref = dataset.expand_to_canonical(ref_loaded).reshape(lod_h, lod_w, CANONICAL_NUM_CHANNELS).cpu()

    rows: list[dict[str, str]] = []
    srgb_types = {"diffuse"}
    lod_dir = output_dir / f"level_{level}_lod_{lod_int:02d}"
    if save_images:
        lod_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGrid level {level}, LOD {lod_int}, resolution {lod_w}x{lod_h}")
    for tex_type in dataset.available_textures:
        cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
        pred_tex = pred[:, :, cn_start:cn_end]
        ref_tex = ref[:, :, cn_start:cn_end]
        mse = ((pred_tex - ref_tex) ** 2).mean().item()
        psnr = psnr_from_mse(mse)
        print(f"  {tex_type:12s} MSE={mse:.6f} PSNR={psnr:.2f} dB")
        rows.append(
            {
                "grid_level": str(level),
                "lod": str(lod_int),
                "texture": tex_type,
                "mse": f"{mse:.8f}",
                "psnr_db": f"{psnr:.4f}",
                "width": str(lod_w),
                "height": str(lod_h),
            }
        )

        if save_images:
            save_tex = pred_tex.pow(1.0 / 2.2).clamp(0.0, 1.0) if tex_type in srgb_types else pred_tex.clamp(0.0, 1.0)
            save_np = (save_tex.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            if save_np.shape[-1] == 1:
                image = Image.fromarray(save_np.squeeze(-1), mode="L")
            else:
                image = Image.fromarray(save_np, mode="RGB")
            image.save(lod_dir / f"{tex_type}.png")

    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality when forcing one NTC feature grid level.")
    parser.add_argument("--data_dir", required=True, help="Directory containing the original material texture images")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth or another trained checkpoint")
    parser.add_argument("--output_dir", default="./grid_level_eval", help="Directory for reconstructed images and metrics")
    parser.add_argument("--texture_resolution", type=int, default=4096, help="Base texture resolution used for training")
    parser.add_argument("--grid_config", default=None, help="Path to grid_config.json. Defaults to this script's directory")
    parser.add_argument("--grid_level", type=int, default=1, help="Feature grid level to force during evaluation")
    parser.add_argument("--lods", default=None, help="Comma-separated integer LODs. Defaults to the forced grid level's mip range")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--n_frequencies", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_images", action="store_true", help="Only write metrics.csv, do not save reconstructed PNGs")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_config_path = args.grid_config or str(Path(__file__).with_name("grid_config.json"))

    dataset = TextureDataset(data_dir=args.data_dir, device=device)
    dataset.eval()
    print(f"Loaded dataset: {dataset.texture_width}x{dataset.texture_height}, channels={dataset.num_channels}, num_lods={dataset.num_lods}")

    model = LearnableGridNetwork(
        grid_config_path=grid_config_path,
        texture_resolution=args.texture_resolution,
        output_dim=CANONICAL_NUM_CHANNELS,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        n_frequencies=args.n_frequencies,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Level mip ranges: {model.level_mip_ranges}")

    if args.grid_level < 0 or args.grid_level >= len(model.grid_configs):
        raise ValueError(f"--grid_level must be in [0, {len(model.grid_configs) - 1}]")

    if args.lods:
        lod_values = [float(x.strip()) for x in args.lods.split(",") if x.strip()]
    else:
        mip_lo, mip_hi = model.level_mip_ranges[args.grid_level]
        lod_values = [float(lod) for lod in range(int(mip_lo), int(mip_hi))]

    rows: list[dict[str, str]] = []
    for lod_value in lod_values:
        rows.extend(
            evaluate_lod(
                model=model,
                dataset=dataset,
                level=args.grid_level,
                lod_value=lod_value,
                output_dir=output_dir,
                batch_size=args.batch_size,
                save_images=not args.no_images,
            )
        )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["grid_level", "lod", "texture", "mse", "psnr_db", "width", "height"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
