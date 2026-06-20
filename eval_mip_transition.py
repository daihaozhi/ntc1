import argparse
import csv
import math
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
def decode_forced_level(
    model: LearnableGridNetwork,
    uv: torch.Tensor,
    mip: float,
    level: int,
    batch_size: int,
) -> torch.Tensor:
    device = uv.device
    out = torch.empty((uv.shape[0], CANONICAL_NUM_CHANNELS), device=device)
    lod_norm_value = float(mip) / float(model.num_mip_levels - 1)
    for start in range(0, uv.shape[0], batch_size):
        end = min(start + batch_size, uv.shape[0])
        lod_norm = torch.full((end - start, 1), lod_norm_value, device=device)
        out[start:end] = model.network(make_forced_level_input(model, uv[start:end], lod_norm, level))
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def parse_transition_specs(raw: str) -> list[tuple[int, int, float, float]]:
    specs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 4:
            raise ValueError(f"Invalid transition spec '{part}', expected left:right:mip_before:mip_after")
        left_level, right_level, mip_before, mip_after = fields
        specs.append((int(left_level), int(right_level), float(mip_before), float(mip_after)))
    if not specs:
        raise ValueError("No transition specs were provided")
    return specs


def random_uv(num_samples: int, device: torch.device) -> torch.Tensor:
    return torch.rand((num_samples, 2), device=device)


def preview_uv(preview_size: int, device: torch.device) -> torch.Tensor:
    ys = torch.arange(preview_size, device=device).view(-1, 1).expand(preview_size, preview_size).reshape(-1)
    xs = torch.arange(preview_size, device=device).view(1, -1).expand(preview_size, preview_size).reshape(-1)
    return torch.stack(
        [
            (xs.float() + 0.5) / float(preview_size),
            (ys.float() + 0.5) / float(preview_size),
        ],
        dim=1,
    )


def save_texture_image(path: Path, tex_type: str, tex: torch.Tensor) -> None:
    save_tex = tex.pow(1.0 / 2.2).clamp(0.0, 1.0) if tex_type == "diffuse" else tex.clamp(0.0, 1.0)
    save_np = (save_tex.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if save_np.shape[-1] == 1:
        image = Image.fromarray(save_np.squeeze(-1), mode="L")
    else:
        image = Image.fromarray(save_np, mode="RGB")
    image.save(path)


@torch.no_grad()
def evaluate_transition(
    model: LearnableGridNetwork,
    dataset: TextureDataset,
    left_level: int,
    right_level: int,
    mip_before: float,
    mip_after: float,
    num_samples: int,
    batch_size: int,
) -> list[dict[str, str]]:
    device = dataset.device
    uv = random_uv(num_samples, device)
    before = decode_forced_level(model, uv, mip_before, left_level, batch_size)
    after = decode_forced_level(model, uv, mip_after, right_level, batch_size)

    gt_before = dataset.expand_to_canonical(
        dataset.sample_trilinear_lod(uv, torch.full((num_samples,), mip_before, device=device))
    )
    gt_after = dataset.expand_to_canonical(
        dataset.sample_trilinear_lod(uv, torch.full((num_samples,), mip_after, device=device))
    )

    rows: list[dict[str, str]] = []
    print(f"\nTransition L{left_level}@mip {mip_before:.3f} -> L{right_level}@mip {mip_after:.3f}")
    for tex_type in dataset.available_textures:
        cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
        before_tex = before[:, cn_start:cn_end]
        after_tex = after[:, cn_start:cn_end]
        gt_before_tex = gt_before[:, cn_start:cn_end]
        gt_after_tex = gt_after[:, cn_start:cn_end]

        before_mse = ((before_tex - gt_before_tex) ** 2).mean().item()
        after_mse = ((after_tex - gt_after_tex) ** 2).mean().item()
        temporal_mse = ((before_tex - after_tex) ** 2).mean().item()
        gt_temporal_mse = ((gt_before_tex - gt_after_tex) ** 2).mean().item()

        print(
            f"  {tex_type:12s} "
            f"before_gt={psnr_from_mse(before_mse):6.2f} dB  "
            f"after_gt={psnr_from_mse(after_mse):6.2f} dB  "
            f"before_after={psnr_from_mse(temporal_mse):6.2f} dB  "
            f"gt_delta={psnr_from_mse(gt_temporal_mse):6.2f} dB"
        )
        rows.append(
            {
                "left_level": str(left_level),
                "right_level": str(right_level),
                "mip_before": f"{mip_before:.6f}",
                "mip_after": f"{mip_after:.6f}",
                "texture": tex_type,
                "before_mse": f"{before_mse:.8f}",
                "before_psnr_db": f"{psnr_from_mse(before_mse):.4f}",
                "after_mse": f"{after_mse:.8f}",
                "after_psnr_db": f"{psnr_from_mse(after_mse):.4f}",
                "before_after_mse": f"{temporal_mse:.8f}",
                "before_after_psnr_db": f"{psnr_from_mse(temporal_mse):.4f}",
                "gt_delta_mse": f"{gt_temporal_mse:.8f}",
                "gt_delta_psnr_db": f"{psnr_from_mse(gt_temporal_mse):.4f}",
                "samples": str(num_samples),
            }
        )
    return rows


@torch.no_grad()
def save_transition_previews(
    model: LearnableGridNetwork,
    dataset: TextureDataset,
    output_dir: Path,
    left_level: int,
    right_level: int,
    mip_before: float,
    mip_after: float,
    preview_size: int,
    batch_size: int,
) -> None:
    device = dataset.device
    uv = preview_uv(preview_size, device)
    before = decode_forced_level(model, uv, mip_before, left_level, batch_size)
    after = decode_forced_level(model, uv, mip_after, right_level, batch_size)
    preview_dir = output_dir / f"transition_L{left_level}_L{right_level}_{mip_before:.2f}_to_{mip_after:.2f}"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for tex_type in dataset.available_textures:
        cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
        before_tex = before[:, cn_start:cn_end].reshape(preview_size, preview_size, cn_end - cn_start).cpu()
        after_tex = after[:, cn_start:cn_end].reshape(preview_size, preview_size, cn_end - cn_start).cpu()
        diff = (before_tex - after_tex).abs()
        diff = diff / max(float(diff.max().item()), 1e-6)
        save_texture_image(preview_dir / f"{tex_type}_before.png", tex_type, before_tex)
        save_texture_image(preview_dir / f"{tex_type}_after.png", tex_type, after_tex)
        save_texture_image(preview_dir / f"{tex_type}_absdiff.png", tex_type, diff)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NTC output jumps when rendering crosses a grid-level mip boundary."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing the original material texture images")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth or another trained checkpoint")
    parser.add_argument("--output_dir", default="./mip_transition_eval")
    parser.add_argument("--texture_resolution", type=int, default=4096)
    parser.add_argument("--grid_config", default=None, help="Path to grid_config.json. Defaults to this script's directory")
    parser.add_argument(
        "--transitions",
        default="0:1:4.7:5.2",
        help="Comma-separated specs left:right:mip_before:mip_after, e.g. 0:1:4.7:5.2",
    )
    parser.add_argument("--num_samples", type=int, default=65536)
    parser.add_argument("--preview_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--n_frequencies", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_images", action="store_true")
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

    rows: list[dict[str, str]] = []
    for left_level, right_level, mip_before, mip_after in parse_transition_specs(args.transitions):
        if left_level < 0 or right_level >= len(model.grid_configs) or left_level >= right_level:
            raise ValueError(f"Invalid transition level pair {left_level}:{right_level}")
        rows.extend(
            evaluate_transition(
                model=model,
                dataset=dataset,
                left_level=left_level,
                right_level=right_level,
                mip_before=mip_before,
                mip_after=mip_after,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
            )
        )
        if not args.no_images:
            save_transition_previews(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                left_level=left_level,
                right_level=right_level,
                mip_before=mip_before,
                mip_after=mip_after,
                preview_size=args.preview_size,
                batch_size=args.batch_size,
            )

    metrics_path = output_dir / "mip_transition_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "left_level",
                "right_level",
                "mip_before",
                "mip_after",
                "texture",
                "before_mse",
                "before_psnr_db",
                "after_mse",
                "after_psnr_db",
                "before_after_mse",
                "before_after_psnr_db",
                "gt_delta_mse",
                "gt_delta_psnr_db",
                "samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote transition metrics: {metrics_path}")


if __name__ == "__main__":
    main()
