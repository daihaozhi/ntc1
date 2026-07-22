import os
import argparse
import math
import os
from pathlib import Path as Path
import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from engine.dataset import (
    TextureDataset,
    CANONICAL_CHANNEL_SLICES,
    CANONICAL_NUM_CHANNELS,
)
from models.learnable_grid_network import LearnableGridNetwork


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Reconstruct textures from trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing original texture images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='./reconstructed',
                        help='Directory to save reconstructed textures')
    parser.add_argument('--texture_resolution', type=int, default=1024,
                        help='Base texture resolution (must match training config)')
    parser.add_argument('--grid_config', type=str, default=None,
                        help='Path to grid_config.json. Defaults to the file next to inference.py')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden layer width (must match training config)')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers (must match training config)')
    parser.add_argument('--n_frequencies', type=int, default=5,
                        help='Number of frequencies (must match training config)')
    parser.add_argument('--tiled', type=str, default='true',
                        choices=['true', 'false'],
                        help='Tiled positional encoding (must match training config)')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Batch size for reconstruction')
    parser.add_argument('--mlp_type', type=str, default='torch_linear',
                        choices=['torch_linear', 'tcnn_cutlass'],
                        help='MLP type (must match training config)')
    parser.add_argument('--grid_sampler_type', type=str, default='corner_four',
                        choices=['corner_four', 'bilinear', 'custom_cuda', 'dual_fused'],
                        help='Grid sampler type (must match training config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    grid_config_path = args.grid_config or str(_Path(__file__).resolve().parents[1] / 'configs' / 'grid' / f'grid_{args.texture_resolution}.json')

    # ── Load original textures as reference ──────────────────────────
    dataset = TextureDataset(data_dir=args.data_dir, device=device)
    dataset.eval()
    H, W, loaded_C = dataset.textures.shape
    print(f'Original texture: {W}x{H}, loaded_channels={loaded_C}')

    # Reference in canonical 11-channel layout [H, W, 11]
    pixels_flat = dataset.textures.reshape(-1, loaded_C)  # [H*W, C]
    ref_canonical = dataset.expand_to_canonical(pixels_flat).reshape(H, W, CANONICAL_NUM_CHANNELS)
    ref_material = ref_canonical[:, :, [0, 1, 2, 8, 3, 4, 5, 6]]

    # ── Load model ───────────────────────────────────────────────────
    model = LearnableGridNetwork(
        grid_config_path=grid_config_path,
        texture_resolution=args.texture_resolution,
        pe_cfg={"type": "torch_triangle", "n_frequencies": args.n_frequencies, "tiled": args.tiled == 'true', "tile_size": 8},
        mlp_cfg={"type": args.mlp_type, "hidden_dim": args.hidden_dim, "num_hidden_layers": args.num_hidden_layers, "output_dim": 8},
        grid_sampler_cfg={"high_res": args.grid_sampler_type, "low_res": "bilinear"},
        output_dim=8,
        default_save_bits=48,
        default_quantize_bits=4,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    print(f'Loaded checkpoint: {args.checkpoint}')

    # ── Reconstruct full texture ─────────────────────────────────────
    # Generate all pixel coordinates (lod=0 for highest quality)
    ys = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)
    xs = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)
    num_pixels = H * W

    pred_material = torch.zeros((num_pixels, 8), device=device)

    for start in range(0, num_pixels, args.batch_size):
        end = min(start + args.batch_size, num_pixels)
        idx = slice(start, end)
        b = end - start

        u = xs[idx].float() / W
        v = ys[idx].float() / H

        model_input = torch.stack([u, v, torch.zeros(b, device=device)], dim=1)
        pred_material[idx] = model(model_input)

    pred_material = pred_material.reshape(H, W, 8).cpu()
    ref_material = ref_material.cpu()

    # ── Save each texture type ───────────────────────────────────────
    # Conversion rules:
    #   diffuse: model outputs linear -> convert to sRGB for PNG save
    #   normal:  model outputs [0,1] (sigmoid), save as-is (standard normal map encoding)
    #   others:  linear, save as-is
    srgb_types = {"diffuse"}

    # Replace any NaN / inf with 0 before saving
    pred_material = torch.nan_to_num(pred_material, nan=0.0, posinf=1.0, neginf=0.0)

    print('\n--- Reconstruction Results ---')
    output_slices = {
        'diffuse': (0, 3),
        'metallic': (3, 4),
        'normal': (4, 7),
        'roughness': (7, 8),
    }
    for tex_type, (cn_start, cn_end) in output_slices.items():
        if tex_type not in dataset.available_textures and tex_type != 'metallic':
            continue
        n_ch = cn_end - cn_start

        pred_tex = pred_material[:, :, cn_start:cn_end]
        ref_tex = ref_material[:, :, cn_start:cn_end]

        if tex_type in srgb_types:
            save_tex = pred_tex.pow(1.0 / 2.2).clamp(0.0, 1.0)
        else:
            save_tex = pred_tex.clamp(0.0, 1.0)

        save_np = (save_tex.numpy() * 255).round().clip(0, 255).astype(np.uint8)

        if n_ch == 1:
            img = Image.fromarray(save_np.squeeze(-1), mode='L')
        else:
            img = Image.fromarray(save_np, mode='RGB')

        save_path = os.path.join(args.output_dir, f'{tex_type}.png')
        img.save(save_path)
        print(f'  Saved: {save_path}')

        # PSNR
        mse = ((pred_tex - ref_tex) ** 2).mean().item()
        psnr = 10 * math.log10(1.0 / mse) if mse > 1e-10 else float('inf')
        print(f'    {tex_type}: MSE={mse:.6f}, PSNR={psnr:.2f} dB')

    print(f'\nAll textures saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
