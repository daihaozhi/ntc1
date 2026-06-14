import os
import argparse
import math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from dataset import (
    TextureDataset,
    CANONICAL_CHANNEL_SLICES,
    CANONICAL_NUM_CHANNELS,
)
from learnable_grid_network import LearnableGridNetwork


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
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='MLP hidden layer width (must match training config)')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers (must match training config)')
    parser.add_argument('--n_frequencies', type=int, default=8,
                        help='Number of frequencies (must match training config)')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Batch size for reconstruction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda / cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    grid_config_path = args.grid_config or str(Path(__file__).with_name('grid_config.json'))

    # ── Load original textures as reference ──────────────────────────
    dataset = TextureDataset(data_dir=args.data_dir, device=device)
    dataset.eval()
    H, W, loaded_C = dataset.textures.shape
    print(f'Original texture: {W}x{H}, loaded_channels={loaded_C}')

    # Reference in canonical 11-channel layout [H, W, 11]
    pixels_flat = dataset.textures.reshape(-1, loaded_C)  # [H*W, C]
    ref_canonical = dataset.expand_to_canonical(pixels_flat).reshape(H, W, CANONICAL_NUM_CHANNELS)

    # ── Load model ───────────────────────────────────────────────────
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
    print(f'Loaded checkpoint: {args.checkpoint}')

    # ── Reconstruct full texture ─────────────────────────────────────
    # Generate all pixel coordinates (lod=0 for highest quality)
    ys = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)
    xs = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)
    num_pixels = H * W

    pred_canonical = torch.zeros((num_pixels, CANONICAL_NUM_CHANNELS), device=device)

    for start in range(0, num_pixels, args.batch_size):
        end = min(start + args.batch_size, num_pixels)
        idx = slice(start, end)
        b = end - start

        u = xs[idx].float() / W
        v = ys[idx].float() / H

        model_input = torch.stack([u, v, torch.zeros(b, device=device)], dim=1)
        pred_canonical[idx] = model(model_input)

    pred_canonical = pred_canonical.reshape(H, W, CANONICAL_NUM_CHANNELS).cpu()
    ref_canonical = ref_canonical.cpu()

    # ── Save each texture type ───────────────────────────────────────
    # Conversion rules:
    #   diffuse: model outputs linear -> convert to sRGB for PNG save
    #   normal:  model outputs [0,1] (sigmoid), save as-is (standard normal map encoding)
    #   others:  linear, save as-is
    srgb_types = {"diffuse"}

    # Replace any NaN / inf with 0 before saving
    pred_canonical = torch.nan_to_num(pred_canonical, nan=0.0, posinf=1.0, neginf=0.0)

    print('\n--- Reconstruction Results ---')
    for tex_type in dataset.available_textures:
        cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
        n_ch = cn_end - cn_start

        pred_tex = pred_canonical[:, :, cn_start:cn_end]
        ref_tex = ref_canonical[:, :, cn_start:cn_end]

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
