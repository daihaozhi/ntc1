import os
import json
import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from learnable_grid_network import LearnableGridNetwork


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Export feature grids to RGBA8 textures and MLP weights to .bin')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='./exported',
                        help='Directory to save exported files')
    parser.add_argument('--texture_resolution', type=int, default=1024,
                        help='Base texture resolution')
    parser.add_argument('--grid_config', type=str, default=None,
                        help='Path to grid_config.json. Defaults to the file next to export.py')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden layer width (must match training)')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers (must match training)')
    parser.add_argument('--n_frequencies', type=int, default=5,
                        help='Number of frequencies (must match training)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for loading model')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    grid_config_path = args.grid_config or str(Path(__file__).with_name('grid_config.json'))

    model = LearnableGridNetwork(
        grid_config_path=grid_config_path,
        texture_resolution=args.texture_resolution,
        output_dim=8,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        n_frequencies=args.n_frequencies,
        use_tiled_encoding=True,
        default_save_bits=192,
        default_quantize_bits=16,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    print(f'Loaded checkpoint: {args.checkpoint}')

    metadata = {
        'texture_resolution': args.texture_resolution,
        'hidden_dim': args.hidden_dim,
        'num_hidden_layers': args.num_hidden_layers,
        'n_frequencies': args.n_frequencies,
        'output_dim': 8,
        'n_input_dims': model.network.n_input_dims,
        'positional_encoding_dims': 12,
        'lod_dims': 1,
        'grid_feature_dim': 12,
        'large_samples_per_level': 4,
        'small_samples_per_level': 1,
        'quantize_bits': 16,
        'channel_order': [
            'basecolor_r', 'basecolor_g', 'basecolor_b',
            'metalness',
            'normal_r', 'normal_g', 'normal_b',
            'roughness',
        ],
        'feature_grids': [],
        'mlp_layers': [],
    }

    # ══════════════════════════════════════════════════════════════════
    # 1. Export feature grids as R8G8B8A8 PNG textures
    # ══════════════════════════════════════════════════════════════════
    print('\n=== Exporting Feature Grids ===')

    for level in range(len(model.grid_configs)):
        grid_cfg_list = model.grid_configs[level]
        for grid_idx in range(len(grid_cfg_list)):
            cfg = grid_cfg_list[grid_idx]
            resolution = cfg['resolution']
            qbits = cfg['quantize_bits']
            sbits = cfg['save_bits']
            feature_dim = sbits // qbits

            grid = model.grids[str(level)][grid_idx]
            params = grid.state_dict()['params'].cpu()  # flat tensor

            # Reshape to [H, W, feature_dim]
            if params.numel() != resolution * resolution * feature_dim:
                raise ValueError(
                    f'Level {level} Grid {grid_idx}: expected {resolution}x{resolution}x{feature_dim} '
                    f'= {resolution*resolution*feature_dim}, got {params.numel()} params'
                )
            params = params.reshape(resolution, resolution, feature_dim)

            # Quantize float params -> uint16 [0, 2^qbits - 1]
            N_k = 2 ** qbits
            min_q = -(N_k - 1) / 2 * (1.0 / N_k)
            ints = torch.round((params + (-min_q)) * N_k)
            ints = torch.clamp(ints, min=0, max=N_k - 1).to(torch.int32)

            # Split 16-bit values into hi / lo bytes
            ints_hi = (ints >> 8).to(torch.uint8)
            ints_lo = (ints & 0xFF).to(torch.uint8)

            # Pack up to 4 feature channels per RGBA set
            # Each set produces 2 PNGs: _hi.png (high bytes) + _lo.png (low bytes)
            num_sets = (feature_dim + 3) // 4

            for s in range(num_sets):
                ch_start = s * 4
                ch_end = min(ch_start + 4, feature_dim)
                ch_count = ch_end - ch_start

                rgba_hi = torch.zeros(resolution, resolution, 4, dtype=torch.uint8)
                rgba_lo = torch.zeros(resolution, resolution, 4, dtype=torch.uint8)
                rgba_hi[:, :, :ch_count] = ints_hi[:, :, ch_start:ch_end]
                rgba_lo[:, :, :ch_count] = ints_lo[:, :, ch_start:ch_end]

                set_suffix = f'_set{s}' if num_sets > 1 else ''
                name = f'grid_L{level}_G{grid_idx}_r{resolution}{set_suffix}'

                png_hi_path = os.path.join(args.output_dir, f'{name}_hi.png')
                png_lo_path = os.path.join(args.output_dir, f'{name}_lo.png')
                Image.fromarray(rgba_hi.numpy(), mode='RGBA').save(png_hi_path)
                Image.fromarray(rgba_lo.numpy(), mode='RGBA').save(png_lo_path)

                metadata['feature_grids'].append({
                    'name': name,
                    'png_hi_file': f'{name}_hi.png',
                    'png_lo_file': f'{name}_lo.png',
                    'level': level,
                    'grid_index': grid_idx,
                    'resolution': resolution,
                    'channels': feature_dim,
                    'channel_start': ch_start,
                    'channel_count': ch_count,
                    'quantize_bits': qbits,
                    'save_bits': sbits,
                    'is_high_res': model._is_high_res_grid(level, grid_idx),
                })
                print(f'  {name}_hi.png  +  {name}_lo.png  ({resolution}x{resolution}, channels [{ch_start}:{ch_end}))')

    # ══════════════════════════════════════════════════════════════════
    # 2. Export MLP weights as .bin (float32, row-major)
    # ══════════════════════════════════════════════════════════════════
    print('\n=== Exporting MLP Weights ===')

    mlp_params = dict(model.network.named_parameters())
    mlp_state = model.network.state_dict()

    weight_count = 0
    for name, tensor in mlp_state.items():
        t = tensor.cpu().to(torch.float32)
        bin_path = os.path.join(args.output_dir, f'mlp_{name}.bin')
        t.numpy().tofile(bin_path)

        shape = list(t.shape)
        numel = t.numel()
        weight_count += numel
        metadata['mlp_layers'].append({
            'name': f'mlp_{name}',
            'bin_file': f'mlp_{name}.bin',
            'shape': shape,
            'numel': numel,
        })
        print(f'  mlp_{name}.bin  shape={shape}  ({numel} floats, {numel * 4} bytes)')

    metadata['mlp_total_floats'] = weight_count
    metadata['mlp_total_bytes'] = weight_count * 4

    # ══════════════════════════════════════════════════════════════════
    # 3. Save metadata
    # ══════════════════════════════════════════════════════════════════
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nMetadata: {meta_path}')

    # ══════════════════════════════════════════════════════════════════
    # 4. Print Vulkan binding summary
    # ══════════════════════════════════════════════════════════════════
    print('\n=== Vulkan Descriptor Set Layout ===')
    print('Feature Grids (Combined Image Samplers, 16-bit stored as hi/lo byte pair):')
    for fg in metadata['feature_grids']:
        print(f'  binding  →  {fg["png_hi_file"]} + {fg["png_lo_file"]}  (RGBA8×2, {fg["resolution"]}x{fg["resolution"]}, channels=[{fg["channel_start"]}:{fg["channel_start"]+fg["channel_count"]})')
    print(f'\nMLP Weights (Storage Buffer / Uniform Buffer):')
    print(f'  Total: {weight_count} floats = {weight_count * 4} bytes')
    for layer in metadata['mlp_layers']:
        print(f'  {layer["name"]}.bin  —  {layer["shape"]}  ({layer["numel"]} floats)')

    print(f'\nAll files exported to: {args.output_dir}')


if __name__ == '__main__':
    main()
