import os
import argparse
import json
import math
from pathlib import Path
import torch
from dataset import (
    TextureDataset,
    CANONICAL_CHANNEL_SLICES,
)
from learnable_grid_network import LearnableGridNetwork


def boundary_loss_weight_config(preset: str) -> dict[str, float]:
    if preset == 'reconstruction':
        return {}
    if preset == 'normal_roughness':
        return {
            'diffuse': 0.5,
            'normal': 2.0,
            'roughness': 4.0,
            'occlusion': 0.25,
            'metallic': 0.5,
            'specular': 0.25,
            'displacement': 0.25,
        }
    if preset == 'roughness':
        return {
            'diffuse': 0.25,
            'normal': 1.0,
            'roughness': 6.0,
            'occlusion': 0.25,
            'metallic': 0.25,
            'specular': 0.25,
            'displacement': 0.25,
        }
    raise ValueError(f"Unknown boundary loss preset: {preset}")


def forward_forced_level(model: LearnableGridNetwork, uv: torch.Tensor, lod_norm: torch.Tensor, level: int) -> torch.Tensor:
    pos_encoding = model._compute_positional_encoding(uv)
    features = model.sample_features(uv, level=level).to(pos_encoding.dtype)
    combined = torch.cat([pos_encoding, features, lod_norm], dim=1)
    return model.network(combined)


def boundary_continuity_loss(
    model: LearnableGridNetwork,
    batch_size: int,
    device: torch.device,
    loss_weights: torch.Tensor,
    band_width: float,
) -> torch.Tensor:
    if not model.level_mip_ranges or len(model.level_mip_ranges) < 2:
        return torch.zeros((), device=device)

    per_boundary = max(1, batch_size // (8 * (len(model.level_mip_ranges) - 1)))
    half_band = max(0.0, float(band_width) * 0.5)
    max_mip = float(model.num_mip_levels - 1)
    losses = []
    for left_level in range(len(model.level_mip_ranges) - 1):
        right_level = left_level + 1
        boundary_mip = float(model.level_mip_ranges[right_level][0])
        if half_band > 0.0:
            mip_min = max(0.0, boundary_mip - half_band)
            mip_max = min(max_mip, boundary_mip + half_band)
            mip_values = torch.empty((per_boundary, 1), device=device).uniform_(mip_min, mip_max)
        else:
            mip_values = torch.full((per_boundary, 1), boundary_mip, device=device)
        uv = torch.rand((per_boundary, 2), device=device)
        lod_norm = mip_values / max_mip
        left = forward_forced_level(model, uv, lod_norm, left_level)
        right = forward_forced_level(model, uv, lod_norm, right_level)
        losses.append(((left - right) ** 2 * loss_weights).sum(dim=1).mean())

    return torch.stack(losses).mean()


def transition_delta_loss(
    model: LearnableGridNetwork,
    dataset: TextureDataset,
    batch_size: int,
    device: torch.device,
    loss_weights: torch.Tensor,
    band_width: float,
) -> torch.Tensor:
    if not model.level_mip_ranges or len(model.level_mip_ranges) < 2:
        return torch.zeros((), device=device)

    half_band = max(0.0, float(band_width) * 0.5)
    if half_band <= 0.0:
        return torch.zeros((), device=device)

    per_boundary = max(1, batch_size // (8 * (len(model.level_mip_ranges) - 1)))
    max_mip = float(model.num_mip_levels - 1)
    losses = []
    for left_level in range(len(model.level_mip_ranges) - 1):
        right_level = left_level + 1
        boundary_mip = float(model.level_mip_ranges[right_level][0])
        offset = torch.rand((per_boundary, 1), device=device) * half_band
        mip_left = torch.clamp(boundary_mip - offset, 0.0, max_mip)
        mip_right = torch.clamp(boundary_mip + offset, 0.0, max_mip)
        uv = torch.rand((per_boundary, 2), device=device)

        left = forward_forced_level(model, uv, mip_left / max_mip, left_level)
        right = forward_forced_level(model, uv, mip_right / max_mip, right_level)

        with torch.no_grad():
            gt_left = dataset.expand_to_canonical(dataset.sample_trilinear_lod(uv, mip_left.squeeze(1)))
            gt_right = dataset.expand_to_canonical(dataset.sample_trilinear_lod(uv, mip_right.squeeze(1)))

        pred_delta = right - left
        gt_delta = gt_right - gt_left
        losses.append(((pred_delta - gt_delta) ** 2 * loss_weights).sum(dim=1).mean())

    return torch.stack(losses).mean()


def main():
    parser = argparse.ArgumentParser(description='Train neural texture compression')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing texture images (diffuse.png, normal.png, ...)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save checkpoints and results')
    parser.add_argument('--texture_resolution', type=int, default=1024,
                        help='Base texture resolution (must match grid_config.json key)')
    parser.add_argument('--grid_config', type=str, default=None,
                        help='Path to grid_config.json. Defaults to the file next to train.py')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Number of random pixel samples per iteration')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Side length of each random training crop')
    parser.add_argument('--crops_per_batch', type=int, default=8,
                        help='Number of random crops per batch; set to 0 for random texels')
    parser.add_argument('--max_iter', type=int, default=20000,
                        help='Total training iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Adam learning rate for feature grids')
    parser.add_argument('--network_lr', type=float, default=0.005,
                        help='Adam learning rate for MLP network')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden layer width')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers')
    parser.add_argument('--n_frequencies', type=int, default=5,
                        help='Number of frequencies for triangle wave positional encoding')
    parser.add_argument('--lr_patience', type=int, default=1000,
                        help='Iterations to wait before LR reduction')
    parser.add_argument('--lr_factor', type=float, default=0.7,
                        help='LR reduction factor')
    parser.add_argument('--lod_sampling', type=str, default='exp',
                        choices=['uniform', 'exp', 'fixed0'],
                        help='LOD sampling strategy: uniform, exp (exponential decay), fixed0')
    parser.add_argument('--mip_target_mode', type=str, default='discrete',
                        choices=['discrete', 'trilinear'],
                        help='Training target mode: discrete integer mips or trilinear continuous LOD targets')
    parser.add_argument('--boundary_continuity_weight', type=float, default=0.0,
                        help='Weight for forcing adjacent grid levels to match at mip boundaries')
    parser.add_argument('--boundary_band_width', type=float, default=0.0,
                        help='Mip interval width around each grid-level boundary for continuity loss. '
                             'For example, 1.0 samples [boundary-0.5, boundary+0.5]. '
                             'The default 0.0 preserves exact-boundary-only training.')
    parser.add_argument('--boundary_loss_preset', type=str, default='normal_roughness',
                        choices=['reconstruction', 'normal_roughness', 'roughness'],
                        help='Channel weights used inside the boundary continuity term')
    parser.add_argument('--boundary_loss_weights', type=str, default=None,
                        help='Optional JSON object overriding boundary channel weights, e.g. {"normal":2,"roughness":5}')
    parser.add_argument('--transition_delta_weight', type=float, default=0.0,
                        help='Weight for matching signed cross-level output jumps to GT mip-chain jumps')
    parser.add_argument('--transition_delta_band_width', type=float, default=1.0,
                        help='Mip interval width around each boundary for transition delta loss. '
                             'For example, 1.0 samples paired transitions like boundary-0.3 -> boundary+0.3.')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate PSNR every N iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (cuda / cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    grid_config_path = args.grid_config or str(Path(__file__).with_name('grid_config.json'))

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = TextureDataset(data_dir=args.data_dir, device=device)
    dataset.eval()

    H = dataset.texture_height
    W = dataset.texture_width
    num_lods = dataset.num_lods
    print(f'Texture: {W}x{H}, loaded_channels={dataset.num_channels}, canonical=11, num_lods={num_lods}')

    # ── Model ────────────────────────────────────────────────────────
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
        max_iter=args.max_iter,
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model trainable params: {total_params:,}')

    # ── Optimizer with separate LRs for grid and network ────────────
    network_params = list(model.network.parameters())
    optimizer_params = [{'params': network_params, 'lr': args.network_lr}]
    for level_key in model.grids:
        for grid in model.grids[level_key]:
            optimizer_params.append({'params': grid.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(optimizer_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=0.0,
    )
    loss_weights = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5], device=device
    )  # [8]: basecolor, metalness, normal, roughness
    boundary_weight_config = boundary_loss_weight_config(args.boundary_loss_preset)
    if args.boundary_loss_weights:
        boundary_weight_config.update(json.loads(args.boundary_loss_weights))
    boundary_loss_weights = loss_weights

    best_psnr = 0.0

    # ── LOD Sampling Weights ─────────────────────────────────────────
    # Exponential decay: LOD 0 gets highest probability, LOD n gets lowest
    lod_probs = torch.exp(-torch.arange(num_lods, dtype=torch.float32) * 1.0)
    lod_probs = lod_probs / lod_probs.sum()
    print(f'LOD sampling ({args.lod_sampling}): probs={lod_probs.cpu().numpy().round(4)}')
    print(f'Mip target mode: {args.mip_target_mode}')
    if args.boundary_continuity_weight > 0.0:
        print(f'Boundary continuity weight: {args.boundary_continuity_weight}')
        print(f'Boundary band width: {args.boundary_band_width}')
        print(f'Boundary channel weights: {boundary_weight_config if boundary_weight_config else "reconstruction"}')
    if args.transition_delta_weight > 0.0:
        print(f'Transition delta weight: {args.transition_delta_weight}')
        print(f'Transition delta band width: {args.transition_delta_band_width}')

    # ── Training Loop ────────────────────────────────────────────────
    for step in range(args.max_iter):
        model.current_iter = step

        # Sample eight random spatial crops by default, matching the paper.
        # A zero crops_per_batch keeps the legacy random-texel fallback.
        if args.crops_per_batch > 0:
            crop_size = min(args.crop_size, H, W)
            crop_count = args.crops_per_batch
            origin_y = torch.randint(0, max(1, H - crop_size + 1), (crop_count,), device=device)
            origin_x = torch.randint(0, max(1, W - crop_size + 1), (crop_count,), device=device)
            local_y = torch.arange(crop_size, device=device).view(1, -1).expand(crop_count, -1)
            local_x = torch.arange(crop_size, device=device).view(1, -1).expand(crop_count, -1)
            ys = (origin_y.view(-1, 1) + local_y).reshape(-1)
            xs = (origin_x.view(-1, 1) + local_x).reshape(-1)
        else:
            ys = torch.randint(0, H, (args.batch_size,), device=device)
            xs = torch.randint(0, W, (args.batch_size,), device=device)
        sample_count = ys.shape[0]

        if args.crops_per_batch > 0:
            if args.lod_sampling == 'fixed0':
                batch_lod = torch.zeros((), dtype=torch.long, device=device)
            elif args.lod_sampling == 'exp':
                if torch.rand(()) < 0.05:
                    batch_lod = torch.randint(0, num_lods, (), device=device)
                else:
                    batch_lod = torch.multinomial(lod_probs, 1).squeeze(0).to(device)
            else:
                batch_lod = torch.randint(0, num_lods, (), device=device)
            lods = torch.full((sample_count,), batch_lod, dtype=torch.long, device=device)
        elif args.lod_sampling == 'fixed0':
            lods = torch.zeros(sample_count, dtype=torch.long, device=device)
        elif args.lod_sampling == 'exp':
            lods = torch.multinomial(lod_probs, sample_count, replacement=True).to(device)
        else:  # uniform
            lods = torch.randint(0, num_lods, (sample_count,), device=device)

        u = xs.float() / W
        v = ys.float() / H
        uv = torch.stack([u, v], dim=1)
        if args.mip_target_mode == 'trilinear' and args.lod_sampling != 'fixed0':
            lod_values = torch.clamp(lods.float() + torch.rand(sample_count, device=device), max=float(num_lods - 1))
        else:
            lod_values = lods.float()

        # Ground truth from dataset's pre-computed mipmap chain
        with torch.no_grad():
            if args.mip_target_mode == 'trilinear':
                gt_data = dataset.sample_trilinear_lod(uv, lod_values)
            else:
                batch_index = torch.stack([ys, xs, lods], dim=1)  # [B, 3]
                gt_data = dataset.sample_discrete_lod(batch_index)
            canonical_gt = dataset.expand_to_canonical(gt_data)  # [B, 11]
            gt = canonical_gt[:, [0, 1, 2, 8, 3, 4, 5, 6]]       # [B, 8]

        # Model input:
        #   dataset pixel coords -> normalized UV [0, 1)
        #   integer or continuous lod -> normalized lod [0, 1]
        lod_norm = lod_values / (num_lods - 1)
        model_input = torch.stack([u, v, lod_norm], dim=1)    # [B, 3]

        pred = model(model_input)                              # [B, 8]

        # Weighted MSE loss
        reconstruction_loss = ((pred - gt) ** 2 * loss_weights).mean()
        if args.boundary_continuity_weight > 0.0:
            continuity_loss = boundary_continuity_loss(
                model,
                sample_count,
                device,
                boundary_loss_weights,
                args.boundary_band_width,
            )
        else:
            continuity_loss = torch.zeros((), device=device)

        if args.transition_delta_weight > 0.0:
            delta_loss = transition_delta_loss(
                model,
                dataset,
                sample_count,
                device,
                boundary_loss_weights,
                args.transition_delta_band_width,
            )
        else:
            delta_loss = torch.zeros((), device=device)

        loss = (
            reconstruction_loss
            + args.boundary_continuity_weight * continuity_loss
            + args.transition_delta_weight * delta_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clamp_value()

        scheduler.step()

        # Keep console output manageable for long 250k-iteration runs while
        # still reporting the optimization progress regularly.
        if step % 10000 == 0 or step == 1:
            current_lr = optimizer.param_groups[0]['lr']
            if args.boundary_continuity_weight > 0.0 or args.transition_delta_weight > 0.0:
                print(
                    f'[{step:5d}/{args.max_iter}]  loss={loss.item():.6f}  '
                    f'recon={reconstruction_loss.item():.6f}  '
                    f'boundary={continuity_loss.item():.6f}  '
                    f'transition={delta_loss.item():.6f}  lr={current_lr:.2e}',
                    flush=True,
                )
            else:
                print(f'[{step:5d}/{args.max_iter}]  loss={loss.item():.6f}  lr={current_lr:.2e}', flush=True)

        # Evaluate full PSNR periodically
        if args.eval_interval > 0 and step % args.eval_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                ys_all = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)
                xs_all = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)
                num_pixels = H * W
                pred_all = torch.zeros((num_pixels, 8), device=device)
                for s in range(0, num_pixels, args.batch_size):
                    e = min(s + args.batch_size, num_pixels)
                    u_b = xs_all[s:e].float() / W
                    v_b = ys_all[s:e].float() / H
                    inp = torch.stack([u_b, v_b, torch.zeros(e - s, device=device)], dim=1)
                    pred_all[s:e] = model(inp)
                pred_all = pred_all.reshape(H, W, 8).cpu()
                pixels_flat = dataset.textures.reshape(-1, dataset.num_channels)
                canonical_ref = dataset.expand_to_canonical(pixels_flat).reshape(H, W, 11)
                ref = canonical_ref[:, :, [0, 1, 2, 8, 3, 4, 5, 6]].cpu()
                avg_mse = ((pred_all - ref) ** 2).mean().item()
                psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 1e-10 else float('inf')
                print(f'  [EVAL]  avg PSNR={psnr:.2f} dB')
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_path = os.path.join(args.output_dir, 'model_best.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f'  [EVAL]  New best model saved: {best_path}')
            model.train()

        if args.save_interval > 0 and step % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f'model_{step}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved checkpoint: {ckpt_path}')

    # Materialize the final 16-bit grid values, freeze them, and adapt only the
    # MLP to the discrete deployment representation for the final 5% steps.
    finetune_steps = max(1, args.max_iter // 20)
    model.quantize_grids_and_freeze()
    model.train()
    for finetune_step in range(finetune_steps):
        sample_count = args.batch_size
        ys = torch.randint(0, H, (sample_count,), device=device)
        xs = torch.randint(0, W, (sample_count,), device=device)
        lods = torch.multinomial(lod_probs, sample_count, replacement=True).to(device)
        u = xs.float() / W
        v = ys.float() / H
        uv = torch.stack([u, v], dim=1)
        lod_values = lods.float()
        with torch.no_grad():
            batch_index = torch.stack([ys, xs, lods], dim=1)
            canonical_gt = dataset.expand_to_canonical(dataset.sample_discrete_lod(batch_index))
            gt = canonical_gt[:, [0, 1, 2, 8, 3, 4, 5, 6]]
        model_input = torch.stack([u, v, lod_values / (num_lods - 1)], dim=1)
        pred = model(model_input)
        loss = ((pred - gt) ** 2 * loss_weights).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # The deployment/evaluation checkpoint must always contain the actual
    # discrete grid values, not an earlier pre-quantization best checkpoint.
    quantized_path = os.path.join(args.output_dir, 'model_quantized.pth')
    best_path = os.path.join(args.output_dir, 'model_best.pth')
    state = model.state_dict()
    torch.save(state, quantized_path)
    torch.save(state, best_path)
    print(f'  -> Saved quantized deployment model: {quantized_path}')
    print(f'  -> Updated evaluation checkpoint: {best_path}')

    print(f'Training complete. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
