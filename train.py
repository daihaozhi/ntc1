import os
import argparse
import math
import torch
from dataset import (
    TextureDataset,
    CANONICAL_CHANNEL_SLICES,
)
from learnable_grid_network import LearnableGridNetwork


def main():
    parser = argparse.ArgumentParser(description='Train neural texture compression')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing texture images (diffuse.png, normal.png, ...)')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save checkpoints and results')
    parser.add_argument('--texture_resolution', type=int, default=1024,
                        help='Base texture resolution (must match grid_config.json key)')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Number of random pixel samples per iteration')
    parser.add_argument('--max_iter', type=int, default=20000,
                        help='Total training iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Adam learning rate for feature grids')
    parser.add_argument('--network_lr', type=float, default=0.002,
                        help='Adam learning rate for MLP network')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden layer width')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers')
    parser.add_argument('--n_frequencies', type=int, default=8,
                        help='Number of frequencies for triangle wave positional encoding')
    parser.add_argument('--lr_patience', type=int, default=2000,
                        help='Iterations to wait before LR reduction')
    parser.add_argument('--lr_factor', type=float, default=0.85,
                        help='LR reduction factor')
    parser.add_argument('--lod_sampling', type=str, default='exp',
                        choices=['uniform', 'exp', 'fixed0'],
                        help='LOD sampling strategy: uniform, exp (exponential decay), fixed0')
    parser.add_argument('--save_interval', type=int, default=2000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate PSNR every N iterations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (cuda / cpu)')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = TextureDataset(data_dir=args.data_dir, device=device)
    dataset.eval()

    H = dataset.texture_height
    W = dataset.texture_width
    num_lods = dataset.num_lods
    print(f'Texture: {W}x{H}, loaded_channels={dataset.num_channels}, canonical=11, num_lods={num_lods}')

    # ── Model ────────────────────────────────────────────────────────
    model = LearnableGridNetwork(
        grid_config_path='grid_config.json',
        texture_resolution=args.texture_resolution,
        output_dim=11,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        n_frequencies=args.n_frequencies,
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_factor, patience=args.lr_patience // 100,
        threshold=1e-4,
    )
    loss_weights = torch.tensor(
        dataset.get_canonical_loss_weights(), device=device
    )  # [11]

    best_psnr = 0.0

    # ── LOD Sampling Weights ─────────────────────────────────────────
    # Exponential decay: LOD 0 gets highest probability, LOD n gets lowest
    lod_probs = torch.exp(-torch.arange(num_lods, dtype=torch.float32) * 1.5)
    lod_probs = lod_probs / lod_probs.sum()
    print(f'LOD sampling ({args.lod_sampling}): probs={lod_probs.cpu().numpy().round(4)}')

    # ── Training Loop ────────────────────────────────────────────────
    for step in range(args.max_iter):
        model.current_iter = step

        # Randomly sample pixel positions and mip levels
        ys = torch.randint(0, H, (args.batch_size,), device=device)
        xs = torch.randint(0, W, (args.batch_size,), device=device)

        if args.lod_sampling == 'fixed0':
            lods = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        elif args.lod_sampling == 'exp':
            lods = torch.multinomial(lod_probs, args.batch_size, replacement=True).to(device)
        else:  # uniform
            lods = torch.randint(0, num_lods, (args.batch_size,), device=device)

        # Ground truth from dataset's pre-computed mipmap chain
        with torch.no_grad():
            batch_index = torch.stack([ys, xs, lods], dim=1)  # [B, 3]
            gt_data = dataset(batch_index)                    # [B, C]
            gt = dataset.expand_to_canonical(gt_data)         # [B, 11]

        # Model input:
        #   dataset pixel coords → normalized UV [0, 1)
        #   dataset integer lod  → normalized lod [0, 1]
        u = xs.float() / W
        v = ys.float() / H
        lod_norm = lods.float() / (num_lods - 1)
        model_input = torch.stack([u, v, lod_norm], dim=1)    # [B, 3]

        pred = model(model_input)                              # [B, 11]

        # Weighted MSE loss
        loss = ((pred - gt) ** 2 * loss_weights).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clamp_value()

        scheduler.step(loss)

        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'[{step:5d}/{args.max_iter}]  loss={loss.item():.6f}  lr={current_lr:.2e}')

        # Evaluate full PSNR periodically
        if step % args.eval_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                ys_all = torch.arange(H, device=device).view(-1, 1).expand(H, W).reshape(-1)
                xs_all = torch.arange(W, device=device).view(1, -1).expand(H, W).reshape(-1)
                num_pixels = H * W
                pred_all = torch.zeros((num_pixels, 11), device=device)
                for s in range(0, num_pixels, args.batch_size):
                    e = min(s + args.batch_size, num_pixels)
                    u_b = xs_all[s:e].float() / W
                    v_b = ys_all[s:e].float() / H
                    inp = torch.stack([u_b, v_b, torch.zeros(e - s, device=device)], dim=1)
                    pred_all[s:e] = model(inp)
                pred_all = pred_all.reshape(H, W, 11).cpu()
                # Compute PSNR on available channels only
                total_mse = 0.0
                total_weight = 0.0
                for tex_type in dataset.available_textures:
                    cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
                    pixels_flat = dataset.textures.reshape(-1, dataset.num_channels)
                    ref = dataset.expand_to_canonical(pixels_flat).reshape(H, W, 11)[:, :, cn_start:cn_end]
                    pr = pred_all[:, :, cn_start:cn_end]
                    mse = ((pr - ref.cpu()) ** 2).mean().item()
                    w = dataset.texture_configs[tex_type].get("loss_weight", 1.0)
                    total_mse += mse * w * (cn_end - cn_start)
                    total_weight += w * (cn_end - cn_start)
                avg_mse = total_mse / total_weight if total_weight > 0 else 0
                psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 1e-10 else float('inf')
                print(f'  [EVAL]  avg PSNR={psnr:.2f} dB')
                if psnr > best_psnr:
                    best_psnr = psnr
                    best_path = os.path.join(args.output_dir, 'model_best.pth')
                    torch.save(model.state_dict(), best_path)
                    print(f'  [EVAL]  New best model saved: {best_path}')
            model.train()

        if step % args.save_interval == 0 or step == args.max_iter - 1:
            ckpt_path = os.path.join(args.output_dir, f'model_{step}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved checkpoint: {ckpt_path}')

    print(f'Training complete. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()
