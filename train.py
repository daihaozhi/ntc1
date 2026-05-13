import os
import argparse
import torch
from dataset import TextureDataset
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
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Total training iterations')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Adam learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='MLP hidden layer width')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='Number of MLP hidden layers')
    parser.add_argument('--n_frequencies', type=int, default=8,
                        help='Number of frequencies for triangle wave positional encoding')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save checkpoint every N iterations')
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
    ).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model trainable params: {total_params:,}')

    # ── Optimizer & Loss Weights ─────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_weights = torch.tensor(
        dataset.get_canonical_loss_weights(), device=device
    )  # [11]

    # ── Training Loop ────────────────────────────────────────────────
    for step in range(args.max_iter):
        model.current_iter = step

        # Randomly sample pixel positions and mip levels
        ys = torch.randint(0, H, (args.batch_size,), device=device)
        xs = torch.randint(0, W, (args.batch_size,), device=device)
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

        if step % 100 == 0:
            print(f'[{step:5d}/{args.max_iter}]  loss = {loss.item():.6f}')

        if step % args.save_interval == 0 or step == args.max_iter - 1:
            ckpt_path = os.path.join(args.output_dir, f'model_{step}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> Saved checkpoint: {ckpt_path}')

    print('Training complete.')


if __name__ == '__main__':
    main()
