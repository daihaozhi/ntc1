"""Unified training loop for NTC models.

Model-agnostic: works with any model that implements the NTCModel interface.
Supports optional CUDA graph capture for reduced kernel launch overhead.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Optional

import torch

from engine.dataset import TextureDataset
from models.base import NTCModel


def normal_angular_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """FNTC-style angular loss for XYZ normal maps encoded in [0, 1]."""
    pred_n = pred * 2.0 - 1.0
    target_n = target * 2.0 - 1.0
    pred_n = pred_n / torch.sqrt(torch.clamp((pred_n ** 2).sum(dim=1, keepdim=True), min=eps))
    target_n = target_n / torch.sqrt(torch.clamp((target_n ** 2).sum(dim=1, keepdim=True), min=eps))
    dot = (pred_n * target_n).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - dot).mean()


class Trainer:
    """Model-agnostic training loop for NTC.

    Args:
        use_cuda_graph: If True, wraps the model with torch.compile(mode='reduce-overhead')
            which uses CUDA graphs internally for compatible subgraphs, reducing
            kernel launch overhead without requiring manual graph management.
    """

    def __init__(
        self,
        model: NTCModel,
        dataset: TextureDataset,
        *,
        # Optimization
        lr: float = 0.01,
        network_lr: float = 0.005,
        max_iter: int = 40000,
        train_steps: int | None = None,
        scheduler_t_max: int | None = None,
        quantized_finetune_steps: int | None = None,
        loss_mode: str = "mse",
        texture_loss_weights: dict[str, float] | None = None,
        use_cuda_graph: bool = False,
        graph_pool_steps: int = 50,
        # Data
        batch_size: int = 65536,
        crop_size: int = 256,
        crops_per_batch: int = 8,
        # LoD
        lod_sampling: str = "exp",
        mip_target_mode: str = "discrete",
        # Regularization
        boundary_continuity_weight: float = 0.0,
        boundary_band_width: float = 0.0,
        boundary_loss_preset: str = "normal_roughness",
        transition_delta_weight: float = 0.0,
        transition_delta_band_width: float = 1.0,
        # Logging
        eval_interval: int = 1000,
        save_interval: int = 5000,
        output_dir: str = "./output",
        # Device
        device: torch.device = None,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_iter = max_iter
        self.train_steps = int(train_steps if train_steps is not None else max_iter)
        if self.train_steps < 1 or self.train_steps > self.max_iter:
            raise ValueError("train_steps must be in [1, max_iter]")
        self.scheduler_t_max = int(scheduler_t_max or max_iter)
        self.quantized_finetune_steps = (
            max_iter // 20 if quantized_finetune_steps is None
            else max(0, int(quantized_finetune_steps))
        )
        self.loss_mode = str(loss_mode).strip().lower().replace("-", "_")
        if self.loss_mode in {"plain", "plain_mse", "mse_only"}:
            self.loss_mode = "mse"
        elif self.loss_mode in {"fntc_grouped", "fntc_normal", "grouped"}:
            self.loss_mode = "fntc"
        if self.loss_mode not in {"mse", "fntc"}:
            raise ValueError("loss_mode must be 'mse' or 'fntc'")
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.crops_per_batch = crops_per_batch
        self.lod_sampling = lod_sampling
        self.mip_target_mode = mip_target_mode
        self.boundary_continuity_weight = boundary_continuity_weight
        self.boundary_band_width = boundary_band_width
        self.boundary_loss_preset = boundary_loss_preset
        self.transition_delta_weight = transition_delta_weight
        self.transition_delta_band_width = transition_delta_band_width
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)

        self.H = dataset.texture_height
        self.W = dataset.texture_width
        self.num_lods = dataset.num_lods
        self.output_dim = dataset.model_output_dim

        # Build optimizer
        self._build_optimizer(lr, network_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.scheduler_t_max, eta_min=0.0,
        )

        # Keep both historical plain-MSE weights and FNTC grouped weights.
        # Neither mode changes the model or compression budget.
        if self.loss_mode == "mse":
            default_weights = {
                "diffuse": 1.0, "metallic": 0.5, "normal": 1.0,
                "roughness": 0.5, "occlusion": 0.5,
                "displacement": 0.5, "specular": 0.5,
            }
        else:
            default_weights = {
                "diffuse": 1.0, "normal": 0.3, "roughness": 0.2,
                "occlusion": 0.2, "metallic": 0.2,
                "specular": 0.2, "displacement": 0.2,
            }
        self.texture_loss_weights = dict(default_weights)
        if texture_loss_weights:
            self.texture_loss_weights.update(
                {str(k): float(v) for k, v in texture_loss_weights.items()}
            )
        self.loss_weights = torch.tensor(
            [self.texture_loss_weights[name]
             for name in dataset.model_channel_types],
            device=self.device,
        )
        print(f"Loss: {self.loss_mode} ({self.texture_loss_weights})")

        # LoD sampling probabilities
        lod_probs = torch.exp(-torch.arange(self.num_lods, dtype=torch.float32, device=self.device) * 1.0)
        self.lod_probs = lod_probs / lod_probs.sum()

        self.best_psnr = 0.0
        self._step_times: list[float] = []

        # ── torch.compile support ───────────────────────────────
        self.use_cuda_graph = use_cuda_graph and self.device.type == "cuda"
        if self.use_cuda_graph and hasattr(torch, 'compile'):
            print('[torch.compile] Compiling model with mode="reduce-overhead"...')
            self.model = torch.compile(self.model, mode="reduce-overhead")
            # Warmup compile
            N = self._get_sample_count()
            dummy = torch.randn(N, 3, device=self.device)
            for _ in range(3):
                _ = self.model(dummy)
                _ = ((_ - torch.randn(N, self.output_dim, device=self.device)) ** 2).mean()
            torch.cuda.synchronize()
            print('[torch.compile] Compile warmup complete.')

    def _build_optimizer(self, lr: float, network_lr: float) -> None:
        """Build optimizer with separate LRs for MLP and grids."""
        network_params = list(self.model.network.parameters())
        optimizer_params = [{"params": network_params, "lr": network_lr}]

        # Grid parameters — works for LearnableGridNetwork
        if hasattr(self.model, "grids"):
            for level_key in self.model.grids:
                for grid in self.model.grids[level_key]:
                    optimizer_params.append({"params": grid.parameters(), "lr": lr})
        elif hasattr(self.model, "hash_grids"):
            for hash_grid in self.model.hash_grids:
                optimizer_params.append({"params": hash_grid.parameters(), "lr": lr})

        self.optimizer = torch.optim.Adam(optimizer_params)

    # ── Data sampling ────────────────────────────────────────────────

    def _sample_coords(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample pixel coordinates and LOD values.

        Returns (u, v, ys, xs, lods, lod_values):
            u, v: normalized pixel-center coordinates [(0.5/W), (W-0.5)/W]
            ys, xs: integer pixel indices
            lods:    integer LOD levels
            lod_values: continuous LOD values for model input
        """
        if self.crops_per_batch > 0:
            crop_size = min(self.crop_size, self.H, self.W)
            crop_count = self.crops_per_batch
            origin_y = torch.randint(0, max(1, self.H - crop_size + 1), (crop_count,), device=self.device)
            origin_x = torch.randint(0, max(1, self.W - crop_size + 1), (crop_count,), device=self.device)
            # Build the full Cartesian crop.  Pairing two 1-D ranges here only
            # samples the crop diagonal and leaves two large triangular regions
            # of the texture almost entirely unsupervised.
            local_y = torch.arange(crop_size, device=self.device).view(1, crop_size, 1)
            local_x = torch.arange(crop_size, device=self.device).view(1, 1, crop_size)
            ys = (
                origin_y.view(-1, 1, 1) + local_y
            ).expand(-1, -1, crop_size).reshape(-1)
            xs = (
                origin_x.view(-1, 1, 1) + local_x
            ).expand(-1, crop_size, -1).reshape(-1)
        else:
            ys = torch.randint(0, self.H, (self.batch_size,), device=self.device)
            xs = torch.randint(0, self.W, (self.batch_size,), device=self.device)

        sample_count = ys.shape[0]

        # LOD sampling
        if self.crops_per_batch > 0:
            if self.lod_sampling == "fixed0":
                batch_lod = torch.zeros((), dtype=torch.long, device=self.device)
            elif self.lod_sampling == "exp":
                if torch.rand((), device=self.device) < 0.05:
                    batch_lod = torch.randint(0, self.num_lods, (), device=self.device)
                else:
                    batch_lod = torch.multinomial(self.lod_probs, 1).squeeze(0)
            else:
                batch_lod = torch.randint(0, self.num_lods, (), device=self.device)
            lods = torch.full((sample_count,), batch_lod, dtype=torch.long, device=self.device)
        elif self.lod_sampling == "fixed0":
            lods = torch.zeros(sample_count, dtype=torch.long, device=self.device)
        elif self.lod_sampling == "exp":
            lods = torch.multinomial(self.lod_probs, sample_count, replacement=True)
        else:
            lods = torch.randint(0, self.num_lods, (sample_count,), device=self.device)

        # Use texel centers consistently for both training and evaluation.
        u = (xs.float() + 0.5) / self.W
        v = (ys.float() + 0.5) / self.H

        if self.mip_target_mode == "trilinear" and self.lod_sampling != "fixed0":
            lod_values = torch.clamp(lods.float() + torch.rand(sample_count, device=self.device), max=float(self.num_lods - 1))
        else:
            lod_values = lods.float()

        return u, v, ys, xs, lods, lod_values

    def _get_ground_truth(self, uv: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, lods: torch.Tensor, lod_values: torch.Tensor) -> torch.Tensor:
        """Sample ground-truth from the dataset's mipmap chain."""
        with torch.no_grad():
            if self.mip_target_mode == "trilinear":
                gt_data = self.dataset.sample_trilinear_lod(uv, lod_values)
            else:
                batch_index = torch.stack([ys, xs, lods], dim=1)
                gt_data = self.dataset.sample_discrete_lod(batch_index)
            canonical_gt = self.dataset.expand_to_canonical(gt_data)
            # Select the output channels represented by this dataset.
            return canonical_gt[:, self.dataset.model_channel_indices]

    # ── Training step ─────────────────────────────────────────────────

    def _get_sample_count(self) -> int:
        """Number of samples per training step."""
        if self.crops_per_batch > 0:
            crop_size = min(self.crop_size, self.H, self.W)
            return crop_size * crop_size * self.crops_per_batch
        return self.batch_size

    def _compute_mse_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Historical NTC loss: weighted per-channel mean squared error."""
        return ((pred.float() - gt.float()).pow(2) * self.loss_weights).mean()

    def _compute_fntc_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """FNTC loss: diffuse/ROMD MSE and angular XYZ normal loss."""
        groups = {
            "diffuse": ["diffuse"],
            "normal": ["normal"],
            "romd": ["roughness", "occlusion", "metallic", "specular", "displacement"],
        }
        total = pred.new_zeros(())
        for group_name, texture_names in groups.items():
            indices = []
            for tex_type in texture_names:
                if tex_type in self.dataset.model_texture_slices:
                    start, end = self.dataset.model_texture_slices[tex_type]
                    indices.extend(range(start, end))
            if not indices:
                continue

            idx = torch.tensor(indices, device=pred.device, dtype=torch.long)
            group_pred = pred.index_select(1, idx).float()
            group_gt = gt.index_select(1, idx).float()
            weights = self.loss_weights.index_select(0, idx).float()

            if group_name == "normal":
                # For XYZ normals, compare directions instead of RGB values.
                total = total + normal_angular_loss(group_pred, group_gt) * weights.sum()
            else:
                channel_mse = (group_pred - group_gt).pow(2).mean(dim=0)
                total = total + (channel_mse * weights).sum()
        return total

    def _compute_reconstruction_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "mse":
            return self._compute_mse_loss(pred, gt)
        return self._compute_fntc_loss(pred, gt)

    def train_step(self) -> dict[str, float]:
        """Single training step. Returns dict of metrics."""
        t0 = time.perf_counter()

        u, v, ys, xs, lods, lod_values = self._sample_coords()
        uv = torch.stack([u, v], dim=1)

        gt = self._get_ground_truth(uv, ys, xs, lods, lod_values)

        # Model input: (u, v, lod_normalized)
        lod_norm = lod_values / (self.num_lods - 1)
        model_input = torch.stack([u, v, lod_norm], dim=1)

        pred = self.model(model_input)
        reconstruction_loss = self._compute_reconstruction_loss(pred, gt)

        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        self.optimizer.step()
        self.model.clamp_value()

        self.scheduler.step()
        self.model.current_iter += 1

        t1 = time.perf_counter()
        self._step_times.append(t1 - t0)

        return {
            "loss": reconstruction_loss.item(),
            "recon": reconstruction_loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # ── Evaluation ────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Full-image PSNR evaluation at LOD 0."""
        self.model.eval()

        ys_all = torch.arange(self.H, device=self.device).view(-1, 1).expand(self.H, self.W).reshape(-1)
        xs_all = torch.arange(self.W, device=self.device).view(1, -1).expand(self.H, self.W).reshape(-1)
        num_pixels = self.H * self.W
        pred_all = torch.zeros((num_pixels, self.output_dim), device=self.device)

        for s in range(0, num_pixels, self.batch_size):
            e = min(s + self.batch_size, num_pixels)
            u_b = (xs_all[s:e].float() + 0.5) / self.W
            v_b = (ys_all[s:e].float() + 0.5) / self.H
            inp = torch.stack([u_b, v_b, torch.zeros(e - s, device=self.device)], dim=1)
            pred_all[s:e] = self.model(inp)

        # Match the stored/reconstructed texture domain used for PSNR.
        pred_all = pred_all.reshape(self.H, self.W, self.output_dim).cpu().clamp(0.0, 1.0)
        pixels_flat = self.dataset.textures.reshape(-1, self.dataset.num_channels)
        canonical_ref = self.dataset.expand_to_canonical(pixels_flat).reshape(self.H, self.W, 11)
        ref = canonical_ref[:, :, self.dataset.model_channel_indices].cpu()

        channel_mse = ((pred_all - ref) ** 2).mean(dim=(0, 1))
        channel_names = self.dataset.model_channel_names
        channel_psnr = {
            name: 10 * math.log10(1.0 / float(mse)) if float(mse) > 1e-10 else float("inf")
            for name, mse in zip(channel_names, channel_mse)
        }

        avg_mse = channel_mse.mean().item()
        avg_psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 1e-10 else float("inf")
        channel_psnr["avg"] = avg_psnr

        self.model.train()
        return channel_psnr

    # ── Checkpointing ─────────────────────────────────────────────────

    def save_checkpoint(self, name: str = "model_best.pth") -> str:
        path = self.output_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_checkpoint(str(path))
        return str(path)

    # ── Full training loop ────────────────────────────────────────────

    def run(self) -> dict:
        """Run the full training loop. Returns summary dict."""
        print(f"Training: {self.model.model_type}, {self.W}x{self.H}, {self.num_lods} LODs")
        print(f"Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"LoD sampling: {self.lod_sampling}, target: {self.mip_target_mode}")

        t_start = time.perf_counter()

        for step in range(self.train_steps):
            metrics = self.train_step()

            # Logging
            if step % 10000 == 0 or step <= 1:
                print(
                    f"[{step:5d}/{self.train_steps}]  "
                    f"loss={metrics['loss']:.6f}  lr={metrics['lr']:.2e}",
                    flush=True,
                )

            # Evaluation
            if self.eval_interval > 0 and step > 0 and step % self.eval_interval == 0:
                psnr_dict = self.evaluate()
                print(
                    "  [EVAL]  " + "  ".join(
                        f"{k}={v:.2f} dB" for k, v in psnr_dict.items()
                    ),
                    flush=True,
                )
                if psnr_dict["avg"] > self.best_psnr:
                    self.best_psnr = psnr_dict["avg"]
                    self.save_checkpoint("model_best.pth")
                    print(f"  [EVAL]  New best model saved (PSNR={self.best_psnr:.2f})")

            # Checkpoint
            if self.save_interval > 0 and step > 0 and step % self.save_interval == 0:
                self.save_checkpoint(f"model_{step}.pth")
                print(f"  -> Saved checkpoint: model_{step}.pth")

        # Optional final quantization/fine-tuning. A short baseline can
        # disable this so the requested train_steps is exact.
        self._finetune_quantized()

        t_end = time.perf_counter()
        total_time = t_end - t_start

        # Save final
        self.save_checkpoint("model_final.pth")

        summary = {
            "model_type": self.model.model_type,
            "resolution": f"{self.W}x{self.H}",
            "max_iter": self.max_iter,
            "train_steps": self.train_steps,
            "loss_mode": self.loss_mode,
            "best_psnr": self.best_psnr,
            "total_time_min": total_time / 60,
            "avg_step_ms": sum(self._step_times) / len(self._step_times) * 1000 if self._step_times else 0,
            "params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        print(f"Training complete. Best PSNR: {self.best_psnr:.2f} dB, Time: {total_time / 60:.1f} min")
        return summary

    def _finetune_quantized(self) -> None:
        """Quantize grid and fine-tune MLP for the final 5% of steps."""
        if hasattr(self.model, "quantize_grids_and_freeze") and self.quantized_finetune_steps > 0:
            finetune_steps = self.quantized_finetune_steps
            self.model.quantize_grids_and_freeze()
            self.model.train()
            for _ in range(finetune_steps):
                u, v, ys, xs, lods, lod_values = self._sample_coords()
                uv = torch.stack([u, v], dim=1)
                gt = self._get_ground_truth(uv, ys, xs, lods, lod_values)
                lod_norm = lod_values / (self.num_lods - 1)
                model_input = torch.stack([u, v, lod_norm], dim=1)
                pred = self.model(model_input)
                loss = self._compute_reconstruction_loss(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
