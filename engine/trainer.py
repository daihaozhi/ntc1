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


class Trainer:
    """Model-agnostic training loop for NTC.

    Args:
        use_cuda_graph: If True, captures the training step as a CUDA graph
            after warmup, eliminating per-step kernel launch overhead.
            Requires fixed batch size and static model forward.
        graph_pool_steps: Number of pre-generated steps in the coordinate pool.
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

        # Build optimizer
        self._build_optimizer(lr, network_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_iter, eta_min=0.0,
        )

        # Loss weights: basecolor(3) + metalness(1) + normal(3) + roughness(1) = 8
        self.loss_weights = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5],
            device=self.device,
        )

        # LoD sampling probabilities
        lod_probs = torch.exp(-torch.arange(self.num_lods, dtype=torch.float32, device=self.device) * 1.0)
        self.lod_probs = lod_probs / lod_probs.sum()

        self.best_psnr = 0.0
        self._step_times: list[float] = []

        # ── CUDA Graph support ─────────────────────────────────
        self.use_cuda_graph = use_cuda_graph and self.device.type == "cuda"
        self.graph_pool_steps = graph_pool_steps
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_static_input: torch.Tensor | None = None
        self._graph_static_gt: torch.Tensor | None = None
        self._graph_pool_input: torch.Tensor | None = None   # [pool_steps, B, 3]
        self._graph_pool_gt: torch.Tensor | None = None       # [pool_steps, B, 8]
        self._graph_pool_idx: int = 0
        self._graph_captured: bool = False

    def _build_optimizer(self, lr: float, network_lr: float) -> None:
        """Build optimizer with separate LRs for MLP and grids."""
        network_params = list(self.model.network.parameters())
        optimizer_params = [{"params": network_params, "lr": network_lr}]

        # Grid parameters — works for both LearnableGridNetwork and TCNNModel
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
            u, v:  normalized [0,1) coordinates
            ys, xs: integer pixel indices
            lods:    integer LOD levels
            lod_values: continuous LOD values for model input
        """
        if self.crops_per_batch > 0:
            crop_size = min(self.crop_size, self.H, self.W)
            crop_count = self.crops_per_batch
            origin_y = torch.randint(0, max(1, self.H - crop_size + 1), (crop_count,), device=self.device)
            origin_x = torch.randint(0, max(1, self.W - crop_size + 1), (crop_count,), device=self.device)
            local_y = torch.arange(crop_size, device=self.device).view(1, -1).expand(crop_count, -1)
            local_x = torch.arange(crop_size, device=self.device).view(1, -1).expand(crop_count, -1)
            ys = (origin_y.view(-1, 1) + local_y).reshape(-1)
            xs = (origin_x.view(-1, 1) + local_x).reshape(-1)
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

        u = xs.float() / self.W
        v = ys.float() / self.H

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
            # Select 8 output channels: basecolor.rgb, metallic, normal.rgb, roughness
            return canonical_gt[:, [0, 1, 2, 8, 3, 4, 5, 6]]

    # ── CUDA Graph support ────────────────────────────────────────

    def _get_sample_count(self) -> int:
        """Number of samples per training step (fixed for graph capture)."""
        if self.crops_per_batch > 0:
            crop_size = min(self.crop_size, self.H, self.W)
            return crop_size * crop_size * self.crops_per_batch
        return self.batch_size

    def _refill_pool(self) -> None:
        """Pre-generate a pool of (model_input, ground_truth) pairs."""
        assert self._graph_pool_input is not None and self._graph_pool_gt is not None
        N = self._get_sample_count()
        for i in range(self.graph_pool_steps):
            u, v, ys, xs, lods, lod_values = self._sample_coords()
            uv = torch.stack([u, v], dim=1)
            gt = self._get_ground_truth(uv, ys, xs, lods, lod_values)
            lod_norm = lod_values / (self.num_lods - 1)
            self._graph_pool_input[i] = torch.stack([u, v, lod_norm], dim=1)
            self._graph_pool_gt[i] = gt
        self._graph_pool_idx = 0
        torch.cuda.synchronize()

    def _init_cuda_graph(self) -> None:
        """Warmup and capture the training step as a CUDA graph."""
        if self._graph_captured:
            return

        N = self._get_sample_count()
        print(f"[CUDA Graph] Capturing training step (sample_count={N:,}, pool={self.graph_pool_steps} steps)...")

        # Allocate static tensors and pool
        self._graph_static_input = torch.empty(N, 3, device=self.device, dtype=torch.float32)
        self._graph_static_gt = torch.empty(N, 8, device=self.device, dtype=torch.float32)
        self._graph_pool_input = torch.empty(self.graph_pool_steps, N, 3, device=self.device, dtype=torch.float32)
        self._graph_pool_gt = torch.empty(self.graph_pool_steps, N, 8, device=self.device, dtype=torch.float32)

        # Fill pool
        self._refill_pool()

        # Warmup: run a few real steps to autotune cuDNN etc.
        for _ in range(3):
            inp = self._graph_pool_input[self._graph_pool_idx]
            gt = self._graph_pool_gt[self._graph_pool_idx]
            self._graph_pool_idx += 1
            self._graph_static_input.copy_(inp)
            self._graph_static_gt.copy_(gt)
            pred = self.model(self._graph_static_input)
            loss = ((pred - self._graph_static_gt) ** 2 * self.loss_weights).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.model.clamp_value()
        torch.cuda.synchronize()

        # Reset pool for capture
        self._graph_pool_idx = 0
        self._refill_pool()

        # Capture: model forward + loss + backward + optimizer step
        self._graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self._graph):
            pred = self.model(self._graph_static_input)
            loss = ((pred - self._graph_static_gt) ** 2 * self.loss_weights).mean()
            loss.backward()
            self.optimizer.step()

        self._graph_captured = True
        print("[CUDA Graph] Capture complete.")

    def _train_step_cuda_graph(self) -> dict[str, float]:
        """Training step using CUDA graph replay."""
        # Refill pool if exhausted
        if self._graph_pool_idx >= self.graph_pool_steps:
            self._refill_pool()

        # Copy new data into static buffers
        inp = self._graph_pool_input[self._graph_pool_idx]
        gt = self._graph_pool_gt[self._graph_pool_idx]
        self._graph_pool_idx += 1
        self._graph_static_input.copy_(inp)
        self._graph_static_gt.copy_(gt)

        # Replay captured graph (forward + loss + backward + optimizer)
        self._graph.replay()

        # Steps that must run outside the graph
        self.model.clamp_value()
        self.scheduler.step()
        self.model.current_iter += 1

        return {}  # loss values from graph are not accessible without a pool copy

    # ── Training step ─────────────────────────────────────────────────

    def train_step(self) -> dict[str, float]:
        """Single training step. Returns dict of metrics."""
        # Initialize CUDA graph on first call if enabled
        if self.use_cuda_graph and not self._graph_captured:
            self._init_cuda_graph()

        if self._graph_captured:
            t0 = time.perf_counter()
            metrics = self._train_step_cuda_graph()
            t1 = time.perf_counter()
            self._step_times.append(t1 - t0)
            lr = self.optimizer.param_groups[0]["lr"]
            metrics.setdefault("loss", 0.0)
            metrics.setdefault("recon", 0.0)
            metrics.setdefault("lr", lr)
            return metrics

        t0 = time.perf_counter()

        u, v, ys, xs, lods, lod_values = self._sample_coords()
        uv = torch.stack([u, v], dim=1)

        gt = self._get_ground_truth(uv, ys, xs, lods, lod_values)

        # Model input: (u, v, lod_normalized)
        lod_norm = lod_values / (self.num_lods - 1)
        model_input = torch.stack([u, v, lod_norm], dim=1)

        pred = self.model(model_input)
        reconstruction_loss = ((pred - gt) ** 2 * self.loss_weights).mean()

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
        pred_all = torch.zeros((num_pixels, 8), device=self.device)

        for s in range(0, num_pixels, self.batch_size):
            e = min(s + self.batch_size, num_pixels)
            u_b = xs_all[s:e].float() / self.W
            v_b = ys_all[s:e].float() / self.H
            inp = torch.stack([u_b, v_b, torch.zeros(e - s, device=self.device)], dim=1)
            pred_all[s:e] = self.model(inp)

        pred_all = pred_all.reshape(self.H, self.W, 8).cpu()
        pixels_flat = self.dataset.textures.reshape(-1, self.dataset.num_channels)
        canonical_ref = self.dataset.expand_to_canonical(pixels_flat).reshape(self.H, self.W, 11)
        ref = canonical_ref[:, :, [0, 1, 2, 8, 3, 4, 5, 6]].cpu()

        channel_mse = ((pred_all - ref) ** 2).mean(dim=(0, 1))
        channel_names = [
            "basecolor.r", "basecolor.g", "basecolor.b", "metallic",
            "normal.r", "normal.g", "normal.b", "roughness",
        ]
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

        for step in range(self.max_iter):
            metrics = self.train_step()

            # Logging
            if step % 10000 == 0 or step <= 1:
                print(
                    f"[{step:5d}/{self.max_iter}]  "
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

        # Final: quantize and fine-tune MLP
        self._finetune_quantized()

        t_end = time.perf_counter()
        total_time = t_end - t_start

        # Save final
        self.save_checkpoint("model_final.pth")

        summary = {
            "model_type": self.model.model_type,
            "resolution": f"{self.W}x{self.H}",
            "max_iter": self.max_iter,
            "best_psnr": self.best_psnr,
            "total_time_min": total_time / 60,
            "avg_step_ms": sum(self._step_times) / len(self._step_times) * 1000 if self._step_times else 0,
            "params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

        print(f"Training complete. Best PSNR: {self.best_psnr:.2f} dB, Time: {total_time / 60:.1f} min")
        return summary

    def _finetune_quantized(self) -> None:
        """Quantize grid and fine-tune MLP for the final 5% of steps."""
        if hasattr(self.model, "quantize_grids_and_freeze"):
            finetune_steps = max(1, self.max_iter // 20)
            self.model.quantize_grids_and_freeze()
            self.model.train()
            for _ in range(finetune_steps):
                u, v, ys, xs, lods, lod_values = self._sample_coords()
                uv = torch.stack([u, v], dim=1)
                gt = self._get_ground_truth(uv, ys, xs, lods, lod_values)
                lod_norm = lod_values / (self.num_lods - 1)
                model_input = torch.stack([u, v, lod_norm], dim=1)
                pred = self.model(model_input)
                loss = ((pred - gt) ** 2 * self.loss_weights).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
