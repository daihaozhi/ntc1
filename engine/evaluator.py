"""Evaluation utilities for NTC models."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from engine.dataset import CANONICAL_CHANNEL_SLICES, CANONICAL_NUM_CHANNELS, TextureDataset
from models.learnable_grid_network import LearnableGridNetwork


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / mse) if mse > 1e-10 else float("inf")


def save_texture_image(path: Path, tex_type: str, tex: torch.Tensor) -> None:
    """Save a single-channel or RGB texture as PNG."""
    save_tex = tex.pow(1.0 / 2.2).clamp(0.0, 1.0) if tex_type == "diffuse" else tex.clamp(0.0, 1.0)
    save_np = (save_tex.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    if save_np.shape[-1] == 1:
        Image.fromarray(save_np.squeeze(-1), mode="L").save(path)
    else:
        Image.fromarray(save_np, mode="RGB").save(path)


@torch.no_grad()
def reconstruct_texture(
    model: LearnableGridNetwork,
    dataset: TextureDataset,
    lod_value: float = 0.0,
    batch_size: int = 65536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full texture at given LOD.

    Returns (predicted, reference) as [H, W, num_channels].
    """
    device = dataset.device
    H, W = dataset.texture_height, dataset.texture_width

    # Adjust for LOD
    lod_int = int(round(lod_value))
    lod_h = max(H >> lod_int, 1)
    lod_w = max(W >> lod_int, 1)
    num_pixels = lod_h * lod_w

    ys = torch.arange(lod_h, device=device).view(-1, 1).expand(lod_h, lod_w).reshape(-1)
    xs = torch.arange(lod_w, device=device).view(1, -1).expand(lod_h, lod_w).reshape(-1)

    pred = torch.zeros((num_pixels, CANONICAL_NUM_CHANNELS), device=device)
    lod_norm_value = lod_value / float(model.num_mip_levels - 1)

    for start in range(0, num_pixels, batch_size):
        end = min(start + batch_size, num_pixels)
        idx = slice(start, end)
        uv = torch.stack(
            [(xs[idx].float() + 0.5) / float(lod_w),
             (ys[idx].float() + 0.5) / float(lod_h)], dim=1,
        )
        lod_norm = torch.full((end - start, 1), lod_norm_value, device=device)
        model_input = torch.stack([uv[:, 0], uv[:, 1], lod_norm[:, 0]], dim=1)
        pred[idx] = model(model_input)

    pred = torch.nan_to_num(pred.reshape(lod_h, lod_w, CANONICAL_NUM_CHANNELS).cpu(), nan=0.0, posinf=1.0, neginf=0.0)

    ref_loaded = dataset.lod_cache[lod_int, :lod_h, :lod_w, :].reshape(-1, dataset.num_channels)
    ref = dataset.expand_to_canonical(ref_loaded).reshape(lod_h, lod_w, CANONICAL_NUM_CHANNELS).cpu()

    return pred, ref


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    ref: torch.Tensor,
    dataset: TextureDataset,
) -> list[dict]:
    """Compute per-texture-type PSNR/MSE metrics."""
    rows = []
    for tex_type in dataset.available_textures:
        cn_start, cn_end = CANONICAL_CHANNEL_SLICES[tex_type]
        pred_tex = pred[:, :, cn_start:cn_end]
        ref_tex = ref[:, :, cn_start:cn_end]
        mse = ((pred_tex - ref_tex) ** 2).mean().item()
        psnr = psnr_from_mse(mse)
        rows.append({
            "texture": tex_type,
            "mse": mse,
            "psnr_db": psnr,
        })
    return rows


def save_evaluation_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """Save evaluation metrics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
