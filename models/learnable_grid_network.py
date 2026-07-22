"""Learnable Grid Network — modular NTC model.

Uses pluggable components for positional encoding, MLP decoder, and grid
sampling. Backward-compatible with the original flat-parameter constructor.

Architecture:
    UV (2D) → [PE] → features (12D)
    UV (2D) → [Grid Pyramid, LOD-gated] → features (60D)
    LOD (1D) → passthrough
    → concat → 73D → [MLP] → 8D material channels
"""

from __future__ import annotations

import json
import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False

from models.base import NTCModel
from models.components.mlp import HardGELU, build_mlp
from models.components.pe import build_pe, PositionalEncoding
from models.components.grid_sampler import build_grid_sampler, GridSampler, DualGridSampler


class LearnableGridNetwork(NTCModel):
    """NTC model with learnable feature grids + MLP decoder.

    Uses pluggable PE, MLP, and grid sampler components selectable via
    config dicts or backward-compatible flat parameters.
    """

    model_type = "learnable_grid"

    DEFAULT_GRID_RESOLUTIONS = {
        0: [256, 128],
        1: [64, 32],
        2: [16, 8],
        3: [4, 2],
    }

    def __init__(
        self,
        # Grid configuration
        grid_configs: dict[int, list[dict]] | None = None,
        grid_config_path: str | None = None,
        texture_resolution: int = 1024,
        default_save_bits: int = 192,
        default_quantize_bits: int = 16,
        # Component configs (new, preferred API)
        pe_cfg: dict | None = None,
        mlp_cfg: dict | None = None,
        grid_sampler_cfg: dict | None = None,
        # --- Backward-compatible flat params (mapped to component configs) ---
        output_dim: int = 8,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        quantize: bool = True,
        max_iter: int = 10000,
        qat_noise_schedule: str = "cosine",
        qat_noise_mult_start: float = 1.0,
        qat_noise_mult_end: float = 0.25,
        qat_noise_warmup_frac: float = 0.0,
        use_tiled_encoding: bool = False,
        tile_size: int = 8,
        n_frequencies: int = 5,
        # --- Boundary / wrap constraints ---
        wrap_boundary_constraint: bool = True,
        wrap_boundary_strength: float = 1.0,
        wrap_boundary_interval: int = 16,
    ):
        super().__init__()

        # ── Component configs: new dict API takes precedence over flat params ──
        if pe_cfg is None:
            pe_cfg = {
                "type": "torch_triangle",
                "n_frequencies": n_frequencies,
                "tiled": use_tiled_encoding,
                "tile_size": tile_size,
            }
        if mlp_cfg is None:
            mlp_cfg = {
                "type": "torch_linear",
                "hidden_dim": hidden_dim,
                "num_hidden_layers": num_hidden_layers,
                "output_dim": output_dim,
            }
        if grid_sampler_cfg is None:
            grid_sampler_cfg = {
                "high_res": "corner_four",
                "low_res": "bilinear",
            }

        self.pe_cfg = pe_cfg
        self.mlp_cfg = mlp_cfg
        self.grid_sampler_cfg = grid_sampler_cfg

        # ── Build PE component ─────────────────────────────────────────
        self.pe: PositionalEncoding = build_pe(pe_cfg)
        print(f"[PE] {pe_cfg['type']}: output_dim={self.pe.output_dim}")

        # ── Wrap boundary ───────────────────────────────────────────────
        self.wrap_boundary_constraint = bool(wrap_boundary_constraint)
        self.wrap_boundary_strength = float(wrap_boundary_strength)
        self.wrap_boundary_interval = max(1, int(wrap_boundary_interval))

        # ── Parse grid configuration ────────────────────────────────────
        if grid_config_path is not None:
            with open(grid_config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            tex_key = str(texture_resolution)
            if tex_key not in raw_config:
                raise ValueError(
                    f"Texture resolution {texture_resolution} not found in {grid_config_path}. "
                    f"Available: {list(raw_config.keys())}"
                )
            tex_cfg = raw_config[tex_key]
            parsed = {}
            self.level_mip_ranges = []
            self.num_mip_levels = 0
            for fg_idx, fg_cfg in enumerate(tex_cfg["feature_grids"]):
                mip_lo, mip_hi = fg_cfg["mip_range"]
                self.level_mip_ranges.append([mip_lo, mip_hi])
                self.num_mip_levels = max(self.num_mip_levels, mip_hi)
                level_configs = []
                # Per-feature-grid fields: scalar = shared, list = per-resolution
                resolutions = fg_cfg["resolutions"]
                n_grids = len(resolutions)

                def _per_grid(field, default):
                    """If field is a list, index by grid; if scalar, broadcast."""
                    val = fg_cfg.get(field, default)
                    if isinstance(val, list):
                        if len(val) != n_grids:
                            raise ValueError(
                                f"{field} list length {len(val)} != resolutions count {n_grids}"
                            )
                        return val
                    return [val] * n_grids

                channels_list = _per_grid("channels", None)
                save_bits_list = _per_grid("save_bits", default_save_bits)
                quantize_bits_list = _per_grid("quantize_bits", default_quantize_bits)

                for i, res in enumerate(resolutions):
                    cfg = {"resolution": res}
                    if channels_list[i] is not None:
                        cfg["channels"] = channels_list[i]
                        qb = quantize_bits_list[i]
                        cfg["quantize_bits"] = qb
                        cfg["save_bits"] = channels_list[i] * qb
                    else:
                        cfg["save_bits"] = save_bits_list[i]
                        cfg["quantize_bits"] = quantize_bits_list[i]
                    level_configs.append(cfg)
                parsed[fg_idx] = level_configs
            grid_configs = parsed
        else:
            self.level_mip_ranges = None
            self.num_mip_levels = 0

        self.grid_configs = self._build_grid_configs(
            grid_configs=grid_configs,
            default_save_bits=default_save_bits,
            default_quantize_bits=default_quantize_bits,
        )

        # Identify high-res grid per level
        self.high_res_grid_indices = {}
        for level, grid_cfg_list in self.grid_configs.items():
            if len(grid_cfg_list) >= 2:
                max_res_idx = max(
                    range(len(grid_cfg_list)),
                    key=lambda i: grid_cfg_list[i]["resolution"],
                )
                self.high_res_grid_indices[level] = max_res_idx
            elif len(grid_cfg_list) == 1:
                self.high_res_grid_indices[level] = 0

        # ── State ──────────────────────────────────────────────────
        self.output_dim = output_dim
        self.quantize = quantize
        self.max_iter = max(1, int(max_iter))
        self.current_iter = 0
        self._qat_noise_schedule = str(qat_noise_schedule).strip().lower()
        self._qat_noise_mult_start = float(qat_noise_mult_start)
        self._qat_noise_mult_end = float(qat_noise_mult_end)
        self._qat_noise_warmup_frac = float(qat_noise_warmup_frac)

        # Fallback mip ranges
        if self.level_mip_ranges is None:
            if self.num_mip_levels == 0:
                self.num_mip_levels = 11
            num_fg_levels = len(self.grid_configs)
            self.level_mip_ranges = []
            for l in range(num_fg_levels):
                start = int(round(l * self.num_mip_levels / num_fg_levels))
                end = int(round((l + 1) * self.num_mip_levels / num_fg_levels))
                self.level_mip_ranges.append([start, end])

        # Cache mip boundaries as persistent GPU buffer (avoids CPU tensor
        # allocation inside CUDA graph capture).
        boundaries_list = [r[0] for r in self.level_mip_ranges] + [self.num_mip_levels]
        self.register_buffer(
            '_mip_boundaries',
            torch.tensor(boundaries_list, dtype=torch.float32),
        )

        # ── Build grid samplers ────────────────────────────────────
        high_res_type = grid_sampler_cfg.get("high_res", "corner_four")
        low_res_type = grid_sampler_cfg.get("low_res", "bilinear")
        self.high_res_sampler: GridSampler = build_grid_sampler(high_res_type)
        self.low_res_sampler: GridSampler = build_grid_sampler(low_res_type)
        print(
            f"[GridSampler] high_res={high_res_type} (x{self.high_res_sampler.output_multiplier}), "
            f"low_res={low_res_type} (x{self.low_res_sampler.output_multiplier})"
        )

        # ── Build feature grids ────────────────────────────────────
        self.grids = nn.ModuleDict()
        self.grid_save_bits = nn.ModuleDict()
        self.grid_quantize_bits = nn.ModuleDict()
        self.grid_feature_dims = nn.ModuleDict()
        self.level_feature_dims = []

        for level, grid_cfg_list in self.grid_configs.items():
            level_grids = nn.ModuleList()
            level_save_bits = nn.ParameterList()
            level_quantize_bits = nn.ParameterList()
            level_feature_dims = nn.ParameterList()
            level_dim = 0
            is_dual = isinstance(self.high_res_sampler, DualGridSampler)

            for i, grid_cfg in enumerate(grid_cfg_list):
                resolution = int(grid_cfg["resolution"])
                save_bits = int(grid_cfg["save_bits"])
                quantize_bits = int(grid_cfg["quantize_bits"])
                feature_dim = grid_cfg.get("channels", save_bits // quantize_bits)

                is_high_res = self._is_high_res_grid(level, i)
                if is_dual:
                    # DualGridSampler produces full concatenated output
                    mult = 4 if is_high_res else 1
                else:
                    mult = (
                        self.high_res_sampler.output_multiplier
                        if is_high_res
                        else self.low_res_sampler.output_multiplier
                    )
                total_feature_dim = feature_dim * mult
                level_dim += total_feature_dim

                # High-res grids use Nearest for manual corner sampling;
                # low-res grids use Linear for built-in bilinear.
                interp_mode = "Nearest" if is_high_res else "Linear"

                # tcnn packing: feature_dim must be 1, 2, 4, 8 or multiple of 4
                if feature_dim in (1, 2, 4, 8):
                    packed_levels, packed_features = 1, feature_dim
                elif feature_dim % 4 == 0:
                    packed_levels, packed_features = feature_dim // 4, 4
                else:
                    raise ValueError(
                        f"feature_dim={feature_dim} not supported by tcnn; "
                        "use 1, 2, 4, 8 or a multiple of 4"
                    )

                level_grids.append(
                    tcnn.Encoding(
                        n_input_dims=2,
                        encoding_config={
                            "otype": "Grid",
                            "type": "Dense",
                            "n_levels": packed_levels,
                            "n_features_per_level": packed_features,
                            "base_resolution": resolution,
                            "per_level_scale": 1.0,
                            "interpolation": interp_mode,
                        },
                    )
                )
                level_save_bits.append(nn.Parameter(torch.tensor(float(save_bits)), requires_grad=False))
                level_quantize_bits.append(nn.Parameter(torch.tensor(float(quantize_bits)), requires_grad=False))
                level_feature_dims.append(nn.Parameter(torch.tensor(float(feature_dim)), requires_grad=False))

            self.grids[str(level)] = level_grids
            self.grid_save_bits[str(level)] = level_save_bits
            self.grid_quantize_bits[str(level)] = level_quantize_bits
            self.grid_feature_dims[str(level)] = level_feature_dims
            self.level_feature_dims.append(level_dim)

        # ── Build MLP ──────────────────────────────────────────────
        # If using dual_fused sampler, PE is already included in feature output
        is_dual = isinstance(self.high_res_sampler, DualGridSampler)
        if is_dual:
            for i in range(len(self.level_feature_dims)):
                self.level_feature_dims[i] += self.pe.output_dim

        single_level_dim = self.level_feature_dims[0]
        n_input_dims = single_level_dim + 1  # features (+PE if dual) + LOD
        self.n_input_dims = n_input_dims
        self.mlp = build_mlp(mlp_cfg, input_dim=n_input_dims)
        # Keep .network alias for backward compat (trainer.py references it)
        self.network = self.mlp.network
        print(f"[MLP] {mlp_cfg['type']}: {n_input_dims} → {mlp_cfg.get('hidden_dim', 64)}^"
              f"{mlp_cfg.get('num_hidden_layers', 2)} → {output_dim}")

    # ═══════════════════════════════════════════════════════════════════
    # Grid construction helpers
    # ═══════════════════════════════════════════════════════════════════

    def _build_grid_configs(
        self,
        grid_configs: dict[int, list[dict]] | None,
        default_save_bits: int,
        default_quantize_bits: int,
    ) -> dict[int, list[dict]]:
        if grid_configs is None:
            return {
                level: [
                    {"resolution": res, "save_bits": default_save_bits, "quantize_bits": default_quantize_bits}
                    for res in resolutions
                ]
                for level, resolutions in self.DEFAULT_GRID_RESOLUTIONS.items()
            }

        normalized = {}
        for level_key, cfg_list in grid_configs.items():
            level = int(level_key)
            normalized[level] = []
            for grid_cfg in cfg_list:
                resolution = int(grid_cfg["resolution"])
                save_bits = int(grid_cfg.get("save_bits", default_save_bits))
                quantize_bits = int(grid_cfg.get("quantize_bits", default_quantize_bits))
                channels = grid_cfg.get("channels", None)
                if channels is not None:
                    save_bits = channels * quantize_bits
                if resolution <= 0:
                    raise ValueError(f"resolution must be positive, got {resolution}")
                if save_bits <= 0:
                    raise ValueError(f"save_bits must be positive, got {save_bits}")
                if quantize_bits <= 0:
                    raise ValueError(f"quantize_bits must be positive, got {quantize_bits}")
                if save_bits % quantize_bits != 0:
                    raise ValueError(f"save_bits must be divisible by quantize_bits")
                entry = {
                    "resolution": resolution,
                    "save_bits": save_bits,
                    "quantize_bits": quantize_bits,
                }
                if channels is not None:
                    entry["channels"] = channels
                normalized[level].append(entry)
        return normalized

    def _is_high_res_grid(self, level: int, grid_index: int) -> bool:
        return self.high_res_grid_indices.get(level, 0) == grid_index

    # ═══════════════════════════════════════════════════════════════════
    # QAT helpers
    # ═══════════════════════════════════════════════════════════════════

    def _qat_noise_multiplier(self) -> float:
        if self._qat_noise_schedule == "none":
            return 1.0
        T = max(1, self.max_iter - 1)
        t = min(max(int(self.current_iter), 0), T)
        w = int(self._qat_noise_warmup_frac * T)
        if t <= w:
            return self._qat_noise_mult_start
        p = (t - w) / max(1, T - w)
        c = 0.5 * (1.0 + math.cos(math.pi * p))
        return self._qat_noise_mult_end + (self._qat_noise_mult_start - self._qat_noise_mult_end) * c

    def _simulate_quantize(self, features: torch.Tensor, quantize_bits: int) -> torch.Tensor:
        N_k = 2 ** quantize_bits
        Q_k = 1.0 / N_k
        noise_range = 0.5 * Q_k
        mult = self._qat_noise_multiplier()
        noise = (torch.rand_like(features) * 2 - 1) * noise_range * mult
        return features + noise

    # ═══════════════════════════════════════════════════════════════════
    # Grid sampling
    # ═══════════════════════════════════════════════════════════════════

    def _sample_grid(self, grid, uv: torch.Tensor, resolution: int, is_high_res: bool, feature_dim: int = 0) -> torch.Tensor:
        """Sample a single grid using the configured sampler."""
        if is_high_res:
            return self.high_res_sampler.sample(grid, uv, resolution, feature_dim)
        else:
            return self.low_res_sampler.sample(grid, uv, resolution, feature_dim)

    def sample_features(
        self,
        uv: torch.Tensor,
        level: int | None = None,
        grid_index: int | None = None,
    ) -> torch.Tensor:
        """Sample feature vectors from grids.

        Args:
            uv: [B, 2] normalized coordinates
            level: specific grid level (or None for all)
            grid_index: specific grid within level (or None for all)
        """
        uv = torch.remainder(uv, 1.0)

        if level is not None:
            if level not in self.grid_configs:
                raise ValueError(f"level {level} not in {list(self.grid_configs)}")

            level_grids = self.grids[str(level)]
            level_qbits = self.grid_quantize_bits[str(level)]

            if grid_index is not None:
                grid_cfg = self.grid_configs[level][grid_index]
                resolution = grid_cfg["resolution"]
                is_high_res = self._is_high_res_grid(level, grid_index)
                fdim = int(self.grid_feature_dims[str(level)][grid_index].item())
                feat = self._sample_grid(level_grids[grid_index], uv, resolution, is_high_res, fdim)
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[grid_index].item()))
                return feat

            feats = []
            # Check for dual_fused sampler (fuses PE + high-res + low-res in one kernel)
            if self.grid_sampler_cfg.get("high_res") == "dual_fused":
                hi_idx = self.high_res_grid_indices[level]
                lo_idx = 1 - hi_idx
                cfg_hi = self.grid_configs[level][hi_idx]
                cfg_lo = self.grid_configs[level][lo_idx]
                fdim_hi = int(self.grid_feature_dims[str(level)][hi_idx].item())
                fdim_lo = int(self.grid_feature_dims[str(level)][lo_idx].item())
                feat = DualGridSampler.sample_dual(
                    uv,
                    level_grids[hi_idx], level_grids[lo_idx],
                    cfg_hi["resolution"], fdim_hi,
                    cfg_lo["resolution"], fdim_lo,
                    n_freq=self.pe_cfg.get("n_frequencies", 5),
                    tiled=self.pe_cfg.get("tiled", True),
                    tile_size=self.pe_cfg.get("tile_size", 8),
                )
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[hi_idx].item()))
                return feat

            for i, grid in enumerate(level_grids):
                grid_cfg = self.grid_configs[level][i]
                resolution = grid_cfg["resolution"]
                is_high_res = self._is_high_res_grid(level, i)
                fdim = int(self.grid_feature_dims[str(level)][i].item())
                feat = self._sample_grid(grid, uv, resolution, is_high_res, fdim)
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[i].item()))
                feats.append(feat)
            return torch.cat(feats, dim=1)

        if grid_index is not None:
            raise ValueError("grid_index requires level")

        features = []
        for level_key in self.grids:
            level_int = int(level_key)
            level_grids = self.grids[level_key]
            level_qbits = self.grid_quantize_bits[level_key]

            for i, grid in enumerate(level_grids):
                grid_cfg = self.grid_configs[level_int][i]
                resolution = grid_cfg["resolution"]
                is_high_res = self._is_high_res_grid(level_int, i)
                fdim = int(self.grid_feature_dims[level_key][i].item())
                feat = self._sample_grid(grid, uv, resolution, is_high_res, fdim)
                if self.quantize and self.training:
                    feat = self._simulate_quantize(feat, int(level_qbits[i].item()))
                features.append(feat)
        return torch.cat(features, dim=1)

    def _compute_level_index(self, mip: torch.Tensor) -> torch.Tensor:
        """Map continuous mip → grid level index via bucketize."""
        boundaries = self._mip_boundaries.to(device=mip.device, dtype=torch.float32)
        idx = torch.bucketize(mip, boundaries) - 1
        return idx.clamp(min=0, max=len(self.grid_configs) - 1)

    # ═══════════════════════════════════════════════════════════════════
    # Forward
    # ═══════════════════════════════════════════════════════════════════

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3] — (u, v, lod_normalized) in [0,1) × [0,1) × [0,1]

        Returns:
            [B, output_dim] material channels
        """
        uv = torch.remainder(x[:, 0:2], 1.0)
        lod = x[:, 2:3]

        mip = lod * (self.num_mip_levels - 1)
        level_idx = self._compute_level_index(mip)

        # Grid features — dual_fused already includes PE in output
        is_dual = isinstance(self.high_res_sampler, DualGridSampler)

        if is_dual:
            # PE is fused inside the kernel
            pos_encoding = None
        else:
            pos_encoding = self.pe(uv)

        max_dim = self.level_feature_dims[0]
        features = torch.zeros(
            x.shape[0], max_dim,
            device=x.device, dtype=torch.float32,
        )
        for l in range(len(self.grid_configs)):
            mask = (level_idx.squeeze(1) == l)
            if mask.any():
                feat = self.sample_features(uv[mask], level=l)
                if feat.shape[1] < max_dim:
                    pad = max_dim - feat.shape[1]
                    feat = torch.cat([
                        feat.to(features.dtype),
                        torch.zeros(feat.shape[0], pad, device=feat.device, dtype=features.dtype),
                    ], dim=1)
                features[mask] = feat.to(features.dtype)

        if is_dual:
            combined = torch.cat([features, lod], dim=1)
        else:
            combined = torch.cat([pos_encoding, features, lod], dim=1)

        return self.mlp(combined)

    # ═══════════════════════════════════════════════════════════════════
    # Wrap boundary constraint
    # ═══════════════════════════════════════════════════════════════════

    def clamp_value(self):
        apply_wrap = (
            self.wrap_boundary_constraint
            and self.wrap_boundary_strength > 0.0
            and (self.current_iter % self.wrap_boundary_interval == 0)
        )

        with torch.no_grad():
            for level_key in self.grids:
                level = int(level_key)
                level_grids = self.grids[level_key]
                level_qbits = self.grid_quantize_bits[level_key]

                for i, grid in enumerate(level_grids):
                    qbits = int(level_qbits[i].item())
                    N_k = 2 ** qbits
                    Q_k = 1.0 / N_k
                    min_q = -(N_k - 1) / 2 * Q_k
                    max_q = 0.5

                    params = self._get_grid_params(grid)
                    params.clamp_(min=min_q, max=max_q)

                    if apply_wrap:
                        grid_cfg = self.grid_configs[level][i]
                        resolution = int(grid_cfg["resolution"])
                        fdim = int(self.grid_feature_dims[level_key][i].item())
                        self._enforce_wrap_boundary_constraint_inplace(
                            params,
                            resolution=resolution,
                            feature_dim=fdim,
                            strength=self.wrap_boundary_strength,
                        )

    @torch.no_grad()
    def quantize_grids_and_freeze(self):
        """Materialize final scalar quantization before MLP fine-tuning."""
        for level_key in self.grids:
            for i, grid in enumerate(self.grids[level_key]):
                qbits = int(self.grid_quantize_bits[level_key][i].item())
                n = float(2 ** qbits)
                q = 1.0 / n
                min_q = -((n - 1.0) * 0.5) * q
                params = self._get_grid_params(grid)
                indices = torch.round((params - min_q) / q).clamp(0.0, n - 1.0)
                params.copy_(min_q + indices * q)
                params.requires_grad_(False)

    def _get_grid_params(self, grid: nn.Module) -> nn.Parameter:
        for name, p in grid.named_parameters():
            if name == 'params':
                return p
        return next(grid.parameters())

    def _enforce_wrap_boundary_constraint_inplace(
        self,
        flat_params: torch.Tensor,
        resolution: int,
        feature_dim: int,
        strength: float = 1.0,
    ) -> None:
        strength = float(max(0.0, min(1.0, strength)))
        if strength <= 0.0:
            return

        total = flat_params.numel()
        area = resolution * resolution
        padded_fdim = total // area
        if padded_fdim * area != total:
            raise ValueError(
                f"flat_params size {total} not divisible by resolution^2={area}"
            )

        grid_tex = flat_params.view(resolution, resolution, padded_fdim)
        one_minus = 1.0 - strength

        left, right = grid_tex[:, 0, :].clone(), grid_tex[:, -1, :].clone()
        lr_avg = 0.5 * (left + right)
        grid_tex[:, 0, :] = left * one_minus + lr_avg * strength
        grid_tex[:, -1, :] = right * one_minus + lr_avg * strength

        top, bottom = grid_tex[0, :, :].clone(), grid_tex[-1, :, :].clone()
        tb_avg = 0.5 * (top + bottom)
        grid_tex[0, :, :] = top * one_minus + tb_avg * strength
        grid_tex[-1, :, :] = bottom * one_minus + tb_avg * strength

        # Note: corner constraint removed. Edge matching (left=right, top=bottom)
        # already guarantees seamless wrapping at corners by transitivity.
        # Explicitly forcing all four corners identical creates triangular artifacts
        # when the texture content differs between corners.

    # ═══════════════════════════════════════════════════════════════════
    # Backward-compatible attribute access
    # ═══════════════════════════════════════════════════════════════════

    def _compute_positional_encoding(self, uv: torch.Tensor) -> torch.Tensor:
        """Backward-compatible PE accessor (used by eval scripts)."""
        return self.pe(uv)

    # ═══════════════════════════════════════════════════════════════════
    # Grid info accessors (used by export.py)
    # ═══════════════════════════════════════════════════════════════════

    def get_grid_params(self, level: int, grid_index: int) -> nn.Parameter:
        if level not in self.grid_configs:
            raise ValueError(f"level must be one of {list(self.grid_configs.keys())}, got {level}")
        grid = self.grids[str(level)][grid_index]
        for name, param in grid.named_parameters():
            if name == "params":
                return param
        return next(grid.parameters())

    def get_grid_config(self, level: int, grid_index: int) -> dict[str, int]:
        if level not in self.grid_configs:
            raise ValueError(f"level must be one of {list(self.grid_configs.keys())}, got {level}")
        return dict(self.grid_configs[level][grid_index])


# ═══════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = LearnableGridNetwork(
        grid_config_path="configs/grid/grid_1024.json",
        texture_resolution=1024,
        pe_cfg={"type": "torch_triangle", "n_frequencies": 5, "tiled": True, "tile_size": 8},
        mlp_cfg={"type": "torch_linear", "hidden_dim": 64, "num_hidden_layers": 2, "output_dim": 8},
        grid_sampler_cfg={"high_res": "corner_four", "low_res": "bilinear"},
    ).cuda()

    x = torch.rand(8, 3, device="cuda")
    out = model(x)
    print(f"Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    print(f"PE type: {model.pe_cfg['type']}, MLP type: {model.mlp_cfg['type']}")
    print(f"Samplers: high_res={model.grid_sampler_cfg['high_res']}, low_res={model.grid_sampler_cfg['low_res']}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test backward compat
    model2 = LearnableGridNetwork(
        use_tiled_encoding=True,
        n_frequencies=5,
        hidden_dim=64,
        num_hidden_layers=2,
        output_dim=8,
        default_save_bits=192,
        default_quantize_bits=16,
    )
    print(f"Backward-compat model created: PE={model2.pe_cfg}, MLP={model2.mlp_cfg}")
