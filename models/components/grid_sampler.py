"""Grid sampler components for NTC.

Controls how feature vectors are read from tcnn Grid encodings.
"""

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError:
    pass


class GridSampler(nn.Module):
    """Abstract grid sampling strategy.

    Each grid sampler defines how to read features from one tcnn Grid encoding
    given UV coordinates.

    Subclasses must define:
        output_multiplier: int — how many feature vectors per grid query
        sample(grid, uv, resolution) → [B, feature_dim * output_multiplier]
    """

    output_multiplier: int = 1

    def sample(
        self,
        grid,
        uv: torch.Tensor,
        resolution: int,
        feature_dim: int = 0,
    ) -> torch.Tensor:
        """Sample features from the grid.

        Args:
            grid: tcnn.Encoding object
            uv: [B, 2] in [0, 1)
            resolution: grid resolution
            feature_dim: logical feature dimension (for custom CUDA kernels)

        Returns:
            [B, feature_dim * output_multiplier]
        """
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════
# Bilinear interpolation (standard, used for low-res grids)
# ═════════════════════════════════════════════════════════════════════════

class BilinearSampler(GridSampler):
    """Standard bilinear interpolation via tcnn (fast CUDA path)."""

    output_multiplier = 1

    def sample(self, grid, uv: torch.Tensor, resolution: int, feature_dim: int = 0) -> torch.Tensor:
        return grid(uv)


# ═════════════════════════════════════════════════════════════════════════
# Corner-four nearest sampling (used for high-res grids)
# ═════════════════════════════════════════════════════════════════════════

class CornerFourSampler(GridSampler):
    """Sample 4 corner points from a dense grid (nearest, no interpolation).

    For each UV coordinate, returns features at:
        (floor_u, floor_v), (ceil_u, floor_v), (floor_u, ceil_v), (ceil_u, ceil_v)

    This captures more spatial information than bilinear at the cost of 4x
    separate tcnn queries (one per corner).
    """

    output_multiplier = 4

    def sample(self, grid, uv: torch.Tensor, resolution: int, feature_dim: int = 0) -> torch.Tensor:
        batch_size = uv.shape[0]

        # Convert UV to grid coordinates
        grid_uv = uv * resolution  # [B, 2]
        floor_u = torch.floor(grid_uv[:, 0]).long()
        floor_v = torch.floor(grid_uv[:, 1]).long()
        ceil_u = (floor_u + 1) % resolution
        ceil_v = (floor_v + 1) % resolution

        # 4 corner UVs in normalized space
        corners_uv = torch.stack([
            torch.stack([floor_u.float() / resolution, floor_v.float() / resolution], dim=1),
            torch.stack([ceil_u.float() / resolution,  floor_v.float() / resolution], dim=1),
            torch.stack([floor_u.float() / resolution, ceil_v.float() / resolution],  dim=1),
            torch.stack([ceil_u.float() / resolution,  ceil_v.float() / resolution],  dim=1),
        ], dim=1)  # [B, 4, 2]

        corners_flat = corners_uv.reshape(-1, 2)  # [B*4, 2]
        corner_features = grid(corners_flat)       # [B*4, feature_dim]

        feature_dim = corner_features.shape[1]
        corner_features = corner_features.reshape(batch_size, 4, feature_dim)
        return corner_features.reshape(batch_size, -1)  # [B, 4 * feature_dim]


# ═════════════════════════════════════════════════════════════════════════
# Fused corner-four (optimization: single query + manual indexing)
# ═════════════════════════════════════════════════════════════════════════

class FusedCornerFourSampler(GridSampler):
    """Custom CUDA kernel: single fused corner-four grid lookup.

    Replaces tcnn grid() entirely with a dedicated CUDA kernel that:
    - Computes 4 corner indices from UV
    - Looks up grid values directly from flat parameter tensor
    - Handles multi-level packed grids
    - Implements custom backward pass

    This eliminates tcnn dispatch overhead and intermediate tensor
    allocations, and reduces Python-side torch.stack/reshape costs.
    """

    output_multiplier = 4

    def sample(self, grid, uv: torch.Tensor, resolution: int, feature_dim: int = 0) -> torch.Tensor:
        from models.components.corner_lookup_cuda import corner_four_lookup

        # Get grid params from tcnn's state_dict
        params = None
        for name, p in grid.named_parameters():
            if name == 'params':
                params = p
                break
        if params is None:
            params = next(grid.parameters())

        # Compute n_levels and n_features from feature_dim and resolution
        # tcnn packs feature_dim into: n_levels * n_features_per_level = feature_dim
        # For feature_dim <= 8: n_levels=1, n_features=feature_dim
        # For feature_dim % 4 == 0: n_features=4, n_levels=feature_dim//4
        if feature_dim <= 8:
            n_levels = 1
            n_features = feature_dim
        else:
            n_features = 4
            n_levels = feature_dim // 4

        return corner_four_lookup(uv, params, resolution, n_levels, n_features)


# ═════════════════════════════════════════════════════════════════════════
# Dual-grid fused sampler (single kernel for high-res + low-res)
# ═════════════════════════════════════════════════════════════════════════

class DualGridSampler(GridSampler):
    """Fused dual-grid: PE + high-res corner-four + low-res bilinear in one kernel.

    Single kernel launch replaces:
      1. Triangle-wave positional encoding
      2. High-res corner-four grid lookup
      3. Low-res bilinear grid lookup
      4. torch.cat of all three

    Output: [B, pe_dim + 4*fdim_high + fdim_low]
    """

    output_multiplier = 0  # Not used; per-call output size varies

    @staticmethod
    def sample_dual(uv, grid_high, grid_low, res_high, fdim_high, res_low, fdim_low,
                    n_freq=5, tiled=True, tile_size=8):
        from models.components.dual_grid_lookup_cuda import pe_dual_grid_lookup

        def _params(g):
            for n, p in g.named_parameters():
                if n == 'params':
                    return p
            return next(g.parameters())

        return pe_dual_grid_lookup(
            uv,
            _params(grid_high), _params(grid_low),
            n_freq, tiled, tile_size,
            res_high, fdim_high, res_low, fdim_low,
        )


# ═════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════

_SAMPLER_REGISTRY = {
    "bilinear": BilinearSampler,
    "corner_four": CornerFourSampler,
    "fused_corner_four": FusedCornerFourSampler,
    "custom_cuda": FusedCornerFourSampler,
    "dual_fused": DualGridSampler,
}


def build_grid_sampler(sampler_type: str) -> GridSampler:
    """Build a grid sampler instance.

    Args:
        sampler_type: one of "bilinear", "corner_four", "fused_corner_four"
    """
    if sampler_type not in _SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler type: {sampler_type}. "
            f"Available: {list(_SAMPLER_REGISTRY)}"
        )
    return _SAMPLER_REGISTRY[sampler_type]()
