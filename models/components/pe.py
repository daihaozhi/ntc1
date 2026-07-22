"""Positional Encoding components for NTC.

All PE implementations map UV coordinates → fixed-dimensional feature vectors.
"""

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False


class PositionalEncoding(nn.Module):
    """Abstract PE interface.

    Subclasses must define self.output_dim and implement forward(uv).
    """

    def __init__(self, n_frequencies: int):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.output_dim: int = n_frequencies * 2 + 2  # f × (sin+cos-equivalent) + 2 constants

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Map UV to positional encoding.

        Args:
            uv: [B, 2] in range [0, 1).

        Returns:
            [B, output_dim] feature vector.
        """
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════
# PyTorch Triangle Wave (current baseline)
# ═════════════════════════════════════════════════════════════════════════

class TorchTriangleWavePE(PositionalEncoding):
    """Pure PyTorch triangle-wave positional encoding with optional tiling.

    Divides texture into tile_size × tile_size blocks and computes
    local triangle-wave frequencies within each tile.
    """

    def __init__(
        self,
        n_frequencies: int = 5,
        tiled: bool = True,
        tile_size: int = 8,
    ):
        super().__init__(n_frequencies)
        self.tiled = tiled
        self.tile_size = tile_size

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Compute triangle-wave encoding.

        For each frequency i, computes:
            tri(2^i * local_uv)
        where local_uv is the fractional position within each tile (if tiled)
        or global UV (if not tiled).
        """
        if self.tiled:
            local_uv = torch.remainder(uv * self.tile_size, 1.0)
        else:
            local_uv = uv

        batch_size = local_uv.shape[0]
        encoded = []

        for i in range(self.n_frequencies):
            freq_scale = 2.0 ** i
            scaled = local_uv * freq_scale
            # triangle wave: tri(x) = 2 * |x - floor(x + 0.5)|
            triangle = 2.0 * torch.abs(scaled - torch.floor(scaled + 0.5))
            encoded.append(triangle)

        encoded = torch.cat(encoded, dim=1)  # [B, n_frequencies * 2]
        constants = torch.ones((batch_size, 2), device=uv.device, dtype=uv.dtype)
        return torch.cat([encoded, constants], dim=1)


# ═════════════════════════════════════════════════════════════════════════
# tcnn Triangle Wave (CUDA-optimized)
# ═════════════════════════════════════════════════════════════════════════

class TcnnTriangleWavePE(PositionalEncoding):
    """tcnn CUDA-accelerated triangle-wave encoding.

    Much faster than PyTorch version, but requires tinycudann.

    NOTE: tcnn TriangleWave does NOT support tiled encoding natively.
    Tiling must be done as a pre-processing step on UV before encoding,
    or this PE should be used with tiled=False.
    """

    def __init__(
        self,
        n_frequencies: int = 5,
        tiled: bool = False,
        tile_size: int = 8,
    ):
        if not HAS_TCNN:
            raise ImportError(
                "tinycudann is required for TcnnTriangleWavePE. "
                "Install with: pip install tinycudann"
            )
        super().__init__(n_frequencies)
        self.tiled = tiled
        self.tile_size = tile_size
        self._encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "n_dims_to_encode": 2,
                "otype": "TriangleWave",
                "n_frequencies": n_frequencies,
            },
        )
        # tcnn TriangleWave produces 2*n_frequencies channels.
        # We add 2 constant channels to match TorchTriangleWavePE's output_dim.
        self.output_dim = n_frequencies * 2 + 2

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        if self.tiled:
            uv = torch.remainder(uv * self.tile_size, 1.0)

        encoded = self._encoding(uv)  # [B, n_frequencies * 2]
        constants = torch.ones((uv.shape[0], 2), device=uv.device, dtype=uv.dtype)
        return torch.cat([encoded, constants], dim=1)


# ═════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════

_PE_REGISTRY = {
    "torch_triangle": TorchTriangleWavePE,
    "tcnn_triangle": TcnnTriangleWavePE,
}


def build_pe(cfg: dict) -> PositionalEncoding:
    """Build positional encoding from config dict.

    Example:
        cfg = {"type": "torch_triangle", "n_frequencies": 5, "tiled": True, "tile_size": 8}
    """
    pe_type = cfg.get("type", "torch_triangle")
    if pe_type not in _PE_REGISTRY:
        raise ValueError(f"Unknown PE type: {pe_type}. Available: {list(_PE_REGISTRY)}")

    cls = _PE_REGISTRY[pe_type]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return cls(**kwargs)
