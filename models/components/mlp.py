"""MLP decoder components for NTC."""

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
    HAS_TCNN = True
except ImportError:
    HAS_TCNN = False


class HardGELU(nn.Module):
    """Hard-GELU activation: piecewise quadratic approximation.

    hardGELU(x) = 0          if x < -1.5
                  x          if x >  1.5
                  x/3*(x+1.5) otherwise
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(
            x < -1.5,
            torch.zeros_like(x),
            torch.where(x > 1.5, x, (x / 3.0) * (x + 1.5)),
        )


class MLPDecoder(nn.Module):
    """Abstract MLP decoder interface.

    Subclasses must set self.input_dim, self.output_dim and implement forward().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════
# PyTorch MLP with HardGELU (current baseline)
# ═════════════════════════════════════════════════════════════════════════

class TorchMLP(MLPDecoder):
    """Pure PyTorch MLP: Linear → HardGELU → Linear → HardGELU → Linear.

    Uses nn.Sequential for simplicity. Compatible with torch.compile().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        output_dim: int = 8,
        compile: bool = False,
    ):
        super().__init__(input_dim, hidden_dim, num_hidden_layers, output_dim)

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), HardGELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

        if compile and hasattr(torch, 'compile'):
            self.network = torch.compile(self.network, mode="reduce-overhead")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═════════════════════════════════════════════════════════════════════════
# tcnn CutlassMLP (CUDA-optimized)
# ═════════════════════════════════════════════════════════════════════════

class TcnnCutlassMLP(MLPDecoder):
    """tcnn CutlassMLP: fused CUDA kernels for Linear layers.

    Much faster than PyTorch nn.Linear, but:
    - Only supports ReLU / LeakyReLU / None activations (no HardGELU).
    - Uses LeakyReLU as the closest available approximation.
    - Input dimension is padded to 16-aligned internally.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        output_dim: int = 8,
    ):
        if not HAS_TCNN:
            raise ImportError(
                "tinycudann is required for TcnnCutlassMLP. "
                "Install with: pip install tinycudann"
            )
        # tcnn requires input dim to be 16-aligned
        aligned_input = ((input_dim + 15) // 16) * 16
        if aligned_input != input_dim:
            print(
                f"TcnnCutlassMLP: padding input dim from {input_dim} to {aligned_input}"
            )

        super().__init__(aligned_input, hidden_dim, num_hidden_layers, output_dim)
        self.logical_input_dim = input_dim

        network_config = {
            "otype": "CutlassMLP",
            "activation": "LeakyReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": num_hidden_layers,
        }
        self.network = tcnn.Network(
            n_input_dims=aligned_input,
            n_output_dims=output_dim,
            network_config=network_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input to 16-aligned if needed
        if self.logical_input_dim != self.input_dim:
            pad = self.input_dim - self.logical_input_dim
            x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)], dim=1)
        return self.network(x)


# ═════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════

_MLP_REGISTRY = {
    "torch_linear": TorchMLP,
    "tcnn_cutlass": TcnnCutlassMLP,
}


def build_mlp(cfg: dict, input_dim: int) -> MLPDecoder:
    """Build MLP decoder from config dict.

    Example:
        cfg = {"type": "torch_linear", "hidden_dim": 64, "num_hidden_layers": 2, "output_dim": 8}
    """
    mlp_type = cfg.get("type", "torch_linear")
    if mlp_type not in _MLP_REGISTRY:
        raise ValueError(f"Unknown MLP type: {mlp_type}. Available: {list(_MLP_REGISTRY)}")

    cls = _MLP_REGISTRY[mlp_type]
    kwargs = {k: v for k, v in cfg.items() if k not in ("type",)}
    kwargs["input_dim"] = input_dim
    return cls(**kwargs)
