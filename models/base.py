"""Abstract base class for NTC models.

All NTC model variants must implement:
- forward(x): (u, v, lod) → output channels
- clamp_value(): clamp grid params to quantization range
- save/load checkpoint
"""

import torch


class NTCModel(torch.nn.Module):
    """Base class for Neural Texture Compression models."""

    # Subclasses should override these
    model_type: str = "base"

    def __init__(self):
        super().__init__()
        self.current_iter: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3] tensor with columns (u, v, lod_normalized),
               where u,v in [0,1) and lod_normalized in [0,1].

        Returns:
            [B, output_dim] decoded material channels.
        """
        raise NotImplementedError

    def clamp_value(self) -> None:
        """Clamp feature grid parameters to quantization range.

        Called after each optimizer.step() to prevent parameter drift.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str) -> None:
        """Save model state dictionary."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, device: torch.device = None) -> None:
        """Load model state dictionary."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(ckpt)
