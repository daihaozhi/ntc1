"""Export Sponza 1K NTC checkpoints to the renderer's runtime assets."""

from pathlib import Path
import argparse
import struct
import zlib
import numpy as np


def write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    height, width, channels = rgba.shape
    if rgba.dtype != np.uint8 or channels != 4:
        raise ValueError("PNG input must be HxWx4 uint8")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgba)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )


def export_checkpoint(material_dir: Path) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("checkpoint export requires PyTorch") from exc

    checkpoint = torch.load(
        material_dir / "model_best.pth",
        map_location="cpu",
        weights_only=False,
    )
    params = checkpoint["mlp.network.params"].detach().float().cpu().numpy()
    if params.ndim != 1 or params.size == 0:
        raise ValueError(f"{material_dir}: expected a non-empty flat MLP parameter array")
    np.savez_compressed(material_dir / "mlp_weights.npz", params=params)

    for grid_index, resolution in enumerate((256, 128)):
        grid = checkpoint[f"grids.0.{grid_index}.params"].detach().float().cpu().numpy()
        expected = resolution * resolution * 8
        if grid.size != expected:
            raise ValueError(f"{material_dir}: grid {grid_index} expected {expected} values, got {grid.size}")
        quantized = np.rint((grid + 15.0 / 32.0) * 16.0).clip(0, 15).astype(np.uint8)
        planes = quantized.reshape(8, resolution * resolution).T
        rgba = np.empty((resolution * resolution, 4), dtype=np.uint8)
        rgba[:, 0] = planes[:, 0] | (planes[:, 1] << 4)
        rgba[:, 1] = planes[:, 2] | (planes[:, 3] << 4)
        rgba[:, 2] = planes[:, 4] | (planes[:, 5] << 4)
        rgba[:, 3] = planes[:, 6] | (planes[:, 7] << 4)
        write_rgba_png(
            material_dir / f"grid_{grid_index}.png",
            rgba.reshape(resolution, resolution, 4),
        )


def write_runtime_weights(material_dir: Path) -> None:
    source = material_dir / "mlp_weights.npz"
    target = material_dir / "mlp_weights.bin"
    with np.load(source) as archive:
        if archive.files != ["params"]:
            raise ValueError(f"{source}: expected a single flat 'params' array, got {archive.files}")
        values = np.asarray(archive["params"], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{source}: MLP parameter array is empty")
    target.write_bytes(values.tobytes())
    print(f"{material_dir}: exported 2 grids and {values.size} MLP floats")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("external/sponza_20260725"))
    args = parser.parse_args()
    material_dirs = sorted(args.root.glob("material_*"))
    if not material_dirs:
        material_dirs = sorted(args.root.glob("sponza_material_*"))
    if not material_dirs:
        material_dirs = sorted(args.root.glob("*_material_*"))
    for material_dir in material_dirs:
        runtime_sources = [
            material_dir / "mlp_weights.npz",
            material_dir / "grid_0.png",
            material_dir / "grid_1.png",
        ]
        if not all(path.exists() for path in runtime_sources):
            if not (material_dir / "model_best.pth").exists():
                continue
            export_checkpoint(material_dir)
        if not all(path.exists() for path in runtime_sources):
            continue
        write_runtime_weights(material_dir)


if __name__ == "__main__":
    main()
