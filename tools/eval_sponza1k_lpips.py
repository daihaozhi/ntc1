"""Compute standard AlexNet LPIPS for Sponza1K diffuse RGB reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def wrapped_bilinear(source: np.ndarray, size: int = 1024) -> np.ndarray:
    height, width = source.shape[:2]
    xcoord = (np.arange(size, dtype=np.float32) + 0.5) * width / size - 0.5
    ycoord = (np.arange(size, dtype=np.float32) + 0.5) * height / size - 0.5
    xfloor = np.floor(xcoord)
    yfloor = np.floor(ycoord)
    x0 = xfloor.astype(np.int32) % width
    y0 = yfloor.astype(np.int32) % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx = (xcoord - xfloor).reshape(1, size, 1)
    wy = (ycoord - yfloor).reshape(size, 1, 1)
    top = source[y0][:, x0] * (1.0 - wx) + source[y0][:, x1] * wx
    bottom = source[y1][:, x0] * (1.0 - wx) + source[y1][:, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1))).unsqueeze(0) * 2.0 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path("external/sponza_20260725"))
    parser.add_argument("--ntc-root", type=Path, default=Path("artifacts/quality_eval/ntc"))
    parser.add_argument("--bc-root", type=Path, default=Path("artifacts/quality_eval/bc"))
    parser.add_argument("--astc-root", type=Path, default=Path("external/sponza1k_astc12x12"))
    parser.add_argument("--metrics-json", type=Path, default=Path("artifacts/quality_eval/quality_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/quality_eval/lpips_diffuse.csv"))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = lpips.LPIPS(net="alex", version="0.1").eval()
    materials = sorted(p.name for p in args.reference_root.glob("material_*") if p.is_dir())
    roots = {"ntc": args.ntc_root, "bc": args.bc_root, "astc": args.astc_root}
    records: list[dict] = []
    for index, material in enumerate(materials, 1):
        reference = tensor(load_rgb(args.reference_root / material / "diffuse.png"))
        for codec, root in roots.items():
            reconstructed = load_rgb(root / material / "diffuse.png")
            if codec == "bc":
                reconstructed = wrapped_bilinear(reconstructed)
            value = float(model(reference, tensor(reconstructed)).item())
            records.append({"material": material, "codec": codec, "lpips_alex": value})
            print(f"[{index:02d}/{len(materials)}] {material} {codec:4s} LPIPS={value:.8f}")

    summary = {}
    for codec in roots:
        values = np.asarray(
            [record["lpips_alex"] for record in records if record["codec"] == codec],
            dtype=np.float64,
        )
        summary[codec] = {
            "diffuse_lpips_alex_mean": float(np.mean(values)),
            "diffuse_lpips_alex_std": float(np.std(values)),
            "diffuse_lpips_alex_min": float(np.min(values)),
            "diffuse_lpips_alex_max": float(np.max(values)),
        }

    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    metrics["methodology"]["lpips"] = (
        "LPIPS v0.1, AlexNet/ImageNet trunk, diffuse RGB in sRGB space mapped to [-1,1]; "
        "arithmetic mean over 24 materials. Lower is better."
    )
    for codec, values in summary.items():
        metrics["summary"][codec].update(values)
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("material", "codec", "lpips_alex"))
        writer.writeheader()
        writer.writerows(records)
    print("\n=== Diffuse RGB LPIPS (AlexNet, lower is better) ===")
    for codec, values in summary.items():
        print(f"{codec:4s} mean={values['diffuse_lpips_alex_mean']:.8f} "
              f"std={values['diffuse_lpips_alex_std']:.8f}")


if __name__ == "__main__":
    main()
