"""Build an equal-storage BC baseline for the Sponza 1K NTC materials.

Each material is resized to 348x348 and encoded as:
  diffuse.dds              BC7_UNORM_SRGB
  normal.dds               BC5_UNORM (encoded tangent-space X/Y)
  metallic_roughness.dds   BC5_UNORM (R=roughness, G=metallic)

At 348x348 each BC5/BC7 payload is 87*87*16 = 121,104 bytes.
The three payloads total 363,312 bytes per material, within 0.34% of
the 364,544-byte Sponza 1K NTC representation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


RESOLUTION = 348
BLOCK_PAYLOAD_BYTES = (RESOLUTION // 4) ** 2 * 16
MATERIAL_PAYLOAD_BYTES = BLOCK_PAYLOAD_BYTES * 3


def run_texconv(
    texconv: Path,
    source: Path,
    output_dir: Path,
    output_format: str,
    srgb: bool,
) -> Path:
    command = [
        str(texconv),
        "-nologo",
        "-y",
        "-dx10",
        "-m",
        "1",
        "-w",
        str(RESOLUTION),
        "-h",
        str(RESOLUTION),
        "-if",
        "CUBIC",
        "-f",
        output_format,
        "-bc",
        "x",
        "-o",
        str(output_dir),
    ]
    if srgb:
        command.append("-srgb")
    command.append(str(source))
    subprocess.run(command, check=True)
    output = output_dir / f"{source.stem}.dds"
    if not output.exists():
        raise FileNotFoundError(output)
    return output


def dds_payload_bytes(path: Path, expected_dxgi_format: int) -> int:
    data = path.read_bytes()
    if len(data) < 148 or data[:4] != b"DDS " or data[84:88] != b"DX10":
        raise ValueError(f"{path}: expected a DDS file with a DX10 header")
    width = int.from_bytes(data[16:20], "little")
    height = int.from_bytes(data[12:16], "little")
    mip_count = int.from_bytes(data[28:32], "little")
    dxgi_format = int.from_bytes(data[128:132], "little")
    if (width, height, mip_count) != (RESOLUTION, RESOLUTION, 1):
        raise ValueError(f"{path}: unexpected dimensions/mips {(width, height, mip_count)}")
    if dxgi_format != expected_dxgi_format:
        raise ValueError(f"{path}: expected DXGI format {expected_dxgi_format}, got {dxgi_format}")
    payload = len(data) - 148
    if payload != BLOCK_PAYLOAD_BYTES:
        raise ValueError(f"{path}: expected {BLOCK_PAYLOAD_BYTES} payload bytes, got {payload}")
    return payload


def encode_material(
    material_dir: Path,
    output_root: Path,
    texconv: Path,
    force: bool,
) -> dict[str, object]:
    output_dir = output_root / material_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    diffuse_source = material_dir / "diffuse.png"
    normal_source = material_dir / "normal.png"
    roughness_source = material_dir / "roughness.png"
    metallic_source = material_dir / "metallic.png"
    for required in (diffuse_source, normal_source, roughness_source, metallic_source):
        if not required.exists():
            raise FileNotFoundError(required)

    diffuse_dds = output_dir / "diffuse.dds"
    normal_dds = output_dir / "normal.dds"
    mr_dds = output_dir / "metallic_roughness.dds"
    with tempfile.TemporaryDirectory(prefix=f"{material_dir.name}_bc_") as temp_name:
        temp = Path(temp_name)
        packed_source = temp / "metallic_roughness.png"
        roughness = np.asarray(Image.open(roughness_source).convert("L"), dtype=np.uint8)
        metallic = np.asarray(Image.open(metallic_source).convert("L"), dtype=np.uint8)
        if roughness.shape != (1024, 1024) or metallic.shape != (1024, 1024):
            raise ValueError(f"{material_dir}: roughness and metallic must be 1024x1024")
        packed = np.zeros((1024, 1024, 4), dtype=np.uint8)
        packed[:, :, 0] = roughness
        packed[:, :, 1] = metallic
        packed[:, :, 3] = 255
        Image.fromarray(packed, mode="RGBA").save(packed_source)

        jobs = [
            (diffuse_source, diffuse_dds, "BC7_UNORM_SRGB", True),
            (normal_source, normal_dds, "BC5_UNORM", False),
            (packed_source, mr_dds, "BC5_UNORM", False),
        ]
        for source, destination, output_format, srgb in jobs:
            if force or not destination.exists():
                generated = run_texconv(texconv, source, output_dir, output_format, srgb)
                if generated != destination:
                    generated.replace(destination)

    formats = {
        "diffuse": ("BC7_UNORM_SRGB", 99, diffuse_dds),
        "normal": ("BC5_UNORM", 83, normal_dds),
        "metallic_roughness": ("BC5_UNORM", 83, mr_dds),
    }
    textures: dict[str, object] = {}
    total = 0
    for name, (output_format, dxgi_format, path) in formats.items():
        payload = dds_payload_bytes(path, dxgi_format)
        textures[name] = {
            "format": output_format,
            "payload_bytes": payload,
            "file_bytes": path.stat().st_size,
        }
        total += payload
    if total != MATERIAL_PAYLOAD_BYTES:
        raise AssertionError((material_dir, total, MATERIAL_PAYLOAD_BYTES))
    return {
        "material": material_dir.name,
        "resolution": [RESOLUTION, RESOLUTION],
        "textures": textures,
        "payload_bytes": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("external/sponza_20260725"))
    parser.add_argument("--output-root", type=Path, default=Path("external/sponza1k_bc_equal"))
    parser.add_argument("--texconv", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    texconv = args.texconv
    if texconv is None:
        discovered = shutil.which("texconv")
        if not discovered:
            raise FileNotFoundError(
                "texconv was not found. Install it with "
                "'winget install Microsoft.DirectXTex.Texconv' or pass --texconv.")
        texconv = Path(discovered)
    if not texconv.exists():
        raise FileNotFoundError(texconv)
    material_dirs = sorted(
        path
        for path in args.source_root.glob("material_*")
        if path.is_dir() and (path / "model_best.pth").exists()
    )
    if not material_dirs:
        raise FileNotFoundError(f"No NTC materials in {args.source_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [
            executor.submit(encode_material, material, args.output_root, texconv, args.force)
            for material in material_dirs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['material']}: {result['payload_bytes']} payload bytes")
    results.sort(key=lambda item: str(item["material"]))
    total_payload = sum(int(item["payload_bytes"]) for item in results)
    manifest = {
        "format": "BC7_SRGB + BC5 normal XY + BC5 roughness/metallic",
        "resolution": [RESOLUTION, RESOLUTION],
        "material_count": len(results),
        "payload_bytes_per_material": MATERIAL_PAYLOAD_BYTES,
        "total_payload_bytes": total_payload,
        "ntc_bytes_per_material": 364_544,
        "difference_from_ntc_percent": (MATERIAL_PAYLOAD_BYTES / 364_544 - 1.0) * 100.0,
        "materials": results,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}: {len(results)} materials, {total_payload} payload bytes")


if __name__ == "__main__":
    main()
