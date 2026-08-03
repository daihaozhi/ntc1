"""Evaluate Sponza1K NTC, equal-storage BC, and ASTC 12x12 quality.

The NTC path mirrors shaders/dx12/sponza_ntc.frag and the grid repacking in
src/dx12/NtcMaterialResources.cpp.  Its FP16 MLP is executed with ONNX Runtime
DirectML so evaluating all 24 1K materials is practical on an AMD GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from PIL import Image
from skimage.metrics import structural_similarity


TEXTURES = {
    "diffuse": ("diffuse.png", "RGB"),
    "normal": ("normal.png", "RGB"),
    "roughness": ("roughness.png", "L"),
    "metallic": ("metallic.png", "L"),
}
CHANNEL_COUNTS = {"diffuse": 3, "normal": 3, "roughness": 1, "metallic": 1}
CHUNK_PIXELS = 65536


def load_image(path: Path, mode: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0


def save_image(path: Path, values: np.ndarray, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(quantized, mode=mode).save(path)


def unpack_exported_grid(path: Path) -> np.ndarray:
    """Reproduce NtcMaterialResources.cpp's feature-major to texel-major copy."""
    packed = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    height, width, _ = packed.shape
    texel_count = width * height
    source = packed.reshape(texel_count, 4)
    texels = np.arange(texel_count, dtype=np.int64)[:, None]
    features = np.arange(8, dtype=np.int64)[None, :]
    source_flat = texels * 8 + features
    packed_channel = source_flat // texel_count
    source_texel = source_flat % texel_count
    source_byte = source[source_texel, packed_channel // 2]
    nibble = np.where(
        (packed_channel & 1) != 0,
        source_byte >> 4,
        source_byte & 15,
    )
    return (nibble.astype(np.float16) / np.float16(16.0)
            - np.float16(15.0 / 32.0)).reshape(height, width, 8)


def create_ntc_model(path: Path) -> None:
    shape_x = [CHUNK_PIXELS, 64]
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["X", "W0T"], ["M0"]),
            helper.make_node("LeakyRelu", ["M0"], ["H0"], alpha=0.01),
            helper.make_node("MatMul", ["H0", "W1T"], ["M1"]),
            helper.make_node("LeakyRelu", ["M1"], ["H1"], alpha=0.01),
            helper.make_node("MatMul", ["H1", "W2T"], ["M2"]),
            helper.make_node("Clip", ["M2", "ClipMin", "ClipMax"], ["Y"]),
        ],
        "sponza_ntc_fp16",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT16, shape_x),
            helper.make_tensor_value_info("W0T", TensorProto.FLOAT16, [64, 64]),
            helper.make_tensor_value_info("W1T", TensorProto.FLOAT16, [64, 64]),
            helper.make_tensor_value_info("W2T", TensorProto.FLOAT16, [64, 8]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [CHUNK_PIXELS, 8])],
        [
            helper.make_tensor("ClipMin", TensorProto.FLOAT16, [], [0.0]),
            helper.make_tensor("ClipMax", TensorProto.FLOAT16, [], [1.0]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
        producer_name="eval_sponza1k_codec_quality",
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def create_ntc_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.exists():
        create_ntc_model(model_path)
    available = ort.get_available_providers()
    providers = ["DmlExecutionProvider"] if "DmlExecutionProvider" in available else ["CPUExecutionProvider"]
    options = ort.SessionOptions()
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    print(f"NTC inference provider: {providers[0]}")
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def ntc_features(
    start: int,
    high: np.ndarray,
    low: np.ndarray,
    coordinate_mode: str,
) -> np.ndarray:
    indices = np.arange(start, start + CHUNK_PIXELS, dtype=np.int64)
    x = indices % 1024
    y = indices // 1024
    if coordinate_mode == "endpoints":
        u = x.astype(np.float32) / 1023.0
        v = y.astype(np.float32) / 1023.0
    else:
        u = (x.astype(np.float32) + 0.5) / 1024.0
        v = (y.astype(np.float32) + 0.5) / 1024.0
    # The shader applies frac().  Keep exact endpoint behavior for auditability.
    u = np.remainder(u, 1.0)
    v = np.remainder(v, 1.0)

    result = np.zeros((CHUNK_PIXELS, 64), dtype=np.float16)
    positional_u = np.remainder(u * 8.0, 1.0)
    positional_v = np.remainder(v * 8.0, 1.0)
    for frequency in range(5):
        scale = float(1 << frequency)
        su = positional_u * scale
        sv = positional_v * scale
        result[:, frequency * 2] = 2.0 * np.abs(su - np.floor(su + 0.5))
        result[:, frequency * 2 + 1] = 2.0 * np.abs(sv - np.floor(sv + 0.5))
    result[:, 10:12] = np.float16(1.0)

    hh, hw, _ = high.shape
    hx = np.minimum(np.floor(u * hw).astype(np.int32), hw - 1)
    hy = np.minimum(np.floor(v * hh).astype(np.int32), hh - 1)
    offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    output = 12
    for ox, oy in offsets:
        cx = np.minimum(hx + ox, hw - 1)
        cy = np.minimum(hy + oy, hh - 1)
        result[:, output:output + 8] = high[cy, cx]
        output += 8

    lh, lw, _ = low.shape
    px = u * lw
    py = v * lh
    x0 = np.minimum(np.floor(px).astype(np.int32), lw - 1)
    y0 = np.minimum(np.floor(py).astype(np.int32), lh - 1)
    x1 = np.minimum(x0 + 1, lw - 1)
    y1 = np.minimum(y0 + 1, lh - 1)
    wx = (px - np.floor(px)).astype(np.float16)[:, None]
    wy = (py - np.floor(py)).astype(np.float16)[:, None]
    top = low[y0, x0] * (np.float16(1.0) - wx) + low[y0, x1] * wx
    bottom = low[y1, x0] * (np.float16(1.0) - wx) + low[y1, x1] * wx
    result[:, 44:52] = top * (np.float16(1.0) - wy) + bottom * wy
    return result


def reconstruct_ntc_material(
    material_dir: Path,
    output_dir: Path,
    session: ort.InferenceSession,
    coordinate_mode: str,
) -> None:
    if all((output_dir / filename).exists() for filename, _ in TEXTURES.values()):
        return
    high = unpack_exported_grid(material_dir / "grid_0.png")
    low = unpack_exported_grid(material_dir / "grid_1.png")
    weights = np.fromfile(material_dir / "mlp_weights.bin", dtype=np.float32)
    if weights.size != 9216:
        raise ValueError(f"{material_dir}: expected 9216 weights, got {weights.size}")
    weights = weights.astype(np.float16)
    feeds = {
        "W0T": weights[:4096].reshape(64, 64).T.copy(),
        "W1T": weights[4096:8192].reshape(64, 64).T.copy(),
        "W2T": weights[8192:8704].reshape(8, 64).T.copy(),
    }
    decoded = np.empty((1024 * 1024, 8), dtype=np.float16)
    for start in range(0, 1024 * 1024, CHUNK_PIXELS):
        feeds["X"] = ntc_features(start, high, low, coordinate_mode)
        decoded[start:start + CHUNK_PIXELS] = session.run(["Y"], feeds)[0]
    decoded = decoded.reshape(1024, 1024, 8).astype(np.float32)
    save_image(output_dir / "diffuse.png", decoded[..., 0:3], "RGB")
    save_image(output_dir / "metallic.png", decoded[..., 3], "L")
    save_image(output_dir / "normal.png", decoded[..., 4:7], "RGB")
    save_image(output_dir / "roughness.png", decoded[..., 7], "L")


def decode_bc_material(texconv: Path, material_dir: Path, output_dir: Path) -> None:
    names = ("diffuse", "normal", "metallic_roughness")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = output_dir / f"{name}.png"
        # Re-run diffuse so an older linear-UNORM decode cannot silently poison
        # the comparison; -y makes this deterministic and inexpensive.
        if target.exists() and name != "diffuse":
            continue
        output_format = (
            "R8G8B8A8_UNORM_SRGB" if name == "diffuse"
            else "R8G8B8A8_UNORM"
        )
        subprocess.run(
            [
                str(texconv), "-nologo", "-y", "-f", output_format,
                "-ft", "png", "-o", str(output_dir), str(material_dir / f"{name}.dds"),
            ],
            check=True,
        )


def wrapped_bilinear(source: np.ndarray, size: int = 1024) -> np.ndarray:
    """Emulate Sample() with a linear wrap sampler from 348x348 to 1024x1024."""
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
    work = source if source.ndim == 3 else source[..., None]
    top = work[y0][:, x0] * (1.0 - wx) + work[y0][:, x1] * wx
    bottom = work[y1][:, x0] * (1.0 - wx) + work[y1][:, x1] * wx
    result = top * (1.0 - wy) + bottom * wy
    return result if source.ndim == 3 else result[..., 0]


def load_reconstruction(codec: str, root: Path, material: str) -> dict[str, np.ndarray]:
    directory = root / material
    if codec == "bc":
        diffuse = wrapped_bilinear(load_image(directory / "diffuse.png", "RGB"))
        normal_xy = wrapped_bilinear(load_image(directory / "normal.png", "RGB"))[..., :2]
        xy = normal_xy * 2.0 - 1.0
        z = np.sqrt(np.maximum(1.0 - np.sum(xy * xy, axis=2), 0.0))
        normal = np.concatenate((normal_xy, (z * 0.5 + 0.5)[..., None]), axis=2)
        mr = wrapped_bilinear(load_image(directory / "metallic_roughness.png", "RGB"))
        return {"diffuse": diffuse, "normal": normal, "roughness": mr[..., 0], "metallic": mr[..., 1]}
    if codec == "astc":
        mr = load_image(directory / "metallic_roughness.png", "RGB")
        return {
            "diffuse": load_image(directory / "diffuse.png", "RGB"),
            "normal": load_image(directory / "normal.png", "RGB"),
            "roughness": mr[..., 1],
            "metallic": mr[..., 2],
        }
    return {
        name: load_image(directory / filename, mode)
        for name, (filename, mode) in TEXTURES.items()
    }


def psnr_from_mse(mse: float) -> float:
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def evaluate_pair(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    diff = reference.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if reference.ndim == 2:
        ssim = float(structural_similarity(reference, reconstructed, data_range=1.0))
    else:
        # Equal channel weighting, also used by the aggregate eight-channel score.
        ssim = float(np.mean([
            structural_similarity(reference[..., c], reconstructed[..., c], data_range=1.0)
            for c in range(reference.shape[2])
        ]))
    return {"mse": mse, "psnr_db": psnr_from_mse(mse), "ssim": ssim}


def mean_and_std(values: list[float]) -> tuple[float, float]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    return float(np.mean(finite)), float(np.std(finite))


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path("external/sponza_20260725"))
    parser.add_argument("--ntc-root", type=Path, default=Path("artifacts/quality_eval/ntc"))
    parser.add_argument("--bc-source-root", type=Path, default=Path("external/sponza1k_bc_equal"))
    parser.add_argument("--bc-root", type=Path, default=Path("artifacts/quality_eval/bc"))
    parser.add_argument("--astc-root", type=Path, default=Path("external/sponza1k_astc12x12"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/quality_eval"))
    parser.add_argument("--texconv", type=Path, default=Path(
        r"C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links\texconv.exe"))
    parser.add_argument("--coordinate-mode", choices=("centers", "endpoints"), default="centers")
    args = parser.parse_args()
    materials = sorted(p.name for p in args.reference_root.glob("material_*") if p.is_dir())
    if not materials:
        raise SystemExit("No material directories found")

    session = create_ntc_session(args.output_root / "ntc_fp16.onnx")
    for index, material in enumerate(materials, 1):
        print(f"[{index:02d}/{len(materials)}] preparing {material}")
        decode_bc_material(args.texconv, args.bc_source_root / material, args.bc_root / material)
        reconstruct_ntc_material(
            args.reference_root / material,
            args.ntc_root / material,
            session,
            args.coordinate_mode,
        )

    records: list[dict] = []
    aggregate_mse: dict[str, list[float]] = {codec: [] for codec in ("ntc", "bc", "astc")}
    aggregate_ssim: dict[str, list[float]] = {codec: [] for codec in ("ntc", "bc", "astc")}
    roots = {"ntc": args.ntc_root, "bc": args.bc_root, "astc": args.astc_root}
    for material in materials:
        reference = {
            name: load_image(args.reference_root / material / filename, mode)
            for name, (filename, mode) in TEXTURES.items()
        }
        for codec, root in roots.items():
            reconstructed = load_reconstruction(codec, root, material)
            channel_mses: list[float] = []
            channel_ssims: list[float] = []
            for texture in TEXTURES:
                metric = evaluate_pair(reference[texture], reconstructed[texture])
                records.append({"material": material, "codec": codec, "texture": texture, **metric})
                channel_mses.extend([metric["mse"]] * CHANNEL_COUNTS[texture])
                if reference[texture].ndim == 2:
                    channel_ssims.append(metric["ssim"])
                else:
                    for channel in range(reference[texture].shape[2]):
                        channel_ssims.append(float(structural_similarity(
                            reference[texture][..., channel],
                            reconstructed[texture][..., channel],
                            data_range=1.0,
                        )))
            aggregate_mse[codec].append(float(np.mean(channel_mses)))
            aggregate_ssim[codec].append(float(np.mean(channel_ssims)))

    summary: dict[str, dict] = {}
    for codec in roots:
        mean_psnr, std_psnr = mean_and_std([psnr_from_mse(v) for v in aggregate_mse[codec]])
        mean_ssim, std_ssim = mean_and_std(aggregate_ssim[codec])
        summary[codec] = {
            "material_count": len(materials),
            "eight_channel_mean_psnr_db": mean_psnr,
            "eight_channel_std_psnr_db": std_psnr,
            "eight_channel_global_psnr_db": psnr_from_mse(float(np.mean(aggregate_mse[codec]))),
            "eight_channel_mean_ssim": mean_ssim,
            "eight_channel_std_ssim": std_ssim,
            "textures": {},
        }
        for texture in TEXTURES:
            selected = [r for r in records if r["codec"] == codec and r["texture"] == texture]
            pmean, pstd = mean_and_std([r["psnr_db"] for r in selected])
            smean, sstd = mean_and_std([r["ssim"] for r in selected])
            summary[codec]["textures"][texture] = {
                "mean_psnr_db": pmean,
                "std_psnr_db": pstd,
                "global_psnr_db": psnr_from_mse(float(np.mean([r["mse"] for r in selected]))),
                "mean_ssim": smean,
                "std_ssim": sstd,
            }

    result = {
        "methodology": {
            "materials": materials,
            "reference_resolution": [1024, 1024],
            "ntc_coordinate_mode": args.coordinate_mode,
            "aggregate": "Arithmetic mean over material-level metrics; 8 channels weighted equally.",
            "bc_sampling": "348x348 decoded BC textures sampled to 1024x1024 with bilinear wrap.",
            "lpips": "Computed separately on diffuse RGB only.",
        },
        "summary": summary,
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "quality_metrics.json"
    csv_path = args.output_root / "quality_metrics.csv"
    json_path.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("material", "codec", "texture", "mse", "psnr_db", "ssim"))
        writer.writeheader()
        writer.writerows(records)
    print("\n=== Eight-channel mean over materials ===")
    for codec, values in summary.items():
        print(f"{codec:4s} PSNR={values['eight_channel_mean_psnr_db']:.4f} dB "
              f"SSIM={values['eight_channel_mean_ssim']:.6f}")
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
