import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_") or "material"


def texture_image_path(gltf_dir: Path, gltf: dict, texture_info: dict | None) -> Path | None:
    if not texture_info:
        return None
    texture_index = int(texture_info["index"])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    if texture_index < 0 or texture_index >= len(textures):
        return None
    image_index = int(textures[texture_index].get("source", -1))
    if image_index < 0 or image_index >= len(images):
        return None
    uri = images[image_index].get("uri")
    if not uri:
        return None
    return gltf_dir / uri


def first_existing_size(paths: list[Path | None], fallback: int | None) -> tuple[int, int]:
    for path in paths:
        if path and path.exists():
            with Image.open(path) as image:
                return image.size
    if fallback:
        return fallback, fallback
    raise RuntimeError("Cannot infer output size. Provide --resolution or use a material with at least one texture.")


def resize_if_needed(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def save_rgb_texture(src: Path | None, dst: Path, size: tuple[int, int], fallback_rgb: tuple[int, int, int]) -> bool:
    if src and src.exists():
        with Image.open(src) as image:
            image = resize_if_needed(image.convert("RGB"), size)
            image.save(dst)
        return True
    Image.new("RGB", size, fallback_rgb).save(dst)
    return False


def save_mr_channels(src: Path | None, roughness_dst: Path, metallic_dst: Path, size: tuple[int, int], roughness: float, metallic: float) -> dict:
    if src and src.exists():
        with Image.open(src) as image:
            image = resize_if_needed(image.convert("RGB"), size)
            channels = image.split()
            channels[1].save(roughness_dst)
            channels[2].save(metallic_dst)
        return {"roughness_source": str(src), "metallic_source": str(src), "source_channels": {"roughness": "G", "metallic": "B"}}

    roughness_value = int(round(max(0.0, min(1.0, roughness)) * 255.0))
    metallic_value = int(round(max(0.0, min(1.0, metallic)) * 255.0))
    Image.new("L", size, roughness_value).save(roughness_dst)
    Image.new("L", size, metallic_value).save(metallic_dst)
    return {"roughness_source": None, "metallic_source": None, "source_channels": {"roughness": "constant", "metallic": "constant"}}


def save_gray_texture(src: Path | None, dst: Path, size: tuple[int, int], fallback_value: int, source_channel: int = 0) -> bool:
    if src and src.exists():
        with Image.open(src) as image:
            image = resize_if_needed(image.convert("RGB"), size)
            image.split()[source_channel].save(dst)
        return True
    Image.new("L", size, fallback_value).save(dst)
    return False


def select_material(gltf: dict, material_id: int | None, material_name: str | None) -> tuple[int, dict]:
    materials = gltf.get("materials", [])
    if material_name is not None:
        matches = [(i, m) for i, m in enumerate(materials) if m.get("name") == material_name]
        if not matches:
            raise RuntimeError(f"Material name not found: {material_name}")
        return matches[0]
    if material_id is None:
        raise RuntimeError("Provide --material-id, --material-name, or --list-materials.")
    if material_id < 0 or material_id >= len(materials):
        raise RuntimeError(f"Material id {material_id} out of range 0..{len(materials) - 1}")
    return material_id, materials[material_id]


def print_materials(gltf: dict) -> None:
    for i, material in enumerate(gltf.get("materials", [])):
        pbr = material.get("pbrMetallicRoughness", {})
        refs = []
        if "baseColorTexture" in pbr:
            refs.append("base")
        if "metallicRoughnessTexture" in pbr:
            refs.append("metallicRoughness")
        if "normalTexture" in material:
            refs.append("normal")
        if "occlusionTexture" in material:
            refs.append("occlusion")
        print(f"{i:02d}: {material.get('name', '<unnamed>')} ({', '.join(refs) if refs else 'no textures'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one Sponza4K glTF material into an ntc1 training directory.")
    parser.add_argument("--gltf", required=True, help="Path to NewSponza_Main_glTF_003.gltf")
    parser.add_argument("--output_dir", default="sponza4k_material_dataset", help="Directory to write diffuse/normal/roughness/metallic/occlusion textures")
    parser.add_argument("--material-id", type=int, default=None, help="glTF material index")
    parser.add_argument("--material-name", type=str, default=None, help="Exact glTF material name")
    parser.add_argument("--resolution", type=int, default=None, help="Optional square output resolution, e.g. 1024/2048/4096")
    parser.add_argument("--list-materials", action="store_true", help="Print material ids and exit")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it already exists")
    args = parser.parse_args()

    gltf_path = Path(args.gltf)
    gltf_dir = gltf_path.parent
    with gltf_path.open("r", encoding="utf-8") as f:
        gltf = json.load(f)

    if args.list_materials:
        print_materials(gltf)
        return

    material_id, material = select_material(gltf, args.material_id, args.material_name)
    material_name = material.get("name", f"material_{material_id:03d}")
    pbr = material.get("pbrMetallicRoughness", {})

    base_path = texture_image_path(gltf_dir, gltf, pbr.get("baseColorTexture"))
    mr_path = texture_image_path(gltf_dir, gltf, pbr.get("metallicRoughnessTexture"))
    normal_path = texture_image_path(gltf_dir, gltf, material.get("normalTexture"))
    ao_path = texture_image_path(gltf_dir, gltf, material.get("occlusionTexture"))
    size = first_existing_size([base_path, normal_path, mr_path, ao_path], args.resolution)
    if args.resolution:
        size = (args.resolution, args.resolution)

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"Output directory exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
    roughness_factor = float(pbr.get("roughnessFactor", 1.0))
    metallic_factor = float(pbr.get("metallicFactor", 1.0))

    wrote = {}
    wrote["diffuse"] = save_rgb_texture(base_path, out_dir / "diffuse.png", size, tuple(int(max(0.0, min(1.0, c)) * 255.0) for c in base_factor[:3]))
    wrote["normal"] = save_rgb_texture(normal_path, out_dir / "normal.png", size, (128, 128, 255))
    mr_meta = save_mr_channels(mr_path, out_dir / "roughness.png", out_dir / "metallic.png", size, roughness_factor, metallic_factor)
    wrote["occlusion"] = save_gray_texture(ao_path, out_dir / "occlusion.png", size, 255, source_channel=0)

    metadata = {
        "source_gltf": str(gltf_path),
        "material_id": material_id,
        "material_name": material_name,
        "safe_name": sanitize_name(material_name),
        "resolution": {"width": size[0], "height": size[1]},
        "base_color_factor": base_factor,
        "roughness_factor": roughness_factor,
        "metallic_factor": metallic_factor,
        "sources": {
            "diffuse": str(base_path) if base_path else None,
            "normal": str(normal_path) if normal_path else None,
            "occlusion": str(ao_path) if ao_path else None,
            **mr_meta,
        },
        "generated_fallbacks": {k: not v for k, v in wrote.items()},
        "notes": [
            "glTF metallicRoughness texture is split into roughness=G and metallic=B.",
            "Base color factors are recorded in metadata but not baked into diffuse.png.",
            "Missing maps are generated as constants so train.py can run on any material.",
        ],
    }
    with (out_dir / "material.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared material {material_id}: {material_name}")
    print(f"Output: {out_dir}")
    print("Files: diffuse.png, normal.png, roughness.png, metallic.png, occlusion.png, material.json")


if __name__ == "__main__":
    main()
