import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_") or "material"


def material_texture_refs(material: dict) -> list[str]:
    refs = []
    pbr = material.get("pbrMetallicRoughness", {})
    if "baseColorTexture" in pbr:
        refs.append("base")
    if "metallicRoughnessTexture" in pbr:
        refs.append("metallicRoughness")
    if "normalTexture" in material:
        refs.append("normal")
    if "occlusionTexture" in material:
        refs.append("occlusion")
    return refs


def parse_material_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def run_command(cmd: list[str], log_path: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return

    if log_path is None:
        subprocess.run(cmd, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch prepare/train/export NTC models for Sponza4K glTF materials."
    )
    parser.add_argument("--gltf", required=True, help="Path to NewSponza_Main_glTF_003.gltf")
    parser.add_argument("--work_dir", default="sponza4k_ntc_batch", help="Root output directory")
    parser.add_argument("--resolution", type=int, default=4096, choices=[1024, 2048, 4096])
    parser.add_argument("--material-ids", default=None, help="Comma/range list, e.g. 0,3,8-12")
    parser.add_argument("--include-untextured", action="store_true", help="Also train materials with no texture refs")
    parser.add_argument("--prepare-only", action="store_true", help="Only create datasets")
    parser.add_argument("--export", action="store_true", help="Run export.py after each successful training")
    parser.add_argument("--overwrite-dataset", action="store_true", help="Recreate existing dataset directories")
    parser.add_argument("--retrain", action="store_true", help="Train even if model_best.pth already exists")
    parser.add_argument("--reexport", action="store_true", help="Export even if metadata.json already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")

    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--max_iter", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--network_lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--n_frequencies", type=int, default=8)
    parser.add_argument("--lod_sampling", default="exp", choices=["uniform", "exp", "fixed0"])
    parser.add_argument("--mip_target_mode", default="discrete", choices=["discrete", "trilinear"])
    parser.add_argument("--boundary_continuity_weight", type=float, default=0.0)
    parser.add_argument("--boundary_band_width", type=float, default=0.0,
                        help="Mip interval width around each grid-level boundary for continuity loss")
    parser.add_argument("--boundary_loss_preset", default="normal_roughness", choices=["reconstruction", "normal_roughness", "roughness"])
    parser.add_argument("--boundary_loss_weights", default=None, help='Optional JSON object, e.g. {"normal":2,"roughness":5}')
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid_config", default=None, help="Defaults to grid_config.json next to this script")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    prepare_script = script_dir / "prepare_sponza4k_material.py"
    train_script = script_dir / "train.py"
    export_script = script_dir / "export.py"
    grid_config = Path(args.grid_config).resolve() if args.grid_config else script_dir / "grid_config.json"

    gltf_path = Path(args.gltf).resolve()
    with gltf_path.open("r", encoding="utf-8") as f:
        gltf = json.load(f)

    selected_ids = parse_material_ids(args.material_ids)
    work_dir = Path(args.work_dir).resolve()
    dataset_root = work_dir / f"datasets_{args.resolution}"
    run_root = work_dir / f"runs_{args.resolution}"
    export_root = work_dir / f"exported_{args.resolution}"
    log_root = work_dir / "logs"

    materials = gltf.get("materials", [])
    print(f"Sponza4K materials: {len(materials)}")
    print(f"Work dir: {work_dir}")
    print(f"Resolution: {args.resolution}")

    completed = 0
    skipped = 0
    failed = 0

    for material_id, material in enumerate(materials):
        if selected_ids is not None and material_id not in selected_ids:
            continue

        name = material.get("name", f"material_{material_id:03d}")
        refs = material_texture_refs(material)
        if not refs and not args.include_untextured:
            print(f"\n[{material_id:02d}] {name}: skipped, no texture refs")
            skipped += 1
            continue

        safe = f"{material_id:03d}_{sanitize_name(name)}"
        dataset_dir = dataset_root / safe
        run_dir = run_root / safe
        exported_dir = export_root / safe
        best_model = run_dir / "model_best.pth"
        exported_meta = exported_dir / "metadata.json"

        print(f"\n[{material_id:02d}] {name} refs={','.join(refs) if refs else 'none'}")

        try:
            if args.overwrite_dataset or not (dataset_dir / "material.json").exists():
                prepare_cmd = [
                    sys.executable,
                    str(prepare_script),
                    "--gltf",
                    str(gltf_path),
                    "--material-id",
                    str(material_id),
                    "--resolution",
                    str(args.resolution),
                    "--output_dir",
                    str(dataset_dir),
                ]
                if args.overwrite_dataset:
                    prepare_cmd.append("--overwrite")
                run_command(prepare_cmd, log_root / f"{safe}_prepare.log", args.dry_run)
            else:
                print(f"Dataset exists: {dataset_dir}")

            if args.prepare_only:
                completed += 1
                continue

            if args.retrain or not best_model.exists():
                train_cmd = [
                    sys.executable,
                    str(train_script),
                    "--data_dir",
                    str(dataset_dir),
                    "--output_dir",
                    str(run_dir),
                    "--texture_resolution",
                    str(args.resolution),
                    "--grid_config",
                    str(grid_config),
                    "--batch_size",
                    str(args.batch_size),
                    "--max_iter",
                    str(args.max_iter),
                    "--lr",
                    str(args.lr),
                    "--network_lr",
                    str(args.network_lr),
                    "--hidden_dim",
                    str(args.hidden_dim),
                    "--num_hidden_layers",
                    str(args.num_hidden_layers),
                    "--n_frequencies",
                    str(args.n_frequencies),
                    "--lod_sampling",
                    args.lod_sampling,
                    "--mip_target_mode",
                    args.mip_target_mode,
                    "--boundary_continuity_weight",
                    str(args.boundary_continuity_weight),
                    "--boundary_band_width",
                    str(args.boundary_band_width),
                    "--boundary_loss_preset",
                    args.boundary_loss_preset,
                    "--eval_interval",
                    str(args.eval_interval),
                    "--save_interval",
                    str(args.save_interval),
                    "--device",
                    args.device,
                ]
                if args.boundary_loss_weights:
                    train_cmd.extend(["--boundary_loss_weights", args.boundary_loss_weights])
                run_command(train_cmd, log_root / f"{safe}_train.log", args.dry_run)
            else:
                print(f"Model exists: {best_model}")

            if args.export:
                if args.reexport or not exported_meta.exists():
                    export_cmd = [
                        sys.executable,
                        str(export_script),
                        "--checkpoint",
                        str(best_model),
                        "--output_dir",
                        str(exported_dir),
                        "--texture_resolution",
                        str(args.resolution),
                        "--grid_config",
                        str(grid_config),
                        "--hidden_dim",
                        str(args.hidden_dim),
                        "--num_hidden_layers",
                        str(args.num_hidden_layers),
                        "--n_frequencies",
                        str(args.n_frequencies),
                        "--device",
                        args.device,
                    ]
                    run_command(export_cmd, log_root / f"{safe}_export.log", args.dry_run)
                else:
                    print(f"Export exists: {exported_meta}")

            completed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"FAILED material {material_id} ({name}) with exit code {exc.returncode}", flush=True)
            print("Continuing with next material.", flush=True)

    print("\nBatch complete")
    print(f"  completed: {completed}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    print(f"  datasets:  {dataset_root}")
    print(f"  runs:      {run_root}")
    if args.export:
        print(f"  exported:  {export_root}")


if __name__ == "__main__":
    main()
