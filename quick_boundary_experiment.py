import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], dry_run: bool = False) -> None:
    printable = " ".join(f'"{x}"' if " " in x else x for x in cmd)
    print(f"\n$ {printable}", flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_boundary_summary(metrics_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "material",
        "left_level",
        "right_level",
        "lod",
        "texture",
        "left_right_psnr_db",
        "left_psnr_db",
        "right_psnr_db",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small Sponza4K NTC training experiment and evaluate grid boundary continuity."
    )
    parser.add_argument("--gltf", required=True, help="Path to NewSponza_Main_glTF_003.gltf")
    parser.add_argument("--work_dir", default="./quick_boundary_experiment")
    parser.add_argument("--material-ids", default="0", help="Comma/range list passed to batch_train_sponza4k.py")
    parser.add_argument("--resolution", type=int, default=1024, choices=[1024, 2048, 4096])
    parser.add_argument("--max_iter", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--boundary_continuity_weight", type=float, default=0.05)
    parser.add_argument("--boundary_loss_preset", default="normal_roughness", choices=["reconstruction", "normal_roughness", "roughness"])
    parser.add_argument("--boundary_loss_weights", default=None)
    parser.add_argument("--lod_sampling", default="uniform", choices=["uniform", "exp", "fixed0"])
    parser.add_argument("--mip_target_mode", default="trilinear", choices=["discrete", "trilinear"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid_config", default=None)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no_images", action="store_true", help="Skip boundary PNGs during evaluation")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    work_dir = Path(args.work_dir).resolve()
    batch_script = script_dir / "batch_train_sponza4k.py"
    eval_script = script_dir / "eval_grid_level.py"
    grid_config = Path(args.grid_config).resolve() if args.grid_config else script_dir / "grid_config.json"

    train_cmd = [
        sys.executable,
        str(batch_script),
        "--gltf",
        str(Path(args.gltf).resolve()),
        "--work_dir",
        str(work_dir),
        "--resolution",
        str(args.resolution),
        "--material-ids",
        args.material_ids,
        "--export",
        "--batch_size",
        str(args.batch_size),
        "--max_iter",
        str(args.max_iter),
        "--lod_sampling",
        args.lod_sampling,
        "--mip_target_mode",
        args.mip_target_mode,
        "--boundary_continuity_weight",
        str(args.boundary_continuity_weight),
        "--boundary_loss_preset",
        args.boundary_loss_preset,
        "--eval_interval",
        str(max(args.max_iter + 1, 1000000)),
        "--save_interval",
        str(max(args.max_iter, 1)),
        "--device",
        args.device,
        "--grid_config",
        str(grid_config),
    ]
    if args.boundary_loss_weights:
        train_cmd.extend(["--boundary_loss_weights", args.boundary_loss_weights])
    if args.retrain:
        train_cmd.append("--retrain")
    if args.overwrite_dataset:
        train_cmd.append("--overwrite-dataset")

    run_command(train_cmd, args.dry_run)

    dataset_root = work_dir / f"datasets_{args.resolution}"
    run_root = work_dir / f"runs_{args.resolution}"
    eval_root = work_dir / f"boundary_eval_{args.resolution}"
    summary_rows: list[dict[str, str]] = []

    for dataset_dir in sorted(dataset_root.iterdir() if dataset_root.exists() else []):
        if not dataset_dir.is_dir():
            continue
        material_name = dataset_dir.name
        checkpoint = run_root / material_name / "model_best.pth"
        if not checkpoint.exists():
            print(f"Skipping {material_name}: missing {checkpoint}")
            continue

        output_dir = eval_root / material_name
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--data_dir",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--texture_resolution",
            str(args.resolution),
            "--grid_config",
            str(grid_config),
            "--boundary",
            "--device",
            args.device,
        ]
        if args.no_images:
            eval_cmd.append("--no_images")
        run_command(eval_cmd, args.dry_run)

        metrics_path = output_dir / "boundary_metrics.csv"
        if args.dry_run or not metrics_path.exists():
            continue
        for row in read_boundary_summary(metrics_path):
            summary_rows.append(
                {
                    "experiment": work_dir.name,
                    "material": material_name,
                    "left_level": row["left_level"],
                    "right_level": row["right_level"],
                    "lod": row["lod"],
                    "texture": row["texture"],
                    "left_right_psnr_db": row["left_right_psnr_db"],
                    "left_psnr_db": row["left_psnr_db"],
                    "right_psnr_db": row["right_psnr_db"],
                }
            )

    summary_path = work_dir / "boundary_summary.csv"
    if summary_rows:
        write_summary(summary_rows, summary_path)
        print(f"\nWrote summary: {summary_path}")

        roughness_rows = [r for r in summary_rows if r["texture"] == "roughness"]
        if roughness_rows:
            print("\nRoughness boundary left_vs_right PSNR:")
            for row in roughness_rows:
                print(
                    f"  {row['material']} L{row['left_level']}/L{row['right_level']} "
                    f"lod {row['lod']}: {float(row['left_right_psnr_db']):.2f} dB"
                )
    elif not args.dry_run:
        print("\nNo boundary metrics were produced.")

    config_path = work_dir / "quick_boundary_experiment_config.json"
    if not args.dry_run:
        config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Wrote config: {config_path}")


if __name__ == "__main__":
    main()
