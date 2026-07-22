import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], log_path: Path | None, dry_run: bool) -> None:
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
        description="Batch reconstruct Sponza4K material textures from trained NTC checkpoints."
    )
    parser.add_argument("--batch_dir", default="../runs_sponza4k_batch",
                        help="Batch root containing datasets_4096 and runs_4096")
    parser.add_argument("--resolution", type=int, default=4096, choices=[1024, 2048, 4096])
    parser.add_argument("--output_name", default=None,
                        help="Output folder name under batch_dir. Defaults to reconstructed_<resolution>")
    parser.add_argument("--material-ids", default=None,
                        help="Comma/range list, e.g. 0,3,8-12")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=65536)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    inference_script = script_dir / "inference.py"
    batch_dir = Path(args.batch_dir).resolve()
    dataset_root = batch_dir / f"datasets_{args.resolution}"
    run_root = batch_dir / f"runs_{args.resolution}"
    output_root = batch_dir / (args.output_name or f"reconstructed_{args.resolution}")
    log_root = batch_dir / "logs"

    selected = None
    if args.material_ids:
        selected = set()
        for part in args.material_ids.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                selected.update(range(int(lo), int(hi) + 1))
            else:
                selected.add(int(part))

    completed = 0
    skipped = 0
    failed = 0
    for dataset_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        try:
            material_id = int(dataset_dir.name.split("_", 1)[0])
        except ValueError:
            continue
        if selected is not None and material_id not in selected:
            continue

        run_dir = run_root / dataset_dir.name
        checkpoint = run_dir / "model_best.pth"
        output_dir = output_root / dataset_dir.name
        if not checkpoint.exists():
            print(f"\n[{dataset_dir.name}] skipped, missing {checkpoint}")
            skipped += 1
            continue
        if output_dir.exists() and not args.overwrite:
            print(f"\n[{dataset_dir.name}] exists: {output_dir}")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(inference_script),
            "--data_dir",
            str(dataset_dir),
            "--checkpoint",
            str(checkpoint),
            "--output_dir",
            str(output_dir),
            "--texture_resolution",
            str(args.resolution),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
        try:
            run_command(cmd, log_root / f"{dataset_dir.name}_reconstruct.log", args.dry_run)
            completed += 1
        except subprocess.CalledProcessError as exc:
            print(f"FAILED {dataset_dir.name} with exit code {exc.returncode}")
            failed += 1

    print("\nReconstruction batch complete")
    print(f"  completed: {completed}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")
    print(f"  output:    {output_root}")


if __name__ == "__main__":
    main()
