#!/usr/bin/env python3
"""Train and evaluate NTC models for the main1_sponza asset on Ubuntu.

The script supports training every material or one deterministic random material.
It delegates preparation/training/reconstruction to the repository's existing
batch entry points and writes a machine-readable PSNR summary.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path


def find_gltf(root: Path) -> Path:
    if root.is_file() and root.suffix.lower() == ".gltf":
        return root
    candidates = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".gltf"
    )
    if not candidates:
        raise FileNotFoundError(f"No .gltf file found under {root}")
    return candidates[0]


def material_count(gltf_path: Path) -> int:
    with gltf_path.open("r", encoding="utf-8") as handle:
        return len(json.load(handle).get("materials", []))


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def collect_psnr(log_root: Path) -> dict[str, dict[str, float]]:
    pattern = re.compile(r"^\s*(\w+): MSE=([0-9.eE+-]+), PSNR=([0-9.+-]+) dB")
    summary: dict[str, dict[str, float]] = {}
    for log_path in sorted(log_root.glob("*_reconstruct.log")):
        material = log_path.stem.removesuffix("_reconstruct")
        values: dict[str, float] = {}
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if match:
                values[f"{match.group(1)}_mse"] = float(match.group(2))
                values[f"{match.group(1)}_psnr_db"] = float(match.group(3))
        if values:
            summary[material] = values
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("main1_sponza"))
    parser.add_argument("--gltf", type=Path, default=None,
                        help="Explicit .gltf path; overrides --data-root scanning")
    parser.add_argument("--mode", choices=["all", "random"], default="all")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--work-dir", type=Path, default=Path("runs_main1_sponza"))
    parser.add_argument("--resolution", type=int, default=4096, choices=[1024, 2048, 4096])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-iter", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crops-per-batch", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite-dataset", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    gltf_path = find_gltf(
        args.gltf.resolve() if args.gltf is not None else args.data_root.resolve()
    )
    count = material_count(gltf_path)
    if count == 0:
        raise RuntimeError(f"No materials found in {gltf_path}")

    selected_ids: list[int]
    if args.mode == "random":
        selected_ids = [random.Random(args.seed).randrange(count)]
    else:
        selected_ids = list(range(count))
    material_arg = ",".join(str(value) for value in selected_ids)
    print(f"Using glTF: {gltf_path}")
    print(f"Selected material ids: {material_arg}")

    batch_script = script_dir / "batch_train_sponza4k.py"
    train_command = [
        sys.executable, str(batch_script),
        "--gltf", str(gltf_path),
        "--work_dir", str(args.work_dir),
        "--resolution", str(args.resolution),
        "--material-ids", material_arg,
        "--export",
        "--max_iter", str(args.max_iter),
        "--batch_size", str(args.batch_size),
        "--crop_size", str(args.crop_size),
        "--crops_per_batch", str(args.crops_per_batch),
        "--device", args.device,
    ]
    if args.overwrite_dataset:
        train_command.append("--overwrite-dataset")
    if not args.skip_train:
        run(train_command)

    if not args.skip_eval:
        reconstruct_script = script_dir / "batch_reconstruct_sponza4k.py"
        run([
            sys.executable, str(reconstruct_script),
            "--batch_dir", str(args.work_dir),
            "--resolution", str(args.resolution),
            "--material-ids", material_arg,
            "--overwrite",
            "--device", args.device,
            "--batch_size", str(args.batch_size),
        ])
        summary = collect_psnr(args.work_dir / "logs")
        summary_path = args.work_dir / "evaluation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote PSNR summary: {summary_path}")
        for material, values in summary.items():
            print(f"{material}: {values}")


if __name__ == "__main__":
    main()
