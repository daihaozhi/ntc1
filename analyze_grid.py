import os
import argparse
import math
import numpy as np
from PIL import Image
from collections import Counter


def analyze_grid(path, name):
    img = Image.open(path)
    arr = np.array(img)  # [H, W, 4] uint8

    print(f'\n{"=" * 60}')
    print(f'  {name}  ({arr.shape[1]}x{arr.shape[0]}, {arr.shape[2]}ch)')
    print(f'{"=" * 60}')

    # Per-channel stats
    for c in range(arr.shape[2]):
        ch = arr[:, :, c].ravel()
        unique = len(np.unique(ch))
        mean = ch.mean()
        std = ch.std()
        hist = np.bincount(ch, minlength=256)

        # Entropy of distribution
        hist_norm = hist / hist.sum()
        nonzero = hist_norm[hist_norm > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))

        # Peak concentration: what fraction of pixels fall in the most common 10% of bins
        top_bins = np.sort(hist)[-26:]  # top ~10% of 256 bins
        concentration = top_bins.sum() / hist.sum()

        # Uniformity test: chi-squared vs uniform distribution
        expected = ch.size / 256
        chi2 = np.sum((hist - expected) ** 2 / expected) if expected > 0 else 0

        # Spatial check: mean of top-left vs bottom-right quarters
        hh, hw = arr.shape[0] // 2, arr.shape[1] // 2
        tl = arr[:hh, :hw, c].mean()
        tr = arr[:hh, hw:, c].mean()
        bl = arr[hh:, :hw, c].mean()
        br = arr[hh:, hw:, c].mean()
        spatial_range = max(tl, tr, bl, br) - min(tl, tr, bl, br)

        print(f'\n  Channel {c}:')
        print(f'    mean={mean:.1f}  std={std:.1f}  unique_values={unique}')
        print(f'    entropy={entropy:.2f}/8.0 (100%={entropy / 8.0 * 100:.1f}%)')
        print(f'    top10%_concentration={concentration:.3f}')
        print(f'    chi2_vs_uniform={chi2:.0f}')
        print(f'    spatial_range_quarters={spatial_range:.1f}')

    # Overall verdict
    all_vals = arr.ravel()
    overall_std = all_vals.std()
    overall_unique = len(np.unique(all_vals))
    overall_entropy_val = -np.sum((np.bincount(all_vals, minlength=256) / all_vals.size) * 0)
    hist_all = np.bincount(all_vals, minlength=256)
    hist_all_norm = hist_all / hist_all.sum()
    nz = hist_all_norm[hist_all_norm > 0]
    overall_entropy = -np.sum(nz * np.log2(nz))

    # Spatial gradient check: average absolute difference between adjacent pixels
    diff_h = np.abs(arr[1:, :, :].astype(float) - arr[:-1, :, :].astype(float)).mean()
    diff_v = np.abs(arr[:, 1:, :].astype(float) - arr[:, :-1, :].astype(float)).mean()
    grad = (diff_h + diff_v) / 2

    print(f'\n  ── Overall ──')
    print(f'    std={overall_std:.1f}  unique={overall_unique}  entropy={overall_entropy:.2f}/8.0')
    print(f'    mean_gradient={grad:.2f}  (0=constant, 20+ = busy texture)')

    # Verdict
    issues = []
    if overall_std < 3:
        issues.append('Nearly constant — model learned almost nothing')
    if overall_unique < 10:
        issues.append('Very few distinct values — collapsed representation')
    if overall_entropy < 2:
        issues.append(f'Extremely low entropy ({overall_entropy:.2f}/8) — feature diversity dead')
    if grad < 1:
        issues.append('No spatial variation — all pixels identical')
    if grad < 4:
        issues.append(f'Weak spatial variation (grad={grad:.1f}) — features barely change spatially')

    if issues:
        print(f'\n  ⚠ VERDICT: PROBLEMS DETECTED')
        for issue in issues:
            print(f'    → {issue}')
    else:
        print(f'\n  ✓ VERDICT: Looks healthy — features have diversity and spatial structure')


def main():
    parser = argparse.ArgumentParser(description='Analyze exported feature grid textures')
    parser.add_argument('--export_dir', type=str, required=True,
                        help='Directory containing exported grid PNG files')
    parser.add_argument('--output_dir', type=str, default='./analysis',
                        help='Directory for analysis reports')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.export_dir) if f.startswith('grid_') and f.endswith('.png')])
    if not files:
        print(f'No grid_*.png files found in {args.export_dir}')
        return

    for fname in files:
        path = os.path.join(args.export_dir, fname)
        analyze_grid(path, fname)


if __name__ == '__main__':
    main()
