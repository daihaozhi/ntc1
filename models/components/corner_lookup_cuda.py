"""Custom CUDA kernel for fused corner-four grid lookup.

Replaces tcnn grid() with a single kernel that:
1. Computes 4 corner indices from UV
2. Looks up grid values directly from flat parameter tensor
3. Handles multi-level packed grids
4. Implements custom backward pass

Usage:
    from models.components.corner_lookup_cuda import corner_four_lookup
    features = corner_four_lookup(uv, grid_params, resolution, n_levels, n_features)
"""

import torch
from torch.utils.cpp_extension import load_inline


_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward: UV [B,2] + grid [L*R*R*F] -> output [B, 4*L*F]
__global__ void corner_lookup_forward_kernel(
    const float* __restrict__ uv,
    const float* __restrict__ grid,
    float* __restrict__ output,
    int B, int R, int L, int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float u = uv[idx * 2] * (float)R;
    float v = uv[idx * 2 + 1] * (float)R;

    int fu = (int)floorf(u);
    int fv = (int)floorf(v);
    int cu = min(fu + 1, R - 1);
    int cv = min(fv + 1, R - 1);

    int level_stride = R * R * F;
    int out_dim = L * F;

    for (int l = 0; l < L; l++) {
        const float* level_grid = grid + l * level_stride;

        int idx00 = (fv * R + fu) * F;
        int idx10 = (fv * R + cu) * F;
        int idx01 = (cv * R + fu) * F;
        int idx11 = (cv * R + cu) * F;

        int out_base = idx * 4 * out_dim + l * F;

        for (int f = 0; f < F; f++) {
            output[out_base + 0 * out_dim + f] = level_grid[idx00 + f];
            output[out_base + 1 * out_dim + f] = level_grid[idx10 + f];
            output[out_base + 2 * out_dim + f] = level_grid[idx01 + f];
            output[out_base + 3 * out_dim + f] = level_grid[idx11 + f];
        }
    }
}

// Backward: grad_output [B, 4*L*F] -> grad_grid [L*R*R*F]
__global__ void corner_lookup_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ uv,
    float* __restrict__ grad_grid,
    int B, int R, int L, int F
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float u = uv[idx * 2] * (float)R;
    float v = uv[idx * 2 + 1] * (float)R;

    int fu = (int)floorf(u);
    int fv = (int)floorf(v);
    int cu = min(fu + 1, R - 1);
    int cv = min(fv + 1, R - 1);

    int level_stride = R * R * F;
    int out_dim = L * F;

    for (int l = 0; l < L; l++) {
        float* level_grad = grad_grid + l * level_stride;

        int idx00 = (fv * R + fu) * F;
        int idx10 = (fv * R + cu) * F;
        int idx01 = (cv * R + fu) * F;
        int idx11 = (cv * R + cu) * F;

        int in_base = idx * 4 * out_dim + l * F;

        for (int f = 0; f < F; f++) {
            atomicAdd(&level_grad[idx00 + f], grad_output[in_base + 0 * out_dim + f]);
            atomicAdd(&level_grad[idx10 + f], grad_output[in_base + 1 * out_dim + f]);
            atomicAdd(&level_grad[idx01 + f], grad_output[in_base + 2 * out_dim + f]);
            atomicAdd(&level_grad[idx11 + f], grad_output[in_base + 3 * out_dim + f]);
        }
    }
}

// Launcher wrappers
torch::Tensor corner_lookup_forward(
    torch::Tensor uv,
    torch::Tensor grid,
    int64_t resolution,
    int64_t n_levels,
    int64_t n_features_per_level
) {
    int B = uv.size(0);
    int R = (int)resolution;
    int L = (int)n_levels;
    int F = (int)n_features_per_level;
    int out_dim = L * F;

    auto output = torch::empty({B, 4 * out_dim}, uv.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    corner_lookup_forward_kernel<<<blocks, threads>>>(
        uv.data_ptr<float>(),
        grid.data_ptr<float>(),
        output.data_ptr<float>(),
        B, R, L, F
    );
    return output;
}

torch::Tensor corner_lookup_backward(
    torch::Tensor grad_output,
    torch::Tensor uv,
    int64_t resolution,
    int64_t n_levels,
    int64_t n_features_per_level,
    int64_t total_params
) {
    int B = grad_output.size(0);
    int R = (int)resolution;
    int L = (int)n_levels;
    int F = (int)n_features_per_level;

    auto grad_grid = torch::zeros({total_params}, grad_output.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    corner_lookup_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        uv.data_ptr<float>(),
        grad_grid.data_ptr<float>(),
        B, R, L, F
    );
    return grad_grid;
}
"""

# C++ declarations only (no PYBIND11_MODULE — load_inline generates it via `functions`)
_cpp_source = """
#include <torch/extension.h>
torch::Tensor corner_lookup_forward(torch::Tensor uv, torch::Tensor grid, int64_t resolution, int64_t n_levels, int64_t n_features_per_level);
torch::Tensor corner_lookup_backward(torch::Tensor grad_output, torch::Tensor uv, int64_t resolution, int64_t n_levels, int64_t n_features_per_level, int64_t total_params);
"""

# Lazy-init compiled module
_corner_module = None

def _get_module():
    global _corner_module
    if _corner_module is None:
        _corner_module = load_inline(
            name="corner_lookup_cuda",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["corner_lookup_forward", "corner_lookup_backward"],
            extra_cuda_cflags=["-O2", "--use_fast_math"],
            verbose=False,
        )
    return _corner_module


class CornerLookupFunction(torch.autograd.Function):
    """Fused corner-four grid lookup with custom backward."""

    @staticmethod
    def forward(ctx, uv, grid_params, resolution, n_levels, n_features_per_level):
        mod = _get_module()
        output = mod.corner_lookup_forward(
            uv.contiguous(), grid_params.contiguous(),
            resolution, n_levels, n_features_per_level,
        )
        ctx.save_for_backward(uv)
        ctx.resolution = resolution
        ctx.n_levels = n_levels
        ctx.n_features_per_level = n_features_per_level
        ctx.grid_shape = grid_params.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        uv, = ctx.saved_tensors
        mod = _get_module()
        grad_grid = mod.corner_lookup_backward(
            grad_output.contiguous(), uv.contiguous(),
            ctx.resolution, ctx.n_levels, ctx.n_features_per_level,
            ctx.grid_shape[0],
        )
        return None, grad_grid, None, None, None


def corner_four_lookup(uv, grid_params, resolution, n_levels=1, n_features_per_level=None):
    """Fused corner-four grid lookup.

    Args:
        uv: [B, 2] normalized [0, 1)
        grid_params: [n_levels * R * R * n_features_per_level]
        resolution: grid resolution R
        n_levels: number of packed levels
        n_features_per_level: features per level (auto if None)
    Returns:
        [B, 4 * n_levels * n_features_per_level]
    """
    if not uv.is_cuda:
        raise RuntimeError("corner_four_lookup requires CUDA tensors")
    if n_features_per_level is None:
        n_features_per_level = grid_params.numel() // (n_levels * resolution * resolution)
    return CornerLookupFunction.apply(uv, grid_params, resolution, n_levels, n_features_per_level)
