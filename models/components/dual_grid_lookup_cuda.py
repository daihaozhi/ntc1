"""Fused CUDA kernel for dual-grid lookup: high-res corner-four + low-res bilinear.

Single kernel launch replaces:
  1. custom CUDA corner-four on high-res grid → [B, 4*Dh]
  2. tcnn bilinear on low-res grid → [B, Dl]
  3. torch.cat → [B, 4*Dh + Dl]

The bilinear part computes standard 4-corner weighted interpolation.
"""

import torch
from torch.utils.cpp_extension import load_inline


_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ── Forward kernel ─────────────────────────────────────────────────
__global__ void dual_grid_forward_kernel(
    const float* __restrict__ uv,           // [B, 2]
    const float* __restrict__ grid_high,    // [Lh*Rh*Rh*Fh]
    const float* __restrict__ grid_low,     // [Ll*Rl*Rl*Fl]
    float* __restrict__ output,             // [B, 4*Lh*Fh + Ll*Fl]
    int B,
    int Rh, int Lh, int Fh,                 // high-res grid params
    int Rl, int Ll, int Fl                  // low-res grid params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    // ── High-res: corner-four nearest lookup ──
    float uh = u * (float)Rh;
    float vh = v * (float)Rh;
    int fu_h = (int)floorf(uh);
    int fv_h = (int)floorf(vh);
    int cu_h = (fu_h + 1) % Rh;
    int cv_h = (fv_h + 1) % Rh;

    int level_stride_h = Rh * Rh * Fh;
    int out_dim_h = Lh * Fh;
    int offset_high = 0;  // high-res features start at output[0]

    for (int l = 0; l < Lh; l++) {
        const float* level_grid = grid_high + l * level_stride_h;

        int i00 = (fv_h * Rh + fu_h) * Fh;
        int i10 = (fv_h * Rh + cu_h) * Fh;
        int i01 = (cv_h * Rh + fu_h) * Fh;
        int i11 = (cv_h * Rh + cu_h) * Fh;

        int out_base = idx * (4 * out_dim_h + Ll * Fl) + offset_high + l * Fh;

        for (int f = 0; f < Fh; f++) {
            output[out_base + 0 * out_dim_h + f] = level_grid[i00 + f];
            output[out_base + 1 * out_dim_h + f] = level_grid[i10 + f];
            output[out_base + 2 * out_dim_h + f] = level_grid[i01 + f];
            output[out_base + 3 * out_dim_h + f] = level_grid[i11 + f];
        }
    }

    // ── Low-res: bilinear interpolation ──
    float ul = u * (float)Rl;
    float vl = v * (float)Rl;
    int fu_l = (int)floorf(ul);
    int fv_l = (int)floorf(vl);
    int cu_l = (fu_l + 1) % Rl;
    int cv_l = (fv_l + 1) % Rl;
    float wu = ul - (float)fu_l;
    float wv = vl - (float)fv_l;

    int level_stride_l = Rl * Rl * Fl;
    int out_dim_l = Ll * Fl;
    int offset_low = 4 * out_dim_h;  // low-res features after high-res

    for (int l = 0; l < Ll; l++) {
        const float* level_grid = grid_low + l * level_stride_l;

        int i00 = (fv_l * Rl + fu_l) * Fl;
        int i10 = (fv_l * Rl + cu_l) * Fl;
        int i01 = (cv_l * Rl + fu_l) * Fl;
        int i11 = (cv_l * Rl + cu_l) * Fl;

        int out_base = idx * (4 * out_dim_h + out_dim_l) + offset_low + l * Fl;

        float w00 = (1.0f - wu) * (1.0f - wv);
        float w10 = wu * (1.0f - wv);
        float w01 = (1.0f - wu) * wv;
        float w11 = wu * wv;

        for (int f = 0; f < Fl; f++) {
            output[out_base + f] = w00 * level_grid[i00 + f]
                                 + w10 * level_grid[i10 + f]
                                 + w01 * level_grid[i01 + f]
                                 + w11 * level_grid[i11 + f];
        }
    }
}

// ── Backward kernel ────────────────────────────────────────────────
__global__ void dual_grid_backward_kernel(
    const float* __restrict__ grad_output,  // [B, 4*Lh*Fh + Ll*Fl]
    const float* __restrict__ uv,           // [B, 2]
    float* __restrict__ grad_high,          // [Lh*Rh*Rh*Fh]
    float* __restrict__ grad_low,           // [Ll*Rl*Rl*Fl]
    int B,
    int Rh, int Lh, int Fh,
    int Rl, int Ll, int Fl
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    // ── High-res gradient (nearest → atomicAdd to corners) ──
    float uh = u * (float)Rh;
    float vh = v * (float)Rh;
    int fu_h = (int)floorf(uh);
    int fv_h = (int)floorf(vh);
    int cu_h = (fu_h + 1) % Rh;
    int cv_h = (fv_h + 1) % Rh;

    int level_stride_h = Rh * Rh * Fh;
    int out_dim_h = Lh * Fh;
    int offset_high = 0;

    for (int l = 0; l < Lh; l++) {
        float* level_grad = grad_high + l * level_stride_h;

        int i00 = (fv_h * Rh + fu_h) * Fh;
        int i10 = (fv_h * Rh + cu_h) * Fh;
        int i01 = (cv_h * Rh + fu_h) * Fh;
        int i11 = (cv_h * Rh + cu_h) * Fh;

        int in_base = idx * (4 * out_dim_h + Ll * Fl) + offset_high + l * Fh;

        for (int f = 0; f < Fh; f++) {
            atomicAdd(&level_grad[i00 + f], grad_output[in_base + 0 * out_dim_h + f]);
            atomicAdd(&level_grad[i10 + f], grad_output[in_base + 1 * out_dim_h + f]);
            atomicAdd(&level_grad[i01 + f], grad_output[in_base + 2 * out_dim_h + f]);
            atomicAdd(&level_grad[i11 + f], grad_output[in_base + 3 * out_dim_h + f]);
        }
    }

    // ── Low-res gradient (bilinear → weighted atomicAdd to corners) ──
    float ul = u * (float)Rl;
    float vl = v * (float)Rl;
    int fu_l = (int)floorf(ul);
    int fv_l = (int)floorf(vl);
    int cu_l = (fu_l + 1) % Rl;
    int cv_l = (fv_l + 1) % Rl;
    float wu = ul - (float)fu_l;
    float wv = vl - (float)fv_l;

    int level_stride_l = Rl * Rl * Fl;
    int out_dim_l = Ll * Fl;
    int offset_low = 4 * out_dim_h;

    for (int l = 0; l < Ll; l++) {
        float* level_grad = grad_low + l * level_stride_l;

        int i00 = (fv_l * Rl + fu_l) * Fl;
        int i10 = (fv_l * Rl + cu_l) * Fl;
        int i01 = (cv_l * Rl + fu_l) * Fl;
        int i11 = (cv_l * Rl + cu_l) * Fl;

        int in_base = idx * (4 * out_dim_h + out_dim_l) + offset_low + l * Fl;

        float w00 = (1.0f - wu) * (1.0f - wv);
        float w10 = wu * (1.0f - wv);
        float w01 = (1.0f - wu) * wv;
        float w11 = wu * wv;

        for (int f = 0; f < Fl; f++) {
            float g = grad_output[in_base + f];
            atomicAdd(&level_grad[i00 + f], w00 * g);
            atomicAdd(&level_grad[i10 + f], w10 * g);
            atomicAdd(&level_grad[i01 + f], w01 * g);
            atomicAdd(&level_grad[i11 + f], w11 * g);
        }
    }
}

// ── Launcher wrappers ──────────────────────────────────────────────
torch::Tensor dual_grid_forward(
    torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
) {
    int B = uv.size(0);
    int out_dim_h = (int)n_levels_high * (int)n_feat_high;
    int out_dim_l = (int)n_levels_low  * (int)n_feat_low;
    int total_out = 4 * out_dim_h + out_dim_l;

    auto output = torch::empty({B, total_out}, uv.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    dual_grid_forward_kernel<<<blocks, threads>>>(
        uv.data_ptr<float>(),
        grid_high.data_ptr<float>(), grid_low.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        (int)res_high, (int)n_levels_high, (int)n_feat_high,
        (int)res_low,  (int)n_levels_low,  (int)n_feat_low
    );
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> dual_grid_backward(
    torch::Tensor grad_output,
    torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
) {
    int B = grad_output.size(0);

    auto grad_high = torch::zeros({grid_high.numel()}, grad_output.options());
    auto grad_low  = torch::zeros({grid_low.numel()}, grad_output.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    dual_grid_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        uv.data_ptr<float>(),
        grad_high.data_ptr<float>(), grad_low.data_ptr<float>(),
        B,
        (int)res_high, (int)n_levels_high, (int)n_feat_high,
        (int)res_low,  (int)n_levels_low,  (int)n_feat_low
    );
    return std::make_tuple(grad_high, grad_low);
}
"""

_cpp_source = """
#include <torch/extension.h>
torch::Tensor dual_grid_forward(
    torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
);
std::tuple<torch::Tensor, torch::Tensor> dual_grid_backward(
    torch::Tensor grad_output, torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
);
"""

_dual_module = None

def _get_module():
    global _dual_module
    if _dual_module is None:
        _dual_module = load_inline(
            name="dual_grid_lookup_cuda",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["dual_grid_forward", "dual_grid_backward"],
            extra_cuda_cflags=["-O2", "--use_fast_math"],
            verbose=False,
        )
    return _dual_module


# ═════════════════════════════════════════════════════════════════════
# PyTorch autograd Function
# ═════════════════════════════════════════════════════════════════════

def _compute_packing(feature_dim: int):
    """Compute n_levels, n_features from tcnn packing convention."""
    if feature_dim <= 8:
        return 1, feature_dim
    return feature_dim // 4, 4


class DualGridLookupFunction(torch.autograd.Function):
    """Fused high-res (corner-four) + low-res (bilinear) grid lookup."""

    @staticmethod
    def forward(ctx, uv, grid_high, grid_low,
                res_high, fdim_high, res_low, fdim_low):
        Lh, Fh = _compute_packing(fdim_high)
        Ll, Fl = _compute_packing(fdim_low)

        mod = _get_module()
        output = mod.dual_grid_forward(
            uv.contiguous(),
            grid_high.contiguous(), grid_low.contiguous(),
            res_high, Lh, Fh,
            res_low,  Ll, Fl,
        )
        ctx.save_for_backward(uv, grid_high, grid_low)
        ctx.res_high, ctx.Lh, ctx.Fh = res_high, Lh, Fh
        ctx.res_low,  ctx.Ll, ctx.Fl = res_low,  Ll, Fl
        return output

    @staticmethod
    def backward(ctx, grad_output):
        uv, grid_high, grid_low = ctx.saved_tensors
        mod = _get_module()
        grad_high, grad_low = mod.dual_grid_backward(
            grad_output.contiguous(), uv.contiguous(),
            grid_high, grid_low,
            ctx.res_high, ctx.Lh, ctx.Fh,
            ctx.res_low,  ctx.Ll, ctx.Fl,
        )
        return None, grad_high, grad_low, None, None, None, None


def dual_grid_lookup(uv, grid_high, grid_low, res_high, fdim_high, res_low, fdim_low):
    """Fused dual-grid lookup.

    Args:
        uv: [B, 2] normalized [0, 1)
        grid_high: flat params for high-res grid [Lh*Rh*Rh*Fh]
        grid_low:  flat params for low-res grid  [Ll*Rl*Rl*Fl]
        res_high, fdim_high: resolution and logical feature dim for high-res
        res_low,  fdim_low:  resolution and logical feature dim for low-res

    Returns:
        [B, 4*Lh*Fh + Ll*Fl] — concatenated high-res (4-corner) + low-res (bilinear)
    """
    if not uv.is_cuda:
        raise RuntimeError("dual_grid_lookup requires CUDA tensors")
    return DualGridLookupFunction.apply(
        uv, grid_high, grid_low,
        res_high, fdim_high, res_low, fdim_low,
    )
