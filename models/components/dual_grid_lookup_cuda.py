"""Fused CUDA kernel: triangle-wave PE + dual-grid lookup in one launch.

Single kernel replaces:
  1. Triangle-wave positional encoding (5 freqs, tiled)
  2. High-res corner-four grid lookup
  3. Low-res bilinear grid lookup
  4. torch.cat(PE, grid_high, grid_low)

Output layout: [PE(12D) | high_res(4*Dh) | low_res(Dl)] = [B, 112D for 4096 config]
"""

import torch
from torch.utils.cpp_extension import load_inline


_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ── Forward kernel: PE + dual-grid ────────────────────────────────
__global__ void pe_dual_grid_forward_kernel(
    const float* __restrict__ uv,           // [B, 2]
    const float* __restrict__ grid_high,    // [Lh*Rh*Rh*Fh]
    const float* __restrict__ grid_low,     // [Ll*Rl*Rl*Fl]
    float* __restrict__ output,             // [B, pe_dim + 4*Lh*Fh + Ll*Fl]
    int B,
    int n_freq, int tiled, int tile_size,   // PE params
    int Rh, int Lh, int Fh,                 // high-res grid
    int Rl, int Ll, int Fl                  // low-res grid
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    int pe_dim = n_freq * 2 + 2;             // e.g. 5*2+2 = 12
    int out_dim_h = Lh * Fh;                 // high-res per-corner feature dim
    int out_dim_l = Ll * Fl;                 // low-res feature dim
    int total_out = pe_dim + 4 * out_dim_h + out_dim_l;

    // ── Positional encoding (triangle wave) ──
    float lu = u, lv = v;
    if (tiled) {
        lu = lu * (float)tile_size - floorf(lu * (float)tile_size);
        lv = lv * (float)tile_size - floorf(lv * (float)tile_size);
    }

    for (int f = 0; f < n_freq; f++) {
        float scale = exp2f((float)f);
        float su = lu * scale;
        float sv = lv * scale;
        // tri(x) = 2 * |x - floor(x + 0.5)|
        output[idx * total_out + f * 2]     = 2.0f * fabsf(su - floorf(su + 0.5f));
        output[idx * total_out + f * 2 + 1] = 2.0f * fabsf(sv - floorf(sv + 0.5f));
    }
    // Constant channels
    output[idx * total_out + n_freq * 2]     = 1.0f;
    output[idx * total_out + n_freq * 2 + 1] = 1.0f;

    // ── High-res: corner-four nearest ──
    float uh = u * (float)Rh;
    float vh = v * (float)Rh;
    int fu_h = (int)floorf(uh);
    int fv_h = (int)floorf(vh);
    int cu_h = (fu_h + 1) % Rh;
    int cv_h = (fv_h + 1) % Rh;

    int level_stride_h = Rh * Rh * Fh;

    for (int l = 0; l < Lh; l++) {
        const float* level_grid = grid_high + l * level_stride_h;
        int i00 = (fv_h * Rh + fu_h) * Fh;
        int i10 = (fv_h * Rh + cu_h) * Fh;
        int i01 = (cv_h * Rh + fu_h) * Fh;
        int i11 = (cv_h * Rh + cu_h) * Fh;

        int out_base = idx * total_out + pe_dim + l * Fh;
        for (int f = 0; f < Fh; f++) {
            output[out_base + 0 * out_dim_h + f] = level_grid[i00 + f];
            output[out_base + 1 * out_dim_h + f] = level_grid[i10 + f];
            output[out_base + 2 * out_dim_h + f] = level_grid[i01 + f];
            output[out_base + 3 * out_dim_h + f] = level_grid[i11 + f];
        }
    }

    // ── Low-res: bilinear ──
    float ul = u * (float)Rl;
    float vl = v * (float)Rl;
    int fu_l = (int)floorf(ul);
    int fv_l = (int)floorf(vl);
    int cu_l = (fu_l + 1) % Rl;
    int cv_l = (fv_l + 1) % Rl;
    float wu = ul - (float)fu_l;
    float wv = vl - (float)fv_l;

    int level_stride_l = Rl * Rl * Fl;
    int offset_low = pe_dim + 4 * out_dim_h;

    for (int l = 0; l < Ll; l++) {
        const float* level_grid = grid_low + l * level_stride_l;
        int i00 = (fv_l * Rl + fu_l) * Fl;
        int i10 = (fv_l * Rl + cu_l) * Fl;
        int i01 = (cv_l * Rl + fu_l) * Fl;
        int i11 = (cv_l * Rl + cu_l) * Fl;

        int out_base = idx * total_out + offset_low + l * Fl;
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

// ── Backward kernel (PE has no parameters → gradient flows to grids only) ──
__global__ void pe_dual_grid_backward_kernel(
    const float* __restrict__ grad_output,  // [B, pe_dim + 4*Lh*Fh + Ll*Fl]
    const float* __restrict__ uv,
    float* __restrict__ grad_high,
    float* __restrict__ grad_low,
    int B,
    int n_freq, int tiled, int tile_size,
    int Rh, int Lh, int Fh,
    int Rl, int Ll, int Fl
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;

    int pe_dim = n_freq * 2 + 2;
    int out_dim_h = Lh * Fh;
    int out_dim_l = Ll * Fl;
    int total_out = pe_dim + 4 * out_dim_h + out_dim_l;
    int offset_high = pe_dim;             // grid gradients start after PE
    int offset_low  = pe_dim + 4 * out_dim_h;

    float u = uv[idx * 2];
    float v = uv[idx * 2 + 1];

    // ── High-res gradient (atomicAdd to 4 corners) ──
    float uh = u * (float)Rh;
    float vh = v * (float)Rh;
    int fu_h = (int)floorf(uh);
    int fv_h = (int)floorf(vh);
    int cu_h = (fu_h + 1) % Rh;
    int cv_h = (fv_h + 1) % Rh;
    int level_stride_h = Rh * Rh * Fh;

    for (int l = 0; l < Lh; l++) {
        float* level_grad = grad_high + l * level_stride_h;
        int i00 = (fv_h * Rh + fu_h) * Fh;
        int i10 = (fv_h * Rh + cu_h) * Fh;
        int i01 = (cv_h * Rh + fu_h) * Fh;
        int i11 = (cv_h * Rh + cu_h) * Fh;
        int in_base = idx * total_out + offset_high + l * Fh;

        for (int f = 0; f < Fh; f++) {
            atomicAdd(&level_grad[i00 + f], grad_output[in_base + 0 * out_dim_h + f]);
            atomicAdd(&level_grad[i10 + f], grad_output[in_base + 1 * out_dim_h + f]);
            atomicAdd(&level_grad[i01 + f], grad_output[in_base + 2 * out_dim_h + f]);
            atomicAdd(&level_grad[i11 + f], grad_output[in_base + 3 * out_dim_h + f]);
        }
    }

    // ── Low-res gradient (bilinear weighted atomicAdd) ──
    float ul = u * (float)Rl;
    float vl = v * (float)Rl;
    int fu_l = (int)floorf(ul);
    int fv_l = (int)floorf(vl);
    int cu_l = (fu_l + 1) % Rl;
    int cv_l = (fv_l + 1) % Rl;
    float wu = ul - (float)fu_l;
    float wv = vl - (float)fv_l;
    int level_stride_l = Rl * Rl * Fl;

    for (int l = 0; l < Ll; l++) {
        float* level_grad = grad_low + l * level_stride_l;
        int i00 = (fv_l * Rl + fu_l) * Fl;
        int i10 = (fv_l * Rl + cu_l) * Fl;
        int i01 = (cv_l * Rl + fu_l) * Fl;
        int i11 = (cv_l * Rl + cu_l) * Fl;
        int in_base = idx * total_out + offset_low + l * Fl;

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
torch::Tensor pe_dual_grid_forward(
    torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t n_freq, bool tiled, int64_t tile_size,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
) {
    int B = uv.size(0);
    int pe_dim = (int)n_freq * 2 + 2;
    int out_dim_h = (int)n_levels_high * (int)n_feat_high;
    int out_dim_l = (int)n_levels_low  * (int)n_feat_low;
    int total = pe_dim + 4 * out_dim_h + out_dim_l;

    auto output = torch::empty({B, total}, uv.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    pe_dual_grid_forward_kernel<<<blocks, threads>>>(
        uv.data_ptr<float>(),
        grid_high.data_ptr<float>(), grid_low.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        (int)n_freq, (int)tiled, (int)tile_size,
        (int)res_high, (int)n_levels_high, (int)n_feat_high,
        (int)res_low,  (int)n_levels_low,  (int)n_feat_low
    );
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> pe_dual_grid_backward(
    torch::Tensor grad_output, torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t n_freq, bool tiled, int64_t tile_size,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
) {
    int B = grad_output.size(0);
    auto grad_high = torch::zeros({grid_high.numel()}, grad_output.options());
    auto grad_low  = torch::zeros({grid_low.numel()},  grad_output.options());

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    pe_dual_grid_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(), uv.data_ptr<float>(),
        grad_high.data_ptr<float>(), grad_low.data_ptr<float>(),
        B,
        (int)n_freq, (int)tiled, (int)tile_size,
        (int)res_high, (int)n_levels_high, (int)n_feat_high,
        (int)res_low,  (int)n_levels_low,  (int)n_feat_low
    );
    return std::make_tuple(grad_high, grad_low);
}
"""

_cpp_source = """
#include <torch/extension.h>
torch::Tensor pe_dual_grid_forward(
    torch::Tensor uv, torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t n_freq, bool tiled, int64_t tile_size,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
);
std::tuple<torch::Tensor, torch::Tensor> pe_dual_grid_backward(
    torch::Tensor grad_output, torch::Tensor uv,
    torch::Tensor grid_high, torch::Tensor grid_low,
    int64_t n_freq, bool tiled, int64_t tile_size,
    int64_t res_high, int64_t n_levels_high, int64_t n_feat_high,
    int64_t res_low,  int64_t n_levels_low,  int64_t n_feat_low
);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="pe_dual_grid_cuda",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["pe_dual_grid_forward", "pe_dual_grid_backward"],
            extra_cuda_cflags=["-O2", "--use_fast_math"],
            verbose=False,
        )
    return _module


def _compute_packing(feature_dim: int):
    if feature_dim <= 8:
        return 1, feature_dim
    return feature_dim // 4, 4


class PEDualGridFunction(torch.autograd.Function):
    """Fused PE + dual-grid lookup. Backward only flows through grid params."""

    @staticmethod
    def forward(ctx, uv, grid_high, grid_low,
                n_freq, tiled, tile_size,
                res_high, fdim_high, res_low, fdim_low):
        Lh, Fh = _compute_packing(fdim_high)
        Ll, Fl = _compute_packing(fdim_low)

        mod = _get_module()
        output = mod.pe_dual_grid_forward(
            uv.contiguous(),
            grid_high.contiguous(), grid_low.contiguous(),
            n_freq, tiled, tile_size,
            res_high, Lh, Fh,
            res_low,  Ll, Fl,
        )
        ctx.save_for_backward(uv, grid_high, grid_low)
        ctx.n_freq, ctx.tiled, ctx.tile_size = n_freq, tiled, tile_size
        ctx.res_high, ctx.Lh, ctx.Fh = res_high, Lh, Fh
        ctx.res_low,  ctx.Ll, ctx.Fl = res_low,  Ll, Fl
        ctx.pe_dim = n_freq * 2 + 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        uv, grid_high, grid_low = ctx.saved_tensors
        mod = _get_module()
        grad_high, grad_low = mod.pe_dual_grid_backward(
            grad_output.contiguous(), uv.contiguous(),
            grid_high, grid_low,
            ctx.n_freq, ctx.tiled, ctx.tile_size,
            ctx.res_high, ctx.Lh, ctx.Fh,
            ctx.res_low,  ctx.Ll, ctx.Fl,
        )
        # No gradient for uv, n_freq, tiled, tile_size, resolutions, feature dims
        return (None, grad_high, grad_low,
                None, None, None, None, None, None, None)


def pe_dual_grid_lookup(uv, grid_high, grid_low,
                         n_freq, tiled, tile_size,
                         res_high, fdim_high, res_low, fdim_low):
    """Fused PE + dual-grid lookup.

    Returns [B, PE_DIM + 4*Dh + Dl] where PE_DIM = n_freq*2 + 2.
    """
    if not uv.is_cuda:
        raise RuntimeError("pe_dual_grid_lookup requires CUDA tensors")
    return PEDualGridFunction.apply(
        uv, grid_high, grid_low,
        n_freq, tiled, tile_size,
        res_high, fdim_high, res_low, fdim_low,
    )
