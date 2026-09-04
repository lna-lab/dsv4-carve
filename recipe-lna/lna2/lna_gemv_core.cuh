#pragma once
//
// F3 inner: the small-R GEMV core of exllamav3's `exl3_gemv_kernel`, factored
// into a device function so a ticket scheduler can call it per expert.
//
// Source of truth (copied structure, not re-derived):
//   /run/media/tonoken3/DATA1/vllm-exl3-lab/exllamav3-src/exllamav3/
//     exllamav3_ext/quant/exl3_gemv_kernel.cuh:140-380
// The upstream kernel is a __global__ that does
//   [input Hadamard] grid.sync [ this core ] grid.sync [output Hadamard]
// and strides its `group` loop by blockIdx.x / gridDim.x.  The only changes
// here are:
//   * the two grid.sync stages are removed -- the F3 outer runs the Hadamards
//     itself under *team* barriers, so there is no grid-wide coordination
//   * the `group` loop bounds are parameters (group_begin / group_stride)
//     instead of blockIdx.x / gridDim.x, so the CTAs of one expert team split
//     the n-tiles between them
//   * `A` is already Hadamard-transformed on entry (upstream sets A = A_had)
//
// Everything numerically load-bearing is untouched: the per-lane trellis window
// constants (exl3_dq.cuh dq8_aligned_2bits / dq8<3,cb,4>), the register decode
// helpers exl3_gemv_ns::dq8_regs_{2,3}bits, the FP16 MMA with the FOLD-cadence
// FP32 fold, and the cross-warp k-split reduction through sh_red.
//
// Why this is the right inner for this machine: MEASURED R_e distribution on
// uniform routing is ~1.04 rows/expert at m=4 and ~1.16 at m=16, so the
// canonical exl3_gemm_kernel_inner spends 15/16 of its m16n8k16 MMA on padding
// and carries a 16xN FP32 C buffer for one live row.  This core carries
// ROWS=1 (MMODE 0) or ROWS=8 (MMODE 1) and streams B straight to registers.

#ifndef LNA2_NULL_DECODE
#define LNA2_NULL_DECODE 0
#endif

#include "quant/exl3_gemv_kernel.cuh"

namespace lna_gemv {

// Shared-memory footprint of one instantiation, so the host can reason about it.
// ROWS_CAP replaces upstream's MMODE: upstream has only ROWS 1 (m==1) and 8
// (2<=m<=8), but the MEASURED R_e histogram on this pack is R=1 for ~96% of
// experts and R<=3 for essentially all of them under uniform routing, so the
// 8-row sh_red is 4x larger than any real route needs.  ROWS_CAP in {1,2,4,8}
// gives the F3 order's R_e buckets and shrinks the only large shared allocation.
// PFD is the prefetch ring depth: the direct lever on bytes-in-flight, i.e. on
// the pressure Ken is asking for.  It must divide evenly into FOLD's cadence,
// so FOLD is tied to it.
template <int ROWS_CAP, int CFG, int PFD>
struct core_traits {
    static constexpr int WK   = CFG == 0 ? 16 : 8;
    static constexpr int WNT  = CFG == 0 ? 2 : 4;
    static constexpr int PF   = PFD;
    static constexpr int FOLD = PFD;
    static constexpr int THREADS = WK * 32;
    static constexpr int ROWS = ROWS_CAP;
    static constexpr int COLS = WNT * 16;
    static constexpr int SH_RED_FLOATS = WK * ROWS * COLS;
};

// One expert projection.  A: [size_m, size_k] half, already suh-scaled and
// Hadamard-transformed.  C: [size_m, size_n] float, written (not accumulated).
// The caller splits the n-tile groups across its team via group_begin/stride.
template <int bits, int cb, int ROWS_CAP, int CFG, int PFD>
__device__ __forceinline__ void gemv_core(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    float* __restrict__ C,
    int size_m, int size_k, int size_n,
    int group_begin, int group_stride,
    float* sh_red_raw)
{
    using T = core_traits<ROWS_CAP, CFG, PFD>;
    constexpr int WK = T::WK, WNT = T::WNT, PF = T::PF, FOLD = T::FOLD;
    constexpr int THREADS = T::THREADS, ROWS = T::ROWS, COLS = T::COLS;
    constexpr int TWORDS = 8 * bits;
    constexpr int LOADS = bits == 2 ? WNT / 2 : WNT;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    static_assert(bits == 2 || bits == 3, "F3 pack is K2/K3 only");
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    // sh_red[WK][ROWS][COLS], supplied by the caller so the two MMODE
    // instantiations can share one allocation.
    auto RED = [&](int w, int r, int c) -> float& {
        return sh_red_raw[(w * ROWS + r) * COLS + c];
    };

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = ROWS_CAP == 1 ? lane < 4 : r0 < size_m;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3) {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    for (int group = group_begin; group < num_groups; group += group_stride)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        auto ld_b = [&] (int i, int l) -> uint32_t {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l) bw[l] = pf[d][l];

            if (i + PF < myn) {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l) pf[d][l] = ld_b(i + PF, l);
            }

            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

#if LNA2_NULL_DECODE
            // Pressure gate instrument: keep the exact load pattern, ring depth
            // and scheduling, drop trellis decode + MMA.  The gap between this
            // and the full kernel IS the compute ceiling, measured rather than
            // argued.  The loads must stay live, hence the accumulate.
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                acc0[l % WNT][0].x += (float) (bw[l] & 1u);
#else
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (bits == 2) {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                } else {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }
                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }
#endif

            if ((d + 1) % FOLD == 0 || i + 1 == myn) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
        }

        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = ROWS_CAP == 1 ? lane < 4 : r0 < ROWS;
            const int sr0 = ROWS_CAP == 1 ? 0 : r0;
            if (store0) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int col = t * 16 + f * 8 + c0;
                        RED(warp, sr0, col + 0) = acc0[t][f].x;
                        RED(warp, sr0, col + 1) = acc0[t][f].y;
                    }
            }
        }
        __syncthreads();

        const int rows_out = ROWS_CAP == 1 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j) sum += RED(j, r, c);
            C[(size_t) r * size_n + group * COLS + c] = sum;
        }
        __syncthreads();
    }
}

}  // namespace lna_gemv
