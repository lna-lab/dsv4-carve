#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <mutex>

#include <ATen/cuda/CUDAContext.h>

#include "util.h"
#include "util.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_gemm_inner.cuh"

namespace cg = cooperative_groups;

namespace lna_detail {

constexpr int HIDDEN = 4096;
constexpr int INTERMEDIATE = 256;
constexpr int TOPK = 6;
constexpr int MAX_M = 16;
constexpr int MAX_ROUTES = MAX_M * TOPK;
// Pointer tables are indexed by rank-local expert id (0..255).  MAX_GROUPS is a
// different quantity: the number of *distinct* experts a single route can name,
// bounded by 6m <= 96.  Conflating the two rejected every real layer.
constexpr int MAX_EXPERTS = 256;
constexpr int MAX_GROUPS = MAX_ROUTES;
constexpr int MAX_PAIRS = MAX_ROUTES;
constexpr int MAX_BLOCKS = 512;
// The inner GEMM indexes locks[slice_m * blocks_n + slice2_n] with slice_m = 0
// and slice2_n < tiles_n (exl3_gemm_inner.cuh:589).  tiles_n is 2 for gate/up
// (256/128) and 8 for down (4096/512), so 16 is ample.
constexpr int MAX_LOCKS = 16;
constexpr float HAD_NORM = 0.088388347648f;

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

constexpr std::size_t O_GROUP_COUNT = 0;
constexpr std::size_t O_PAIR_COUNT = O_GROUP_COUNT + sizeof(int);
constexpr std::size_t O_GROUP_EXPERT = O_PAIR_COUNT + sizeof(int);
constexpr std::size_t O_GROUP_ROWS_COUNT = O_GROUP_EXPERT + MAX_GROUPS * sizeof(int);
constexpr std::size_t O_GROUP_PAIR_BASE = O_GROUP_ROWS_COUNT + MAX_GROUPS * sizeof(int);
constexpr std::size_t O_GROUP_ROWS = O_GROUP_PAIR_BASE + MAX_GROUPS * sizeof(int);
constexpr std::size_t O_SLOT_PAIR = O_GROUP_ROWS + MAX_GROUPS * MAX_M * sizeof(int);
constexpr std::size_t O_PAIR_GROUP = O_SLOT_PAIR + MAX_ROUTES * sizeof(int);
constexpr std::size_t O_ERROR = O_PAIR_GROUP + MAX_PAIRS * sizeof(int);

constexpr std::size_t O_HAD_GATE = align_up(O_ERROR + sizeof(int), 16);
constexpr std::size_t O_HAD_UP = O_HAD_GATE + MAX_PAIRS * HIDDEN * sizeof(half);
constexpr std::size_t O_GATE = O_HAD_UP + MAX_PAIRS * HIDDEN * sizeof(half);
constexpr std::size_t O_UP = O_GATE + MAX_PAIRS * INTERMEDIATE * sizeof(float);
constexpr std::size_t O_ACT = O_UP + MAX_PAIRS * INTERMEDIATE * sizeof(float);
constexpr std::size_t O_DOWN_IN = O_ACT + MAX_PAIRS * INTERMEDIATE * sizeof(half);
constexpr std::size_t O_DOWN = O_DOWN_IN + MAX_PAIRS * INTERMEDIATE * sizeof(half);
constexpr std::size_t O_LOCKS = O_DOWN + MAX_PAIRS * HIDDEN * sizeof(float);
// Phase clocks: 12 slots of clock64() taken by linear block 0, thread 0.
constexpr std::size_t O_CLOCK = O_LOCKS + MAX_BLOCKS * MAX_LOCKS * sizeof(int);
constexpr std::size_t O_END = O_CLOCK + 16 * sizeof(long long);

static_assert(O_END < (8u << 20), "F1 scratch unexpectedly large");

template <typename T>
__device__ __forceinline__ T* at(char* base, std::size_t offset) {
    return reinterpret_cast<T*>(base + offset);
}

template <typename T>
__device__ __forceinline__ const T* at(const char* base, std::size_t offset) {
    return reinterpret_cast<const T*>(base + offset);
}

__device__ __forceinline__ int block_linear() {
    return blockIdx.z * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ int blocks_linear() {
    return gridDim.x * gridDim.z;
}

__device__ __forceinline__ int global_warp() {
    return ((blockIdx.z * gridDim.x + blockIdx.x) * (blockDim.x / 32)) +
           (threadIdx.x / 32);
}

__device__ __forceinline__ int global_warps() {
    return gridDim.x * gridDim.z * (blockDim.x / 32);
}

__device__ __forceinline__ half4 load_x_half4(
    const void* x, int row, int input_block, bool bf16) {
    const int offset = row * HIDDEN + input_block * 128 +
                       (threadIdx.x & 31) * 4;
    half4 result;
    if (bf16) {
        const auto* src = reinterpret_cast<const __nv_bfloat16*>(x) + offset;
        result.x = __floats2half2_rn(__bfloat162float(src[0]),
                                     __bfloat162float(src[1]));
        result.y = __floats2half2_rn(__bfloat162float(src[2]),
                                     __bfloat162float(src[3]));
    } else {
        // Load four half values as two half2 values so the BF16 and FP16 paths
        // share the same Hadamard body and rounding boundary.
        const auto* src = reinterpret_cast<const half*>(x) + offset;
        result.x = __halves2half2(src[0], src[1]);
        result.y = __halves2half2(src[2], src[3]);
    }
    return result;
}

__device__ __forceinline__ void had_from_half4(
    half4 v, half* output, const half* scale, bool debug_finite, int* error) {
    const int lane = threadIdx.x & 31;
    const half4 scales = reinterpret_cast<const half4*>(scale)[lane];
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    if (debug_finite && (!isfinite(h0) || !isfinite(h1) ||
                         !isfinite(h2) || !isfinite(h3))) {
        atomicExch(error, 1);
    }
    v.x = __floats2half2_rn(h0 * HAD_NORM, h1 * HAD_NORM);
    v.y = __floats2half2_rn(h2 * HAD_NORM, h3 * HAD_NORM);
    reinterpret_cast<half4*>(output)[lane] = v;
}

// Route plan.
//
// MEASURED (run-27, f1/phase_probe.py): the original single-threaded plan cost
// 262,650 cycles at m=4 (29.8% of the kernel) and 2,827,074 cycles at m=16
// (66.9%) -- it walked O(routes x groups x rows) linear scans through *global*
// memory.  This version is one 256-thread block, one thread per expert id:
//   1. a 16-bit row bitmask per expert, built with shared-memory atomics
//   2. two Hillis-Steele exclusive scans over the 256 experts, giving the group
//      index and the pair base
//   3. slot -> pair via popcount of the rows below this one
// Groups are now numbered in ascending expert id rather than first-appearance
// order.  That is still fully deterministic, and the gather still walks the
// original [m,6] slot order, so the FP32 accumulation order is unchanged.
__device__ __forceinline__ void route_plan(
    const int32_t* ids, int m, int n_experts, char* scratch) {
    int* group_count = at<int>(scratch, O_GROUP_COUNT);
    int* pair_count = at<int>(scratch, O_PAIR_COUNT);
    int* group_expert = at<int>(scratch, O_GROUP_EXPERT);
    int* group_rows_count = at<int>(scratch, O_GROUP_ROWS_COUNT);
    int* group_pair_base = at<int>(scratch, O_GROUP_PAIR_BASE);
    int* group_rows = at<int>(scratch, O_GROUP_ROWS);
    int* slot_pair = at<int>(scratch, O_SLOT_PAIR);
    int* pair_group = at<int>(scratch, O_PAIR_GROUP);
    int* error = at<int>(scratch, O_ERROR);

    __shared__ int sh_rowmask[MAX_EXPERTS];
    __shared__ int sh_gid[MAX_EXPERTS];
    __shared__ int sh_base[MAX_EXPERTS];
    __shared__ int sc_a[MAX_EXPERTS];
    __shared__ int sc_b[MAX_EXPERTS];

    // blockDim.x may be 512; only the first 256 threads own an expert id, but
    // every thread must reach every __syncthreads below.
    const int e = threadIdx.x;
    const bool own = e < MAX_EXPERTS;
    if (own) sh_rowmask[e] = 0;
    if (e == 0) *error = 0;
    __syncthreads();

    const int n_slots = m * TOPK;
    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        const int id = ids[slot];
        if (id >= 0 && id < n_experts)
            atomicOr(&sh_rowmask[id], 1 << (slot / TOPK));
    }
    __syncthreads();

    const int mask = own ? sh_rowmask[e] : 0;
    const int has = mask ? 1 : 0;
    const int np = __popc(mask);
    if (own) { sc_a[e] = has; sc_b[e] = np; }
    __syncthreads();
    for (int off = 1; off < MAX_EXPERTS; off <<= 1) {
        const int va = (own && e >= off) ? sc_a[e - off] : 0;
        const int vb = (own && e >= off) ? sc_b[e - off] : 0;
        __syncthreads();
        if (own) { sc_a[e] += va; sc_b[e] += vb; }
        __syncthreads();
    }
    const int gexcl = own ? sc_a[e] - has : 0;
    const int pexcl = own ? sc_b[e] - np : 0;
    if (e == MAX_EXPERTS - 1) {
        *group_count = sc_a[e];
        *pair_count = sc_b[e];
        if (sc_a[e] > MAX_GROUPS || sc_b[e] > MAX_PAIRS) *error = 1;
    }
    if (own) { sh_gid[e] = gexcl; sh_base[e] = pexcl; }
    if (own && has && gexcl < MAX_GROUPS && pexcl + np <= MAX_PAIRS) {
        group_expert[gexcl] = e;
        group_rows_count[gexcl] = np;
        group_pair_base[gexcl] = pexcl;
        int rest = mask;
        for (int k = 0; rest; ++k) {
            const int row = __ffs(rest) - 1;
            rest &= rest - 1;
            group_rows[gexcl * MAX_M + k] = row;
            pair_group[pexcl + k] = gexcl;
        }
    }
    __syncthreads();

    for (int slot = threadIdx.x; slot < n_slots; slot += blockDim.x) {
        const int id = ids[slot];
        if (id < 0 || id >= n_experts) { slot_pair[slot] = -1; continue; }
        const int row = slot / TOPK;
        slot_pair[slot] = sh_base[id] + __popc(sh_rowmask[id] & ((1 << row) - 1));
    }
}

// LNA_THREADS is not free: the canonical inner GEMM needs
// EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16 threads, and splits them into sub_k
// pipelines (exl3_gemm_inner.cuh:75).  TILE_K=16 -> 256 threads, TILE_K=32 ->
// 512.  Both GEMMs in this kernel must agree on the block size, so GU and DOWN
// share one TILE_K.
#ifndef LNA_TILE_K
#define LNA_TILE_K 32
#endif
#define LNA_THREADS (256 * LNA_TILE_K / 16)

#ifndef LNA_GU_TILE_N
#define LNA_GU_TILE_N 256
#endif
#ifndef LNA_GU_STAGES
#define LNA_GU_STAGES 4
#endif
#ifndef LNA_GU_FSTAGES
#define LNA_GU_FSTAGES 3
#endif
#ifndef LNA_DOWN_TILE_N
#define LNA_DOWN_TILE_N 512
#endif
#ifndef LNA_DOWN_STAGES
#define LNA_DOWN_STAGES 4
#endif
#ifndef LNA_DOWN_FSTAGES
#define LNA_DOWN_FSTAGES 3
#endif

template <int K, int TILE_N, int STAGES, int FSTAGES>
__device__ __forceinline__ void run_gemm(
    const half* input, const uint16_t* trellis, float* output,
    int rows, int size_k, int size_n, int* locks) {
    // The canonical EXL3 inner GEMM performs the circular trellis extraction
    // and FP32 MMA accumulation; LNA only changes the scheduling around it.
    exl3_gemm_kernel_inner<K, true, 1, 16, LNA_TILE_K, TILE_N, STAGES, FSTAGES, false>(
        input, trellis, output, rows, size_k, size_n, locks, nullptr);
}

// MEASURED (run-27/28): the gate/up GEMM is ~69% of the kernel at m=4, and with
// TILE_N=128 each pipeline stage carries only ~512 B of B, so a block keeps
// ~1 KB in flight and reaches 0.6 B/cycle/SM (~50 GB/s over 34 SMs).  N=256
// doubles the tile and 6/5 stages deepen the pipeline; N=256 also makes
// tiles_n=1 for the 256-wide TP slice, so gridDim.x slicing splits k instead of
// n and the lock chain (exl3_gemm_inner.cuh:592) does the cross-block reduce.

template <int K, int M_CAP>
__global__ __launch_bounds__(LNA_THREADS)
void lna_moe_kernel(
    const void* __restrict__ x,
    void* __restrict__ out,
    const int64_t* __restrict__ gate_t_ptrs,
    const int64_t* __restrict__ gate_suh_ptrs,
    const int64_t* __restrict__ gate_svh_ptrs,
    const int64_t* __restrict__ up_t_ptrs,
    const int64_t* __restrict__ up_suh_ptrs,
    const int64_t* __restrict__ up_svh_ptrs,
    const int64_t* __restrict__ down_t_ptrs,
    const int64_t* __restrict__ down_suh_ptrs,
    const int64_t* __restrict__ down_svh_ptrs,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int m,
    int n_experts,
    int x_bf16,
    int out_kind,
    float limit,
    int debug_finite,
    char* __restrict__ scratch) {
    static_assert(M_CAP == 4 || M_CAP == 8 || M_CAP == 16);
    auto grid = cg::this_grid();
    int phase_i = 0;
    long long* clocks = at<long long>(scratch, O_CLOCK);
    const bool timing = block_linear() == 0 && threadIdx.x == 0;
    #define LNA_MARK() do { if (timing) clocks[phase_i] = clock64(); ++phase_i; } while (0)
    LNA_MARK();
    const bool bf16 = x_bf16 != 0;
    const bool dbg = debug_finite != 0;
    const int warp = global_warp();
    const int warps = global_warps();
    int* group_count = at<int>(scratch, O_GROUP_COUNT);
    int* pair_count = at<int>(scratch, O_PAIR_COUNT);
    int* group_expert = at<int>(scratch, O_GROUP_EXPERT);
    int* group_rows_count = at<int>(scratch, O_GROUP_ROWS_COUNT);
    int* group_pair_base = at<int>(scratch, O_GROUP_PAIR_BASE);
    int* group_rows = at<int>(scratch, O_GROUP_ROWS);
    int* pair_group = at<int>(scratch, O_PAIR_GROUP);
    int* error = at<int>(scratch, O_ERROR);

    // gridDim.x > 1 means several blocks share blockIdx.z, so the plan must be
    // pinned to one *linear* block or the single-threaded pass races itself.
    if (block_linear() == 0)
        route_plan(ids, m, n_experts, scratch);
    // The canonical GEMM barriers self-reset (ptx.cuh:131 barrier_release with
    // reset=true on the last slice), so the lock array is zeroed exactly once
    // here instead of per wave -- with S>1 blocks per group a per-wave memset
    // would race against slices already spinning on the same lock.
    for (int i = block_linear() * blockDim.x + threadIdx.x;
         i < MAX_BLOCKS * MAX_LOCKS; i += blocks_linear() * blockDim.x)
        at<int>(scratch, O_LOCKS)[i] = 0;
    grid.sync();
    LNA_MARK();

    // Input transform: each unique (expert,row) owns 32 independent 128-wide
    // blocks. The x load is shared by the gate and up transforms in this CTA.
    const int total_input_warps = *pair_count * (HIDDEN / 128);
    for (int w = warp; w < total_input_warps; w += warps) {
        const int pair = w / (HIDDEN / 128);
        const int ib = w % (HIDDEN / 128);
        const int g = pair_group[pair];
        const int e = group_expert[g];
        const int row = group_rows[g * MAX_M + (pair - group_pair_base[g])];
        const half4 v = load_x_half4(x, row, ib, bf16);
        half* hg = at<half>(scratch, O_HAD_GATE) + pair * HIDDEN + ib * 128;
        half* hu = at<half>(scratch, O_HAD_UP) + pair * HIDDEN + ib * 128;
        const half* gs = reinterpret_cast<const half*>(gate_suh_ptrs[e]) + ib * 128;
        const half* us = reinterpret_cast<const half*>(up_suh_ptrs[e]) + ib * 128;
        had_from_half4(v, hg, gs, dbg, error);
        had_from_half4(v, hu, us, dbg, error);
    }
    grid.sync();
    LNA_MARK();

    // One block is one expert group. Groups run in persistent waves, so the
    // launch remains exactly P blocks even when m=16 has more than P active
    // experts. With one block per group the canonical GEMM's lock range is
    // still supplied and remains correct for future wider-group variants.
    for (int wave = 0; wave * gridDim.z < *group_count; ++wave) {
        const int g = wave * gridDim.z + blockIdx.z;
        int* locks = at<int>(scratch, O_LOCKS) + blockIdx.z * MAX_LOCKS;
        if (g < *group_count) {
            const int e = group_expert[g];
            const int rows = group_rows_count[g];
            const int pair = group_pair_base[g];
            run_gemm<K, LNA_GU_TILE_N, LNA_GU_STAGES, LNA_GU_FSTAGES>(
                at<half>(scratch, O_HAD_GATE) + pair * HIDDEN,
                reinterpret_cast<const uint16_t*>(gate_t_ptrs[e]),
                at<float>(scratch, O_GATE) + pair * INTERMEDIATE,
                rows, HIDDEN, INTERMEDIATE, locks);
            run_gemm<K, LNA_GU_TILE_N, LNA_GU_STAGES, LNA_GU_FSTAGES>(
                at<half>(scratch, O_HAD_UP) + pair * HIDDEN,
                reinterpret_cast<const uint16_t*>(up_t_ptrs[e]),
                at<float>(scratch, O_UP) + pair * INTERMEDIATE,
                rows, HIDDEN, INTERMEDIATE, locks);
        }
        grid.sync();
    LNA_MARK();
    }

    // Output Hadamard/scales for gate and up, followed by the FP32 clamp and
    // SwiGLU. The activation is the only value converted to FP16 before down.
    const int total_inter_warps = *pair_count * (INTERMEDIATE / 128);
    for (int w = warp; w < total_inter_warps; w += warps) {
        const int pair = w / (INTERMEDIATE / 128);
        const int ib = w % (INTERMEDIATE / 128);
        const int g = pair_group[pair];
        const int e = group_expert[g];
        const int offset = pair * INTERMEDIATE + ib * 128;
        had_ff_r_128_inner<false, true>(
            at<float>(scratch, O_GATE) + offset,
            at<float>(scratch, O_GATE) + offset,
            reinterpret_cast<const half*>(gate_svh_ptrs[e]) + ib * 128,
            HAD_NORM);
        had_ff_r_128_inner<false, true>(
            at<float>(scratch, O_UP) + offset,
            at<float>(scratch, O_UP) + offset,
            reinterpret_cast<const half*>(up_svh_ptrs[e]) + ib * 128,
            HAD_NORM);
    }
    grid.sync();
    LNA_MARK();

    const int total_act = *pair_count * INTERMEDIATE;
    const int tid = block_linear() * blockDim.x + threadIdx.x;
    const int threads = blocks_linear() * blockDim.x;
    for (int idx = tid; idx < total_act; idx += threads) {
        float g = at<float>(scratch, O_GATE)[idx];
        float u = at<float>(scratch, O_UP)[idx];
        if (dbg && (!isfinite(g) || !isfinite(u))) atomicExch(error, 1);
        if (g > limit) g = limit;
        if (u < -limit) u = -limit;
        if (u > limit) u = limit;
        const float act = (g / (1.0f + expf(-g))) * u;
        at<half>(scratch, O_ACT)[idx] = __float2half_rn(act);
    }
    grid.sync();
    LNA_MARK();

    // Down input Hadamard has its own expert-specific suh.
    for (int w = warp; w < total_inter_warps; w += warps) {
        const int pair = w / (INTERMEDIATE / 128);
        const int ib = w % (INTERMEDIATE / 128);
        const int g = pair_group[pair];
        const int e = group_expert[g];
        const int offset = pair * INTERMEDIATE + ib * 128;
        had_hf_r_128_inner<true, false>(
            at<half>(scratch, O_ACT) + offset,
            at<half>(scratch, O_DOWN_IN) + offset,
            reinterpret_cast<const half*>(down_suh_ptrs[e]) + ib * 128,
            HAD_NORM);
    }
    grid.sync();
    LNA_MARK();

    for (int wave = 0; wave * gridDim.z < *group_count; ++wave) {
        const int g = wave * gridDim.z + blockIdx.z;
        int* locks = at<int>(scratch, O_LOCKS) + blockIdx.z * MAX_LOCKS;
        if (g < *group_count) {
            const int e = group_expert[g];
            const int rows = group_rows_count[g];
            const int pair = group_pair_base[g];
            run_gemm<K, LNA_DOWN_TILE_N, LNA_DOWN_STAGES, LNA_DOWN_FSTAGES>(
                at<half>(scratch, O_DOWN_IN) + pair * INTERMEDIATE,
                reinterpret_cast<const uint16_t*>(down_t_ptrs[e]),
                at<float>(scratch, O_DOWN) + pair * HIDDEN,
                rows, INTERMEDIATE, HIDDEN, locks);
        }
        grid.sync();
    LNA_MARK();
    }

    // Down output Hadamard/scales are in-place FP32. Gather is a separate
    // fixed-order pass over the original [m,6] route slots; no atomics.
    const int total_down_warps = *pair_count * (HIDDEN / 128);
    for (int w = warp; w < total_down_warps; w += warps) {
        const int pair = w / (HIDDEN / 128);
        const int ib = w % (HIDDEN / 128);
        const int g = pair_group[pair];
        const int e = group_expert[g];
        const int offset = pair * HIDDEN + ib * 128;
        had_ff_r_128_inner<false, true>(
            at<float>(scratch, O_DOWN) + offset,
            at<float>(scratch, O_DOWN) + offset,
            reinterpret_cast<const half*>(down_svh_ptrs[e]) + ib * 128,
            HAD_NORM);
    }
    grid.sync();
    LNA_MARK();

    const int total_out = m * HIDDEN;
    for (int idx = tid; idx < total_out; idx += threads) {
        const int row = idx / HIDDEN;
        const int col = idx % HIDDEN;
        float sum = 0.0f;
        for (int k = 0; k < TOPK; ++k) {
            const int pair = at<int>(scratch, O_SLOT_PAIR)[row * TOPK + k];
            if (pair >= 0)
                sum += weights[row * TOPK + k] * at<float>(scratch, O_DOWN)[pair * HIDDEN + col];
        }
        if (dbg && !isfinite(sum)) atomicExch(error, 1);
        if (out_kind == 0)
            reinterpret_cast<half*>(out)[idx] = __float2half_rn(sum);
        else if (out_kind == 1)
            reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(sum);
        else
            reinterpret_cast<float*>(out)[idx] = sum;
    }
    grid.sync();
    LNA_MARK();
    #undef LNA_MARK
}

struct Prepared {
    bool ready = false;
    int sms = 0;
    int resident = 0;
    int slices = 1;   // gridDim.x: cooperating blocks per expert group
    int groups = 1;   // gridDim.z: expert groups resident at once
    std::size_t smem = 0;
};

// LNA_MOE_SLICES overrides the blocks-per-expert split for the gate-E sweep.
// It is read once, before any graph capture, in prepare_one().
inline int slices_env() {
    const char* v = std::getenv("LNA_MOE_SLICES");
    if (!v) return 0;
    int n = std::atoi(v);
    return (n >= 1 && n <= 32) ? n : 0;
}

Prepared prepared[2][3];
std::mutex prepared_mutex;

template <int K>
constexpr std::size_t dynamic_smem_bytes() {
    // Size for the larger of the two calls actually made.
    constexpr int a_halfs = 16 * LNA_TILE_K;             // TILESIZE_M * TILESIZE_K
    // gate/up: TILE_N x STAGES
    constexpr int gu_b = (LNA_TILE_K / 16) * (LNA_GU_TILE_N / 16) * 16 * K;
    constexpr int gu_c = 4 * 256 * (2 * (LNA_GU_TILE_N / 16) / 8);
    constexpr int gu = LNA_GU_STAGES * (2 * a_halfs + 2 * gu_b) + 4 * gu_c;
    // down: TILE_N x STAGES
    constexpr int dn_b = (LNA_TILE_K / 16) * (LNA_DOWN_TILE_N / 16) * 16 * K;
    constexpr int dn_c = 4 * 256 * (2 * (LNA_DOWN_TILE_N / 16) / 8);
    constexpr int dn = LNA_DOWN_STAGES * (2 * a_halfs + 2 * dn_b) + 4 * dn_c;
    return gu > dn ? gu : dn;
}

template <int K, int CAP>
void prepare_one(int device) {
    constexpr int ki = K - 2;
    constexpr int ci = CAP == 4 ? 0 : (CAP == 8 ? 1 : 2);
    auto& entry = prepared[ki][ci];
    if (entry.ready) return;
    void* kernel = reinterpret_cast<void*>(lna_moe_kernel<K, CAP>);
    int cooperative = 0;
    cuda_check(cudaDeviceGetAttribute(&cooperative,
                                      cudaDevAttrCooperativeLaunch, device));
    TORCH_CHECK(cooperative, "LNA MoE requires cooperative launch support");
    cuda_check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(dynamic_smem_bytes<K>())));
    int sms = 0;
    cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    int resident = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, kernel, LNA_THREADS, dynamic_smem_bytes<K>()));
    TORCH_CHECK(resident > 0, "LNA MoE has no resident CTA at requested shared memory");
    TORCH_CHECK(sms <= MAX_BLOCKS, "LNA MoE scratch block capacity is too small");
    entry.sms = sms;
    entry.resident = resident;
    entry.smem = dynamic_smem_bytes<K>();
    // Cooperative launch needs every block co-resident, so the grid is exactly
    // sms * resident blocks.  Those blocks are split into `groups` expert teams
    // of `slices` blocks each; the canonical inner GEMM slices its k*n tile
    // space by blockIdx.x / gridDim.x (exl3_gemm_inner.cuh:86-88), which is the
    // same mechanism exl3_moe_kernel uses for its per-expert SM groups.
    const int total = sms * resident;
    int slices = slices_env();
    if (!slices) {
        // Default: aim for ~24 expert teams, which covers the m=4 unique-expert
        // expectation (U ~= 23) in one wave while still giving each expert more
        // than one SM.
        slices = total / 24;
        if (slices < 1) slices = 1;
    }
    while (slices > 1 && total % slices) --slices;
    entry.slices = slices;
    entry.groups = total / slices;
    TORCH_CHECK(entry.groups >= 1 && entry.groups <= MAX_BLOCKS,
                "LNA MoE group count out of range");
    entry.ready = true;
}

template <int K, int CAP>
void launch_one(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gsu, const at::Tensor& gsv,
    const at::Tensor& ut, const at::Tensor& usu, const at::Tensor& usv,
    const at::Tensor& dt, const at::Tensor& dsu, const at::Tensor& dsv,
    const at::Tensor& ids, const at::Tensor& weights, float limit,
    const at::Tensor& scratch, int n_experts, int debug_finite) {
    const int device = x.get_device();
    {
        std::lock_guard<std::mutex> lock(prepared_mutex);
        prepare_one<K, CAP>(device);
    }
    const auto& entry = prepared[K - 2][CAP == 4 ? 0 : (CAP == 8 ? 1 : 2)];
    const int m = static_cast<int>(x.size(0));
    const int x_bf16 = x.scalar_type() == at::kBFloat16;
    const int out_kind = out.scalar_type() == at::kHalf ? 0 :
                         (out.scalar_type() == at::kBFloat16 ? 1 : 2);
    void* xp = const_cast<void*>(x.data_ptr());
    void* op = out.data_ptr();
    void* gtp = gt.data_ptr<int64_t>();
    void* gsup = gsu.data_ptr<int64_t>();
    void* gsvp = gsv.data_ptr<int64_t>();
    void* utp = ut.data_ptr<int64_t>();
    void* usup = usu.data_ptr<int64_t>();
    void* usvp = usv.data_ptr<int64_t>();
    void* dtp = dt.data_ptr<int64_t>();
    void* dsup = dsu.data_ptr<int64_t>();
    void* dsvp = dsv.data_ptr<int64_t>();
    void* idp = ids.data_ptr<int32_t>();
    void* wp = weights.data_ptr<float>();
    void* sp = scratch.data_ptr<uint8_t>();
    int m_arg = m;
    int n_experts_arg = n_experts;
    int x_bf16_arg = x_bf16;
    int out_kind_arg = out_kind;
    int debug_finite_arg = debug_finite;
    float limit_arg = limit;
    void* args[] = {
        &xp, &op, &gtp, &gsup, &gsvp, &utp, &usup, &usvp,
        &dtp, &dsup, &dsvp, &idp, &wp, &m_arg, &n_experts_arg,
        &x_bf16_arg, &out_kind_arg, &limit_arg, &debug_finite_arg, &sp};
    cuda_check(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(lna_moe_kernel<K, CAP>),
        dim3(entry.slices, 1, entry.groups), dim3(LNA_THREADS), args, entry.smem,
        at::cuda::getCurrentCUDAStream(device).stream()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace lna_detail

int64_t lna_moe_scratch_bytes_cuda() {
    return static_cast<int64_t>(lna_detail::O_END);
}

void lna_moe_prepare_cuda(int64_t bits, int64_t m_cap) {
    TORCH_CHECK(bits == 2 || bits == 3, "LNA supports K2 and K3 only");
    TORCH_CHECK(m_cap == 4 || m_cap == 8 || m_cap == 16,
                "LNA M_CAP must be 4, 8, or 16");
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(lna_detail::prepared_mutex);
    if (bits == 2) {
        if (m_cap == 4) lna_detail::prepare_one<2, 4>(device);
        else if (m_cap == 8) lna_detail::prepare_one<2, 8>(device);
        else lna_detail::prepare_one<2, 16>(device);
    } else {
        if (m_cap == 4) lna_detail::prepare_one<3, 4>(device);
        else if (m_cap == 8) lna_detail::prepare_one<3, 8>(device);
        else lna_detail::prepare_one<3, 16>(device);
    }
}

std::vector<int64_t> lna_moe_info_cuda(int64_t bits, int64_t m_cap) {
    lna_moe_prepare_cuda(bits, m_cap);
    const int ki = static_cast<int>(bits) - 2;
    const int ci = m_cap == 4 ? 0 : (m_cap == 8 ? 1 : 2);
    const auto& e = lna_detail::prepared[ki][ci];
    return {e.sms, e.resident, e.slices, e.groups,
            static_cast<int64_t>(e.smem)};
}

at::Tensor lna_moe_decode_cuda(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gsu, const at::Tensor& gsv,
    const at::Tensor& ut, const at::Tensor& usu, const at::Tensor& usv,
    const at::Tensor& dt, const at::Tensor& dsu, const at::Tensor& dsv,
    const at::Tensor& ids, const at::Tensor& weights, int64_t bits,
    double swiglu_limit, const at::Tensor& scratch) {
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kHalf ||
                                x.scalar_type() == at::kBFloat16),
                "LNA input must be CUDA fp16 or bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == lna_detail::HIDDEN,
                "LNA input must be [m,4096]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= lna_detail::MAX_M,
                "LNA supports 1 <= m <= 16");
    TORCH_CHECK(out.is_cuda() && out.device() == x.device() &&
                out.sizes() == x.sizes(), "LNA output must match x shape/device");
    TORCH_CHECK(out.scalar_type() == at::kHalf ||
                out.scalar_type() == at::kBFloat16 ||
                out.scalar_type() == at::kFloat,
                "LNA output must be fp16, bf16, or fp32");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt &&
                ids.is_contiguous() && ids.dim() == 2 &&
                ids.size(0) == x.size(0) && ids.size(1) == lna_detail::TOPK,
                "LNA expert_ids must be contiguous int32 [m,6]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat &&
                weights.is_contiguous() && weights.sizes() == ids.sizes(),
                "LNA routing_weights must be contiguous fp32 [m,6]");
    TORCH_CHECK(bits == 2 || bits == 3, "LNA supports K2/K3");
    const at::Tensor* tables[] = {&gt, &gsu, &gsv, &ut, &usu, &usv,
                                  &dt, &dsu, &dsv};
    const int64_t n_experts = gt.numel();
    TORCH_CHECK(n_experts >= 1 && n_experts <= lna_detail::MAX_EXPERTS,
                "LNA pointer table has invalid expert count");
    for (const at::Tensor* table : tables) {
        TORCH_CHECK(table->is_cuda() && table->device() == x.device() &&
                    table->scalar_type() == at::kLong && table->is_contiguous() &&
                    table->numel() == n_experts,
                    "LNA pointer tables must be contiguous CUDA int64 [experts]");
    }
    TORCH_CHECK(scratch.is_cuda() && scratch.device() == x.device() &&
                scratch.scalar_type() == at::kByte && scratch.is_contiguous() &&
                scratch.numel() >= lna_detail::O_END,
                "LNA scratch is too small or has the wrong dtype/device");

    const int m = static_cast<int>(x.size(0));
    const int cap = m <= 4 ? 4 : (m <= 8 ? 8 : 16);
    const int debug_finite = std::getenv("LNA_EXL3_DEBUG_FINITE") ? 1 : 0;
    const float limit = static_cast<float>(swiglu_limit);
    TORCH_CHECK(limit > 0.0f && isfinite(limit),
                "LNA requires a finite positive SwiGLU limit");
    if (bits == 2) {
        if (cap == 4) lna_detail::launch_one<2, 4>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
        else if (cap == 8) lna_detail::launch_one<2, 8>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
        else lna_detail::launch_one<2, 16>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
    } else {
        if (cap == 4) lna_detail::launch_one<3, 4>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
        else if (cap == 8) lna_detail::launch_one<3, 8>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
        else lna_detail::launch_one<3, 16>(x, out, gt, gsu, gsv, ut, usu, usv, dt, dsu, dsv, ids, weights, limit, scratch, n_experts, debug_finite);
    }
    return out;
}
