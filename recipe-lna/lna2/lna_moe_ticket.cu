// F3 second machine: resident expert-team + dynamic ticket (outer, modelled on
// exllamav3 exl3_moe_kernel.cuh) around the small-R GEMV core (inner,
// lna_gemv_core.cuh).  No grid-wide barriers anywhere; only team barriers.
//
// Numerical boundaries follow the F1 v2 order, NOT the incumbent:
//   x bf16 -> fp16 in-kernel; gate/up C in FP32; output Hadamard + svh in FP32;
//   silu(min(g,L)) * clamp(u,-L,L) in FP32; activation -> fp16 for down;
//   down C and its output Hadamard in FP32; expert sum in FP32 in fixed
//   top-k slot order (no atomics), then one cast to the consumer dtype.
// The one deliberate deviation is the inner MMA, which accumulates in FP16 and
// folds to FP32 every FOLD k-slices (exl3_gemv_kernel.cuh:14). The F1 review
// allows that only with margin on every adversarial test; see REPORT-F3.md.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>

#include <cuda/atomic>
#include <ATen/cuda/CUDAContext.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/hadamard_inner.cuh"
#include "lna_gemv_core.cuh"

namespace lna2 {

constexpr int HIDDEN = 4096;
constexpr int INTERMEDIATE = 256;
constexpr int TOPK = 6;
// F3b: the seat's 制式 is 4 seqs x DSpark3(1+3) = exactly 16 rows, i.e. sitting
// ON the old ceiling -- one more --max-num-seqs produced 20 rows and the plugin
// reverted to exllamav3 silently (REPORT-F3-seat.md §2).  32 gives 8 seqs of
// headroom.  The only structural cost is the shared route plan (row bitmask is
// still one 32-bit word per expert) and MAX_PAIRS, which scales the scratch.
// F3c: 24, not 32.  The seat boots at 制式 args only with UTIL=0.972 and is
// ~10 MiB/GPU short at 0.97 (REPORT-F3-seat2.md §1), and MAX_PAIRS scales the
// scratch linearly: 32 rows = 6.470 MiB, 24 rows = 4.853 MiB.  制式 is 4 seqs x
// DSpark3 = 16 rows, so 24 still leaves 50% headroom (6 seqs) -- the cliff the
// seat hit at 16 is gone either way.  Build with -DLNA2_MAX_M=32 for 8 seqs.
#ifndef LNA2_MAX_M
#define LNA2_MAX_M 24
#endif
constexpr int MAX_M = LNA2_MAX_M;
static_assert(MAX_M >= 1 && MAX_M <= 32, "row bitmask is one 32-bit word");
constexpr int MAX_EXPERTS = 256;
constexpr int MAX_ROUTES = MAX_M * TOPK;   // 96
constexpr int MAX_GROUPS = MAX_ROUTES;
constexpr int MAX_PAIRS = MAX_ROUTES;
constexpr int MAX_TEAMS = 64;
constexpr float HAD_NORM = 0.088388347648f;

// CFG 1 = "wide": 256 threads, WK=8 k-splits, WNT=4 n-tiles/warp, COLS=64.
// Chosen because both GEMMs here are FFN-shaped and because 256-thread CTAs
// leave room for >1 CTA/SM.  CFG 0 is kept switchable for the sweep.
#ifndef LNA2_PHASE_CLOCK
#define LNA2_PHASE_CLOCK 0
#endif
#ifndef LNA2_CFG
#define LNA2_CFG 1
#endif
#ifndef LNA2_TEAM_W
#define LNA2_TEAM_W 8
#endif
// Minimum CTAs per SM handed to ptxas.  MEASURED (run-41, -Xptxas -v): with no
// bound the K3 kernel takes 205 registers (K2 119), which pins K3 at 1 CTA/SM.
// The three inlined ROWS_CAP variants are what costs it.
#ifndef LNA2_MIN_BLOCKS
#define LNA2_MIN_BLOCKS 2
#endif
#ifndef LNA2_PFD
#define LNA2_PFD 2
#endif
#ifndef LNA2_ROWS_CAP
#define LNA2_ROWS_CAP 4
#endif
constexpr int CFG = LNA2_CFG;
constexpr int THREADS = CFG == 0 ? 512 : 256;
constexpr int COLS = CFG == 0 ? 32 : 64;
constexpr int WK = CFG == 0 ? 16 : 8;
constexpr int PFD = LNA2_PFD;
constexpr int ROWS_CAP = LNA2_ROWS_CAP;   // R_e bucket ceiling handled in one pass
constexpr int SH_RED_FLOATS = WK * ROWS_CAP * COLS;

constexpr std::size_t align_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }

// ---- global scratch layout (per stream) ----------------------------------
constexpr std::size_t O_SCHED    = 0;                                  // [0]=next ticket [1]=retired [2+t]=team ticket
constexpr std::size_t O_BARRIER  = O_SCHED + (2 + MAX_TEAMS) * sizeof(int);
constexpr std::size_t O_TEAMFLAG = O_BARRIER + 2 * MAX_TEAMS * sizeof(int);
constexpr std::size_t O_CTRL_END = O_TEAMFLAG + MAX_TEAMS * sizeof(int);
constexpr std::size_t O_HAD_G  = align_up(O_CTRL_END, 256);
constexpr std::size_t O_HAD_U  = O_HAD_G + (std::size_t) MAX_PAIRS * HIDDEN * sizeof(half);
constexpr std::size_t O_GATE_C = O_HAD_U + (std::size_t) MAX_PAIRS * HIDDEN * sizeof(half);
constexpr std::size_t O_UP_C   = O_GATE_C + (std::size_t) MAX_PAIRS * INTERMEDIATE * sizeof(float);
constexpr std::size_t O_DOWN_I = O_UP_C + (std::size_t) MAX_PAIRS * INTERMEDIATE * sizeof(float);
constexpr std::size_t O_DOWN_C = O_DOWN_I + (std::size_t) MAX_PAIRS * INTERMEDIATE * sizeof(half);
constexpr std::size_t O_CLOCK  = O_DOWN_C + (std::size_t) MAX_PAIRS * HIDDEN * sizeof(float);
// Per-phase cycle accumulators, written by team 0 / CTA 0 / thread 0 only.
// Enabled with LNA2_PHASE_CLOCK=1 at build time; off by default (a clock64 per
// phase per ticket is cheap but it is still instrumentation).
constexpr std::size_t O_END    = O_CLOCK + 16 * sizeof(long long);

// Control region must be zero at the first launch; the scheduler self-resets
// after that (last team out restores next-ticket and retired to 0).
constexpr std::size_t CTRL_BYTES = O_CTRL_END;

template <typename T> __device__ __forceinline__ T* at(char* b, std::size_t o) {
    return reinterpret_cast<T*>(b + o);
}

// ---- bf16/fp16 x load + suh + 128-block Hadamard (from F1) ----------------
__device__ __forceinline__ void had_x_128(
    const void* x, int row, int ib, bool bf16, const half* scale, half* out) {
    const int lane = threadIdx.x & 31;
    const int off = row * HIDDEN + ib * 128 + lane * 4;
    half4 v;
    if (bf16) {
        const auto* s = reinterpret_cast<const __nv_bfloat16*>(x) + off;
        v.x = __floats2half2_rn(__bfloat162float(s[0]), __bfloat162float(s[1]));
        v.y = __floats2half2_rn(__bfloat162float(s[2]), __bfloat162float(s[3]));
    } else {
        const auto* s = reinterpret_cast<const half*>(x) + off;
        v.x = __halves2half2(s[0], s[1]);
        v.y = __halves2half2(s[2], s[3]);
    }
    const half4 sc = reinterpret_cast<const half4*>(scale)[lane];
    v.x = __hmul2(v.x, sc.x);
    v.y = __hmul2(v.y, sc.y);
    float v0 = __half2float(__low2half(v.x)), v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y)), v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v.x = __floats2half2_rn(h0 * HAD_NORM, h1 * HAD_NORM);
    v.y = __floats2half2_rn(h2 * HAD_NORM, h3 * HAD_NORM);
    reinterpret_cast<half4*>(out)[lane] = v;
}

// ---- shared route plan (parallel; recomputed per CTA so no grid barrier) ---
struct Plan {
    unsigned int rowmask[MAX_EXPERTS];
    int sc_a[MAX_EXPERTS];
    int sc_b[MAX_EXPERTS];
    short gid[MAX_EXPERTS];
    short base[MAX_EXPERTS];
    int gexpert[MAX_GROUPS];
    int grows_cnt[MAX_GROUPS];
    int gpair[MAX_GROUPS];
    unsigned char grows[MAX_GROUPS * MAX_M];
    short slotpair[MAX_ROUTES];
    int group_count;
    int pair_count;
};

// Same algorithm as the repaired F1 route plan: expert-major bitmask + two
// Hillis-Steele exclusive scans.  Deterministic (ascending expert id); the
// gather still walks the original [m,6] slot order, so the FP32 add order is
// the reference's.  Recomputing it in every CTA costs ~2 us of wall time once
// and removes the only reason this kernel would need a grid-wide barrier.
__device__ __forceinline__ void build_plan(
    Plan& p, const int32_t* ids, int m, int n_experts) {
    const int e = threadIdx.x;
    const bool own = e < MAX_EXPERTS;
    if (own) p.rowmask[e] = 0;
    __syncthreads();
    const int n_slots = m * TOPK;
    for (int s = threadIdx.x; s < n_slots; s += blockDim.x) {
        const int id = ids[s];
        if (id >= 0 && id < n_experts) atomicOr(&p.rowmask[id], 1u << (s / TOPK));
    }
    __syncthreads();
    const unsigned int mask = own ? p.rowmask[e] : 0u;
    const int has = mask ? 1 : 0;
    const int np = __popc(mask);
    if (own) { p.sc_a[e] = has; p.sc_b[e] = np; }
    __syncthreads();
    for (int off = 1; off < MAX_EXPERTS; off <<= 1) {
        const int va = (own && e >= off) ? p.sc_a[e - off] : 0;
        const int vb = (own && e >= off) ? p.sc_b[e - off] : 0;
        __syncthreads();
        if (own) { p.sc_a[e] += va; p.sc_b[e] += vb; }
        __syncthreads();
    }
    const int gx = own ? p.sc_a[e] - has : 0;
    const int px = own ? p.sc_b[e] - np : 0;
    if (e == MAX_EXPERTS - 1) { p.group_count = p.sc_a[e]; p.pair_count = p.sc_b[e]; }
    if (own) { p.gid[e] = (short) gx; p.base[e] = (short) px; }
    if (own && has && gx < MAX_GROUPS && px + np <= MAX_PAIRS) {
        p.gexpert[gx] = e;
        p.grows_cnt[gx] = np;
        p.gpair[gx] = px;
        unsigned int rest = mask;
        for (int k = 0; rest; ++k) {
            const int r = __ffs((int) rest) - 1;
            rest &= rest - 1u;
            p.grows[gx * MAX_M + k] = (unsigned char) r;
        }
    }
    __syncthreads();
    for (int s = threadIdx.x; s < n_slots; s += blockDim.x) {
        const int id = ids[s];
        if (id < 0 || id >= n_experts) { p.slotpair[s] = -1; continue; }
        const int row = s / TOPK;
        p.slotpair[s] = (short) (p.base[id] + __popc(p.rowmask[id] & ((1u << row) - 1u)));
    }
    __syncthreads();
}

template <int K>
__device__ __forceinline__ void run_proj(
    const half* A, const uint16_t* B, float* C,
    int R, int size_k, int size_n, int gbeg, int gstride, float* sh_red) {
    // R = 1 uses the ROWS=1 variant (no 16xN staging at all); 2..8 the ROWS=8
    // variant; R > 8 repeats it in row blocks of 8 (re-reads B, but MEASURED
    // R_e > 8 only happens on adversarial all-rows-one-expert routes).
    if (R == 1) {
        lna_gemv::gemv_core<K, 1, 1, CFG, PFD>(A, B, C, 1, size_k, size_n, gbeg, gstride, sh_red);
    } else if (R == 2 && ROWS_CAP >= 2) {
        lna_gemv::gemv_core<K, 1, 2, CFG, PFD>(A, B, C, 2, size_k, size_n, gbeg, gstride, sh_red);
    } else {
        // R > ROWS_CAP repeats in row blocks; that re-reads B, but MEASURED R_e
        // is 1.04 (m=4) / 1.16 (m=16) and R > 4 only occurs on adversarial
        // all-rows-one-expert routes.
        for (int rb = 0; rb < R; rb += ROWS_CAP) {
            const int rows = min(ROWS_CAP, R - rb);
            lna_gemv::gemv_core<K, 1, ROWS_CAP, CFG, PFD>(
                A + (size_t) rb * size_k, B, C + (size_t) rb * size_n,
                rows, size_k, size_n, gbeg, gstride, sh_red);
        }
    }
}

template <int K>
__global__ __launch_bounds__(THREADS, LNA2_MIN_BLOCKS)
void lna2_kernel(
    const void* __restrict__ x,
    void* __restrict__ out,
    const int64_t* __restrict__ gt, const int64_t* __restrict__ gsu, const int64_t* __restrict__ gsv,
    const int64_t* __restrict__ ut, const int64_t* __restrict__ usu, const int64_t* __restrict__ usv,
    const int64_t* __restrict__ dt, const int64_t* __restrict__ dsu, const int64_t* __restrict__ dsv,
    const int32_t* __restrict__ ids,
    const float* __restrict__ weights,
    int m, int n_experts, int x_bf16, int out_kind, float limit,
    char* __restrict__ scratch)
{
    __shared__ Plan plan;
    __shared__ float sh_red[SH_RED_FLOATS];
    __shared__ int sh_ticket;

    const int team = blockIdx.z;
    const int wid = blockIdx.x;
    const int W = gridDim.x;
    const int T = gridDim.z;
    const bool bf16 = x_bf16 != 0;

    build_plan(plan, ids, m, n_experts);

    int* sched = at<int>(scratch, O_SCHED);
    int* bar = at<int>(scratch, O_BARRIER);
    int* teamflag = at<int>(scratch, O_TEAMFLAG);
    half* had_g = at<half>(scratch, O_HAD_G);
    half* had_u = at<half>(scratch, O_HAD_U);
    float* gate_c = at<float>(scratch, O_GATE_C);
    float* up_c = at<float>(scratch, O_UP_C);
    half* down_i = at<half>(scratch, O_DOWN_I);
    float* down_c = at<float>(scratch, O_DOWN_C);

    const int warps_per_cta = THREADS / 32;
    const int warp_idx0 = wid * warps_per_cta + (threadIdx.x / 32);
    const int warps_per_team = W * warps_per_cta;
    const int W2 = W / 2;

#if LNA2_PHASE_CLOCK
    long long* clk = at<long long>(scratch, O_CLOCK);
    const bool timing = (team == 0 && wid == 0 && threadIdx.x == 0);
    long long t_prev = 0;
    if (timing) for (int i = 0; i < 8; ++i) clk[i] = 0;
    #define PH(i) do { if (timing) { long long n = clock64(); clk[i] += n - t_prev; t_prev = n; } } while (0)
    #define PH0() do { if (timing) t_prev = clock64(); } while (0)
#else
    #define PH(i) do {} while (0)
    #define PH0() do {} while (0)
#endif
    int ticket = team;
    while (ticket < plan.group_count)
    {
        PH0();
        const int e = plan.gexpert[ticket];
        const int R = plan.grows_cnt[ticket];
        const int pb = plan.gpair[ticket];

        // 1. input Hadamard for gate and up (x loaded once per (row,block))
        {
            const int total = R * (HIDDEN / 128);
            const half* gs = reinterpret_cast<const half*>(gsu[e]);
            const half* us = reinterpret_cast<const half*>(usu[e]);
            for (int wi = warp_idx0; wi < total; wi += warps_per_team) {
                const int r = wi / (HIDDEN / 128);
                const int ib = wi % (HIDDEN / 128);
                const int row = plan.grows[ticket * MAX_M + r];
                had_x_128(x, row, ib, bf16, gs + ib * 128, had_g + (size_t)(pb + r) * HIDDEN + ib * 128);
                had_x_128(x, row, ib, bf16, us + ib * 128, had_u + (size_t)(pb + r) * HIDDEN + ib * 128);
            }
            group_barrier(team, W, bar);
        }
        PH(0);

        // 2. gate and up GEMV run *concurrently*: the team splits in half.
        //    size_n = 256 gives only 256/COLS n-groups, so a whole 8-CTA team
        //    on one projection would leave half the CTAs idle.
        if (wid < W2)
            run_proj<K>(had_g + (size_t) pb * HIDDEN, reinterpret_cast<const uint16_t*>(gt[e]),
                        gate_c + (size_t) pb * INTERMEDIATE, R, HIDDEN, INTERMEDIATE,
                        wid, W2, sh_red);
        else
            run_proj<K>(had_u + (size_t) pb * HIDDEN, reinterpret_cast<const uint16_t*>(ut[e]),
                        up_c + (size_t) pb * INTERMEDIATE, R, HIDDEN, INTERMEDIATE,
                        wid - W2, W - W2, sh_red);
        group_barrier(team, W, bar);
        PH(1);

        // 3. gate/up output Hadamard + svh (FP32), clamp+SwiGLU (FP32), -> fp16,
        //    then down suh + input Hadamard.  One warp owns one 128-block, so
        //    all of this happens without another barrier.
        {
            const int total = R * (INTERMEDIATE / 128);
            const half* gv = reinterpret_cast<const half*>(gsv[e]);
            const half* uv = reinterpret_cast<const half*>(usv[e]);
            const half* ds = reinterpret_cast<const half*>(dsu[e]);
            const int lane = threadIdx.x & 31;
            for (int wi = warp_idx0; wi < total; wi += warps_per_team) {
                const int r = wi / (INTERMEDIATE / 128);
                const int ib = wi % (INTERMEDIATE / 128);
                const size_t o = (size_t)(pb + r) * INTERMEDIATE + ib * 128;
                had_ff_r_128_inner<false, true>(gate_c + o, gate_c + o, gv + ib * 128, HAD_NORM);
                had_ff_r_128_inner<false, true>(up_c + o, up_c + o, uv + ib * 128, HAD_NORM);
                half* di = down_i + o;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = lane * 4 + j;
                    float g = gate_c[o + c];
                    float u = up_c[o + c];
                    if (g > limit) g = limit;
                    if (u < -limit) u = -limit;
                    if (u > limit) u = limit;
                    di[c] = __float2half_rn((g / (1.0f + __expf(-g))) * u);
                }
                __syncwarp();
                had_hf_r_128_inner<true, false>(di, di, ds + ib * 128, HAD_NORM);
            }
            group_barrier(team, W, bar);
        }
        PH(2);

        // 4. down GEMV across the whole team (size_n = 4096 -> plenty of groups)
        run_proj<K>(down_i + (size_t) pb * INTERMEDIATE, reinterpret_cast<const uint16_t*>(dt[e]),
                    down_c + (size_t) pb * HIDDEN, R, INTERMEDIATE, HIDDEN, wid, W, sh_red);
        group_barrier(team, W, bar);
        PH(3);

        // 5. down output Hadamard + svh, FP32 in place
        {
            const int total = R * (HIDDEN / 128);
            const half* dv = reinterpret_cast<const half*>(dsv[e]);
            for (int wi = warp_idx0; wi < total; wi += warps_per_team) {
                const int r = wi / (HIDDEN / 128);
                const int ib = wi % (HIDDEN / 128);
                const size_t o = (size_t)(pb + r) * HIDDEN + ib * 128;
                had_ff_r_128_inner<false, true>(down_c + o, down_c + o, dv + ib * 128, HAD_NORM);
            }
        }

        PH(4);

        // 6. next ticket; the barrier publishes it and protects the buffers
        if (wid == 0 && threadIdx.x == 0)
            sched[2 + team] = T + atomicAdd(&sched[0], 1);
        group_barrier(team, W, bar);
        if (threadIdx.x == 0) sh_ticket = *(volatile int*) &sched[2 + team];
        __syncthreads();
        ticket = sh_ticket;
        PH(5);
    }
    #undef PH
    #undef PH0

    // Retire.  The acq_rel increment orders every team's down writes before the
    // last team's gather, and the last team out resets the scheduler.
    if (wid == 0 && threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups(sched[1]);
        const int retired = retired_groups.fetch_add(1, cuda::memory_order_acq_rel);
        const int last = (retired == T - 1) ? 1 : 0;
        teamflag[team] = last;
        if (last) {
            next_ticket.store(0, cuda::memory_order_relaxed);
            retired_groups.store(0, cuda::memory_order_relaxed);
        }
    }
    group_barrier(team, W, bar);

    if (*(volatile int*) &teamflag[team]) {
        // Fixed-order gather, done once by the last team to retire: no atomics,
        // no second launch, no grid-wide barrier.
        const int total = m * HIDDEN;
        const int tid = wid * THREADS + threadIdx.x;
        const int nthreads = W * THREADS;
        for (int idx = tid; idx < total; idx += nthreads) {
            const int row = idx / HIDDEN;
            const int col = idx % HIDDEN;
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                const int pair = plan.slotpair[row * TOPK + k];
                if (pair >= 0) sum += weights[row * TOPK + k] * down_c[(size_t) pair * HIDDEN + col];
            }
            if (out_kind == 0) reinterpret_cast<half*>(out)[idx] = __float2half_rn(sum);
            else if (out_kind == 1) reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(sum);
            else reinterpret_cast<float*>(out)[idx] = sum;
        }
    }
}

struct Prepared {
    bool ready = false;
    int sms = 0, resident = 0, teams = 0, width = 0, streams = 1;
};
// Keyed by (device, bits): the F1 version was process-global, which is wrong on
// a mixed-device host even though this house is 8x the same card.
std::map<std::pair<int, int>, Prepared> prepared;
std::mutex prepared_mutex;

inline int env_int(const char* name, int lo, int hi) {
    const char* v = std::getenv(name);
    if (!v) return 0;
    const int n = std::atoi(v);
    return (n >= lo && n <= hi) ? n : 0;
}

template <int K>
void prepare_one(int device) {
    auto key = std::make_pair(device, K);
    auto it = prepared.find(key);
    if (it != prepared.end() && it->second.ready) return;
    Prepared p;
    void* kernel = reinterpret_cast<void*>(lna2_kernel<K>);
    cuda_check(cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, device));
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&p.resident, kernel, THREADS, 0));
    TORCH_CHECK(p.resident > 0, "LNA2 has no resident CTA");
    int width = env_int("LNA2_TEAM_W", 1, 32);
    if (!width) width = LNA2_TEAM_W;
    // Team barriers require every CTA of the launch to be co-resident (the same
    // constraint the incumbent states in exl3_moe.cu:203-222).  Two lna2 kernels
    // overlapping on different streams therefore DEADLOCK: together they want
    // 2 x teams x width CTAs but the device only holds sms x resident.
    // MEASURED: gate4's 2-stream stress hung at the default (run-57).
    // LNA2_CONCURRENT_STREAMS reserves room for that many simultaneous launches.
    // Default 1 = the house setting (VLLM_DISABLE_SHARED_EXPERTS_STREAM=1).
    int streams = env_int("LNA2_CONCURRENT_STREAMS", 1, 8);
    if (!streams) streams = 1;
    const int total = p.sms * p.resident / streams;
    int teams = total / width;
    if (teams < 1) { teams = 1; width = total < 1 ? 1 : total; }
    if (teams > MAX_TEAMS) teams = MAX_TEAMS;
    p.streams = streams;
    p.width = width;
    p.teams = teams;
    p.ready = true;
    prepared[key] = p;
}

}  // namespace lna2

int64_t lna2_moe_scratch_bytes_cuda() { return (int64_t) lna2::O_END; }

// The plugin must not hardcode the row ceiling; it asks the kernel.
int64_t lna2_moe_max_rows_cuda() { return (int64_t) lna2::MAX_M; }

void lna2_moe_prepare_cuda(int64_t bits) {
    TORCH_CHECK(bits == 2 || bits == 3, "LNA2 supports K2/K3 only");
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(lna2::prepared_mutex);
    if (bits == 2) lna2::prepare_one<2>(device); else lna2::prepare_one<3>(device);
}

std::vector<int64_t> lna2_moe_info_cuda(int64_t bits) {
    lna2_moe_prepare_cuda(bits);
    int device = 0;
    cuda_check(cudaGetDevice(&device));
    const auto& p = lna2::prepared[std::make_pair(device, (int) bits)];
    return {p.sms, p.resident, p.width, p.teams, (int64_t) lna2::THREADS,
            (int64_t) lna2::CFG, (int64_t) lna2::PFD, (int64_t) lna2::ROWS_CAP,
            (int64_t) (lna2::SH_RED_FLOATS * 4 + (int) sizeof(lna2::Plan)),
            (int64_t) lna2::CTRL_BYTES, (int64_t) lna2::O_END,
            (int64_t) lna2::MAX_M, (int64_t) lna2::MAX_PAIRS,
            (int64_t) p.streams};
}

at::Tensor lna2_moe_decode_cuda(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gsu, const at::Tensor& gsv,
    const at::Tensor& ut, const at::Tensor& usu, const at::Tensor& usv,
    const at::Tensor& dt, const at::Tensor& dsu, const at::Tensor& dsv,
    const at::Tensor& ids, const at::Tensor& weights, int64_t bits,
    double swiglu_limit, const at::Tensor& scratch)
{
    using namespace lna2;
    TORCH_CHECK(x.is_cuda() && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16),
                "LNA2 input must be CUDA fp16 or bf16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == HIDDEN, "LNA2 input must be [m,4096]");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= MAX_M,
                "LNA2 row count out of range (1 <= m <= LNA2_MAX_M)");
    TORCH_CHECK(x.is_contiguous(), "LNA2 input must be contiguous");
    TORCH_CHECK(out.is_cuda() && out.device() == x.device() && out.sizes() == x.sizes()
                && out.is_contiguous(), "LNA2 output must match x shape/device and be contiguous");
    TORCH_CHECK(out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16
                || out.scalar_type() == at::kFloat, "LNA2 output must be fp16/bf16/fp32");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt && ids.is_contiguous()
                && ids.dim() == 2 && ids.size(0) == x.size(0) && ids.size(1) == TOPK,
                "LNA2 expert_ids must be contiguous int32 [m,6]");
    TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == at::kFloat
                && weights.is_contiguous() && weights.sizes() == ids.sizes(),
                "LNA2 routing_weights must be contiguous fp32 [m,6]");
    TORCH_CHECK(bits == 2 || bits == 3, "LNA2 supports K2/K3");
    const at::Tensor* tables[] = {&gt, &gsu, &gsv, &ut, &usu, &usv, &dt, &dsu, &dsv};
    const int64_t n_experts = gt.numel();
    TORCH_CHECK(n_experts >= 1 && n_experts <= MAX_EXPERTS,
                "LNA2 pointer table has invalid expert count");
    for (const at::Tensor* t : tables)
        TORCH_CHECK(t->is_cuda() && t->device() == x.device() && t->scalar_type() == at::kLong
                    && t->is_contiguous() && t->numel() == n_experts,
                    "LNA2 pointer tables must be contiguous CUDA int64 [experts]");
    TORCH_CHECK(scratch.is_cuda() && scratch.device() == x.device()
                && scratch.scalar_type() == at::kByte && scratch.is_contiguous()
                && scratch.numel() >= (int64_t) O_END, "LNA2 scratch too small");

    const int device = x.get_device();
    {
        std::lock_guard<std::mutex> lock(prepared_mutex);
        if (bits == 2) prepare_one<2>(device); else prepare_one<3>(device);
    }
    const auto& p = prepared[std::make_pair(device, (int) bits)];

    const float limit = (float) swiglu_limit;
    TORCH_CHECK(limit > 0.0f && isfinite(limit), "LNA2 requires a finite positive SwiGLU limit");
    const int m = (int) x.size(0);
    const int x_bf16 = x.scalar_type() == at::kBFloat16;
    const int out_kind = out.scalar_type() == at::kHalf ? 0
                       : (out.scalar_type() == at::kBFloat16 ? 1 : 2);

    void* xp = const_cast<void*>(x.data_ptr());
    void* op = out.data_ptr();
    auto S = at::cuda::getCurrentCUDAStream(device).stream();
    dim3 grid(p.width, 1, p.teams);
    if (bits == 2)
        lna2::lna2_kernel<2><<<grid, THREADS, 0, S>>>(
            xp, op, gt.data_ptr<int64_t>(), gsu.data_ptr<int64_t>(), gsv.data_ptr<int64_t>(),
            ut.data_ptr<int64_t>(), usu.data_ptr<int64_t>(), usv.data_ptr<int64_t>(),
            dt.data_ptr<int64_t>(), dsu.data_ptr<int64_t>(), dsv.data_ptr<int64_t>(),
            ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            m, (int) n_experts, x_bf16, out_kind, limit,
            reinterpret_cast<char*>(scratch.data_ptr<uint8_t>()));
    else
        lna2::lna2_kernel<3><<<grid, THREADS, 0, S>>>(
            xp, op, gt.data_ptr<int64_t>(), gsu.data_ptr<int64_t>(), gsv.data_ptr<int64_t>(),
            ut.data_ptr<int64_t>(), usu.data_ptr<int64_t>(), usv.data_ptr<int64_t>(),
            dt.data_ptr<int64_t>(), dsu.data_ptr<int64_t>(), dsv.data_ptr<int64_t>(),
            ids.data_ptr<int32_t>(), weights.data_ptr<float>(),
            m, (int) n_experts, x_bf16, out_kind, limit,
            reinterpret_cast<char*>(scratch.data_ptr<uint8_t>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
