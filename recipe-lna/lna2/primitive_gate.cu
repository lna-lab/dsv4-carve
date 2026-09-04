#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "/lab/exllamav3-src/exllamav3/exllamav3_ext/util.cuh"
#include "/lab/exllamav3-src/exllamav3/exllamav3_ext/ptx.cuh"
#include <quant/exl3_dq.cuh>

namespace {

constexpr int kPatterns = 4;
constexpr int kOffsets = 256;

inline void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

__device__ __forceinline__ half lna_mcg(uint32_t window) {
    window *= 0xCBAC1FEDu;
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(window));
    const half lo = __ushort_as_half(static_cast<uint16_t>(window));
    const half hi = __ushort_as_half(static_cast<uint16_t>(window >> 16));
    return __hadd(lo, hi);
}

template <int bits>
__device__ __forceinline__ half lna_dq(const uint32_t* ptr, int t_offset) {
    constexpr int words = bits * 256 / 32;
    const int b0 = (t_offset + 257) * bits - 16;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const uint32_t a = ptr[i0 % words];
    const uint32_t b = ptr[i1 % words];
    const uint32_t window = __funnelshift_r(b, a, shift) & 0xffffu;
    return lna_mcg(window);
}

template <int bits>
__global__ void primitive_kernel(const uint32_t* streams, uint16_t* local,
                                 uint16_t* canonical) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = kPatterns * kOffsets;
    if (item >= total) return;
    const int pattern = item / kOffsets;
    const int offset = item % kOffsets;
    constexpr int words = bits * 256 / 32;
    const uint32_t* stream = streams + pattern * words;
    local[item] = __half_as_ushort(lna_dq<bits>(stream, offset));
    canonical[item] = __half_as_ushort(dq<bits, 1>(stream, offset));
}

template <int bits>
void run_one(const std::vector<uint32_t>& host_streams) {
    constexpr int words = bits * 256 / 32;
    constexpr int total = kPatterns * kOffsets;
    uint32_t* streams = nullptr;
    uint16_t* local = nullptr;
    uint16_t* canonical = nullptr;
    check_cuda(cudaMalloc(&streams, sizeof(uint32_t) * kPatterns * words),
               "cudaMalloc streams");
    check_cuda(cudaMalloc(&local, sizeof(uint16_t) * total), "cudaMalloc local");
    check_cuda(cudaMalloc(&canonical, sizeof(uint16_t) * total),
               "cudaMalloc canonical");
    check_cuda(cudaMemcpy(streams, host_streams.data(),
                          sizeof(uint32_t) * kPatterns * words,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy streams");

    primitive_kernel<bits><<<1, 256>>>(streams, local, canonical);
    check_cuda(cudaGetLastError(), "primitive kernel launch");
    check_cuda(cudaDeviceSynchronize(), "primitive kernel synchronize");

    std::vector<uint16_t> local_h(total), canonical_h(total);
    check_cuda(cudaMemcpy(local_h.data(), local, sizeof(uint16_t) * total,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy local");
    check_cuda(cudaMemcpy(canonical_h.data(), canonical,
                          sizeof(uint16_t) * total, cudaMemcpyDeviceToHost),
               "cudaMemcpy canonical");

    for (int i = 0; i < total; ++i) {
        if (local_h[i] != canonical_h[i]) {
            std::fprintf(stderr,
                         "PRIMITIVE FAIL K=%d pattern=%d t_offset=%d local=0x%04x canonical=0x%04x\n",
                         bits, i / kOffsets, i % kOffsets, local_h[i],
                         canonical_h[i]);
            std::exit(3);
        }
    }
    std::printf("PRIMITIVE K=%d PASS patterns=%d offsets=%d\n", bits,
                kPatterns, kOffsets);
    cudaFree(streams);
    cudaFree(local);
    cudaFree(canonical);
}

template <int bits>
std::vector<uint32_t> make_streams() {
    constexpr int words = bits * 256 / 32;
    std::vector<uint32_t> out(kPatterns * words);
    for (int i = 0; i < words; ++i) {
        out[i] = 0u;
        out[words + i] = 0xffffffffu;
        out[2 * words + i] = 1u << ((7 * i + 3) & 31);
        uint32_t x = 0x9e3779b9u ^ static_cast<uint32_t>(i);
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        out[3 * words + i] = x;
    }
    return out;
}

}  // namespace

int main() {
    run_one<2>(make_streams<2>());
    run_one<3>(make_streams<3>());
    std::puts("F1_PRIMITIVE_GATE PASS half_bits=exact");
    return 0;
}
