#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kColdSamples = 7;
constexpr int kWarmSamples = 17;
constexpr std::size_t kFlushBytes = 512ull * 1024ull * 1024ull;

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(2);
    }
}

__global__ void flush_kernel(const uint4* data, std::size_t count,
                             uint32_t* sink) {
    uint32_t acc = 0;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    for (std::size_t i = tid; i < count; i += stride) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (threadIdx.x == 0) atomicXor(sink, acc);
}

__global__ void read_kernel(const uint4* data, std::size_t count,
                            uint32_t* block_sums) {
    uint64_t acc = 0;
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = gridDim.x * blockDim.x;
    for (std::size_t i = tid; i < count; i += stride) {
        const uint4 v = data[i];
        acc += v.x + v.y + v.z + v.w;
    }
    __shared__ uint64_t reduce[kThreads];
    reduce[threadIdx.x] = acc;
    __syncthreads();
    for (int width = kThreads / 2; width; width >>= 1) {
        if (threadIdx.x < width) reduce[threadIdx.x] += reduce[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_sums[blockIdx.x] = static_cast<uint32_t>(reduce[0]);
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void run_case(int bits, int unique, int blocks, const cudaDeviceProp& prop) {
    // F1 local expert byte contract: three 16x16 trellis grids plus three
    // fp16 scale vectors (H+I each), with the exact K2/K3 extents.
    constexpr std::size_t trellis_k2 = 786432;
    constexpr std::size_t trellis_k3 = 1179648;
    constexpr std::size_t scales = 26112;
    const std::size_t per_expert = (bits == 2 ? trellis_k2 : trellis_k3) + scales;
    const std::size_t bytes = per_expert * static_cast<std::size_t>(unique);
    const std::size_t words = bytes / sizeof(uint4);

    uint4* data = nullptr;
    uint4* flush = nullptr;
    uint32_t* sums = nullptr;
    uint32_t* flush_sink = nullptr;
    check(cudaMalloc(&data, bytes), "cudaMalloc data");
    check(cudaMalloc(&flush, kFlushBytes), "cudaMalloc flush");
    check(cudaMalloc(&sums, sizeof(uint32_t) * blocks), "cudaMalloc sums");
    check(cudaMalloc(&flush_sink, sizeof(uint32_t)), "cudaMalloc flush sink");
    check(cudaMemset(data, 0x5a, bytes), "cudaMemset data");
    check(cudaMemset(flush, 0xa5, kFlushBytes), "cudaMemset flush");

    std::vector<float> cold;
    cold.reserve(kColdSamples);
    for (int i = 0; i < kColdSamples; ++i) {
        check(cudaMemsetAsync(flush_sink, 0, sizeof(uint32_t)), "reset flush sink");
        flush_kernel<<<blocks, kThreads>>>(flush, kFlushBytes / sizeof(uint4), flush_sink);
        check(cudaGetLastError(), "flush launch");
        check(cudaDeviceSynchronize(), "flush synchronize");
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "event create start");
        check(cudaEventCreate(&stop), "event create stop");
        check(cudaEventRecord(start), "event record start");
        read_kernel<<<blocks, kThreads>>>(data, words, sums);
        check(cudaEventRecord(stop), "event record stop");
        check(cudaEventSynchronize(stop), "event synchronize stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        cold.push_back(ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Warm samples intentionally omit the flush. They are reported separately
    // and are not used for the acceptance floor.
    std::vector<float> warm;
    warm.reserve(kWarmSamples);
    for (int i = 0; i < kWarmSamples; ++i) {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "event create warm start");
        check(cudaEventCreate(&stop), "event create warm stop");
        check(cudaEventRecord(start), "event record warm start");
        read_kernel<<<blocks, kThreads>>>(data, words, sums);
        check(cudaEventRecord(stop), "event record warm stop");
        check(cudaEventSynchronize(stop), "event synchronize warm stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "event warm elapsed");
        warm.push_back(ms);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    const double cold_ms = median(cold);
    const double warm_ms = median(warm);
    const double cold_gbps = static_cast<double>(bytes) / (cold_ms * 1.0e6);
    const double warm_gbps = static_cast<double>(bytes) / (warm_ms * 1.0e6);
    const double floor_us = static_cast<double>(bytes) / (cold_gbps * 1.0e9) * 1.0e6;
    std::printf(
        "MEASURED COLD_READ bits=%d U=%d bytes=%zu cold_ms=%.3f cold_GBps=%.2f T_floor_us=%.2f warm_ms=%.3f warm_GBps=%.2f\n",
        bits, unique, bytes, cold_ms, cold_gbps, floor_us, warm_ms, warm_gbps);

    cudaFree(data);
    cudaFree(flush);
    cudaFree(sums);
    cudaFree(flush_sink);
    (void)prop;
}

}  // namespace

int main() {
    int device = 0;
    check(cudaSetDevice(device), "cudaSetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    int sms = 0;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
          "cudaDeviceGetAttribute SM count");
    std::printf("MEASURED CARD device=%d name=%s sm_count=%d\n", device,
                prop.name, sms);
    const int blocks = std::max(1, sms * 4);
    for (int bits : {2, 3}) {
        for (int unique : {23, 80}) run_case(bits, unique, blocks, prop);
    }
    return 0;
}
