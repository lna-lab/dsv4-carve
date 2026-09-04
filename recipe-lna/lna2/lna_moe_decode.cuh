#pragma once

#include <torch/extension.h>

at::Tensor lna_moe_decode_cuda(
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    const at::Tensor& expert_ids,
    const at::Tensor& routing_weights,
    int64_t bits,
    double swiglu_limit,
    const at::Tensor& scratch);

int64_t lna_moe_scratch_bytes_cuda();
void lna_moe_prepare_cuda(int64_t bits, int64_t m_cap);
std::vector<int64_t> lna_moe_info_cuda(int64_t bits, int64_t m_cap);
