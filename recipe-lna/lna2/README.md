# lna2 — bespoke routed-experts decode kernel for DSV4-Flash-Vision on 8× RTX PRO 2000 (house kernel since 2026-09-04)

Outer: resident expert teams + dynamic ticket scheduler (same shape as exllamav3 `exl3_moe`, no grid-wide barrier).
Inner: small-R variants (rows per expert 1/2/4/8, m ≤ 24) built on `exl3_gemv_kernel`'s core — the incumbent's M=16 tile wastes 15/16 of its MMA at R_e ≈ 1.04.
One scratch per device; concurrent streams disabled by default (`LNA2_CONCURRENT_STREAMS=1`).

Files: `lna_moe_ticket.cu/.cuh` (the kernel), `lna_gemv_core.cuh`, `lna_moe_decode.cu/.cuh` (F1, the first barrier-based attempt, kept as opt-in `lna`), `bindings.cpp`, `setup.py` (build from vcruz305/vllm-exl3 v0.3.0's csrc tree; needs `EXL3_EXT_INCLUDE` pointing at exllamav3's `exllamav3_ext`), prebuilt `vllm_exl3_c.cpython-312-x86_64-linux-gnu.so` (sm_120, torch cu13), tests (`parity_native.py`, `gate_active.py`, `gate_vram.py`, `bench_lna.py`, `primitive_gate.cu`, `cold_read_bw.cu`).
Plugin side: `../vllm_exl3_exl3.py.lna` (`get_moe_kernel_backend`, `_apply_lna2_moe`, `VLLM_EXL3_MOE_STRICT`).

Measured (rank0, graph on): 232.6 → 104.0 µs/launch (2.24×); single-stream code 86 → 108.6 tok/s; KV window 396,656 → 395,069 tokens at `--gpu-memory-utilization 0.97`; ppl unchanged (4.7630 vs 4.7647); 166k needle TTFT 167 → 119.6 s.
Gates and reports: `../../orders/reports/`. Canon: `../../docs/LNA-CANON.md`.
