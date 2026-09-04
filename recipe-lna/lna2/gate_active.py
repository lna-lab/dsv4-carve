"""F3b regression gate: prove the requested MoE kernel ACTUALLY RAN.

The F3 seat gate passed a full serving run while measuring the incumbent,
because `lna2` declined every call and fell back to exllamav3 in total silence
(REPORT-F3-seat.md §2).  Three defects combined:
  1. _prepare_lna_state() was only called for backend == "lna"
  2. _validate_lna_pointer_contract() read `linear.mcg`, but exllamav3's
     LinearEXL3 stores the marker in `mcg_tensor` and `mcg` is a bool
  3. nothing logged the decline
This file fails if any of the three comes back.
"""
import os, sys, torch
from types import SimpleNamespace

from f1.parity_native import synthetic_layer, input_tensor, route_ids, route_weights, LIMIT, DEVICE
import vllm_exl3.exl3 as X

FAIL = []


def expect(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    bits = 2
    layer = synthetic_layer(bits)
    inners = layer._exl3_inners

    # --- 1. the real LinearEXL3 marker attribute -------------------------
    lin = inners[0]["gate"]
    print(f'MEASURED LinearEXL3 attrs: mcg={type(getattr(lin,"mcg",None)).__name__} '
          f'mcg_tensor={type(getattr(lin,"mcg_tensor",None)).__name__}')
    X._validate_lna_pointer_contract(inners, 4096, 256, bits)   # must not raise
    expect(True, "ABI validation accepts a real exllamav3 LinearEXL3 (mcg_tensor)")

    # --- 2. preflight must succeed and set the ready flag ----------------
    for backend in ("lna", "lna2"):
        probe = SimpleNamespace(prefix=f"probe-{backend}", _exl3_hidden_size=4096,
                                _exl3_intermediate_local=256, _exl3_bits=bits,
                                _exl3_k=bits)
        X._prepare_lna_state(probe, inners, backend)
        expect(getattr(probe, "_exl3_lna_ready", False),
               f"_prepare_lna_state(backend={backend}) sets _exl3_lna_ready "
               f"(error={getattr(probe, '_exl3_lna_error', None)})")
        # backend-specific scratch only
        n_f1 = len(getattr(probe, "_exl3_lna_scratch", {}) or {})
        expect((n_f1 == 0) if backend == "lna2" else (n_f1 == 1),
               f"backend={backend} allocates only its own scratch (F1 entries={n_f1})")

    # --- 3. the kernel actually runs, and the counter proves it ----------
    X.reset_moe_kernel_call_counts()
    m = 4
    x = input_tensor(m, 'real-bf16')
    ids = route_ids(m, 'unique').to(torch.long)
    w = route_weights(m, 'normal')
    out = X._apply_lna2_moe(x, ids, w, layer, inners, None, LIMIT)
    torch.cuda.synchronize()
    expect(out is not None, "_apply_lna2_moe returned a tensor")
    expect(X.moe_kernel_call_counts()["lna2"] == 1,
           f"lna2 call counter incremented: {X.moe_kernel_call_counts()}")

    # --- 4. row ceiling is asked, not assumed ---------------------------
    maxr = X.lna2_max_rows()
    print(f'MEASURED lna2_max_rows={maxr}')
    # 制式 is 4 seqs x DSpark3 = 16 rows.  F3c ships 24 (50% headroom) instead of
    # 32 to give the seat back 1.6 MiB/GPU; the cliff at exactly 16 is what
    # mattered and it is gone either way.
    expect(maxr >= 24, f"row ceiling >= 24 so 制式 (16 rows) has headroom, got {maxr}")
    for rows in (16, 20, maxr):
        xr = input_tensor(rows, 'real-bf16')
        idr = route_ids(rows, 'unique').to(torch.long)
        wr = route_weights(rows, 'normal')
        n0 = X.moe_kernel_call_counts()["lna2"]
        o = X._apply_lna2_moe(xr, idr, wr, layer, inners, None, LIMIT)
        torch.cuda.synchronize()
        expect(o is not None and X.moe_kernel_call_counts()["lna2"] == n0 + 1,
               f"lna2 accepts m={rows}")

    # --- 5. over the ceiling: declines, and says so ---------------------
    X._LNA2_REJECT_LOGGED.clear()
    over = maxr + 8
    o = X._apply_lna2_moe(input_tensor(over, 'real-bf16'),
                          route_ids(over, 'unique').to(torch.long),
                          route_weights(over, 'normal'), layer, inners, None, LIMIT)
    expect(o is None, f"lna2 declines m={over} (> ceiling)")
    reasons = X.moe_kernel_declined_reasons()
    expect(any('geometry' in r for r in reasons),
           f"the decline is LOGGED with a reason: {reasons}")

    # --- 6. strict levels -----------------------------------------------
    # Level 1 must NOT raise on a shape decline: prefill (512 rows) legitimately
    # declines, so level 1 has to be safe to leave on in production.
    def try_over(level):
        os.environ["VLLM_EXL3_MOE_STRICT"] = str(level)
        X._LNA2_REJECT_LOGGED.clear()
        try:
            X._apply_lna2_moe(input_tensor(over, 'real-bf16'),
                              route_ids(over, 'unique').to(torch.long),
                              route_weights(over, 'normal'), layer, inners, None, LIMIT)
            return False
        except RuntimeError:
            return True
        finally:
            os.environ["VLLM_EXL3_MOE_STRICT"] = "0"

    expect(not try_over(1),
           "STRICT=1 does NOT raise on a shape decline (prefill must survive)")
    expect(try_over(2), "STRICT=2 raises on a shape decline (gate-only)")

    # Level 1 DOES raise when the kernel was never made ready -- the exact
    # class of bug the seat hit (mcg/mcg_tensor left _exl3_lna_ready False).
    notready = probe = SimpleNamespace(prefix="notready", _exl3_hidden_size=4096,
                                       _exl3_intermediate_local=256,
                                       _exl3_bits=bits, _exl3_k=bits,
                                       _exl3_lna_ready=False, _exl3_lna_error="simulated",
                                       _exl3_ptrs=layer._exl3_ptrs,
                                       _exl3_inners=inners)
    os.environ["VLLM_EXL3_MOE_STRICT"] = "1"
    try:
        X._LNA2_REJECT_LOGGED.clear()
        raised = False
        try:
            X._apply_lna2_moe(x, ids, w, notready, inners, None, LIMIT)
        except RuntimeError:
            raised = True
        expect(raised, "STRICT=1 raises when the kernel was never prepared")
    finally:
        os.environ["VLLM_EXL3_MOE_STRICT"] = "0"

    # --- 7. scratch is shared across layers, not per layer --------------
    before = X.lna2_scratch_bytes_resident()
    for _ in range(8):
        other = synthetic_layer(bits)
        p = SimpleNamespace(prefix="extra", _exl3_hidden_size=4096,
                            _exl3_intermediate_local=256, _exl3_bits=bits, _exl3_k=bits)
        X._prepare_lna_state(p, other._exl3_inners, "lna2")
        del other
        torch.cuda.empty_cache()
    after = X.lna2_scratch_bytes_resident()
    expect(after == before,
           f"8 more layers add no lna2 scratch ({before} -> {after} bytes)")
    print(f'MEASURED lna2 scratch resident total = {after} B '
          f'({after/1048576:.3f} MiB) for {len(X._LNA2_SCRATCH)} (device,stream) key(s)')

    del layer
    torch.cuda.empty_cache()
    if FAIL:
        print("GATE_ACTIVE FAIL:", len(FAIL))
        raise SystemExit(1)
    print("GATE_ACTIVE PASS")


if __name__ == '__main__':
    main()
