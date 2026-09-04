"""F1 gate E: kernel-alone latency (cold/warm) + CUDA graph capture/replay parity."""
import argparse, torch
from f1.parity_native import (synthetic_layer, production_layer, lna_fp32, input_tensor,
                              route_ids, route_weights, HIDDEN, TOPK, LIMIT, DEVICE,
                              DEGENERATE_ROW_NORM, REL_TOL)
from vllm_exl3.exl3 import apply_exl3_python_loop
import vllm_exl3_c

# MEASURED by Luna (f1/cold_read_bw.cu, run-15/16): 264-275 GB/s cold read on
# this card.  Scale bytes are 26,112 B/expert for both K.
BW_COLD = 270e9
BYTES = {2: 812544, 3: 1205760}
L2_FLUSH = None


def call(layer, x, out, ids, w):
    p = layer._exl3_ptrs
    scratch = next(iter(layer._exl3_lna_scratch.values()))
    vllm_exl3_c.lna_moe_decode(
        x, out, p['gate_trellis'], p['gate_suh'], p['gate_svh'],
        p['up_trellis'], p['up_suh'], p['up_svh'],
        p['down_trellis'], p['down_suh'], p['down_svh'],
        ids, w, int(layer._exl3_bits), float(LIMIT), scratch)


def bench(layer, bits, m, label, iters=50):
    global L2_FLUSH
    if L2_FLUSH is None:
        L2_FLUSH = torch.empty(96 << 20, dtype=torch.uint8, device=DEVICE)
    torch.manual_seed(4242 + m)
    x = input_tensor(m, 'real-bf16')
    out = torch.empty((m, HIDDEN), dtype=torch.float32, device=DEVICE)
    # Independent uniform routing: the realistic decode case.
    ids = torch.randint(0, 256, (m, TOPK), dtype=torch.int32, device=DEVICE)
    w = route_weights(m, 'normal')
    U = int(torch.unique(ids).numel())
    for _ in range(5):
        call(layer, x, out, ids, w)
    torch.cuda.synchronize()
    def timed(cold):
        ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
        for a, b in ev:
            if cold:
                L2_FLUSH.zero_()
            a.record(); call(layer, x, out, ids, w); b.record()
        torch.cuda.synchronize()
        t = sorted(a.elapsed_time(b) * 1e3 for a, b in ev)
        return t[len(t) // 2], t[int(len(t) * 0.95)]
    cold_p50, cold_p95 = timed(True)
    warm_p50, warm_p95 = timed(False)
    floor = U * BYTES[bits] / BW_COLD * 1e6
    print(f'MEASURED LATENCY {label} K={bits} m={m} U={U} '
          f'cold_p50={cold_p50:.1f}us cold_p95={cold_p95:.1f}us '
          f'warm_p50={warm_p50:.1f}us warm_p95={warm_p95:.1f}us '
          f'T_floor={floor:.1f}us ratio_cold={cold_p50/floor:.2f} '
          f'vs229us={229.0/cold_p50:.2f}x')
    return cold_p50, warm_p50, floor, U


def graph_test(layer, bits):
    """Capture/replay each M_CAP bucket; parity and stale-state must both hold."""
    for m in (4, 8, 16):
        x = input_tensor(m, 'real-bf16')
        out = torch.empty((m, HIDDEN), dtype=torch.float32, device=DEVICE)
        ids = torch.randint(0, 256, (m, TOPK), dtype=torch.int32, device=DEVICE)
        w = route_weights(m, 'normal')
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call(layer, x, out, ids, w)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                call(layer, x, out, ids, w)
        except Exception as exc:
            # F3 gate 5: a capture failure is a FAILURE, not a skip.  The old
            # `continue` here meant a kernel that could not be captured at all
            # still reported PASS overall.
            print(f'MEASURED GRAPH K={bits} m={m} CAPTURE_FAILED {type(exc).__name__}: '
                  f'{str(exc).splitlines()[0][:110]}')
            raise SystemExit(1)
        worst = 0.0
        for rep in range(8):
            x.copy_(input_tensor(m, 'real-bf16') * (1.0 + 0.1 * rep))
            ids.copy_(torch.randint(0, 256, (m, TOPK), dtype=torch.int32, device=DEVICE))
            g.replay()
            torch.cuda.synchronize()
            ref = apply_exl3_python_loop(x, ids.to(torch.long), w,
                                         layer._exl3_inners, None, LIMIT)
            n = ref.norm(dim=1)
            rel = float(torch.where(n > DEGENERATE_ROW_NORM,
                                    (out - ref).norm(dim=1) / n.clamp_min(1e-6),
                                    torch.zeros_like(n)).max().item())
            worst = max(worst, rel)
        ok = worst <= REL_TOL
        print(f'MEASURED GRAPH K={bits} m={m} replays=8 worst_rel={worst:.6g} '
              f'{"PASS" if ok else "FAIL"}')
        if not ok:
            # ...and a replay mismatch was previously only printed.
            raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='bench')
    a = ap.parse_args()
    for bits, ln in ((2, 0), (3, 13)):
        layer, b = production_layer(ln)
        assert b == bits, (ln, b)
        if a.mode in ('bench', 'all'):
            for m in (1, 4, 16):
                bench(layer, bits, m, f'prod-L{ln}')
        if a.mode in ('graph', 'all'):
            graph_test(layer, bits)
        del layer
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
