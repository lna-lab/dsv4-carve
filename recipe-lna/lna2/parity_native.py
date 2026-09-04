"""F1 desk parity and production-tensor exercise.

All CUDA work in this file is intentionally launched by f1/RUN.sh through the
relay.  It constructs the rank-0 TP8 slices from the full production pack:
gate/up columns are 16 trellis columns and down rows are 16 trellis rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm_exl3.exl3 import (
    MCG_MARKER_SIGNED_INT32,
    _apply_lna_moe,
    apply_exl3_python_loop,
    make_linear_exl3,
)


HIDDEN = 4096
FULL_INTERMEDIATE = 2048
LOCAL_INTERMEDIATE = 256
EXPERTS = 256
TOPK = 6
LIMIT = 10.0
M_VALUES = (1, 2, 3, 4, 5, 7, 8, 15, 16)
# F3b raised the lna2 row ceiling to 32 so the seat's 制式 (4 seqs x DSpark3 =
# exactly 16 rows) is no longer sitting on the cliff.  F1 is still capped at 16.
def _lna2_ceiling():
    try:
        import vllm_exl3_c
        return int(vllm_exl3_c.lna2_moe_max_rows())
    except Exception:
        return 16


# Sample past 16 up to whatever the build accepts (F3c default is 24).
M_VALUES_LNA2 = M_VALUES + tuple(
    m for m in (17, 20, 23, 24, 31, 32) if m <= _lna2_ceiling())
REL_TOL = 2e-3              # order gate, applied to the FP32 kernel output
# The BF16 consumer output cannot be judged at REL_TOL: BF16 has an 8-bit
# mantissa, so rounding alone costs ~2^-9 = 1.95e-3 per element.  MEASURED
# (run-23): every BF16 "failure" had max_abs exactly one BF16 ULP (0.25 at
# |out|~64, 0.125, 0.000244141).  Three ULP of headroom is the honest gate.
BF16_REL_TOL = 6e-3
DEGENERATE_ROW_NORM = 1e-3  # below this the reference row carries no signal
# ...so such rows are judged on absolute error.  MEASURED (run-23): exactly
# cancelling routes (w = +1,-1,+.25,-.25,...) leave ~1.2e-5 of row L2 residue
# that comes from the REFERENCE's index_add_ FP32 rounding, not the kernel.
DEGENERATE_ABS_TOL = 1e-4
DEVICE = torch.device('cuda:0')


def ptr_table(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([int(t.data_ptr()) for t in tensors], dtype=torch.int64, device=DEVICE)


def make_layer(bits: int, packs: list[tuple[torch.Tensor, ...]], label: str):
    marker = torch.full((1,), MCG_MARKER_SIGNED_INT32, dtype=torch.int32, device=DEVICE)
    inners = []
    gate_t, gate_su, gate_sv = [], [], []
    up_t, up_su, up_sv = [], [], []
    down_t, down_su, down_sv = [], [], []
    for gt, gs, gv, dt, ds, dv, ut, us, uv in packs:
        gate = make_linear_exl3(gt, gs, gv, marker)
        up = make_linear_exl3(ut, us, uv, marker)
        down = make_linear_exl3(dt, ds, dv, marker)
        inners.append({'gate': gate, 'up': up, 'down': down})
        gate_t.append(gt)
        gate_su.append(gs)
        gate_sv.append(gv)
        up_t.append(ut)
        up_su.append(us)
        up_sv.append(uv)
        down_t.append(dt)
        down_su.append(ds)
        down_sv.append(dv)
    import vllm_exl3_c

    for cap in (4, 8, 16):
        vllm_exl3_c.lna_moe_prepare(bits, cap)
    vllm_exl3_c.lna2_moe_prepare(bits)
    stream_key = int(torch.cuda.current_stream(DEVICE).cuda_stream)
    scratch = torch.empty(int(vllm_exl3_c.lna_moe_scratch_bytes()), dtype=torch.uint8, device=DEVICE)
    layer = SimpleNamespace(
        prefix=label,
        _exl3_hidden_size=HIDDEN,
        _exl3_intermediate_local=LOCAL_INTERMEDIATE,
        _exl3_bits=bits,
        _exl3_k=bits,
        _exl3_lna_ready=True,
        _exl3_lna_scratch={stream_key: scratch},
        _exl3_inners=inners,
        _exl3_ptrs={
            'gate_trellis': ptr_table(gate_t), 'gate_suh': ptr_table(gate_su),
            'gate_svh': ptr_table(gate_sv), 'up_trellis': ptr_table(up_t),
            'up_suh': ptr_table(up_su), 'up_svh': ptr_table(up_sv),
            'down_trellis': ptr_table(down_t), 'down_suh': ptr_table(down_su),
            'down_svh': ptr_table(down_sv),
        },
    )
    return layer


def synthetic_layer(bits: int, scale: float = 0.1):
    last = 16 * bits
    torch.manual_seed(7000 + bits)
    # MEASURED (run-20): scale=0.01 drives the whole activation vector into the
    # FP16 subnormal band (|ref| ~ 2e-6, max_abs 6e-9), so relative error is
    # meaningless there.  scale=0.1 puts pre-activations at O(1) with random
    # trellis bits, which is the regime production tensors live in.  The tiny
    # scale is kept as a separate denormal case judged on absolute error.
    packs = []
    for _ in range(EXPERTS):
        raw = lambda shape: torch.randint(0, 65536, shape, dtype=torch.int32, device=DEVICE).to(torch.int16)
        packs.append((
            raw((HIDDEN // 16, LOCAL_INTERMEDIATE // 16, last)),
            torch.full((HIDDEN,), scale, dtype=torch.float16, device=DEVICE),
            torch.full((LOCAL_INTERMEDIATE,), scale, dtype=torch.float16, device=DEVICE),
            raw((LOCAL_INTERMEDIATE // 16, HIDDEN // 16, last)),
            torch.full((LOCAL_INTERMEDIATE,), scale, dtype=torch.float16, device=DEVICE),
            torch.full((HIDDEN,), scale, dtype=torch.float16, device=DEVICE),
            raw((HIDDEN // 16, LOCAL_INTERMEDIATE // 16, last)),
            torch.full((HIDDEN,), scale, dtype=torch.float16, device=DEVICE),
            torch.full((LOCAL_INTERMEDIATE,), scale, dtype=torch.float16, device=DEVICE),
        ))
    return make_layer(bits, packs, f'synthetic-K{bits}-s{scale}')


def production_names(index: dict, layer: int, expert: int) -> dict[str, str]:
    prefix = f'layers.{layer}.ffn.experts.{expert}.'
    names = {}
    for proj in ('w1', 'w2', 'w3'):
        for suffix in ('trellis', 'suh', 'svh', 'mcg'):
            name = prefix + proj + '.' + suffix
            if name not in index:
                raise KeyError(name)
            names[proj + '.' + suffix] = name
    return names


def production_layer(layer_number: int):
    root = Path('/model')
    index_file = next(iter(sorted(root.glob('*.safetensors.index.json'))))
    weight_map = json.loads(index_file.read_text())['weight_map']
    handles = {}
    def get(name: str):
        filename = weight_map[name]
        if filename not in handles:
            handles[filename] = safe_open(str(root / filename), framework='pt', device='cuda')
        return handles[filename].get_tensor(name)

    probe = production_names(weight_map, layer_number, 0)
    bits = int(get(probe['w1.trellis']).shape[-1]) // 16
    if bits not in (2, 3):
        raise ValueError(f'layer {layer_number}: expected K2/K3, got K{bits}')
    packs = []
    for expert in range(EXPERTS):
        names = production_names(weight_map, layer_number, expert)
        w1t, w1u, w1v = get(names['w1.trellis']), get(names['w1.suh']), get(names['w1.svh'])
        w2t, w2u, w2v = get(names['w2.trellis']), get(names['w2.suh']), get(names['w2.svh'])
        w3t, w3u, w3v = get(names['w3.trellis']), get(names['w3.suh']), get(names['w3.svh'])
        # Full pack is [H/16, 2048/16, 16K] for w1/w3 and
        # [2048/16, H/16, 16K] for w2.  This is rank-0 of TP8.
        packs.append((
            w1t[:, :LOCAL_INTERMEDIATE // 16, :].contiguous(), w1u.contiguous(),
            w1v[:LOCAL_INTERMEDIATE].contiguous(),
            w2t[:LOCAL_INTERMEDIATE // 16, :, :].contiguous(), w2u[:LOCAL_INTERMEDIATE].contiguous(),
            w2v.contiguous(),
            w3t[:, :LOCAL_INTERMEDIATE // 16, :].contiguous(), w3u.contiguous(),
            w3v[:LOCAL_INTERMEDIATE].contiguous(),
        ))
    return make_layer(bits, packs, f'production-layer-{layer_number}'), bits


def route_ids(m: int, profile: str) -> torch.Tensor:
    if profile == 'unique':
        return torch.arange(m * TOPK, device=DEVICE, dtype=torch.int32).reshape(m, TOPK)
    if profile == 'hot':
        return torch.zeros((m, TOPK), device=DEVICE, dtype=torch.int32)
    if profile == 'zipf':
        row = torch.tensor([0, 0, 1, 1, 2, 3], device=DEVICE, dtype=torch.int32)
        return row.repeat(m, 1)
    if profile == 'boundary':
        row = torch.tensor([0, 255, 0, 255, 1, 254], device=DEVICE, dtype=torch.int32)
        return row.repeat(m, 1)
    if profile == 'within-row-dup':
        rows = [torch.tensor([(r + j) % EXPERTS for j in (0, 0, 1, 2, 3, 4)], device=DEVICE, dtype=torch.int32)
                for r in range(m)]
        return torch.stack(rows)
    if profile == 'R_e16':
        return torch.stack([
            torch.full((TOPK,), r % 16, device=DEVICE, dtype=torch.int32) for r in range(m)
        ])
    if profile == 'cancel':
        row = torch.tensor([0, 0, 1, 1, 2, 3], device=DEVICE, dtype=torch.int32)
        return row.repeat(m, 1)
    raise ValueError(profile)


def route_weights(m: int, profile: str) -> torch.Tensor:
    if profile == 'zero':
        return torch.zeros((m, TOPK), device=DEVICE, dtype=torch.float32)
    if profile == 'cancel':
        row = torch.tensor([1.0, -1.0, 0.25, -0.25, 0.125, -0.125], device=DEVICE)
        return row.repeat(m, 1)
    row = torch.tensor([0.50, 0.25, 0.125, 0.0625, 0.03125, 0.015625], device=DEVICE)
    return row.repeat(m, 1)


def input_tensor(m: int, name: str) -> torch.Tensor:
    if name == 'zero':
        return torch.zeros((m, HIDDEN), device=DEVICE, dtype=torch.bfloat16)
    if name == 'real-bf16':
        torch.manual_seed(9100 + m)
        return torch.randn((m, HIDDEN), device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
    if name == 'amplified':
        torch.manual_seed(9200 + m)
        return (torch.randn((m, HIDDEN), device=DEVICE) * 8).to(torch.bfloat16)
    if name.startswith('clamp-'):
        value = float(name.split('-', 1)[1])
        return torch.full((m, HIDDEN), value, device=DEVICE, dtype=torch.bfloat16)
    if name == 'large':
        return torch.full((m, HIDDEN), 256.0, device=DEVICE, dtype=torch.bfloat16)
    raise ValueError(name)


def stage_stats(layer, x, ids, weights):
    gate_max = up_max = act_max = down_max = 0.0
    finite = True
    for raw in torch.unique(ids).tolist():
        e = int(raw)
        token_idx, _ = (ids == e).nonzero(as_tuple=True)
        h = x.index_select(0, token_idx).contiguous().half()
        gate = layer._exl3_inners[e]['gate'].forward(h, {}, out_dtype=torch.float32)
        up = layer._exl3_inners[e]['up'].forward(h, {}, out_dtype=torch.float32)
        act = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(min=-LIMIT, max=LIMIT)
        down = layer._exl3_inners[e]['down'].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        gate_max = max(gate_max, float(gate.abs().max().item()))
        up_max = max(up_max, float(up.abs().max().item()))
        act_max = max(act.abs().max().item(), act_max)
        down_max = max(down.abs().max().item(), down_max)
        finite = finite and bool(torch.isfinite(gate).all().item() and torch.isfinite(up).all().item()
                                 and torch.isfinite(act).all().item() and torch.isfinite(down).all().item())
    return gate_max, up_max, float(act_max), float(down_max), finite


BACKEND = os.environ.get('F1_PARITY_BACKEND', 'lna')   # 'lna' (F1) or 'lna2' (F3)


def m_values():
    return M_VALUES_LNA2 if BACKEND == 'lna2' else M_VALUES


def lna_fp32(layer, x, ids, weights):
    """Call the kernel with an FP32 output so parity is not judged at the BF16 floor.

    MEASURED (run-22): with a BF16 consumer output, max_abs pinned at 0.000976562
    = one BF16 ULP and per-row rel sat at ~1.5e-3, which is simply BF16's 8-bit
    mantissa, not kernel error.  The FP32 path is the sharp gate.
    """
    import vllm_exl3_c
    if BACKEND == 'lna2':
        from f1 import lna2
        return lna2.call(layer, x, ids, weights, limit=LIMIT)
    p = layer._exl3_ptrs
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    scratch = next(iter(layer._exl3_lna_scratch.values()))
    vllm_exl3_c.lna_moe_decode(
        x, out,
        p['gate_trellis'], p['gate_suh'], p['gate_svh'],
        p['up_trellis'], p['up_suh'], p['up_svh'],
        p['down_trellis'], p['down_suh'], p['down_svh'],
        ids.to(torch.int32).contiguous(), weights.to(torch.float32).contiguous(),
        int(layer._exl3_bits), float(LIMIT), scratch)
    return out


def one_case(layer, x, ids, weights, label: str, strict: bool = True):
    ref = apply_exl3_python_loop(x, ids.to(torch.long), weights, layer._exl3_inners, None, LIMIT)
    if BACKEND == 'lna2':
        from f1 import lna2
        out = lna2.call(layer, x, ids, weights,
                        out=torch.empty(x.shape, dtype=x.dtype, device=x.device), limit=LIMIT)
    else:
        out = _apply_lna_moe(x, ids.to(torch.long), weights, layer, layer._exl3_inners, None, LIMIT)
    out32 = lna_fp32(layer, x, ids, weights)
    out32b = lna_fp32(layer, x, ids, weights)
    torch.cuda.synchronize(DEVICE)
    bitwise = bool(torch.equal(out32, out32b))
    d32 = (out32 - ref).abs()
    n32 = ref.norm(dim=1)
    rel32 = float(torch.where(n32 > DEGENERATE_ROW_NORM,
                              d32.norm(dim=1) / n32.clamp_min(1e-6),
                              torch.zeros_like(n32)).max().item())
    ref_dtype = ref.to(dtype=x.dtype).float()
    diff = (out.float() - ref_dtype).abs()
    row_norm = ref_dtype.norm(dim=1)
    denom = row_norm.clamp_min(1e-6)
    row_rel = diff.norm(dim=1) / denom
    # Order gate 2: per-row relative error, with an absolute fallback where the
    # reference row is degenerate (all-zero route, cancelling weights, or an
    # activation vector that lives in the FP16 subnormal band).
    degenerate = row_norm <= DEGENERATE_ROW_NORM
    row_abs = diff.norm(dim=1)
    row_rel = torch.where(degenerate, torch.zeros_like(row_rel), row_rel)
    max_rel = float(row_rel.max().item())
    max_degen_abs = float(row_abs[degenerate].max().item()) if bool(degenerate.any()) else 0.0
    if strict and (rel32 > REL_TOL or not bitwise):
        raise AssertionError(f'{label} rel32={rel32} bitwise={bitwise}')
    max_abs = float(diff.max().item())
    finite = bool(torch.isfinite(out).all().item() and torch.isfinite(ref).all().item())
    if strict and (max_rel > BF16_REL_TOL or max_degen_abs > DEGENERATE_ABS_TOL
                   or not finite):
        raise AssertionError(
            f'{label} rel={max_rel} max_abs={max_abs} '
            f'degen_abs={max_degen_abs} finite={finite}')
    return max_rel, max_abs, finite, rel32, bitwise


def run_smoke():
    print('DESK_SMOKE begin')
    for bits in (2, 3):
        layer = synthetic_layer(bits)
        x = input_tensor(4, 'real-bf16')
        ids = route_ids(4, 'unique')
        weights = route_weights(4, 'normal')
        rel, abs_err, finite, rel32, bw = one_case(layer, x, ids, weights, f'synthetic-K{bits}-m4')
        stats = stage_stats(layer, x, ids, weights)
        print(f'MEASURED SMOKE K={bits} m=4 rel_bf16={rel:.6g} rel_fp32={rel32:.6g} '
              f'bitwise={bw} max_abs={abs_err:.6g} '
              f'gate_max={stats[0]:.6g} up_max={stats[1]:.6g} act_max={stats[2]:.6g} '
              f'down_max={stats[3]:.6g} stage_finite={stats[4]}')
        del layer
        torch.cuda.empty_cache()
    print('DESK_SMOKE PASS')


def run_shape_regression(layer):
    """Gate 2: the [m,6] vs flat [6m] confusion must be impossible, not merely unlikely."""
    m = 4
    x = input_tensor(m, 'real-bf16')
    ids = route_ids(m, 'unique')
    w = route_weights(m, 'normal')
    flat = ids.reshape(-1)
    try:
        lna_fp32(layer, x, flat, w.reshape(-1))
        raise AssertionError('flat [6m] ids were accepted; the ABI must reject them')
    except RuntimeError as exc:
        print('MEASURED SHAPE_REGRESSION flat_rejected:', str(exc).splitlines()[0][:90])
    # A [6,m]->[m,6] transposed misread must change the answer (order-sensitive).
    wrong = flat.reshape(TOPK, m).t().contiguous()
    a = lna_fp32(layer, x, ids, w)
    b = lna_fp32(layer, x, wrong, w)
    same = bool(torch.equal(a, b))
    print(f'MEASURED SHAPE_REGRESSION transposed_ids_changes_output={not same}')
    if same:
        raise AssertionError('transposed route ids produced identical output')


def run_poison_replay(layer, bits):
    """Gate 2: NaN-poison scratch/out, replay many times, alternate m and route."""
    scratch = next(iter(layer._exl3_lna_scratch.values()))
    worst = 0.0
    for rep in range(24):
        m = (1, 2, 4, 8, 16)[rep % 5]
        profile = ('unique', 'hot', 'within-row-dup', 'R_e16')[rep % 4]
        x = input_tensor(m, 'real-bf16')
        ids = route_ids(m, profile)
        w = route_weights(m, 'normal')
        # Poison every scratch byte with the NaN bit pattern before the launch.
        scratch.view(torch.int32)[:] = torch.tensor(-1, dtype=torch.int32, device=DEVICE)
        ref = apply_exl3_python_loop(x, ids.to(torch.long), w, layer._exl3_inners, None, LIMIT)
        out = lna_fp32(layer, x, ids, w)
        torch.cuda.synchronize(DEVICE)
        if not bool(torch.isfinite(out).all().item()):
            raise AssertionError(f'poison leaked NaN at rep={rep} m={m} {profile}')
        n = ref.norm(dim=1)
        rel = float(torch.where(n > DEGENERATE_ROW_NORM,
                                (out - ref).norm(dim=1) / n.clamp_min(1e-6),
                                torch.zeros_like(n)).max().item())
        worst = max(worst, rel)
        if rel > REL_TOL:
            raise AssertionError(f'poison replay rel={rel} rep={rep} m={m} {profile}')
    print(f'MEASURED POISON_REPLAY K={bits} reps=24 worst_rel={worst:.6g} PASS')


def run_full():
    route_profiles = ('unique', 'hot', 'zipf', 'boundary', 'within-row-dup', 'R_e16')
    input_profiles = ('zero', 'real-bf16', 'amplified', 'clamp-9.99', 'clamp-10.0', 'clamp-10.01', 'large')
    total = 0
    worst_rel = 0.0
    worst_abs = 0.0
    failures = []
    print('DESK_PARITY synthetic begin')
    for bits in (2, 3):
        layer = synthetic_layer(bits)
        for mi, m in enumerate(m_values()):
            for ri, profile in enumerate(route_profiles):
                ids = route_ids(m, profile)
                # Rotate the weight profile so 'zero' and 'cancel' are exercised
                # against every route shape, not only the one they were named for.
                wprof = ('normal', 'zero', 'cancel')[(mi + ri) % 3]
                weights = route_weights(m, wprof)
                inp = input_profiles[(mi + ri) % len(input_profiles)]
                x = input_tensor(m, inp)
                try:
                    rel, abs_err, finite, rel32, bw = one_case(layer, x, ids, weights,
                                                    f'synthetic-K{bits}-m{m}-{profile}-{inp}-{wprof}')
                    stats = stage_stats(layer, x, ids, weights)
                    if not stats[4]:
                        raise AssertionError('stagewise reference non-finite')
                    total += 1
                    worst_rel = max(worst_rel, rel32)
                    worst_abs = max(worst_abs, abs_err)
                except Exception as exc:
                    failures.append(str(exc))
        del layer
        torch.cuda.empty_cache()
    # Explicit cancellation is separate from the zero-weight route so both
    # fixed-order accumulation and zero scaling are covered.
    layer = synthetic_layer(2)
    for m in ((4, 16, _lna2_ceiling()) if BACKEND == 'lna2' else (4, 16)):
        x = input_tensor(m, 'real-bf16')
        for profile in ('cancel',):
            ids = route_ids(m, profile)
            weights = route_weights(m, profile)
            try:
                rel, abs_err, finite, rel32, bw = one_case(layer, x, ids, weights, f'synthetic-K2-m{m}-cancel')
                total += 1
                worst_rel = max(worst_rel, rel32)
                worst_abs = max(worst_abs, abs_err)
                print(f'MEASURED CANCEL K=2 m={m} rel_bf16={rel:.6g} rel_fp32={rel32:.6g} '
                      f'bitwise={bw} max_abs={abs_err:.6g} finite={finite}')
            except Exception as exc:
                failures.append(str(exc))
    del layer
    torch.cuda.empty_cache()
    print(f'MEASURED DESK_PARITY synthetic_cases={total} worst_rel={worst_rel:.6g} '
          f'worst_abs={worst_abs:.6g} failures={len(failures)}')
    if failures:
        for failure in failures[:8]:
            print('DESK_FAILURE', failure)
        raise SystemExit(1)

    layer = synthetic_layer(2)
    run_shape_regression(layer)
    run_poison_replay(layer, 2)
    del layer
    torch.cuda.empty_cache()
    layer = synthetic_layer(3)
    run_poison_replay(layer, 3)
    del layer
    torch.cuda.empty_cache()

    print('DESK_PARITY production begin')
    for layer_number in (0, 13, 22, 28):
        layer, bits = production_layer(layer_number)
        for m in ((4, 16, _lna2_ceiling()) if BACKEND == 'lna2' else (4, 16)):
            for profile in ('unique', 'boundary', 'R_e16'):
                x = input_tensor(m, 'real-bf16')
                ids = route_ids(m, profile)
                weights = route_weights(m, profile)
                rel, abs_err, finite, rel32, bw = one_case(layer, x, ids, weights,
                                                f'production-L{layer_number}-K{bits}-m{m}-{profile}')
                stats = stage_stats(layer, x, ids, weights)
                print(f'MEASURED PRODUCTION layer={layer_number} K={bits} m={m} route={profile} '
                      f'rel_bf16={rel:.6g} rel_fp32={rel32:.6g} bitwise={bw} '
                      f'max_abs={abs_err:.6g} finite={finite} '
                      f'gate_max={stats[0]:.6g} up_max={stats[1]:.6g} act_max={stats[2]:.6g} '
                      f'down_max={stats[3]:.6g} stage_finite={stats[4]}')
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    print('DESK_PARITY production PASS')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('F1 parity requires the relay CUDA card')
    if args.smoke:
        run_smoke()
    else:
        run_full()


if __name__ == '__main__':
    main()
