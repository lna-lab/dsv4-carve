"""F3b: measure the per-layer VRAM cost of the LNA scratch, old vs new."""
import torch, vllm_exl3_c
from types import SimpleNamespace
from f1.parity_native import synthetic_layer
import vllm_exl3.exl3 as X

MIB = 1048576.0
N_ROUTED = 43     # MEASURED from the pack index (layers.N.ffn.experts.*)
N_SHARED = 43     # every layer also has a shared expert; the seat trace shows
                  # moe_forward_shared going through the same kernel


def probe(prefix, bits=2):
    return SimpleNamespace(prefix=prefix, _exl3_hidden_size=4096,
                           _exl3_intermediate_local=256, _exl3_bits=bits, _exl3_k=bits)


def main():
    bits = 2
    layer = synthetic_layer(bits)      # one set of weights, reused as the ABI source
    inners = layer._exl3_inners
    lna1_b = int(vllm_exl3_c.lna_moe_scratch_bytes())
    lna2_b = int(vllm_exl3_c.lna2_moe_scratch_bytes())
    print(f'MEASURED scratch_bytes lna(F1)={lna1_b} ({lna1_b/MIB:.3f} MiB) '
          f'lna2(F3)={lna2_b} ({lna2_b/MIB:.3f} MiB)')
    print(f'MEASURED lna2 max_rows={vllm_exl3_c.lna2_moe_max_rows()} '
          f'max_pairs={vllm_exl3_c.lna2_moe_info(2)[12]}')

    # --- NEW behaviour: prepare N layers for lna2, measure the real delta ---
    X._LNA2_SCRATCH.clear()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    keep = []
    for i in range(N_ROUTED + N_SHARED):
        p = probe(f'L{i}')
        X._prepare_lna_state(p, inners, "lna2")
        assert p._exl3_lna_ready, p._exl3_lna_error
        keep.append(p)
    torch.cuda.synchronize()
    new_total = torch.cuda.memory_allocated() - base
    print(f'MEASURED NEW  backend=lna2, {N_ROUTED+N_SHARED} layers prepared: '
          f'delta={new_total} B = {new_total/MIB:.3f} MiB '
          f'({new_total/(N_ROUTED+N_SHARED)/MIB:.4f} MiB/layer)')

    # --- OLD behaviour (as shipped to the seat): both scratches, per layer ---
    for n_layers, label in ((N_ROUTED, 'routed only (43)'),
                            (N_ROUTED + N_SHARED, 'routed+shared (86)'),
                            (72, 'implied by the seat deficit (72)')):
        old = n_layers * (lna1_b + lna2_b // 2 * 0 + 3392896)   # seat build had MAX_M=16 lna2
        old_seat = n_layers * (lna1_b + 3392896)
        print(f'  old (seat build, per layer lna+lna2@16rows) {label}: '
              f'{old_seat} B = {old_seat/MIB:.1f} MiB = {old_seat/1024/MIB:.3f} GiB')
    print(f'  new (shared, MAX_M=32): {new_total} B = {new_total/MIB:.3f} MiB '
          f'= {new_total/1024/MIB:.4f} GiB')

    del keep, layer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
