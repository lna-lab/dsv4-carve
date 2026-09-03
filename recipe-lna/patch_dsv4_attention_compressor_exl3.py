#!/usr/bin/env python3
"""LNA-LAB: DSV4 attention.py reads compressor.fused_wkv_wgate.weight directly (torch.mm). When the
module was built by vllm-exl3 (no .weight, has _exl3_linears), run the EXL3 shards instead.
Idempotent, exact anchors, backup attention.py.orig-lna2."""
import shutil, sys
from pathlib import Path

HELPER = '''
def _lna_kv_score(module, hidden_states):
    """LNA-LAB: fused_wkv_wgate as bf16 weight (torch.mm) or as EXL3 shards (vllm-exl3)."""
    linears = getattr(module, "_exl3_linears", None)
    if linears:
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).to(torch.float16).contiguous()
        outs = [lin.forward(x, {}, out_dtype=torch.float32) for lin in linears]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)
        return out.reshape(*hidden_states.shape[:-1], out.shape[-1])
    return torch.mm(hidden_states, module.fused_wkv_wgate.weight.T, out_dtype=torch.float32)

'''
EDITS = [
    ('''                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
''', '''                return _lna_kv_score(compressor.fused_wkv_wgate, hidden_states)  # LNA-LAB
'''),
    ('''                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
''', '''                return _lna_kv_score(indexer.compressor.fused_wkv_wgate, hidden_states)  # LNA-LAB
'''),
]

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        import vllm
        root = Path(vllm.__file__).resolve().parent
    f = root / "models" / "deepseek_v4" / "attention.py"
    text = f.read_text()
    if "_lna_kv_score" in text:
        print(f"{f}: already patched"); return 0
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"{f}: expected 1 anchor, found {text.count(old)}: {old[:60]!r}")
        text = text.replace(old, new)
    # helper: insert before the first class definition
    idx = text.index("\nclass ")
    text = text[:idx] + "\n" + HELPER + text[idx:]
    # the helper only reads .weight in the fallback path; keep it after fused_wkv_wgate module refs
    text = text.replace("return torch.mm(hidden_states, module.fused_wkv_wgate.weight.T, out_dtype=torch.float32)",
                        "return torch.mm(hidden_states, module.weight.T, out_dtype=torch.float32)")
    bak = f.with_suffix(f.suffix + ".orig-lna2")
    if not bak.exists(): shutil.copy2(f, bak)
    f.write_text(text); compile(text, str(f), "exec"); print(f"{f}: patched")
    return 0

if __name__ == "__main__":
    sys.exit(main())
