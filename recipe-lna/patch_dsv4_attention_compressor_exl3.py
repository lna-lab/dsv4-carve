#!/usr/bin/env python3
"""LNA-LAB: DSV4 attention.py reads compressor.fused_wkv_wgate.weight directly (torch.mm). When the
module was built by vllm-exl3 (no .weight, has _exl3_linears), run the EXL3 shards instead.
Idempotent, exact anchors, backup attention.py.orig-lna2."""
import re
import shutil, sys
from pathlib import Path

HELPER_V2 = '''
def _lna_kv_score_v2(module, hidden_states):
    """LNA-LAB: fused_wkv_wgate as bf16 weight (torch.mm) or as EXL3 shards (vllm-exl3)."""
    linears = getattr(module, "_exl3_linears", None)
    if linears:
        source = hidden_states.reshape(-1, hidden_states.shape[-1])
        if not source.is_contiguous():
            source = source.contiguous()
        x = source.to(torch.float16).contiguous()
        try:
            # LNA-LAB: F2 T1 — preserve source identity across separate
            # compressor/indexer calls so they share one owner generation.
            from vllm_exl3.exl3 import run_exl3_group
            outs = run_exl3_group(module, x, torch.float32, source_tensor=source)
        except (ImportError, AttributeError) as exc:
            # Keep the compatibility fallback, but make a missing/incomplete
            # mounted plugin tree observable.  Strict mode is for seat gates
            # and turns this otherwise-compatible fallback into a hard failure.
            import logging, os

            reason = f"{type(exc).__name__}: {exc}"
            logging.getLogger(__name__).warning(
                "F2 DENSE_GROUP declined: %s", reason
            )
            if os.environ.get("LNA_EXL3_DENSE_STRICT") == "1":
                raise
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
''', '''                return _lna_kv_score_v2(compressor.fused_wkv_wgate, hidden_states)  # LNA-LAB
'''),
    ('''                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
''', '''                return _lna_kv_score_v2(indexer.compressor.fused_wkv_wgate, hidden_states)  # LNA-LAB
'''),
]

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if root is None:
        import vllm
        root = Path(vllm.__file__).resolve().parent
    f = root / "models" / "deepseek_v4" / "attention.py"
    text = f.read_text()
    if "def _lna_kv_score_v2(" in text:
        print(f"{f}: already patched"); return 0
    # T0 used _lna_kv_score as its marker. Replace the old helper in place so
    # a helper-body change is not hidden by the old idempotency check.
    if "def _lna_kv_score(" in text:
        start = text.index("\ndef _lna_kv_score(")
        match = re.search(r"\n\nclass ", text[start:])
        if match is None:
            raise SystemExit(f"{f}: old _lna_kv_score helper has no class boundary")
        end = start + match.start() + 2
        text = text[:start] + "\n" + HELPER_V2 + text[end:]
        text = text.replace("_lna_kv_score(", "_lna_kv_score_v2(")
        bak = f.with_suffix(f.suffix + ".orig-lna2")
        if not bak.exists(): shutil.copy2(f, bak)
        f.write_text(text); compile(text, str(f), "exec"); print(f"{f}: replaced old helper with v2")
        return 0
    for old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"{f}: expected 1 anchor, found {text.count(old)}: {old[:60]!r}")
        text = text.replace(old, new)
    # helper: insert before the first class definition
    idx = text.index("\nclass ")
    text = text[:idx] + "\n" + HELPER_V2 + text[idx:]
    # the helper only reads .weight in the fallback path; keep it after fused_wkv_wgate module refs
    text = text.replace("return torch.mm(hidden_states, module.fused_wkv_wgate.weight.T, out_dtype=torch.float32)",
                        "return torch.mm(hidden_states, module.weight.T, out_dtype=torch.float32)")
    bak = f.with_suffix(f.suffix + ".orig-lna2")
    if not bak.exists(): shutil.copy2(f, bak)
    f.write_text(text); compile(text, str(f), "exec"); print(f"{f}: patched")
    return 0

if __name__ == "__main__":
    sys.exit(main())
