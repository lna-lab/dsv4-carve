#!/usr/bin/env python3
"""LNA-LAB: let LNA_DSV4_AUX_STREAMS=0 disable DSV4's attention aux CUDA streams (EXL3 dense GEMMs are
cooperative kernels sharing one device lock buffer; running them concurrently on aux streams deadlocks).
Idempotent exact-anchor patch of nvidia/model.py, backup model.py.orig-lna3."""
import shutil, sys
from pathlib import Path
OLD = "        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]\n"
NEW = ("        import os as _lna_os  # LNA-LAB\n"
       "        aux_stream_list = (None if _lna_os.environ.get(\"LNA_DSV4_AUX_STREAMS\", \"1\") == \"0\"\n"
       "                           else [torch.cuda.Stream() for _ in range(3)])  # LNA-LAB: EXL3 dense needs sequential GEMMs\n")
def main():
    import vllm
    f = Path(vllm.__file__).resolve().parent / "models" / "deepseek_v4" / "nvidia" / "model.py"
    t = f.read_text()
    if "LNA_DSV4_AUX_STREAMS" in t: print(f"{f}: already patched"); return 0
    if t.count(OLD) != 1: raise SystemExit(f"{f}: expected 1 anchor, found {t.count(OLD)}")
    b = f.with_suffix(f.suffix + ".orig-lna3"); b.exists() or shutil.copy2(f, b)
    t = t.replace(OLD, NEW); compile(t, str(f), "exec"); f.write_text(t); print(f"{f}: patched"); return 0
if __name__ == "__main__": sys.exit(main())
