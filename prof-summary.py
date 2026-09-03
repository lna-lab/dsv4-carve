#!/usr/bin/env python3
"""Summarize a torch profiler trace: GPU kernel time by name group, plus CPU-side gaps. usage: prof-summary.py trace.json.gz [top]"""
import gzip, json, sys, re, collections
path = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ev = json.load(gzip.open(path))["traceEvents"]
kern = [e for e in ev if e.get("cat") == "kernel"]
gpu_total = sum(e["dur"] for e in kern)
span = (max(e["ts"] + e["dur"] for e in kern) - min(e["ts"] for e in kern)) if kern else 0
def group(n):
    n = re.sub(r"<.*", "", n); n = re.sub(r"\(.*", "", n)
    for pat, g in [("nccl", "NCCL all-reduce/comm"), ("exl3|trellis|mcg", "EXL3 expert GEMM"), ("gemm|cutlass|cublas|Gemm|sm120|sm90|sm80", "dense GEMM (BF16)"), ("topk|sort|radix|indexer|Indexer", "indexer/topk"), ("mla|flashinfer|attn|attention|sparse", "attention"), ("norm|rms", "norm"), ("silu|act|gelu", "activation"), ("copy|memcpy|Memcpy|cat|elementwise|vectorized|fill|index_|gather|scatter", "elementwise/copy"), ("moe|expert|route|sum", "MoE glue")]:
        if re.search(pat, n): return g
    return "other"
by_group = collections.Counter(); by_name = collections.Counter(); cnt = collections.Counter()
for e in kern:
    by_group[group(e["name"])] += e["dur"]; by_name[e["name"][:90]] += e["dur"]; cnt[e["name"][:90]] += 1
print(f"kernels={len(kern)} gpu_busy={gpu_total/1e3:.1f} ms  wall_span={span/1e3:.1f} ms  busy_ratio={gpu_total/max(span,1):.2f}")
print("--- by group (ms, share of GPU busy)")
for g, d in by_group.most_common(): print(f"{d/1e3:9.1f} ms {d/gpu_total*100:5.1f}%  {g}")
print(f"--- top {top} kernels (ms, count, avg us)")
for n, d in by_name.most_common(top): print(f"{d/1e3:9.1f} ms {cnt[n]:7d} {d/cnt[n]:8.1f}us  {n}")
