#!/usr/bin/env python3
"""Attribute GPU kernel time to the launching CPU op + its input shapes (needs torch_profiler_record_shapes). usage: prof-shapes.py trace.json.gz [top]"""
import gzip, json, sys, bisect, collections
path = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ev = json.load(gzip.open(path))["traceEvents"]
ops = collections.defaultdict(list)      # tid -> [(ts, te, name, dims)]
rt = {}                                   # correlation -> (tid, ts)
kern = []
for e in ev:
    c = e.get("cat"); a = e.get("args", {})
    if c == "cpu_op":
        ops[e["tid"]].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"], str(a.get("Input Dims", ""))[:90]))
    elif c == "cuda_runtime" and "correlation" in a:
        rt[a["correlation"]] = (e["tid"], e["ts"])
    elif c == "kernel":
        kern.append((e["name"], e.get("dur", 0), a.get("correlation")))
for t in ops: ops[t].sort()
starts = {t: [o[0] for o in ops[t]] for t in ops}
def owner(tid, ts):
    if tid not in ops: return None
    i = bisect.bisect_right(starts[tid], ts) - 1
    best = None
    while i >= 0 and ops[tid][i][0] <= ts:
        o = ops[tid][i]
        if o[1] >= ts: best = o                 # innermost = latest start that still contains ts
        if best is not None and o[0] < ts - 5_000_000: break
        i -= 1
        if best is not None and i >= 0 and ops[tid][i][1] < ts: break
    return best
agg = collections.Counter(); cnt = collections.Counter(); total = 0.0
for name, dur, corr in kern:
    total += dur
    o = owner(*rt[corr]) if corr in rt else None
    key = (o[2] if o else "?", o[3] if o else "", name.split("(")[0][:48])
    agg[key] += dur; cnt[key] += 1
print(f"total kernel time {total/1000:.1f} ms")
print(f"{'ms':>8} {'%':>5} {'n':>5}  op | input dims | kernel")
for k, v in agg.most_common(top):
    print(f"{v/1000:8.1f} {100*v/total:5.1f} {cnt[k]:5d}  {k[0]} | {k[1]} | {k[2]}")
