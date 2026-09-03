#!/usr/bin/env python3
"""tok/s + DSpark acceptance meter for the vLLM DSV4 seat. usage: bench-dsv4.py [port] [max_tokens]"""
import json, sys, time, urllib.request, re
port = sys.argv[1] if len(sys.argv) > 1 else "8899"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
U = f"http://127.0.0.1:{port}"
P = {"en": "Explain in detail how a transformer language model generates text, step by step.",
     "ja": "日本の四季それぞれの特徴と、季節ごとの代表的な行事や食べ物について詳しく説明してください。",
     "code": "Write a Python class implementing an LRU cache with get/put and O(1) operations, with docstrings and a small test."}
def metrics():
    try: t = urllib.request.urlopen(U + "/metrics", timeout=10).read().decode()
    except Exception: return {}
    out = {}
    for k in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total"):
        m = re.findall(r"^" + re.escape(k) + r"(?:\{[^}]*\})?\s+([\d.e+]+)", t, re.M)
        if m: out[k.split("num_")[1].replace("_total","")] = sum(float(x) for x in m)
    return out
for name, prompt in P.items():
    body = {"model": "DSV4-Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": N, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    m0 = metrics(); t0 = time.time(); first = None; toks = 0; text = ""
    req = urllib.request.Request(U + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): toks = d["usage"]["completion_tokens"]
            ch = d.get("choices") or []
            if ch and (ch[0]["delta"].get("content") or (ch[0]["delta"].get("reasoning_content") or ch[0]["delta"].get("reasoning"))):
                if first is None: first = time.time()
                text += ch[0]["delta"].get("content") or ""
    wall = time.time() - t0; m1 = metrics()
    acc = ""
    if m1.get("drafts") and m0 is not None:
        dr = m1["drafts"] - m0.get("drafts", 0); ac = m1["accepted_tokens"] - m0.get("accepted_tokens", 0)
        if dr: acc = f" accept_len={1 + ac / dr:.2f} (accepted {ac:.0f}/{m1['draft_tokens'] - m0.get('draft_tokens', 0):.0f} drafted)"
    print(f"{name}: tokens={toks} wall={wall:.2f}s TTFT={(first or t0) - t0:.2f}s decode={toks / (wall - ((first or t0) - t0) + 1e-9):.2f} tok/s{acc}")
    print("   ", repr(text[:100]))
