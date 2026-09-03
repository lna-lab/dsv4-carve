#!/usr/bin/env python3
"""ppl meter through a vLLM OpenAI seat: prompt_logprobs over fixed text chunks. usage: ppl-vllm.py [port] [text_file] [ctx_tokens] [n_chunks]"""
import json, math, sys, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8899"
path = sys.argv[2] if len(sys.argv) > 2 else "/run/media/tonoken3/DATA1/Models/wikitext-2-raw/wiki.test.raw"
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 2048; nch = int(sys.argv[4]) if len(sys.argv) > 4 else 8
U = f"http://127.0.0.1:{port}/v1/completions"
text = open(path, encoding="utf-8", errors="ignore").read()
chars = ctx * 4  # rough char budget per chunk; server truncates nothing, we just pick disjoint windows
tot_nll = 0.0; tot_n = 0
for i in range(nch):
    chunk = text[i * chars:(i + 1) * chars]
    body = {"model": "DSV4-Flash", "prompt": chunk, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0, "echo": False}
    r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(body).encode(), headers={"content-type": "application/json"}), timeout=900))
    pl = r["choices"][0].get("prompt_logprobs") or []
    lps = []
    for tok in pl[1:ctx]:  # skip first token (no context); cap at ctx tokens
        if tok:
            lps.append(max(v["logprob"] if isinstance(v, dict) else v for v in tok.values()))
    n = len(lps); nll = -sum(lps)
    tot_nll += nll; tot_n += n
    print(f"chunk {i}: tokens={n} ppl={math.exp(nll / max(n, 1)):.4f}", flush=True)
print(f"PPL={math.exp(tot_nll / max(tot_n, 1)):.4f} tokens={tot_n} ctx={ctx} chunks={nch} file={path}")
