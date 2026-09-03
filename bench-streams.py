#!/usr/bin/env python3
"""Multi-stream decode meter for the DSV4 seat: per-stream and aggregate tok/s (streaming, wall-clock)."""
import json, sys, time, threading, urllib.request
port = sys.argv[1] if len(sys.argv) > 1 else "8899"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 256
streams_list = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,4").split(",")]
P = {"en": "Write a detailed essay about the history of the printing press and its effects on European society.",
     "ja": "日本の四季それぞれの風物詩と、それが文学に与えた影響について詳しく説明してください。",
     "code": "Write a complete Python implementation of an LRU cache with tests and explanations."}
def one(prompt, out):
    body = {"model": "DSV4-Flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": N, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time(); first = None; n = 0
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        j = json.loads(line[5:])
        if j.get("usage") and j["usage"].get("completion_tokens"): n = j["usage"]["completion_tokens"]; last = time.time(); continue
        ch = j.get("choices") or []
        d = ch[0].get("delta", {}) if ch else {}
        if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
            if first is None: first = time.time()
            last = time.time()
    out.append((first - t0, n, last - first))
for lang, prompt in P.items():
    for s in streams_list:
        res = []; th = [threading.Thread(target=one, args=(prompt, res)) for _ in range(s)]
        t = time.time(); [x.start() for x in th]; [x.join() for x in th]; wall = time.time() - t
        tot = sum(n for _, n, _ in res); per = [n / dt for _, n, dt in res]
        print(f"{lang}: streams={s} tokens={tot} wall={wall:.2f}s TTFT={max(f for f,_,_ in res):.2f}s agg={tot/wall:.1f} tok/s per_stream={min(per):.1f}-{max(per):.1f}", flush=True)
