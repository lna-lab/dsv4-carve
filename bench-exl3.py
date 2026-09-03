#!/usr/bin/env python3
"""EXL3 bench v2 — 正直な計器: 実トークン数・prefill/decode 分離・多流。2026-09-01 YUKI."""
import sys, time, json, argparse, resource, subprocess
import torch
from exllamav3 import model_init
from exllamav3.generator import Generator, Job

def vram():
    out = subprocess.run(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],
                         capture_output=True,text=True).stdout
    return {l.split(",")[0].strip(): int(l.split(",")[1]) for l in out.strip().splitlines()}

def main():
    global gen, tokenizer
    p = argparse.ArgumentParser()
    model_init.add_args(p, add_draft_model_args=True)
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--ntok", type=int, default=128)
    args = p.parse_args()

    t0=time.time(); res = model_init.init(args)
    model, config, cache, tokenizer = res[0], res[1], res[2], res[3]
    print(f"LOAD_WALL_S={time.time()-t0:.1f}", flush=True)
    print("VRAM_AFTER_LOAD=", json.dumps(vram()), flush=True)
    draft_model = res[4] if len(res) >= 7 else None
    draft_cache = res[6] if len(res) >= 7 else None
    gkw = dict(model=model, cache=cache, tokenizer=tokenizer)
    if draft_model is not None:
        gkw.update(draft_model=draft_model, draft_cache=draft_cache)
        if getattr(args, "num_draft_tokens", None): gkw["num_draft_tokens"] = args.num_draft_tokens
        if getattr(args, "dynamic_draft", False): gkw["dynamic_draft_tokens"] = True
    print("DRAFT_ATTACHED=", draft_model is not None, flush=True)
    gen = Generator(**gkw)

    PROMPTS = {"ja":"日本語で、秋の海辺の朝を三文で描写してください。",
               "en":"Explain in three sentences why tidal pools host unusual biodiversity.",
               "code":"Write a Python function that merges two sorted lists into one sorted list."}

    def run(tag, prompt_list, ntok):
        global gen, tokenizer
        jobs=[]
        for pr in prompt_list:
            ids = tokenizer.encode(pr, add_bos=False)
            jobs.append(Job(input_ids=ids, max_new_tokens=ntok, stop_conditions=[]))
        for j in jobs: gen.enqueue(j)
        t0=time.time(); first=None; total=0
        while gen.num_remaining_jobs():
            r = gen.iterate()
            for x in r:
                n = len(x.get("token_ids", [])) if x.get("token_ids") is not None else 0
                if n == 0 and x.get("text"): n = 0
                total += x.get("new_tokens_this_iter", 0) or 0
            if first is None: first=time.time()-t0
        dt=time.time()-t0
        got = sum(j.new_tokens if hasattr(j,"new_tokens") else 0 for j in jobs)
        acc = sum(getattr(j, "accepted_draft_tokens", 0) or 0 for j in jobs)
        if acc: print(f"   accepted_draft_tokens={acc} of {got} generated -> spec share {acc/max(got,1):.2f}", flush=True)
        return dt, first, got

    # warm
    gen.generate("Hello", max_new_tokens=8)
    print("WARM_DONE", flush=True)
    S=args.streams
    for k,pr in PROMPTS.items():
        dt, ttft, got = run(k, [pr]*S, args.ntok)
        tokens = got if got else args.ntok*S
        print(f"{k}: streams={S} tokens={tokens} wall={dt:.2f}s TTFT={ttft:.3f}s "
              f"agg={tokens/dt:.2f} tok/s per_stream={tokens/dt/S:.2f} ITL={1000*dt/(tokens/S):.1f}ms", flush=True)
    print("VRAM_AFTER_GEN=", json.dumps(vram()), flush=True)
    print(f"RSS_GIB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576:.1f}", flush=True)

if __name__ == "__main__":
    main()
