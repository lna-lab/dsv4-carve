# dsv4-carve — DeepSeek-V4-Flash-Vision (305B) EXL3 on 8 × 16 GB, 380K context, DSpark3

Lna-Lab's serving recipe for [vcruz305/DSV4-Flash-Vision-EXL3-MixedK](https://huggingface.co/vcruz305/DSV4-Flash-Vision-EXL3-MixedK)
carved down until a **380K-context, 4-stream, speculative (DSpark3) seat** fits on eight RTX PRO 2000 (16 GB, Blackwell sm_120)
under vLLM. Adopted as the house model of SAZANAMI on 2026-09-03 ("オオタニ").

Weights (private mirror, 86 GB): `sakamakismile/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`.

## What is in the pack

| Part | Format | Source |
|---|---|---|
| Routed experts (43 layers) | EXL3 2/3 bit mixed, layers 3/21/41 replaced by K2 | vcruz305 MixedK + wrldsuksgo2mars K2-v1 (`tools/layer_overlay.py`) |
| MTP draft experts (3 layers) | EXL3 K2 (was fp8) | wrldsuksgo2mars K2.2-D2-v1 (`tools/mtp_overlay.py`) |
| Attention + shared experts (790 dense linears) | **EXL3 6 bit** (was BF16), calibrated 250 × 2048 | baked here (`tools/dense_bake.py`) |
| KV cache | fp8 (≈17 KB/token incl. draft) | vLLM |

Per-GPU (TP8): weights 12.0 GiB, KV pool **396,656 tokens** at `--max-model-len 389120`.

## Measured (2026-09-03, TP8, CUDA graphs on)

| | tok/s |
|---|---|
| ppl (wikitext-2, 512 × 16) | **6.7159** (Cruz original 6.6271, +1.34 %) |
| single stream, no speculation | en 42.1 / ja 42.0 / code 41.9 |
| single stream, DSpark3 | en 62 / ja 55 / code 86 |
| 4 streams, DSpark3 (aggregate) | en 136 / ja 116 / code 186 |
| needle at 166K tokens | found; TTFT 167 s (prefill ≈ 1.0k tok/s), decode at depth 37 |

2 streams cost the same wall time as 4 (graph capture sizes + EXL3 small-batch bucketing): run 1 or 4.

## The three traps (all fixed in this repo)

1. **`dense_bake.py --merge` stamped `bits: 4` into `config.json` for a 6-bit bake** → shape mismatch at load. Merge now reads bits from `work/args.json`.
2. **exllamav3's cooperative-GEMM autotuner runs inside CUDA-graph capture** (`coop_autotune.cu:464`, "operation not permitted when stream is capturing"). Patch in `recipe-lna/exllamav3/exl3_gemm.cu.lna`: skip tuning while `cudaStreamIsCapturing`, fall back to the static heuristic; plus the plugin pre-tunes decode row counts (1,2,4,8,16) right after weight load (`LNA_EXL3_PREWARM_ROWS`).
3. **Decode deadlock, GPU 100 % on all ranks.** vLLM's FusedMoE overlaps *shared experts on a second stream* for ≤ 256 tokens. With shared experts also EXL3, two cooperative kernels share exllamav3's per-device lock buffer (`DevCtx::get_locks`) from two streams and spin forever. Diagnosed with `CUDA_LAUNCH_BLOCKING=1` (no hang → concurrency) and py-spy. Fix: **`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`** (default in `serve-dsv4-tp8.sh`).

## Serving

Image: `Dockerfile` (vllm/vllm-openai:nightly + vLLM 0.28.1rc1.dev337 wheel + exllamav3 1.4.5 built from source with the house patches + vllm-exl3 0.2.3 + `recipe-lna/patch_*.py`). The rebuilt extension and plugin are mounted over the image at run time (`EXT_SO`, `PLUGIN_SRC`) so the image never needs re-baking.

```bash
export EXT_SO=/path/to/exllamav3_ext.cpython-312-x86_64-linux-gnu.so   # built from recipe-lna/exllamav3/*.lna
export PLUGIN_SRC=/path/to/vllm_exl3                                   # vllm-exl3 0.2.3 + recipe-lna/vllm_exl3_exl3.py.lna
AUX_STREAMS=0 IMAGE=dsv4-dense \
NCCL_EXTRA="-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 -e VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1" \
MODEL=/path/to/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6 \
UTIL=0.97 MAXLEN=389120 BT=512 SEQS=4 SPEC='{"method":"dspark","num_speculative_tokens":3}' \
bash serve-dsv4-tp8.sh          # OpenAI API on 127.0.0.1:8899, model name DSV4-Flash
```

Notes: thinking is on by default and eats the budget (700 tokens → empty answer); send `chat_template_kwargs: {"thinking": false}` or a large `max_tokens`. Vision works. `--ulimit core=0` and cache mounts are set because a crashed seat once wrote 27 GB of core dumps into the container layer.

## Layout

- `serve-dsv4-tp8.sh` — the seat. `bench-dsv4.py` / `bench-streams.py` (usage-based token count; SSE chunk counting undercounts with speculation) / `ppl-vllm.py` / `prof-summary.py`.
- `tools/` — `mtp_overlay.py`, `layer_overlay.py`, `dense_bake.py` (+ `README-mtp-overlay.md`).
- `recipe-lna/` — vLLM/exllamav3/plugin patches applied in the image; `exllamav3/*.lna` are the patched source files.
- `orders/` — the work orders given to the craftspeople (codex "Luna", Explore agents). `docs/` — campaign canon (`PLAN.md`) and design notes.
- `exl3-tune-cache/` — autotuner disk cache for the 8 × RTX PRO 2000.

## Credits

vcruz305 (MixedK, vllm-exl3 patches), wrldsuksgo2mars (K2 packs), turboderp (exllamav3), DeepSeek (weights, MIT), vLLM.
Bake, patches and measurements: Lna-Lab / YUKI with Ken ([@Tono_Ken3](https://x.com/Tono_Ken3)). License for this repo: MIT; the weights follow the DeepSeek model license.
