# T4 — dense_bake.py: calibrated EXL3 (4 bpw) bake of ONLY the dense (non-expert) linears of DeepSeek-V4-Flash-Vision-Exp

Read `orders/T4-design.md` first (it cites exllamav3 v1.4.5 source lines; the source tree is at
`/run/media/tonoken3/DATA1/.tmp/exl3src`, read-only, do not modify it — vendor what you need by import path
`sys.path.insert(0, "/run/media/tonoken3/DATA1/.tmp/exl3src")` or by copying functions into the tool).

## Goal
`tools/dense_bake.py` that:
1. Loads the bf16/fp8 source checkpoint (`--src /run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp`, arriving;
   57 files; read its config.json to learn the layout) with the routed experts routed to an existing EXL3 pack
   (`--experts /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2`) via `VariantSafetensorsCollection`
   (`add_stc(["layers.*.ffn.experts.*"], SafetensorsCollection(experts_dir))`).
2. Runs exllamav3's calibrated conversion loop (capture H → quantize → reload quantized → advance calibration state)
   but quantizes ONLY dense linears: per layer `attn.wq_a, attn.wq_b, attn.wkv, attn.wo_a.slice.{0..7}, attn.wo_b,
   attn.compressor.{wkv,wgate}, attn.indexer.compressor.{wkv,wgate}, attn.indexer.wq_b, ffn.shared_experts.{w1,w2,w3}`
   (skip `indexer.weights_proj`, `ffn.gate*`, norms, hc_*, embed, head; skip mtp.* entirely). Bits: `--bits 4`
   default, `--attn-bits/--shared-bits` overrides; all 8 `wo_a` slices must share one K.
3. Writes per-layer `work/qtensors/layers.N.safetensors` holding only the dense EXL3 tensors
   (`.trellis/.suh/.svh/.mcg` — use codebook **mcg** to match the expert pack; check how the converter selects
   mcg vs mul1 and use the same switch), with `--resume` at module granularity, checkpointing after every layer.
4. `--merge <out_dir>`: builds an overlay pack directory on top of `--experts` pack: symlink everything, add
   `model-dense-exl3.safetensors` (all dense EXL3 tensors), rewrite the shards that carried the BF16 versions of the
   replaced dense tensors dropping them (reuse `tools/layer_overlay.py` helpers: header parsing, streaming rewrite,
   index/config writing, and its verify that also checks dropped names are absent), and write `config.json` with a
   `non_routed_exl3` block for vllm-exl3 (read `vllm-exl3/src/vllm_exl3/exl3.py` `_matches_non_routed_exl3`,
   `_bits_for_non_routed`, `Exl3LinearMethod` to learn the exact schema and which module prefixes vLLM builds — the
   vLLM model file is `vllm/models/deepseek_v4/nvidia/model.py`; if docker is reachable read it via
   `docker run --rm --entrypoint cat lna-lab/vllm-exl3:dsv4 /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py`,
   otherwise state the assumption). NOTE the vLLM side fuses `wq_a+wkv` into `attn.fused_wqa_wkv` and
   `compressor.wkv+wgate` into `compressor.fused_wkv_wgate`; document how your per-tensor EXL3 outputs map onto those
   fused modules (one entry per fused module, all shards same bits). Leave `wo_a` handling documented as an open
   question if vLLM's `_o_proj` path cannot consume EXL3 (it may need a plugin change) — do not guess silently.
5. Calibration: use the converter's bundled corpus mix (default `-cr 250 -cc 2048` is fine; expose `--cal-rows/--cal-cols`).
   Devices: `--devices 0,1` (16 GB each). Keep host RAM use reasonable (swap_cpu as the converter does).

## Deliverables
- `tools/dense_bake.py`, `tools/README-dense-bake.md` (what it does, exact key lists, config schema written, how to run,
  what was NOT tested), unit tests for the key filter and the merge planning against the local packs.
- Test what you can offline: import the exllamav3 modules, construct `Config.from_directory` on the D2 pack
  (experts-only pack — it has config.json), dry-run the key filter and module list, dry-run the merge planning.
  You cannot run the real bake (no GPU in your sandbox, the source checkpoint may still be copying) — say so.
- Commit on `master` if git works; else leave files in place.
