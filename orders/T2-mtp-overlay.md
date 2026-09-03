# T2 — mtp_overlay.py: replace the fp8 DSpark draft experts with EXL3 K2 experts range-read from the Hub

## Goal
Write `tools/mtp_overlay.py` (Python 3.12, stdlib + `safetensors` + `torch` + `requests`/`urllib`), that builds an
overlay pack directory in which the three DSpark draft layers (`mtp.0..2`) carry EXL3 2-bit routed experts instead of
the block-FP8 experts that the source pack ships. Everything else is symlinked, never rewritten.

## Facts (measured, do not re-derive)
- Source pack: `/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK` (48 shards, no model.safetensors.index.json;
  scan the safetensors headers). Its draft expert tensors live only in shards 46,47,48 and are named
  `mtp.{L}.ffn.experts.{E}.{w1,w2,w3}.weight` (dtype I8, fp8 e4m3 bytes) + `.scale` (F8_E8M0). L in 0..2, E in 0..255.
  Shapes: w1/w3 weight [2048,2048]?? — verify from headers, print them, and DO NOT assume; w2 weight [4096,1024].
  All other `mtp.*` tensors (attn, norms, hc_*, gate weight/bias/bias_vl, attn_sink) stay as they are.
- Donor pack on the Hub: `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1` (same base model
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp; drafts uniform K2). Its index is already downloaded at
  `/run/media/tonoken3/DATA1/.tmp/d2-index.json` (weight_map). Draft expert tensors are named
  `mtp.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.{trellis,suh,svh,mcg}` and live in
  `model-00002-of-00011.safetensors` and `model-00011-of-00011.safetensors` (8.6 GB each).
  Name mapping to our pack: gate_proj→w1, up_proj→w3, down_proj→w2; `mlp`→`ffn`.
- Range reads: HTTP `Range:` requests against
  `https://huggingface.co/<repo>/resolve/main/<shard>` work (follow redirects to the CDN). Read the 8-byte header
  length + JSON header once per shard, then fetch only the byte ranges of the tensors we need. Total to fetch ≈ 2.4 GB.
  Retry on failure, resume by skipping tensors already present in the partially written output file (write the
  output shard at the end from an in-memory dict is fine: ~2.4 GB).
- The consumer is vLLM + the vllm-exl3 plugin 0.2.3 (source at `vllm-exl3/src/vllm_exl3/exl3.py`, read it). With
  `quantization_config.mtp_experts` absent or "exl3", the plugin builds `Exl3MoEMethod` for draft layers too and loads
  `.trellis/.suh/.svh/.mcg` per expert exactly like the 43 main layers (the loader in `exl3.py` around lines 800-900
  and 1100-1200 shows the expected per-expert tensor names after vLLM's mapper: check what names the mapper produces for
  `mtp.` tensors — `_remap_dspark_name` in the image's `vllm/models/deepseek_v4/nvidia/dspark.py`; the image is
  `lna-lab/vllm-exl3:dsv4`, you can read the file with
  `docker run --rm --entrypoint cat lna-lab/vllm-exl3:dsv4 /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py`
  if docker is reachable; otherwise state the assumption clearly).
- Per-layer bits: config `layer_bits` keys are main-layer indices as strings; draft layers are indices 43,44,45 in
  vLLM's numbering (`mtp_experts_start_layer: 43`). Base `bits` is 2, so K2 drafts need no override, but write
  `"43": 2, "44": 2, "45": 2` explicitly.

## Output contract
`python tools/mtp_overlay.py --src <pack> --donor wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1 --out <overlay-dir> [--dry-run] [--verify]`
- `<overlay-dir>`: symlinks to every source file except shards 46-48 and config.json; three rewritten shards
  46-48 that drop `mtp.*.ffn.experts.*.{weight,scale}` and keep everything else byte-identical; one new shard
  `model-mtp-exl3.safetensors` holding the donor EXL3 expert tensors renamed into our naming; a fresh
  `model.safetensors.index.json`; a `config.json` copied from source with `quantization_config` edited:
  remove `mtp_experts` and `mtp_experts_start_layer`, keep `non_routed_quantization` and `non_routed_dtype_policy`,
  add the 43/44/45 entries to `layer_bits`.
- `--dry-run`: print the plan (tensor counts, bytes to fetch, bytes to rewrite) and exit without network or writes.
- `--verify`: after building, open every tensor of the overlay through the index and check shapes/dtypes; for the
  donor tensors compare shapes with what a main-layer K2 expert has in the source pack (e.g. `layers.5.ffn.experts.0.w1.trellis`).
- Log every step; never silently skip a tensor; exit non-zero on any mismatch.

## Constraints
- Do not touch the source pack in place. Do not download whole shards. Keep peak RAM < 6 GB.
- No network inside your sandbox may be available: implement, unit-test the header parsing and name mapping against
  the local pack and the local d2-index.json, run `--dry-run`, and leave the real fetch for the operator.
- Commit to the repo at `/run/media/tonoken3/DATA1/vllm-exl3-lab` (branch `t2-mtp-overlay`) if git works; otherwise
  leave the files in place and print the paths.
- Write a short `tools/README-mtp-overlay.md`: what it does, the name map, the config edit, how to run, what was NOT tested.
