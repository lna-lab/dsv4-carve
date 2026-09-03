# T6 — make vllm-exl3 + vLLM DSV4 serve EXL3 dense linears at TP=8 (per orders/T5-vllm-side-reading.md)

Read `orders/T5-vllm-side-reading.md` (it has file:line for everything) and `tools/README-dense-bake.md` (what the
bake writes: per-layer EXL3 tensors named with the pre-fusion HF names, incl. `attn.wo_a.slice.0..7`).
Plugin source: `vllm-exl3/src/vllm_exl3/exl3.py` (edit in place — this is our fork; keep a `LNA-LAB` comment on
every change). vLLM model files live only inside the docker image `lna-lab/vllm-exl3:dsv4`; you may not be able to
run docker — then write the vLLM edits as small idempotent exact-anchor patch scripts in `recipe-lna/`
(same style as `recipe/scripts/patch_dsv4_stock028.py`: anchor, patched text, expect count, `.orig-lna` backup),
using the file contents quoted in T5 (line numbers are for dev337). I will apply them when rebuilding the image.

## Changes
1. Plugin — effective TP: `Exl3LinearMethod.create_weights` must record the layer's effective tp_size/tp_rank
   (1/0 when `getattr(layer, "disable_tp", False)` or the layer is a `ReplicatedLinear`), and the weight loader must
   shard with those values instead of the global `get_tensor_model_parallel_world_size()` (exl3.py ~1144-1152,
   `_narrow_tp`, `shard_exl3_col/row`). Also use the effective tp for the `bf16_shards` TP>1 guard (~1018-1026).
2. Plugin — `wo_a` (`is_bmm=True` ColumnParallelLinear, out=8192=8 groups×1024, TP=8 → one group per rank): load
   `layers.N.attn.wo_a.slice.{rank}.{trellis,suh,svh,mcg}` as this rank's whole shard (do NOT narrow a fused tensor:
   each slice has its own suh/svh), build a per-rank `LinearEXL3`, and keep a `layer.weight` shim absent but expose
   what `_o_proj` needs. Since vLLM's `_o_proj` does an einsum on `wo_a.weight`, add in the model patch (item 4) an
   EXL3 branch: when `getattr(self.wo_a, "_exl3_linears", None)` exists and `n_local_groups == 1`, compute
   `z = exl3_linear(o.reshape(b, -1))` reshaped to `(b, 1, o_lora_rank)` then the inverse-RoPE step exactly as the
   bf16 branch does (read the Triton reference path `rocm_inv_rope_einsum` to keep the math identical), then `wo_b`.
   If `n_local_groups != 1` raise a clear NotImplementedError (TP<8 unsupported for EXL3 wo_a for now).
   Checkpoint naming: the loader must accept `wo_a.slice.{r}.*` → map to the `wo_a` param for rank r only.
3. Plugin — config: `non_routed_exl3.layers` entries for the modules in T5's table; treat `wo_a` entries
   (`model.layers.N.attn.wo_a`) as the slice-per-rank kind. Keep `bf16_as_stored` behaviour for anything not listed.
4. vLLM patches (recipe-lna/patch_dsv4_dense_exl3.py): (a) `compressor.py` pass the real quant_config to
   `fused_wkv_wgate` (both attention compressor and indexer compressor); (b) `flashinfer_sparse.py` `_o_proj`
   (both classes, incl. the vcruz305-patched text) — EXL3 branch described in item 2, and make the dtype probe
   tolerant (`getattr(self.wo_a, "weight", None)`); (c) optional `attention.py` `indexer.weights_proj` quant_config
   (skip; it stays bf16).
5. Tests: unit tests for the effective-TP sharding math (fake layer objects), for the wo_a slice loader naming, and
   for config parsing. No GPU/docker needed.
6. `tools/README-dense-serve.md`: exact `non_routed_exl3` config the bake's `--merge` should emit (update
   `tools/dense_bake.py` merge config writer accordingly, including `wo_a` entries and removing the "unresolved"
   status when the plugin supports it), how to rebuild the image (Dockerfile: add `COPY recipe-lna /opt/recipe-lna`
   + RUN the patch), and the env/flags for serving (`VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1`, SP off).
Commit on master if git works. Finish with a concise report: files changed, tests, what could not be verified.
