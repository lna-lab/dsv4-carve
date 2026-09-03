# DSV4 dense partial bake

`dense_bake.py` makes a resumable, calibrated EXL3 work directory for only the
non-routed linears in DeepSeek-V4-Flash-Vision. The routed experts continue to
come from the existing `DSV4-Flash-Vision-EXL3-MixedK-D2` pack through
`VariantSafetensorsCollection`. The source checkout at
`/run/media/tonoken3/DATA1/.tmp/exl3src` is imported read-only; set
`EXL3_SOURCE` to another checkout when needed.

## Targets

For each `layers.N`, the filter admits exactly:

```text
attn.wq_a
attn.wq_b
attn.wkv
attn.wo_a.slice.0 ... attn.wo_a.slice.7
attn.wo_b
attn.compressor.wkv
attn.compressor.wgate
attn.indexer.compressor.wkv
attn.indexer.compressor.wgate
attn.indexer.wq_b
ffn.shared_experts.w1
ffn.shared_experts.w2
ffn.shared_experts.w3
```

`wo_a` is one checkpoint `wo_a.weight` exposed as eight sliced linears. The
filter rejects a partial slice set. It rejects `ffn.experts.*`, all `mtp.*`,
`attn.indexer.weights_proj`, `ffn.gate*`, norms, embeddings, heads, and `hc_*`.
The default is `--bits 4`; `--attn-bits` and `--shared-bits` override the two
dense groups. The eight `wo_a` slices receive one common K (the attention
override, if present). The output codebook is always `mcg`, matching D2, and
each converted linear emits `.trellis`, `.suh`, `.svh`, and `.mcg`.

The calibration path is the converter's capture-H -> quantize -> reload ->
advance-state path. It sets `calibration_all_experts` on the DSV4 model so all
routed experts contribute Hessian data during capture. Prefix modules are
forwarded but never written; the bake stops after the final language block, so
MTP and output-side modules are untouched. `--cal-rows 250 --cal-cols 2048`
and `--devices 0,1` are the defaults. `swap_cpu()` is used before conversion,
matching the converter's host-RAM strategy.

## Run

First do the offline layout check:

```bash
python3 tools/dense_bake.py \
  --src /run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp \
  --experts /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --work /run/media/tonoken3/DATA1/.tmp/t4-dense-work \
  --dry-run
```

The dry-run constructs the D2 Python model graph. On a CPU-only host it uses a
local native-extension stub for construction only; it does not run a model
forward or quantizer and does not write the work directory. A real bake is:

```bash
python3 tools/dense_bake.py \
  --src /run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp \
  --experts /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --work /run/media/tonoken3/DATA1/.tmp/t4-dense-work \
  --bits 4 --devices 0,1
```

The first run saves `args.json`, `qtensors/layers.N.safetensors`, and a
checkpoint after every completed layer under `ckpt/`. A resumed run repeats no
completed module:

Repeat the real-bake command with `--resume` appended.

Resume settings (source, expert pack, K values, and calibration dimensions) are
checked against `args.json`. A layer file is accepted for merging only when
every dense linear in it has the four expected EXL3 tensors.

## Merge

Merge rewrites only the D2 shards that carry the replaced dense `.weight`
tensors. It streams retained payloads byte-for-byte, symlinks every other pack
file, writes one `model-dense-exl3.safetensors`, rebuilds the index, and checks
that dropped names are absent and retained payloads are identical.

An ordinary merge requires all dense layer files. A dry-run may inspect a
partial work directory (and reports the missing target names) so an interrupted
bake can be diagnosed without creating an output pack.

```bash
python3 tools/dense_bake.py \
  --src /run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp \
  --experts /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --work /run/media/tonoken3/DATA1/.tmp/t4-dense-work \
  --merge /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-Dense4
```

The generated `quantization_config.non_routed_exl3` follows the plugin schema:
`codebook: "mcg"` and a `layers` map of vLLM post-mapper module prefixes to
`{"bits": K}`. Defaults assume the DSV4 loader uses these prefixes:

```text
model.layers.N.attn.fused_wqa_wkv
model.layers.N.attn.wq_b
model.layers.N.attn.wo_a
model.layers.N.attn.wo_b
model.layers.N.attn.compressor.fused_wkv_wgate
model.layers.N.attn.indexer.compressor.fused_wkv_wgate
model.layers.N.attn.indexer.wq_b
model.layers.N.mlp.shared_experts.gate_up_proj
model.layers.N.mlp.shared_experts.down_proj
```

The two source tensors of each fused module are represented by one config
entry and must have the same K. `wo_a.slice.0..7` is represented by one
`model.layers.N.attn.wo_a` entry; the serving patch maps each slice number to
its TP rank. Change the two path assumptions with `--vllm-root` and
`--vllm-shared-prefix`. The Docker image containing the DSV4 vLLM model was
not readable in this environment (Docker socket access was denied), and no
local `vllm/models/deepseek_v4/nvidia/model.py` was present; the prefixes above
therefore remain an explicit assumption to validate in the serving image.

The serving recipe supplies the remaining runtime changes: effective TP for
replicated/`disable_tp` linears, rank-local EXL3 `wo_a` slices, the compressor's
real quantization config, and the DSV4 `_o_proj` EXL3 branch. `weights_proj`
remains native BF16 by design because it is not in the target list and the
pack uses `non_routed_dtype_policy: "bf16_as_stored"`.

## Source functions reused

The tool directly reuses these converter functions from the read-only source;
the file:line references are pinned to the audited checkout:

| Function | Source |
| --- | --- |
| `load_tensor` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:130-141` |
| `save_tensor` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:144-157` |
| `quantize_linears_single` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:496-552` |
| `quantize_linears_parallel` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:555-670` |
| `load_parallel_calib_modules` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:701-729` |
| `capture_module_parallel` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:767-863` |
| `advance_state_parallel` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:866-935` |
| `get_default_calibration` | `/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/calibration_data.py:61-99` |

The quantizer's `make_quant_args` codebook switch is visible at
`/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/conversion/convert_model.py:393-406`;
`mcg` is selected here rather than `mul1`. The resulting tensor suffixes are
defined by `LinearEXL3.get_tensors` at
`/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/modules/quant/exl3.py:98-111`,
and its loader selection is `Linear.load` at
`/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/modules/linear.py:437-444`.
The DSV4 fused `wo_a` FP8-block dequantize-then-slice path is
`/run/media/tonoken3/DATA1/.tmp/exl3src/exllamav3/modules/linear.py:342-380`.
`VariantSafetensorsCollection.add_stc` is at
`loader/safetensors.py:884-886`. Since the audited variant collection leaves
`set_new_tensors` unimplemented at `loader/safetensors.py:1028-1029`, the tool
sets reload tensors on its `main` collection: dense keys resolve there while
the routed-expert override remains in the expert collection.

## What was not tested

At implementation time the source checkpoint was arriving; it is now header-
complete, but this host has no usable CUDA device/native exllamav3 extension,
so no real calibrated bake, EXL3 kernel execution, quality measurement, or
vLLM serving load was run. Tests cover the
key filter, exact slice rule, bits planning, D2 model-layout construction with
the offline extension stub, and a small safetensors merge fixture. The merge
implementation uses `tools/layer_overlay.py`'s header, streaming-copy,
metadata/index, payload-compare, and safe-output helpers.
