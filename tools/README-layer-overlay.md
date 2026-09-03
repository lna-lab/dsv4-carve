# Main-layer K2 EXL3 overlay

`layer_overlay.py` replaces the routed-expert EXL3 tensors of selected main
decoder layers with the same layers from the uniform-K2 donor
`wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1`.

The source pack is not modified. The output links every unchanged source file,
rewrites only the numbered source shards containing selected experts, and adds
`model-layers-k2.safetensors` with the donor tensors. Rewrites stream retained
tensor payloads one tensor at a time, and donor reads use only HTTP Range
requests for required headers and tensors. A completed-tensor partial output
can be resumed.

## Name map

```text
model.layers.L.mlp.experts.E.gate_proj.S -> layers.L.ffn.experts.E.w1.S
model.layers.L.mlp.experts.E.up_proj.S   -> layers.L.ffn.experts.E.w3.S
model.layers.L.mlp.experts.E.down_proj.S -> layers.L.ffn.experts.E.w2.S
```

`S` is `trellis`, `suh`, `svh`, or `mcg`. Donor scalar MCG markers (`shape []`)
are emitted as `[1]`, as required by the EXL3 loader. The source selected
layers must be K3 (`trellis` last dimension 48); the replacement ABI is checked
against an unselected local K2 main layer (`trellis` last dimension 32).

## Config and index

The output index retains every non-replaced source tensor and maps all
replacement names to `model-layers-k2.safetensors`. The copied config keeps the
base quantization settings and removes the selected layers from
`quantization_config.layer_bits`; base `bits` remains 2. No other config keys
are edited.

## Usage

Offline planning with the local donor index named in the order:

```bash
python3 tools/layer_overlay.py \
  --src /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --donor wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1 \
  --donor-index /run/media/tonoken3/DATA1/.tmp/k2v1-index.json \
  --out /path/to/layer-overlay \
  --layers 3,13,21,22,28,41 \
  --dry-run
```

A real build fetches the donor index/headers and selected tensor ranges:

```bash
python3 tools/layer_overlay.py \
  --src /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --donor wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1 \
  --donor-index /run/media/tonoken3/DATA1/.tmp/k2v1-index.json \
  --out /path/to/layer-overlay \
  --layers 3,13,21,22,28,41 \
  --verify
```

`--dry-run` never opens a network connection and never creates the output.
`--verify` is for a completed real build. Set `LAYER_OVERLAY_LOCAL_DIR` to a
directory containing pre-downloaded donor shard basenames to use local range
reads instead of HTTP. `--donor-headers` can point at a reusable header cache.

## Verification status

The offline checks for this checkout cover safetensors header parsing, all 48
local source shard headers, source/index agreement, the six selected K3 expert
sets, all 18,432 donor index expert names, all three projection mappings, the
K2 ABI plan, and the requested dry-run. No donor HTTP/local-shard fetch, real
overlay write, post-build `--verify`, vLLM load, or network behavior was tested.
