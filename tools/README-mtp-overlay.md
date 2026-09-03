# MTP EXL3 overlay

`mtp_overlay.py` builds an overlay for the three DSpark draft layers. It leaves
the source directory untouched, links unchanged source files, removes only the
source-format MTP expert tensors from source shards 46–48, and adds one
`model-mtp-exl3.safetensors` shard containing the donor EXL3 experts.

The donor-to-pack name map is:

```text
mtp.L.mlp.experts.E.gate_proj.S  -> mtp.L.ffn.experts.E.w1.S
mtp.L.mlp.experts.E.up_proj.S    -> mtp.L.ffn.experts.E.w3.S
mtp.L.mlp.experts.E.down_proj.S  -> mtp.L.ffn.experts.E.w2.S
```

`S` is one of `trellis`, `suh`, `svh`, or `mcg`. This is the ABI consumed by
`vllm-exl3/src/vllm_exl3/exl3.py`: `Exl3MoEMethod` loads gate/up as the two
`w13_*` slots and down as `w2_*`, with the EXL3 suffix on each packed tensor.
The checked-in loader confirms `I16` trellis, `F16` `suh`/`svh`, and the MCG
`I32` marker path. The vLLM container mapper (`_remap_dspark_name`) could not
be inspected here because Docker access was denied; the overlay therefore uses
the order's stated `mtp.*` mapper assumption.

The copied `config.json` keeps `non_routed_quantization` and
`non_routed_dtype_policy`, removes `mtp_experts` and
`mtp_experts_start_layer`, and explicitly adds `layer_bits` entries
`"43": 2`, `"44": 2`, and `"45": 2`. Existing layer-bit entries are retained.

## Usage

Offline planning, using the local index from the order:

```bash
python3 tools/mtp_overlay.py \
  --src /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK \
  --donor wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1 \
  --out /path/to/mtp-overlay \
  --dry-run
```

The real build reads the donor index, fetches only the required donor shard
headers, then uses HTTP `Range` requests for the selected tensor payloads:

```bash
python3 tools/mtp_overlay.py \
  --src /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK \
  --donor wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1 \
  --out /path/to/mtp-overlay --verify
```

`--donor-index` changes the local/remote weight-map path and `--donor-headers`
selects a reusable remote-header cache. A failed build leaves a partial new
shard that can be resumed at a completed tensor boundary. `--dry-run` performs
no network access and creates no output files.

## Verification status for this checkout

Tested offline: safetensors header parsing against all 48 local source shard
headers; the local source draft expert count, dtypes, and measured shapes;
the donor index's 9,216 expert names and all three projection mappings; and the
requested dry run. Not tested: donor HTTP redirects/range responses, the actual
~2.4 GB overlay write, post-build `--verify`, loading the result in vLLM, or
the container-side `_remap_dspark_name` implementation because network and
Docker are unavailable in this sandbox.
