# DSV4 dense EXL3 serving at TP=8

This is the serving contract for the directory produced by
`tools/dense_bake.py --merge`. It assumes the DSV4 vLLM model files are in the
`lna-lab/vllm-exl3:dsv4` image and that the dense plugin changes in
`vllm-exl3/src/vllm_exl3/exl3.py` are installed.

## Merged configuration

The merge preserves the routed-expert configuration and replaces
`quantization_config.non_routed_exl3` with this shape. `N` means every language
layer index present in the bake. The example uses the default `--bits 4`; when
`--attn-bits` or `--shared-bits` is supplied, the writer changes the
corresponding integer values.

```json
{
  "quantization_config": {
    "quant_method": "exl3",
    "codebook": "mcg",
    "non_routed_dtype_policy": "bf16_as_stored",
    "non_routed_exl3": {
      "codebook": "mcg",
      "layers": {
        "model.layers.N.attn.fused_wqa_wkv": {"bits": 4},
        "model.layers.N.attn.wq_b": {"bits": 4},
        "model.layers.N.attn.wo_a": {"bits": 4},
        "model.layers.N.attn.wo_b": {"bits": 4},
        "model.layers.N.attn.compressor.fused_wkv_wgate": {"bits": 4},
        "model.layers.N.attn.indexer.compressor.fused_wkv_wgate": {"bits": 4},
        "model.layers.N.attn.indexer.wq_b": {"bits": 4},
        "model.layers.N.mlp.shared_experts.gate_up_proj": {"bits": 4},
        "model.layers.N.mlp.shared_experts.down_proj": {"bits": 4}
      }
    }
  }
}
```

The actual output keeps any existing `bits`, `layer_bits`, MTP, and source
quantization fields from the routed pack. The two source tensors represented by
each fused entry must share one K. `wo_a.slice.0` through `.slice.7` are eight
complete `[4096, 1024]` group linears, all represented by the one
`model.layers.N.attn.wo_a` entry. The model patch maps slice `r` to TP rank `r`;
the plugin does not narrow those already-local tensors a second time.

`attn.indexer.weights_proj`, norms, gates, heads, and other modules absent from
the map remain native BF16 through `bf16_as_stored`. Do not add
`weights_proj` to this EXL3 map.

## Bake and merge

Bake as described in `tools/README-dense-bake.md`, then merge only after all 43
language-layer files are complete:

```bash
python3 tools/dense_bake.py \
  --src /run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp \
  --experts /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2 \
  --work /run/media/tonoken3/DATA1/.tmp/t4-dense-work \
  --merge /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-Dense4
```

## Rebuild the serving image

`Dockerfile` copies `recipe-lna` into the image and runs the new patch after
the existing DSV4 patches. For an explicit rebuild:

```bash
docker build -f Dockerfile -t lna-lab/vllm-exl3:dsv4 .
```

The patch script is safe to run again. It edits the common
`deepseek_v4/compressor.py`, both `_o_proj` methods in
`deepseek_v4/nvidia/flashinfer_sparse.py`, and the DSV4 model's eight
`wo_a.slice.N` loader mappings. Each changed runtime file receives a
`*.orig-lna` backup.

## TP=8 serving

The dense `wo_a` path currently requires exactly one local group, therefore
serve with TP=8. Keep sequence parallelism and expert parallelism off: do not
pass an SP/EP enable flag. The shared-expert fusion path must also be disabled.

```bash
export VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -v /run/media/tonoken3/DATA1:/run/media/tonoken3/DATA1 \
  -p 127.0.0.1:8899:8000 lna-lab/vllm-exl3:dsv4 \
  /run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-Dense4 \
  --served-model-name DSV4-Flash \
  --tensor-parallel-size 8 \
  --quantization exl3 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --trust-remote-code
```

Start with `--enforce-eager` if CUDA-graph capture is not known to work for
the image build. Speculative decoding is independent of the dense patch and
can be added later with the image's supported DSpark configuration.

## Verification boundary

The plugin and merge tests do not require a GPU or Docker. A real calibration,
EXL3 kernel execution, eight-rank model load, and quality/speed comparison
require the DSV4 source image, eight usable CUDA ranks, and the compiled
`exllamav3_ext`; those checks must be performed on the serving host.
