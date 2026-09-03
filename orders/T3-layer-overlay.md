# T3 — layer_overlay.py: replace the six 3-bit main-layer expert sets with uniform K2 experts from a donor

## Goal
Generalize the approach of `tools/mtp_overlay.py` (read it first; reuse its helpers by import or copy) into
`tools/layer_overlay.py` that swaps the routed-expert EXL3 tensors of a chosen set of MAIN decoder layers in the
source pack for the same layers' EXL3 tensors from a donor pack, producing a new overlay directory. Purpose: the
source pack has six layers at 3 bits (config `layer_bits` {"3","13","21","22","28","41"}: 3); replacing them
with K2 saves ≈0.4 GiB per GPU at TP=8.

## Facts (measured)
- Source pack (may itself be an overlay produced by mtp_overlay.py; symlinks must be followed and re-linked, not
  copied): `/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2`. Main-layer expert tensors are named
  `layers.{L}.ffn.experts.{E}.{w1,w2,w3}.{trellis,suh,svh,mcg}`; 3-bit trellis has last dim 48 (K3), K2 has 32.
  Source main tensors are spread over shards 1..45 (scan headers; there is an index.json in the overlay dir).
- Donor: `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1` (uniform K2, same base model). Index is
  downloaded at `/run/media/tonoken3/DATA1/.tmp/k2v1-index.json`. Donor names:
  `model.layers.{L}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.{trellis,suh,svh,mcg}` (map: gate→w1, up→w3,
  down→w2, strip the `model.` prefix, `mlp`→`ffn`). Donor shards holding the six layers: 3→shard 6; 13→2,3; 21→4;
  22→4,5; 28→6; 41→8,9 (10 shards of ~8.6 GB; do NOT download whole shards, range-read ≈1.6 GB per layer).
  Donor mcg markers are scalar `[]`; write them as shape `[1]` (as mtp_overlay does now).
- Output contract like mtp_overlay: symlink every source file that is unchanged; rewrite only the shards that
  contain the replaced tensors (dropping them, keeping the rest byte-identical); add one new shard
  `model-layers-k2.safetensors`; new index; config.json with `layer_bits` entries for the replaced layers removed
  (base bits is 2). `--layers 3,13,21,22,28,41` selects layers. `--dry-run`, `--verify` as before. Honour the
  `MTP_OVERLAY_LOCAL_DIR`-style local-shard shortcut (env `LAYER_OVERLAY_LOCAL_DIR`) for pre-downloaded donor shards.
- Keep peak RAM < 8 GB. Rewriting a source shard: stream tensor by tensor, never load a whole 8 GB shard.

## Constraints
No network in your sandbox is expected: implement, unit-test header parsing + name map + plan against the local
files, run `--dry-run`, and report what was verified and what was not. Commit on branch `t3-layer-overlay` if git
works, else leave files in place. Write `tools/README-layer-overlay.md`.
