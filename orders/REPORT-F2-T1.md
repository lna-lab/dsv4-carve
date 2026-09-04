# REPORT F2 T1 — dense launch fusion v2

Date: 2026-09-04  
Scope: `vllm-exl3/src/vllm_exl3/exl3.py`, `f2/`, and the requested
`recipe-lna/patch_dsv4_attention_compressor_exl3.py` marker fix. The
`vllm-exl3-v030-port/` workspace was not touched.

## Result

T1 is implemented behind `LNA_EXL3_DENSE_GROUP=2`. `=1` remains the T0
same-call fusion path and an unset/invalid flag is fail-closed to OFF.

The final relay run is `f2/RUN.log` run 7, with `f2/RELAY.status` reporting
`relay up`; it ended `rc=0`. No seat endpoint, GPU 0–9, or `pkill -f` was
used.

## Changes

- Added load-time registration of layer h-fans (`q_a/wkv`, compressor, and
  indexer-compressor; up to six matrices) and q_a-fans (`wq_b` plus
  `indexer.wq_b`; two matrices).
- Added per-group pointer tables, permanent `A_had`/output scratch for every
  row bucket 1–16, and load-time prewarm. The final run prewarmed 5 T1 fans.
- Added owner-trigger generation caching keyed by source identity, pointer,
  shape, dtype, device, row count, and stream. The first member computes all
  outputs; later members return allocation-free slices.
- Mixed-width h-fans use load-time zero-scaled padded carriers so the widest
  kernel traversal cannot write outside a narrower member. Narrow outputs are
  cropped as views; no padding or pointer-table allocation occurs in forward.
- Added loud fallback warnings, per-reason counters, one-shot T0/T1 ACTIVE
  logs, stream checks, and m>16 fallthrough. `Exl3LinearMethod.apply` also
  routes single-shard q_a members into the T1 fan.
- Versioned the attention helper idempotency marker to
  `_lna_kv_score_v2`; the recipe replaces an already-installed old
  `_lna_kv_score` helper and keeps an `.orig-lna2` backup.
- Added `f2/parity_t1.py` and updated `f2/RUN.sh` for relay-only execution.

## Parity and hazard gates (MEASURED)

The final relay used production tensors from layers 5 and 40 plus synthetic
fans. Each row below covers `m={1,2,4,8,16}` and both owner-order
permutations; references are solo `LinearEXL3.forward` calls at fp32.

| fan | members | k | worst per-row rel | worst max abs | finite |
|---|---:|---:|---:|---:|---|
| L5 h-fan | 4 | 4096 | 1.663e-6 | 2.289e-5 | PASS |
| L40 h-fan | 6 | 4096 | 1.694e-6 | 2.098e-5 | PASS |
| L40 q_a-fan | 2 | 1024 | 1.661e-6 | 4.578e-5 | PASS |
| synthetic h-fan | 4 | 1024 | 5.386e-7 | 5.859e-3 | PASS |
| synthetic q_a-fan | 2 | 1024 | 1.374e-6 | 3.967e-4 | PASS |

Additional measured checks:

- `m=17` and `m=145` fall through to solo bitwise for every fan.
- `apply` path: bf16-boundary rel `7.310e-4`, max abs `3.906e-3`.
- Stream mismatch returns solo bitwise and emits a `stream:` fallback counter.
- 1008 eager generations with NaN-poisoned scratch and alternating inputs /
  owner order pass without stale output.
- CUDA graph capture plus 120 alternating-input replays pass with NaN
  poisoning; allocator delta during replay is `0 B`.
- Default/unset mode is solo bitwise. Mode 1 triggers the T0 local group with
  rel `6.547e-7`, max abs `2.623e-6`, and increments the T0 ACTIVE counter.
- Final observability counters: `t1_active=1061`, `t0_active=1`,
  `fallback_total=11`. The ten row-bucket fallbacks and one stream fallback
  are visible in the log with reasons; ACTIVE logs name all five T1 fans.
- `compute-sanitizer` is not present in the relay image. The permitted
  substitute—solo-vs-fused parity, poisoned replay, stream fallback, and
  graph replay—was run and passed.

## Launch-count estimate

The inherited T0 seat result is **MEASURED**: dense launches `489 -> 341`
per step. T1's requested fan arithmetic is **ESTIMATE** until a seat
profiler run:

```
341 - (20 odd-layer h-fan savings)
    - (42 even-layer h-fan savings)
    - (21 q_a-fan savings)
  = 258 dense launches/step
```

Thus the estimate is `258/step`, inside the `<=265` target and at the stretch
target. The relay parity fixture proves fan activation and correctness, but it
does not claim a seat end-to-end launch measurement.

## Remaining seat gates

Still required on the seat, using the same build and pack, are the T1
OFF/ON/OFF A/B runs (at least three each), profiler proof of `<=265` launches
and no added cat/copy/elementwise launches, per-fan bytes/time versus the
`1.25*T_floor` checks, dense-segment and full-step timing, eager graph-OFF
throughput/gap measurements, all-8-rank P50/P95 skew, NCCL accounting, and
the final numerical/essay/speculative-decode checks. Run with
`LNA_DSV4_AUX_STREAMS=0` and `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`.

## F2 follow-up — active/fallback observability

The attention recipe now warns with the ImportError/AttributeError reason and
honors `LNA_EXL3_DENSE_STRICT=1`; `exl3.py` emits one
`F2 DENSE_GROUP=<n> ACTIVE` marker per active mode/rank, warns on every dense fallback, and exposes
`dense_group_call_counts()` for gates. The OFF path remains the unchanged solo
path.

`f2/gate_active_dense.py` was added to `f2/RUN.sh`. After confirming
`f2/RELAY.status` was `relay up`, relay run 10 (`f2/RUN.request` →
`f2/RUN.done`) passed the strict-missing-import check and observed counter
increments plus ACTIVE markers for both `DENSE_GROUP=1` and `=2`; the complete
T0/T1 parity run also finished `rc=0`.

## VRAM

The seat symptom is confirmed by `/run/media/tonoken3/DATA1/.tmp/f3c-g2.log`
and `f3c-g2b.log`: lna2+T1 reports 0.20/0.23 GiB available KV and lna2+T0
reports 0.52/0.55 GiB. The routed lna2-only reference is the measured 1.86
GiB / 395,069-token result from the seat notes.

### Relay measurement

Measured through `f2/RUN.sh` → `f2/RUN.request` → `f2/RUN.done` with
`f2/RELAY.status` equal to `relay up`, run 16, on the relay's GPU 11
(`NVIDIA RTX PRO 2000 Blackwell`). The harness loaded production tensors from
`/model/model-dense-exl3.safetensors`, used layer-5 and layer-40 shapes, and
scaled the stand-in to 43 layers. Each delta is synchronized
`torch.cuda.memory_allocated`; the first-call probe used four decode rows.

“Before” is a relay reconstruction of the removed allocator, because that
version is no longer installed: each narrow member received a max-N carrier
and each group retained private `m=1..16` A_had/output scratch. “After” is the
real current registration and runner. The old retained storage was allocated
at registration, so there was no separate old lazy-bucket allocation to
measure; current first-call deltas are measured below.

| phase | DENSE_GROUP=1 before | DENSE_GROUP=1 after | DENSE_GROUP=2 before | DENSE_GROUP=2 after |
|---|---:|---:|---:|---:|
| registration total | 540.365 MiB | 6.220 MiB | 1,554.387 MiB | 6.702 MiB |
| first call total | registration-resident | 0.000 MiB | registration-resident | 0.000 MiB |
| fixed shared pool | — | 3.000 MiB | — | 3.000 MiB |

Per-layer registration deltas (the layer sets cover all 43 stand-in layers):

| layer(s) | T0 before | T0 after | T1 before | T1 after |
|---|---:|---:|---:|---:|
| 00 | 8.596 MiB | 3.033 MiB | 8.596 MiB | 3.033 MiB |
| 01 | 8.596 MiB | 0.033 MiB | 8.596 MiB | 0.033 MiB |
| 02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42 | 14.189 MiB | 0.087 MiB | 47.820 MiB | 0.110 MiB |
| 03,05,07,09,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41 | 11.260 MiB | 0.066 MiB | 26.648 MiB | 0.066 MiB |

The current first-call delta is `+0.000 MiB` for every layer `00..42` in
both modes. The registration trace also reports `+0.000 MiB` for every
first-call fan/local phase.

### Cut

- Removed per-member max-width zero-scaled carriers; tables now point directly
  at the loaded trellis/scale tensors. Mixed widths are dispatched as
  homogeneous subgroups, preserving the installed mgemm ABI without padding.
- Replaced per-group, per-row-bucket A_had/output allocations with one shared
  3.000-MiB decode pool per device/dtype. Groups retain only views and small
  pointer tables; capacity is fixed before graph capture, so later layers do
  not grow VRAM.
- T1 owner completion now drops the cached output views and retains only
  source/shape/stream metadata for the stream-safety check. No per-fan output
  generation remains resident after the last consumer.

Thus the measured T1 stand-in adds `6.702 MiB` total per GPU at registration,
well below the `20 MiB` target, with no first-call growth. The relay run ended
`rc=0`: `gate_active_dense`, `parity_t0`, and `parity_t1` all passed, including
default-OFF bitwise parity, poisoned replay, stream fallback, and graph
no-hot-allocation checks.
