# 発注 F1 v2: オオタニ専用 routed-experts decode カーネル（フェラーリ一本目）

v2 = Sol の設計検分（`REVIEW-F1-sol.md`、2026-09-04）を折り込んだ版。v1 は `F1-v1-archive.md`。**検分書は発注書の一部。着手前に全文読むこと。**

発注者: YUKI（Lna-Lab）／決裁: ケン 2026-09-04「量産型でなくフェラーリ。職人が 1 台に 1 人。1 モデルにフィットした V8 用カーネル」
職人: 1 名（Luna または Sol）。設計の検分は Sol。

## この一台（形は固定してよい。汎用性は捨てる）
- モデル: DeepSeek-V4-Flash-Vision 305B、pack `DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`
- 機械: RTX PRO 2000 Blackwell 16 GB × 8（sm_120、cu13、torch 2.x、vLLM 0.28.1rc1.dev337 + vllm-exl3 0.2.3+LNA）。TP8、P2P 無し PCIe。クロック 2600 固定・62 W。
- routed experts（TP8 列分割の 1 rank 分）:
  - hidden H = 4096、expert 中間幅（1 rank）I = **256**、experts n = 256（全 expert が各 rank に「細く」在る）、topk = 6
  - 量子化: EXL3 trellis、**K = 2 または 3（層ごと固定、config `layer_bits`）**、codebook **mcg**（marker −877912083）、`suh`(H) / `svh`(I or H) fp16、テンソルは `w13_trellis[e, 0|1]`（gate|up: [H/16, I/16, 16K] int16）、`w2_trellis[e]`（down: [I/16, H/16, 16K]）
  - 活性化: **silu(min(g, L)) · clamp(u, −L, L)**、L = swiglu_limit = 10.0
  - decode の行数 m: 性能の主標的は **4**（1 流 DSpark3 = 1+3）と **16**（4 流）。**正しさは 1 ≤ m ≤ 16 全部**（自己検査ログに 2,3,5,6 行も現れる）。m > 16 は既存 exllamav3 経路のまま
  - 入力 x: [m, H] bf16 → fp16、routing ids [m, topk] int32（rank 内 local id、非ローカル無し=TP なので全 id 有効）、weights [m, topk] fp32（既に routed_scaling 済）
  - 出力: [m, H] fp32 = Σ_j w[m,j] · down_e(act(gate_e(x), up_e(x)))、**呼び出し側で TP all-reduce**（カーネルの外）
- routing weights は **routed_scaling 1.5 を掛け終えた値**を受け取る（カーネル内では掛けない。ABI に明記し、省略/一回/二回を見分ける番兵テストを置く）
- 参照実装（数値の基準）: exllamav3 `LinearEXL3.forward`（家の像の `exllamav3` 1.4.5+LNA）を gate/up/down に順に掛けた Python ループ = `vllm_exl3.exl3.apply_exl3_python_loop`。**これと相対誤差 ≤ 2e-3 で一致すること**（fp16 中間での自然な差の範囲）。

## いまの数字（実測、rank0、graph ON）
- 現行 = exllamav3 `exl3_moe_kernel<K,256,1>`: **1 層 1 launch、229 µs/launch**（m=4）、1 ステップの **31%**。
- 1 launch が読む重み: m·topk = 24 スロット、expert 1 個（w1+w3+w2、幅 256、K≈2.2）≈ **0.86 MB** → ≈ 21 MB（重複 expert を一度だけ読めばそれ以下）。229 µs で ≈ **90 GB/s**。
- カードの帯域（実測、d2d copy 往復 254 GB/s ≈ 片道 127 GB/s、仕様は ≈288 GB/s 級）。**的: 選ばれた expert の重みを一度だけ読み、≥ 200 GB/s で流す → ≤ 105 µs/launch（≈2×）。**
- 成功の定義（ケン）: 「CUDA graph を入れても速度が変わらない」「56〜62 W に張り付く」= launch は少なく、各 launch がカードを使い切る。

## 設計の基準案（Sol 検分 §2。職人は変えてよい。理由を REPORT に書けば）

### ABI（固定）
- expert 1 個につき **9 本の pointer**: gate/up/down × (trellis, suh, svh)。形は gate/up `[256,16,16K] int16` + suh `[4096] fp16` + svh `[256] fp16`、down `[16,256,16K] int16` + suh `[256]` + svh `[4096]`。**gate と up は形が同じでもデータも suh/svh も別**。alias を仮定しない。
- `mcg` は codebook の標識（int32 −877912083 = 0xCBAC1FED）。**host 側で全 expert・全 projection の標識を検査し、カーネルは compile-time cb=1**。raw mcg pointer を毎回読まない。
- K は層ごと固定（config `layer_bits`）。`trellis.size(-1)==16K` を全 expert・全 projection で検査。**K2 と K3 を同じ pointer table に混ぜない**。
- trellis 復号の正典 = 家の `exllamav3_ext/quant/exl3_dq.cuh` + `codebook.cuh`（16×16 tile、circular bitstream 開始 `((t_offset+257)*K − 16) mod 256K` bit、K3 は 24 word・modulo 24・状態 s2、MCG は unsigned mul 0xCBAC1FED + lop3 0x6a + half 加算の丸めまで同一）。**Cruz `exl3_dequant.cuh` の scalar helper を oracle にしない**（float で返す経路があり half bit-exact でない）。
- Hadamard は **128 要素の独立 block ごと**の Walsh–Hadamard。順序: suh を掛ける → 128-block Hadamard（1/√128）→ 復号行列と積 → 出力 128-block Hadamard（1/√128）→ svh。**suh は行共通でも expert 共通でもなく、gate と up でも別**。共有できるのは同じ (expert,row) の gate/up で元の x の load だけ。
- SwiGLU: `silu(min(g, L)) · clamp(u, −L, L)`、L=10.0。gate は pre-SiLU の片側 clamp、up は対称 clamp。**gate/up の累積・出力 Hadamard・clamp/SwiGLU まで FP32**。activation を FP16 に落とすのはそのあと（ここが down の ABI 境界）。down の累積と出力 Hadamard も FP32。expert 和も FP32。FP16 MMA 累積+定期 FP32 fold は「追加最適化」扱いで、採る条件は全 adversarial test で十分な余裕があること。
- NaN 方針: 入力 finite を契約とし、debug flag で non-finite を検出して fail（fminf/fmaxf が NaN を潰すのを黙認しない）。
- 入出力 dtype: 入力 bf16 → **カーネル内で** fp16 化（別 cast launch を作らない。「1 launch」は cast/sort/zeroing の補助カーネルを外に出して数えない）。出力は内部 FP32、**同じ launch の最後で consumer dtype（x.dtype、通常 bf16）に書く**を第一候補。NCCL の byte 数が既存と同じことを確認する。

### CTA 構成（cooperative persistent、1 層 1 launch）
- grid = P blocks × 256 threads（8 warps）。P = `cudaDevAttrMultiProcessorCount`（推測せず 8 rank 全部で実測して記録）。`cudaOccupancyMaxActiveBlocksPerMultiprocessor` で P blocks 同時常駐を launch 前に保証（超えると cudaErrorCooperativeLaunchTooLarge）。2P や 512 threads は bench variant。
- `<K, M_CAP>` を compile-time variant（M_CAP 4/8/16 を prewarm）、実 m は runtime。occupancy query・variant 選択・allocation・pointer table 構築は **graph capture 前に一度**。
- **全 CTA が全 grid.sync() に同順序で到達する**。仕事が無い phase でも early return しない。
- phase:
  1. route plan: CTA0 が ≤96 個の (expert,row,kpos,weight) を固定順で整列。同一 row 内の重複 expert は weight を FP32 固定順で事前合算。active expert list と row span を scratch へ。scratch counter は世代管理（別 zero kernel を作らない）。grid.sync。
  2. gate/up 入力変換: work = (expert,row,input-128-block)。x を一度 load、gate 用 suh・up 用 suh を別に掛けて 2 本の Hadamard を fp16 scratch へ。grid.sync。
  3. gate+up+activation: 1 CTA = (expert, intermediate-128-block)（I=256 → expert あたり 2 task）。gate/up の各 tile を**一度復号してその expert の全 row に使ってから捨てる**。FP32 累積 → 出力 Hadamard → svh → clamp/SwiGLU → fp16 → **同じ CTA で down 用 suh + 入力 Hadamard** を down-input scratch へ（barrier を一つ節約）。m=4 一様 routing で active expert ≈23 → ≈46 task。grid.sync。
  4. down: work = (expert, output-column-tile)、N tile 256 と 512 を比較（512 なら expert あたり 8 task、m=4 で ≈184）。K=256 を sweep、復号した B fragment をその expert の全 row に適用してから捨てる。**R_e=16 でも二つ目の 8-row panel のために B を再 load/復号しない**（16-row panel か B register 保持）。FP32 累積 → 出力 Hadamard → svh → (row, expert-slot, tile) の FP32 scratch へ。grid.sync。
  5. 固定順 gather: 1 warp = (row, output-128-block)。top-k slot 0..5 の固定順に FP32 weight を掛けて加算、**atomics を使わず上書き**（同一入力の bitwise 再現性と stale scratch 排除）。consumer dtype に変換して書く。
- scratch と route plan は layer 単位でなく **同時 replay し得る stream/graph instance ごとに分離**。既存 process-global lock/scratch（DevCtx）を流用しない。shared-experts の aux stream と常駐 grid の重なりは未対応として明示 error/fallback（現運用は VLLM_DISABLE_SHARED_EXPERTS_STREAM=1）。
- m=1 は active expert ≤6 で並列度が低い。主標的 m=4/16 の設計を複雑化させない（m=1 は既存経路 dispatch も可）。

## 的（Sol §3。105 µs は K2・m=4 の stretch goal に限定）
unique expert 1 個の実 byte: K2 812,544 B（trellis 786,432 + scales 26,112）、K3 1,205,760 B。一様 routing の unique 数 U = 256(1−(255/256)^{6m}): m=4 → 22.95、m=16 → 80.18（dedup 効果はそれぞれ 1.05×・1.20× にすぎない。**実 prompt での U histogram を全層で採る**）。

| 条件 | unique bytes | 288 GB/s 下限 | 200 GB/s 下限 |
|---|---:|---:|---:|
| m=4 K2 | 18.65 MB | 64.8 µs | 93.2 µs |
| m=4 K3 | 27.67 MB | 96.1 µs | 138.4 µs |
| m=16 K2 | 65.15 MB | 226.2 µs | 325.8 µs |
| m=16 K3 | 96.68 MB | 335.7 µs | 483.4 µs |

- 合否は **T ≤ 1.25 × T_floor**（T_floor = 実 unique bytes ÷ 実測 cold read 帯域）＋現行 229 µs 比の倍率で判定。K2・m=4 の 105 µs は stretch goal。
- 先に **read-only cold-cache microbench**（重みと同じ幅・順序）で実測 read 帯域を出す。D2D copy 往復 254 GB/s は read+write 合算で、read-only 上限 127 GB/s の意味ではない（公称 288 GB/s）。
- Nsight Compute で DRAM read bytes/帯域・L2 hit・tensor/ALU 利用・occupancy・long-scoreboard・barrier stall・wave 数。「論理 unique bytes/time」と「実 DRAM bytes/time」を分け、cold と warm も分ける。**56〜62 W は診断情報。合否基準にしない**。

## 門（Sol §4。速さは最後）
1. **primitive**: trellis 復号を all-zero / all-one / walking-bit / 乱数 bitstream × 全 lane × 全 t_offset × wrap 境界 × K2/K3 で、canonical device 実装と **half bit pattern 一致**。
2. **机上 parity**（`F1-ref/parity-native.py` を出発点に拡張）: TP 実形状 I=256、m ∈ {1,2,3,4,5,7,8,15,16} × K2/K3。production tensor（K2 層と K3 の層 13/22/28）と合成 tensor の両方。route: 重複なし / 全 row 同一 hot expert / Zipf / expert 0 と 255 / 同一 row 内重複 / R_e=16 / zero weight / 相殺 weight。**[m,6] と flat [6m] の取り違え regression**。入力: zero / 実 bf16 capture / 増幅 / clamp 境界 9.99, 10.0, 10.01 / 大値。段階別比較（gate/up FP32・act FP32 と FP16 後・down・route 後）。判定 = per-row rel ≤ 2e-3 **＋ max abs ＋ finite 一致**（零付近は abs 閾値）。scratch/出力を NaN poison して graph を多数回 replay し K/m/route を交互に変えて stale 検出。`compute-sanitizer` memcheck/racecheck/synccheck/initcheck を release gate。
3. **席内自己検査**（`LNA_EXL3_NATIVE_SELFCHECK` と同じ仕組みで `VLLM_EXL3_MOE_KERNEL=lna`）: capture 中は走らせない（cudaStreamIsCapturing で分岐、eager で実施）。mismatch は strict mode で fallback か起動失敗。再現用 dump（x, routes, weights, K, limit, 9 pointer の識別）。同一入力二回で **bitwise 一致**を別 gate に。初回 N 回で max|gate|,|up|,|act|,|down| と finite flag。全 9 pointer の dtype/shape/stride/alignment/lifetime/marker/K extent を検証。steady decode の全 bucket と K3 層が通ったことを記録。
4. **給仕の門**: graph ON/OFF の parity と latency（M_CAP 4/8/16 × K2/K3 を全部 capture/replay）。launch trace で補助カーネル（.half/.to/sort/zeroing）が層の外に増えていないこと。spec OFF と DSpark3、1/4 流、作文 7 本 finish=stop、ppl（4k・spec off）6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2。scaling 1.5 が一度だけ・加算後 dtype・NCCL byte が基準と同じ。**8 rank 全部**の P50/P95 と最遅 rank、3 回以上、thermal steady。各計測に K と U を併記。cold-cache 計測を主値、warm replay は別記。
5. そのあとで速度: bench-streams 1,4 三回平均、prof-shapes.py で launch 時間。

## 成果物
- `vllm-exl3-v030-port/csrc/lna_moe_decode.cu(.cuh)` と bindings（`lna_moe_decode(x, out, 9 pointer tables, ids[m,6], weights[m,6] fp32, K, limit, stream-scratch handle)`）
- `src/vllm_exl3/exl3.py` の dispatch（`VLLM_EXL3_MOE_KERNEL=lna`、m ≤ 16 のみ、他は exllamav3）、host 側の標識/K/pointer 検証、prewarm、自己検査
- primitive test、机上 parity テスト、cold read microbench、`REPORT-F1.md`（設計・実測・落ちた道・U histogram・8 rank latency）

## 読むべきもの
- **`orders/REVIEW-F1-sol.md`（Sol の検分。全文）**
- 家の exllamav3: `vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe.cu`, `exl3_moe_kernel.cuh`, `exl3_gemm_kernel.cuh`, `exl3_dq.cuh`, `codebook.cuh`, `hadamard_inner.cuh`, `exl3_gemv_kernel.cuh`, `exl3_devctx.cu`（trellis 復号・Hadamard の正典）
- Cruz: `vllm-exl3-v030-port/csrc/p2b_moe.cu`, `exl3_dequant.cuh`（レジスタ内復号の参考。oracle にはしない）
- plugin: `src/vllm_exl3/exl3.py` の `apply_exl3_python_loop` / `apply_exl3_fused_moe` / `build_exl3_fused_state`
- 正典: `Lna-Factory/veins/dsv4-carve/PLAN.md`（家計簿・門の作法・09-04 の教訓＝平坦 id の罠）
