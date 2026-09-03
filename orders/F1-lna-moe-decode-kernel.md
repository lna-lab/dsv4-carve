# 発注 F1: オオタニ専用 routed-experts decode カーネル（フェラーリ一本目）

発注者: YUKI（Lna-Lab）／決裁: ケン 2026-09-04「量産型でなくフェラーリ。職人が 1 台に 1 人。1 モデルにフィットした V8 用カーネル」
職人: 1 名（Luna または Sol）。設計の検分は Sol。

## この一台（形は固定してよい。汎用性は捨てる）
- モデル: DeepSeek-V4-Flash-Vision 305B、pack `DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`
- 機械: RTX PRO 2000 Blackwell 16 GB × 8（sm_120、cu13、torch 2.x、vLLM 0.28.1rc1.dev337 + vllm-exl3 0.2.3+LNA）。TP8、P2P 無し PCIe。クロック 2600 固定・62 W。
- routed experts（TP8 列分割の 1 rank 分）:
  - hidden H = 4096、expert 中間幅（1 rank）I = **256**、experts n = 256（全 expert が各 rank に「細く」在る）、topk = 6
  - 量子化: EXL3 trellis、**K = 2 または 3（層ごと固定、config `layer_bits`）**、codebook **mcg**（marker −877912083）、`suh`(H) / `svh`(I or H) fp16、テンソルは `w13_trellis[e, 0|1]`（gate|up: [H/16, I/16, 16K] int16）、`w2_trellis[e]`（down: [I/16, H/16, 16K]）
  - 活性化: **silu(min(g, L)) · clamp(u, −L, L)**、L = swiglu_limit = 10.0
  - decode の行数 m: **4**（1 流 DSpark3 = 1+3）または **16**（4 流）。m ≤ 16 だけ相手にする（m > 16 は既存 exllamav3 経路のまま）
  - 入力 x: [m, H] bf16 → fp16、routing ids [m, topk] int32（rank 内 local id、非ローカル無し=TP なので全 id 有効）、weights [m, topk] fp32（既に routed_scaling 済）
  - 出力: [m, H] fp32 = Σ_j w[m,j] · down_e(act(gate_e(x), up_e(x)))、**呼び出し側で TP all-reduce**（カーネルの外）
- 参照実装（数値の基準）: exllamav3 `LinearEXL3.forward`（家の像の `exllamav3` 1.4.5+LNA）を gate/up/down に順に掛けた Python ループ = `vllm_exl3.exl3.apply_exl3_python_loop`。**これと相対誤差 ≤ 2e-3 で一致すること**（fp16 中間での自然な差の範囲）。

## いまの数字（実測、rank0、graph ON）
- 現行 = exllamav3 `exl3_moe_kernel<K,256,1>`: **1 層 1 launch、229 µs/launch**（m=4）、1 ステップの **31%**。
- 1 launch が読む重み: m·topk = 24 スロット、expert 1 個（w1+w3+w2、幅 256、K≈2.2）≈ **0.86 MB** → ≈ 21 MB（重複 expert を一度だけ読めばそれ以下）。229 µs で ≈ **90 GB/s**。
- カードの帯域（実測、d2d copy 往復 254 GB/s ≈ 片道 127 GB/s、仕様は ≈288 GB/s 級）。**的: 選ばれた expert の重みを一度だけ読み、≥ 200 GB/s で流す → ≤ 105 µs/launch（≈2×）。**
- 成功の定義（ケン）: 「CUDA graph を入れても速度が変わらない」「56〜62 W に張り付く」= launch は少なく、各 launch がカードを使い切る。

## 設計の指針（職人は変えてよい。理由を書けば）
1. **1 層 1 launch**（協調 or persistent）。m 行 × topk の (row, expert) 対を expert ごとに束ね、**同じ expert は一度だけ読む**（m=16 で重複が出る）。
2. **レジスタ内 trellis 復号**（Cruz `p2b_moe.cu` の `exl3_dequant.cuh` を借りてよい。K=2/3 の分岐はテンプレートで固定）。
3. 入力側 Hadamard（suh）は **行ごとに一度**（expert 間で共有）。出力側（svh）は expert ごと。gate/up は同じ入力に対する 2 本の GEMV → **1 パスで読み分け**。
4. 幅 256 は小さい: 1 expert の gate/up は [4096→256]×2、down は [256→4096]。CTA の割り当ては「expert × 出力タイル」で、SM 数（PRO 2000: 実測で `cudaDevAttrMultiProcessorCount`）を埋めること。
5. 累積は fp32。行ごとの部分和は shared memory か fp32 atomic（決定性は不要、精度は必要）。
6. m=4 と m=16 の両方で帯域効率を測る（同じカーネルで、m はテンプレートでなく実行時でよい）。

## 門（三段。速さは最後）
1. **机上 parity**: `DATA1/.tmp/parity-native.py` の型で、K=2/3 × m∈{1,4,8,16} × routing（重複 expert あり・重み 0 あり）× 乱数 3 seed、参照= `apply_exl3_python_loop` 相当（LinearEXL3）。相対誤差 ≤ 2e-3。
2. **席内自己検査**: `LNA_EXL3_NATIVE_SELFCHECK=N` と同じ仕組み（plugin 側に `LNA_EXL3_MOE_KERNEL=lna` を足し、最初の N 呼び出し/層で参照と比較・ログ）。43 層すべて rel ≤ 2e-3。
3. **給仕の門**: 貪欲突き合わせ（基準＝同カーネル二度走りの揺れ幅）、作文 7 本 finish=stop、ppl（4k・spec off）6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2。
4. そのあとで速度: bench-streams 1,4 三回平均、家計簿（prof-shapes.py）で launch 時間。

## 成果物
- `vllm-exl3-v030-port/csrc/lna_moe_decode.cu(.cuh)` と bindings（`lna_moe_decode(x, out, ptr tables…, ids, weights, K, limit)`）
- `src/vllm_exl3/exl3.py` の dispatch（`VLLM_EXL3_MOE_KERNEL=lna`、m ≤ 16 のみ、他は exllamav3）
- 机上 parity テスト、`REPORT-F1.md`（設計・実測・落ちた道）

## 読むべきもの
- 家の exllamav3: `vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe.cu`, `exl3_moe_kernel.cuh`, `exl3_gemm_kernel.cuh`, `exl3_dq.cuh`（trellis 復号の正典）
- Cruz: `vllm-exl3-v030-port/csrc/p2b_moe.cu`, `exl3_dequant.cuh`（レジスタ内復号・had_hf_r_128）
- plugin: `src/vllm_exl3/exl3.py` の `apply_exl3_python_loop` / `apply_exl3_fused_moe` / `build_exl3_fused_state`
- 正典: `Lna-Factory/veins/dsv4-carve/PLAN.md`（家計簿・門の作法・09-04 の教訓＝平坦 id の罠）
