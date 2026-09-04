# 発注 F2: オオタニ専用 密 EXL3（K=6）decode の launch 融合（フェラーリ二本目）

発注者: YUKI（Lna-Lab）／決裁: ケン 2026-09-04「フェラーリで行こう。職人が 1 台に 1 人」「CUDA graph を入れても速度が変わらない状態を目指す」
監督（設計検分・報告の検分）: GLM-L。職人: GLM-F（1 名）。F1（routed experts）とは別の職人・別のファイル。F1 の職人が触る `csrc/lna_moe_decode.*` と `exl3.py` の routed 経路には触らない。

## この一台（形は固定してよい）
- モデル: DeepSeek-V4-Flash-Vision 305B、pack `DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`。密線形 790 本（43 層 × 15〜20）はすべて **EXL3 K=6、mcg、trellis 幅 96**（校正付き焼き）。
- 機械: RTX PRO 2000 16 GB × 8（sm_120）、TP8、クロック 2600 固定・62 W。cold read 帯域 **264〜275 GB/s（実測、F1 職人の microbench 09-04）**。
- decode の行数 m = 4（1 流 DSpark3）〜 16（4 流）。
- 1 層の密線形（保存形。TP8 で列/行を 1/8 に分割して各 rank が持つ）:

| 名 | in → out（保存形） | 入力 |
|---|---|---|
| attn.wq_a | 4096 → 1024 | h（層入力） |
| attn.wkv | 4096 → 512 | h |
| attn.compressor.wgate / wkv（compress 層のみ、43 層中 41） | 4096 → 512 ×2 | h |
| attn.wq_b | 1024 → 32768 | q_a の出力 |
| attn.wo_a.slice.{rank} | 4096 → 1024 | 注意の出力（rank 局所） |
| attn.wo_b | 8192 → 4096 | wo_a の出力 |
| ffn.shared_experts.w1 / w3 | 4096 → 2048 ×2 | h'（注意後） |
| ffn.shared_experts.w2 | 2048 → 4096 | act |

## いまの数字（実測、rank0、graph ON、64 tok）
- `exl3_gemm_kernel<6,…>` が **1 ステップ ≈ 255 launch（≈6/層）、平均 17〜18 µs（tile 128/256）、31 µs（tile 512）**。合計で 1 ステップの **25%**（routed 31%、NCCL 18%）。
- 1 launch の重みは 0.8〜3.1 MB（K=6 → 0.75 byte/weight）。3.1 MB なら 270 GB/s で 11.5 µs、実測 18 µs → **帯域効率 ≈ 6 割、残りは launch の固定費と tail**。
- ESTIMATE: 1 層の密の総 byte ≈ 19 MB（rank 局所）→ 帯域下限 ≈ 70 µs/層。現状 ≈ 107 µs/層。**上限利得 ≈ 1.5×（ステップ全体の 25% → 約 17%、ステップで 8% 前後）**。小さく見えるが、launch 数を 6/層 → 2〜3/層 に減らすことは NCCL の「待ち」（rank 間の粒揃え）にも効く。監督はまず「この発注は割に合うか」を判定してよい。

## 設計の基準案（職人は変えてよい。理由を REPORT に書けば）
1. **同じ入力を持つ線形を 1 launch に束ねる**（grouped GEMV）: (a) h → wq_a + wkv + compressor.wgate + compressor.wkv（4 → 1）、(b) h' → w1 + w3（2 → 1）。それぞれ suh は行列ごとに別、出力 svh も別。束ねても各行列の Hadamard/suh/svh は独立に正しく適用する。
2. 直列の対（wq_a→wq_b、wo_a→wo_b、w1/w3→act→w2）は依存があるので同じ launch には束ねない（cooperative の grid.sync で 2 段にする案は「追加最適化」。まず 1 が出てから）。
3. カーネルは exllamav3 の `exl3_gemm_kernel<6,…>`（家の `exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh`）の **K=6 復号をそのまま使い**、複数の (trellis, suh, svh, out) の表を受けて CTA を「行列 × 出力タイル」に割る。m ≤ 16 の行はすべて同じ CTA が担う（重みは一度だけ読む）。
4. vLLM 側の呼び出し: plugin `src/vllm_exl3/exl3.py` の密線形経路（`LinearEXL3` を包む部分）に、層ごとの「同入力グループ」を登録し、最初の forward で束ねて呼ぶ。**vLLM の層コードを書き換えない**（q_a/kv/compressor の forward が別々に呼ばれる形なら、最初の呼び出しで束ねて計算し、残りはキャッシュから返す「遅延束ね」も可。ただし graph capture と整合すること）。
5. 出力 dtype は既存と同じ（fp32 → bf16 の境界を動かさない）。

## 門（三段。速さは最後）
1. **机上 parity**: 束ねた出力 vs `LinearEXL3.forward` を個別に呼んだ参照。m ∈ {1,2,4,8,16}、実 tensor（層 5 と 40）と合成 tensor。per-row rel ≤ 2e-3 ＋ max abs ＋ finite 一致。`compute-sanitizer` memcheck。
2. **席内**: graph ON/OFF で貪欲突き合わせ（基準＝同カーネル二度走りの揺れ幅 49〜118 字）、作文 7 本 finish=stop、ppl（4k・spec off）6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2。
3. 速度: prof-shapes.py で `exl3_gemm_kernel<6>` の launch 数と合計時間（前: ≈255 launch・25%）、bench-streams 1,4 三回平均、**graph OFF（eager）でも落ちない**ことを測る（ケンの完成の定義）。

## 成果物
- `vllm-exl3-v030-port/csrc/lna_dense_group.cu(.cuh)` ＋ bindings（`lna_dense_group_gemm(x, [trellis…], [suh…], [svh…], [out…], K=6)`）
- `src/vllm_exl3/exl3.py` の密線形経路の束ね（環境変数 `LNA_EXL3_DENSE_GROUP=1` で ON、既定 OFF）
- 机上 parity テスト、`REPORT-F2.md`（設計・実測・落ちた道）

## 読むべきもの
- 家の exllamav3: `exl3_gemm.cu`（autotune・capture guard `lna_stream_is_capturing`）、`exl3_gemm_kernel.cuh`、`exl3_dq.cuh`、`codebook.cuh`、`hadamard_inner.cuh`
- plugin: `vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py`（`make_linear_exl3`、密経路、`_is_wo_a_layer`、`non_routed_exl3`）
- 正典: `Lna-Factory/veins/dsv4-carve/PLAN.md`（家計簿・門の作法）、`orders/REVIEW-F1-sol.md`（ABI の正典: 9 pointer・128 要素 Hadamard・suh/svh の独立・FP32 境界。密でも同じ）
- GPU 実行の作法: `vllm-exl3-v030-port/f1/PROTOCOL.md` と同じ型で `f2/` を使う（GPU 11 のみ。GPU10 は F1 の職人）
