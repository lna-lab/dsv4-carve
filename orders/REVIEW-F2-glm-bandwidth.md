# F2 密 EXL3 launch 融合 発注書の検分（帯域と数字のレンズ）

監督: GLM-L（帯域と数字のみ）。対象: `orders/F2-dense-launch-fusion.md`（決裁 2026-09-04）。
入力: 本家 pack の config と safetensors ヘッダ、`prof-graph-0904/` の実プロファイル、
`vllm-exl3/src/vllm_exl3/exl3.py`、`exllamav3-src/.../quant/{exl3_gemm.cu, exl3_gemm_kernel.cuh, exl3_kernel_map.cuh}`、
`orders/REVIEW-F1-sol.md`、`orders/REPORT-F1-opus.md`、`orders/T5-vllm-side-reading.md`。
本検分はファイル書き込み以外の副作用なし（ビルド・GPU・席 :8899 は不使用）。

## 結論

発注の方向性（同じ入力を持つ密線形を 1 launch に束ね、重みを 1 回だけ読む）は帯域の数字から正当化できる。
ただし発注書の「いまの数字」のうち現状量と対象範囲の両方に大きい誤りがあり、**上限利得の 1.5× と ステップ 8% は
どちらも導出が不正確**である。再計算の結果は次の通り。

- **launch 数は ≈255/ステップではなく実測 489/ステップ（11.4/層）**。発注書の表に載っていない indexer 系
  （`attn.indexer.wq_b`、`attn.indexer.compressor` ×2、21 層）が decode の各ステップで走っており、
  その 3 launch/層 × 21 層 を含む構成と実カウントが完全に一致する（§2）。
- **rank 局所の密 byte は ≈19 MB/層ではなく層平均 26.5 MB、1 ステップ計 1.138 GB**。compressor の出力幅は
  512 でなく 1024（×2）、indexer 系が +7.9 MB/層（21 層分）になる（§1）。
- **帯域の床（270 GB/s）は 4.215 ms/ステップ（98.0 µs/層）**。現状実測 8.885 ms/ステップ（206.6 µs/層）との比で
  **上限利得は ≈2.1×、ステップ時間 −13%（スループット +15%）**。発注書の 107 µs/層 ÷ 70 µs/層 = 1.5× は
  分子（launch 数の過少計上）・分母（byte の過少計上）が相殺して偶然 1.5 に見えている（§3）。
- 発注書が実際に設計した範囲（グループ (a)(b) のみ、依存ペアの融合は後回し）で取れるのは **dense 1.15〜1.35×、
  ステップ −3〜6%**。indexer 2 グループを同じ機構で追加すれば上寄り。**「ステップ 8%」はこの scope では過大目標**で、
  到達には後回しにした依存融合か 0.795 MB クラスの抜本的整理が要る（§3.3）。
- 帯域の数字だけで見た最大の泉は 0.795 MB クラス（shared experts + indexer compressor。byte の 12% で時間の 33%、
  実効 47.5 GB/s）。グループ (b) はその 171 launch 中 86 本しか刈らない。スコープ見直しを推奨（§4）。
- 実装の出発点として家には既に `exl3_mgemm`（pointer 表・1 入力複数出力・行列ごとの suh/svh/幅）がある。
  新規に `lna_dense_group` を書く場合は差分理由を REPORT に書くこと（§4.3）。

判定: **修正条件付きで着手可**。「上限 1.5×・ステップ 8%」をそのまま合否基準に使わないこと（§5）。

## 1. 形と byte の積算（pack 実測）

### 1.1 保存形と TP（実測）

根拠はすべて pack `DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6` から取った。

- `config.json:13` head_dim=512、`:26` num_attention_heads=64 → wq_b 出力 32768 = 64 頭 × 512。
  `:32` o_groups=8、`:33` o_lora_rank=1024 → wo_a は 8 group × (8 頭 × 512 = 4096 入力) → 1024 出力の slice。
  `:34` q_lora_rank=1024、`:22` moe_intermediate_size=2048、`:28` num_hidden_layers=43。
  indexer は `:16-17` index_head_dim=128 × index_n_heads=64 → indexer.wq_b 出力 8192。
- `model-dense-exl3.safetensors` ヘッダ（layers.10 で確認、他層同形）:
  wq_a trellis `[256,64,96]` svh `[1024]`、wkv `[256,32,96]` svh `[512]`、
  **compressor.wkv/wgate `[256,64,96]` svh `[1024]`（4096→1024。発注書 F2:16 の「4096→512×2」は誤り）**、
  wq_b `[64,2048,96]` svh `[32768]`、wo_a.slice.N `[256,64,96]` svh `[1024]`、wo_b `[512,256,96]` suh `[8192]`、
  **indexer.wq_b `[64,512,96]` svh `[8192]`、indexer.compressor ×2 `[256,16,96]` svh `[256]`（発注書の表に存在しない）**、
  shared w1/w3 `[256,128,96]`、w2 `[128,256,96]`。
- `config.json` の `quantization_config.non_routed_exl3.layers` 佐証: compressor は 41 層（2..42。F2:16 の
  「43 層中 41」は正しい）、**indexer 系は偶数 21 層（2..42）**。790 本（F2:7）は checkpoint 側の数え方
  （wo_a 8 slice を数え、indexer を含む）で正しい。
- TP 扱いは `orders/T5-vllm-side-reading.md:5-11` と `vllm-exl3/src/vllm_exl3/exl3.py` の実装通り:
  wq_a/wkv/compressor/indexer.compressor は `disable_tp`、indexer.wq_b は ReplicatedLinear →
  実効 TP=1（`exl3.py:74-87`、tp_size 1 では narrow しない `exl3.py:62-66`）。wq_b は列分割、wo_b は行分割
  （`exl3.py:1315-1322`）、wo_a は rank 局所 slice をそのまま（`exl3.py:1056-1063`, `:1236-1252`）。
  **発注書 F2:10 の「TP8 で列/行を 1/8 に分割して各 rank が持つ」は wq_a/wkv/compressor に対して誤り
  （これらは複製で 8 分割されない）**。

### 1.2 rank 局所 byte（実測形状からの計算）

1 行列の重み byte = trellis (in/16)(out/16)(16K)·2 + suh 2·in + svh 2·out。K=6 で 0.75 B/weight
（`orders/REVIEW-F1-sol.md:36-38` の正典、`exl3_gemm.cu:28-33` の B 形、K=`B.size(2)/16` は `exl3_gemm.cu:178`）。
mcg marker は kernel が dereference しない（`REVIEW-F1-sol.md:44-48`）ので除外。

| 行列 | 保存形 in→out | rank 局所 in→out | 1 launch の重み | launch/step | MB/step |
|---|---|---|---:|---:|---:|
| attn.wq_a（複製） | 4096→1024 | 4096→1024 | 3.156 MB | 43 | 135.7 |
| attn.wkv（複製） | 4096→512 | 4096→512 | 1.582 MB | 43 | 68.0 |
| attn.compressor.wkv/wgate（複製、41 層） | 4096→**1024** ×2 | 同左 | 3.156 MB ×2 | 82 | 258.8 |
| attn.wq_b（列/8、64 頭×512 を 8 頭ずつ） | 1024→32768 | 1024→4096 | 3.156 MB | 43 | 135.7 |
| attn.wo_a.slice.{rank}（局所） | 4096→1024 | 4096→1024 | 3.156 MB | 43 | 135.7 |
| attn.wo_b（行/8） | 8192→4096 | 1024→4096 | 3.156 MB | 43 | 135.7 |
| attn.indexer.wq_b（複製、21 層） | 1024→8192 | 1024→8192 | **6.310 MB** | 21 | 132.5 |
| attn.indexer.compressor ×2（複製、21 層） | 4096→256 ×2 | 同左 | 0.795 MB ×2 | 42 | 33.4 |
| ffn.shared w1/w3（列/8、2048 を 8 分割） | 4096→2048 ×2 | 4096→256 ×2 | 0.795 MB ×2 | 86 | 68.4 |
| ffn.shared w2（行/8） | 2048→4096 | 256→4096 | 0.795 MB | 43 | 34.2 |
| **合計** | | | 0.795〜6.310 MB | **489** | **1,138.1** |

層ごとの class（実測形状から計算）:

| 層種 | 層数 | rank 局所 byte | 270 GB/s 床 |
|---|---:|---:|---:|
| 非 compressor（層 0,1） | 2 | 16.59 MB | 61.4 µs |
| compressor（奇数 3..39 + 41） | 20 | 22.90 MB | 84.8 µs |
| compressor + indexer（偶数 2..42） | 21 | 30.80 MB | 114.1 µs |
| **43 層平均 / 1 ステップ計** | 43 | **26.47 MB**（**1,138.1 MB**） | **98.0 µs（4.215 ms）** |

発注書 F2:26 の「1 層 ≈19 MB」は compressor 層から indexer を引いた 22.9 MB に近く、indexer 分 7.9 MB と
compressor の幅の誤り（1024 を 512 と書いた分 −1.6 MB/層）を除けば偶々 19 MB 付近に落ちているが、
**正しい積算は層平均 26.5 MB・ステップ 1.138 GB**。帯域幅（264〜275 GB/s、`REPORT-F1-opus.md:146` の
`f1/cold_read_bw.cu` 実測、270 を使用）に対する床は:

- 264 GB/s: 4.311 ms ／ 270 GB/s: **4.215 ms** ／ 275 GB/s: 4.139 ms（いずれも 1 ステップ、rank 局所）。

## 2. 現状の実測（prof-graph-0904/profiler_out_0.txt、rank0、graph ON）

| kernel（`exl3_kernel_map.cuh:7-15` の並び: bits,fp32,cb,TM,TK,TN,SH,FR） | calls | avg | 計 |
|---|---:|---:|---:|
| `exl3_gemm_kernel<6,true,1,16,32,128,4,3>`（profiler_out_0.txt:7） | 8,179 | 17.971 µs | 146.985 ms |
| `exl3_gemm_kernel<6,true,1,16,32,256,4,3>`（:10） | 3,053 | 16.753 µs | 51.148 ms |
| `exl3_gemm_kernel<6,true,1,16,16,512,4,3>`（:19） | 483 | 31.296 µs | 15.116 ms |
| 計 | 11,715 | 18.19 µs | **213.249 ms** |

- ステップ数は 24（実測からの一意決定）: `exl3_moe_kernel<2>` 1,032 calls = 43 層 × 24（:5。layer_bits の
  13/22/28 が K3、MTP 43-45 が K2 → K2 は 40+3 層。`exl3_moe_kernel<3>` 72 = 3 × 24（:18））。
- **11,715 = 24 × 489 − 21**。489 は §1.2 の構成表の launch 総数そのもの。−21 は indexer.wq_b が
  21 層分ちょうど 1 ステップぶん欠けていることと一致する（tile512 が 483 = 21×23。一方 indexer 系の
  attention kernel は 504 = 21×24。どのステップで欠けるかは profiling 窓の切り方の可能性もあり推定）。
  すなわち **1 ステップ 489 launch、11.4/層**。発注書 F2:24 の「≈255 launch（≈6/層）」を支持するデータは
  このプロファイルに存在しない。
- dense の割合: 213.249 ms ÷ 総 self CUDA 862.200 ms（:106）= **24.7%**。routed 31.0%（29.06+1.99%）、
  NCCL 17.6%（:6 ほか）。F2:24 の 25%/31%/18% はこのまま正しい。
- 1 ステップの dense 時間 = 213.249/24 = **8.885 ms/step = 206.6 µs/層**（発注書 F2:26 の「≈107 µs/層」は
  255×18µs/43 から導かれたもので、launch 数の誤りを引き継いでいる）。
- tile ↔ 行列 class の対応は推定（コール数の一致のみによる）: tile512 = indexer.wq_b（483 = 21×23）、
  tile256 ≈ shared 3 本（127.2/step ≈ 129）、tile128 = 残り（3.156 MB 系 254 本 + wkv 43 + indexer.compressor 42）。
  発注書 F2:24 が引用する「31 µs（tile 512）」はまさに表から落ちている 6.31 MB の indexer.wq_b であり、
  発注書自身の実測が indexer の存在を証明している。また 1 launch の重みは F2:25 の「0.8〜3.1 MB」でなく
  **0.795〜6.310 MB**。

### 2.1 class 別の実効帯域（実測時間 × 推定 class 割当てから導出）

| class | launch/step | byte/step | 時間/step | 実効帯域 | 270 GB/s 比 |
|---|---:|---:|---:|---:|---:|
| 3.156 MB（wq_a/wq_b/wo_a/wo_b/compressor） | 254 | 801.5 MB | 4.57 ms | 175 GB/s* | 65% |
| 1.582 MB（wkv） | 43 | 68.0 MB | 0.77 ms | 88 GB/s* | 33% |
| 0.795 MB（shared 3 本 + indexer.compressor 2 本） | 171 | 136.0 MB | 2.89 ms | 47.5 GB/s | 18% |
| 6.310 MB（indexer.wq_b） | 21 | 132.5 MB | 0.66 ms | 202 GB/s | 75% |
| **計** | **489** | **1,138.1 MB** | **8.89 ms** | **128 GB/s** | **47%** |

\* tile128 バケットは 3 class の混ざりなので、3.156 MB 系単独の実効帯域はこの平均より良い可能性がある
（推定の誤差範囲）。**発注書 F2:25 の「帯域効率 ≈6 割」は 3.156 MB 系にのみ成立し、全体では 47%**。
固定費の実測換算: 3.156 MB 系で +6.3 µs/launch、6.31 MB 系で +7.9 µs、0.795 MB 系で **+13.8 µs/launch**（純帯域時間 2.95 µs の 4.7 倍が固定費で、実測 16.8 µs は純分の 5.7 倍）。0.795 MB クラスが byte の 12% で時間の 33% を占めるのが最大の非効率。

## 3. 床・現状・上限の再計算

### 3.1 上限（帯域床との比）

- 現状（実測）: 8.885 ms/step（206.6 µs/層）。
- 床（270 GB/s、§1.2 の実測 byte）: 4.215 ms/step（98.0 µs/層）。
- **上限利得 = 8.885/4.215 = 2.11×**。dense がステップの 24.7% なので、床まで行けば
  ステップ時間は 0.753 + 0.247/2.11 = 0.870 → **−13.0%（スループット +15%）**。
- 発注書 F2:26 の「上限 ≈1.5×・ステップ 8%」: 分子 107 µs/層は launch 255 の誤り、分母 70 µs/層は
  byte 19 MB の誤り（正しくは 206.6 と 98.0）。比は偶然 1.53 vs 2.11 になっているだけで、
  「上限」としては 1.5× は過小、導出根拠としては両辺とも使えない。

### 3.2 床の前提（推定の明示）

- 床は重み byte を各 rank が 1 ステップに 1 回読む限界。活性（A・A_had・C）の DRAM traffic は含まない。
  L2 が 32 MB（`REPORT-F1-opus.md:145`）で launch 内の A_had 書き戻し・C 読みは L2 に載る想定。
  m=64（このプロファイルの bucket）では活性 slab が 0.1〜2 MB/launch で、L2 を外れる分だけ床は数 % 上がる
  （未検証、誤差項として扱うこと）。
- 現行 kernel は m を 16 行パネルに分け、パネルごとに B を再ストリームする（`exl3_gemm_kernel.cuh:36-49`）。
  m=64 の実測 18 µs 台にはこの再読み（L2 で相当量吸収されるはず）が含まれる。発注書 F2:31 の
  「m ≤ 16 の行はすべて同じ CTA が担う（重みは一度だけ読む）」が守られることが床成立の条件。
- 270 GB/s は cold read microbench（`REPORT-F1-opus.md:146`、264〜275 GB/s）。GEMM の 32B 単位の
  strided access で同値が出るかは F2 の職人が prof-shapes.py と併せて実測すること。

### 3.3 発注書の設計 scope で取れる分（推定）

固定費モデル（class 実測時間 − 純帯域時間）で束ね効果を積算する。融合後の launch は現状の最好 class
（202 GB/s、n が広い場合）から 0.795 MB 系の悲観値（≈100 GB/s、n=256 は狭い）までの帯域を仮定。

| 束ね | launch | 現状（実測） | 融合後（推定） | 節減/step |
|---|---|---:|---:|---:|
| (a) h → wq_a+wkv(+comp ×2) 41+2 層 | 4→1 / 2→1 | 71.9 µs/層 | 55〜63 µs/層 | 0.4〜0.7 ms |
| (b) h' → w1+w3 43 層 | 2→1 | 33.5 µs/層 | 9〜16 µs/層 | 0.8〜1.1 ms |
| （発注書外）idx.compressor 2→1 21 層 | 2→1 | 35.9 µs/層 | 9〜16 µs/層 | 0.4〜0.6 ms |
| （発注書外）q_a → wq_b+indexer.wq_b 21 層 | 2→1 | 49.3 µs/層 | 46〜47 µs/層 | ≈0.04 ms |

- 発注書 scope（(a)+(b) のみ）: 1.2〜1.8 ms 節減 → dense 7.1〜7.7 ms → **dense 1.15〜1.25×、
  ステップ −3〜5%**。
- indexer 2 グループを追加すれば 1.6〜2.4 ms → dense 6.5〜7.3 ms → **dense 1.2〜1.35×、ステップ −4〜6%**。
- **「ステップ 8%」は (a)+(b) の書かれた scope に対しては過大**。8% に届くには、(i) indexer グループ追加に
  加えて (ii) 残る単独 launch（wq_b・wo_a・wo_b・w2、いずれも 65% 前後の実効帯域）か 0.795 MB クラスの
  抜本整理、すなわち発注書 F2:30 が「追加最適化」と後回しにした依存融合（wq_a→wq_b、wo_a→wo_b、w1/w3→w2）が要る。
- 同じ理由で、F2:26 の「launch 数を 6/層 → 2〜3/層」も不整合: 現状は 11.4/層、(a)+(b) で 321/step（7.5/層）、
  indexer 込みで 279/step（6.5/層）。**2〜3/層は依存融合を含まない限り到達不能**。目標は 489→280 前後
  （−43%）と書き直すべきである。

## 4. 帯域レンズからの設計上の注文

1. **0.795 MB クラスを主目標に据えよ**（§2.1: byte 12% で時間 33%、+13.8 µs/launch）。グループ (b) は
   shared gate/up の 86 本のみ。indexer.compressor の 42 本は同じ機構・同じ入力（h）なので (a) に統合できる
   （indexer 層では h 系 6 本 → 1 launch）。ここを外すと scope の割に合いが悪い。
2. **グループ (a) の byte 計算は 11.05 MB/層**（wq_a 3.16 + wkv 1.58 + comp 2×3.16）。発注書の「4 → 1」は
   正しいが、表の幅の誤り（512→1024）を直すと床は 40.9 µs/層、現状 71.9 µs/層。利得の見積りは §3.3 の通り。
3. **家の `exl3_mgemm` を起点に検討せよ**: `exl3_gemm.cu:370-410` は pointer 表（B/suh/svh は device address の
   int64 表）、1 入力の複数出力への broadcast（a_batches==1）、行列ごとの出力幅 `size_n_list`/`c_ptrs`
   （`exl3_gemm.cu:467-477`）を既に持つ。kernel 側も行列ごとに suh/svh を独立適用する
   （`exl3_gemm_kernel.cuh:166-260`）ので ABI の正典（`REVIEW-F1-sol.md` 9 pointer・suh/svh 独立）と整合する。
   新規 `lna_dense_group.cu` を書くなら、mgemm を使わない理由（A_had を行列ごとの slab に書く
   `exl3_gemm.cu:478-479` の scratch traffic、grid.z concurrency、lock 区画の非効率など）を REPORT に数値で書け。
   ただし mgemm の `size_n_list` モードは `num_tokens==1 && min_index<0 && !weights` 限定（`exl3_gemm.cu:472-473`）。
4. **床の証明を acceptance に入れよ**: prof-shapes.py で「launch 数（前 489 → 後）」と「logical unique bytes
   （1.138 GB/step）÷ 実測時間」の実効帯域を報告。融合 kernel が重みを 1 回だけ読んでいることの証明は
   Nsight Compute の DRAM read bytes（unique bytes との一致）で行う（F1 と同じ作法）。
5. m=16 bucket と m=4 bucket の両で測ること。m=4 は現行 kernel でもパネル再読みが 1 回で済むため
   融合の見かけ上の利得が m=64 より小さくなる可能性がある（推定）。
6. NCCL の「待ち」への効果（F2:26）は本レンズでは検証不能の質的主張。acceptance から外し、
   副次的観測（8 rank の P50/P95 揃い）に留めること。

## 5. 発注書への反映事項

- F2:10 表: wq_a/wkv/compressor は複製（実効 TP=1）と修正。compressor を 4096→**1024** ×2 に修正。
  indexer.wq_b（1024→8192、複製、21 層）と indexer.compressor（4096→256 ×2、複製、21 層）の 2 行を追加。
- F2:24: 「≈255 launch（≈6/層）」→「489 launch（11.4/層、実測 11,715 calls/24 steps）」。平均時間と
  25%/31%/18% の内訳はそのまま正しい。
- F2:25: 1 launch の重みは 0.795〜6.310 MB。「帯域効率 6 割」は 3.156 MB 系のみ。全体実効は 128 GB/s（47%）。
- F2:26: 「1 層 ≈19 MB → 70 µs/層・現状 ≈107 µs/層・上限 ≈1.5×・ステップ 8%」を
  「層平均 26.5 MB・1 ステップ 1.138 GB → 床 4.215 ms（98.0 µs/層）・現状実測 8.885 ms（206.6 µs/層）・
  上限 2.11×（ステップ −13%）」に置換。**合否基準は二段に分ける**: 第一目標 = scope (a)+(b)+indexer 系で
  dense 8.89→7.3 ms 以下（−18%、ステップ −4% 以上）、到達目標 = 依存融合込みで dense 1.5×
  （ステップ −8%、旧発注書の「8%」はこちらの意味で成立）。床 −13% は上限参考値。
- F2:26: 「launch 6/層 → 2〜3/層」→「489/step → 280/step 前後（−43%）。2〜3/層は依存融合（追加最適化）込み」。
- F2:29: グループ (a) に indexer.compressor ×2 を加え、(c) q_a → wq_b + indexer.wq_b を追加候補に。
  vLLM 側登録（F2:32）も indexer 系 module を含める。
- F2:41 成果物: 既存 `exl3_mgemm`（`exl3_gemm.cu:370-410`）との差分報告を REPORT-F2 に必須化。

## 参照

- `orders/F2-dense-launch-fusion.md`（検分対象）
- pack: `/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6/` の `config.json`（:13,:16-17,:22,:26,:28,:32-34、`quantization_config.non_routed_exl3.layers`）と `model-dense-exl3.safetensors` ヘッダ（layers.10）
- `prof-graph-0904/profiler_out_0.txt`（:5-10,:18-19,:106）
- `orders/REPORT-F1-opus.md`（:145-146 帯域実測）
- `orders/REVIEW-F1-sol.md`（:36-38,:44-48 ABI 正典）
- `orders/T5-vllm-side-reading.md`（:5-11 module 分類）
- `vllm-exl3/src/vllm_exl3/exl3.py`（:62-66,:74-87,:1056-1063,:1236-1252,:1315-1322,:1486-1509）
- `exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu`（:28-33,:178,:370-410,:467-479）、
  `exl3_gemm_kernel.cuh`（:36-49,:88-310）、`exl3_kernel_map.cuh`（:7-15）
