# 発注 F2 v2: オオタニ専用 密 EXL3（K=6）decode の launch 融合（フェラーリ二本目）

発注者: YUKI（Lna-Lab）／決裁: ケン 2026-09-04「フェラーリで行こう。職人が 1 台に 1 人」「CUDA graph を入れても速度が変わらない状態を目指す」
監督: GLM-L。職人: GLM-F（1 名）。F1/F3（routed experts）とは別の職人・別のファイル。
`csrc/lna_moe_decode.*`・`csrc/lna_moe_ticket.*`・`exl3.py` の routed 経路には触らない。

**v2 の位置づけ**: v1（2026-09-04 08:15）に対する GLM 5 本の検分（overall / bandwidth / kernel / integration / gates）と、
軍師による突き合わせ（`orders/F2-SYNTHESIS.md`）を織り込んだ改訂。**v1 の「いまの数字」は launch 数・byte・
上限利得のすべてが誤りだったので、この v2 が唯一の基準である**。v1 の数字を引用しないこと。

**v1 からの最大の変更**: 新規 CUDA カーネルを書く発注ではなくなった。家の `exl3_mgemm` が本発注の
基準案そのものを既に実装しており、上流 exllamav3 の DSV4 実装がグループ (a)(c) を実戦で使っている。
F2 の本体は **plugin 側の wiring と門**である。

---

## 0. この一台（形は固定してよい）

- モデル: DeepSeek-V4-Flash-Vision 305B、pack `DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`。
- 機械: RTX PRO 2000 16 GB × 8（sm_120）、TP8、クロック 2600 固定・62 W。
  **SM 数 34（MEASURED、F3 実測 `f1/REPORT-F3.md`）**、L2 32 MB。
  cold read 帯域 **264〜275 GB/s（MEASURED、F1 職人の microbench 09-04、`REPORT-F1-opus.md:145-146`。以下 270 を代表値に使う）**。
- decode の行数 m = 4（1 流）〜 16（4 流）。**給仕の graph capture sizes は `[1,2,4]`（MEASURED、`serve-dsv4-tp8.sh:17`）
  なので、4 流 decode の m=8..16 は毎ステップ eager である**（integration 検分 §2.2）。これは設計の前提。
- 前提の環境: `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`・`LNA_DSV4_AUX_STREAMS=0`（単一メインストリーム）。

### 0.1 密線形の実形（MEASURED — 軍師が pack の safetensors ヘッダを直読み、2026-09-04）

v1 の表も、bandwidth 検分の表も、どちらも一部が誤っていた。**正しい形は以下**。
compressor は層によって幅が違う（偶数層 1024・奇数層 512）——これが両者の食い違いの正体である。

| 名 | 保存形 in→out | 層数 | TP 後の rank 局所形 | 1 launch の byte |
|---|---|---:|---|---:|
| attn.wq_a（disable_tp＝複製） | 4096→1024 | 43 | 4096→1024 | 3.156 MB |
| attn.wkv（複製） | 4096→512 | 43 | 4096→512 | 1.582 MB |
| attn.compressor.wkv / wgate（複製）**偶数層** | 4096→**1024** ×2 | **21**（2,4..42） | 同左 | 3.156 MB ×2 |
| attn.compressor.wkv / wgate（複製）**奇数層** | 4096→**512** ×2 | **20**（3,5..41） | 同左 | 1.582 MB ×2 |
| attn.indexer.compressor.wkv / wgate（複製） | 4096→256 ×2 | 21（2,4..42） | 同左 | 0.795 MB ×2 |
| attn.indexer.wq_b（Replicated） | 1024→8192 | 21（2,4..42） | 同左 | 6.310 MB |
| attn.wq_b（列/8） | 1024→32768 | 43 | 1024→4096 | 3.156 MB |
| attn.wo_a.slice.{rank}（rank 局所） | 4096→1024 ×8 | 43 | 4096→1024 | 3.156 MB |
| attn.wo_b（行/8） | 8192→4096 | 43 | 1024→4096 | 3.156 MB |
| ffn.shared_experts.w1 / w3（列/8） | 4096→2048 ×2 | 43 | 4096→256 ×2 | 0.795 MB ×2 |
| ffn.shared_experts.w2（行/8） | 2048→4096 | 43 | 256→4096 | 0.795 MB |

すべて K=6・mcg・trellis 幅 96（MEASURED）。
**v1 の「TP8 で列/行を 1/8 に分割して各 rank が持つ」は誤り**: wq_a・wkv・compressor・indexer 系は
`disable_tp` / `ReplicatedLinear` で 8 分割されない（`exl3.py:78-85`）。
**v1 の「compressor 4096→512 ×2」は 20 層でだけ正しく、21 層では 1024**。
**v1 の表に indexer 系 3 本（21 層）が丸ごと欠けていた**。

---

## 1. いまの数字（MEASURED、rank0、graph ON、`prof-graph-0904/profiler_out_0.txt`）

### 1.1 ステップ数の確定 — **24 ステップ（MEASURED）**

v1 と overall 検分・kernel 検分は 46 ステップ、bandwidth 検分は 24 ステップとした。**24 が正しい。**
根拠（軍師が profiler 表で検算）:

- `exl3_moe_kernel<2,…>` **1,032 回**（`:5`）、`exl3_moe_kernel<3,…>` **72 回**（`:18`）。
  主モデルは 43 層、うち K3 は 3 層。1032 = **43 × 24**、72 = **3 × 24**。
  46 は 1032 も 72 も割り切らない（gcd(1032,72)=24）。**46 はステップ数ではありえない。**
- tile 別 launch を 24 で割ると 340.8 / 127.2 / 20.1 となり、integration 検分が層内トレースから
  読んだ tile 帰属（tile128 = wq_a/wkv/compressor/wo_a/shared、tile256 = wq_b/wo_b/w2、
  tile512 = indexer.wq_b ≈ 21/層集合）と整合する。46 では 10.5/step 等の非整数になり帰属不能。
- `execute_context_0(0)_generation_1(4)` の 46 回（`:4`）は CPU 側の span 数であって step 数ではない
  （CUDA total 1.261 s > self CUDA 合計 862.2 ms で重複計上されている）。**ここが 3 本の検分の誤りの発生源。**

### 1.2 launch と時間

| kernel | 回数 | 平均 | 合計 | Self CUDA % |
|---|---:|---:|---:|---:|
| `exl3_gemm_kernel<6,true,1,16,32,128,4,3>` | 8,179 | 17.971 µs | 146.985 ms | 17.05% |
| `exl3_gemm_kernel<6,true,1,16,32,256,4,3>` | 3,053 | 16.753 µs | 51.148 ms | 5.93% |
| `exl3_gemm_kernel<6,true,1,16,16,512,4,3>` | 483 | 31.296 µs | 15.116 ms | 1.75% |
| 計 | 11,715 | 18.19 µs | **213.249 ms** | **24.73%** |

- **1 ステップ = 489 launch（11.4/層）**。これは §0.1 の構成表から算術で出る数
  （2 層×8 + 20 層×10 + 21 層×13 = 489）と厳密に一致し、11,715 = 24×489 − 21 と合う
  （−21 は profiling 窓の端で indexer.wq_b が 1 ステップ分欠けたもの。tile512 483 = 21×23）。
  **v1 の「≈255 launch（≈6/層）」は誤り。この数と整合する構成はリポジトリから再現できない。**
- **1 ステップの密 EXL3 時間 = 213.249/24 = 8.885 ms = 206.6 µs/層**（MEASURED）。
  v1 の「≈107 µs/層」は誤り（255 launch を前提に導いたもの）。
- 内訳比（MEASURED）: 密 EXL3 24.7%、routed MoE 31.0%、NCCL 17.6%。**v1 の 25/31/18 はそのまま正しい。**
- 1 launch の重みは **0.795〜6.310 MB**（v1 の「0.8〜3.1 MB」は上端が誤り）。

### 1.3 byte の家計簿（MEASURED 形状からの計算。ESTIMATE は TP 規則の適用のみ）

| 層種 | 層数 | launch/層 | rank 局所 byte/層 | 270 GB/s の床 |
|---|---:|---:|---:|---:|
| 層 0,1（compressor 無し） | 2 | 8 | 16.591 MB | 61.4 µs |
| 奇数層 3..41（compressor 512） | 20 | 10 | 19.755 MB | 73.2 µs |
| 偶数層 2..42（compressor 1024 ＋ indexer） | 21 | 13 | 30.803 MB | 114.1 µs |
| **1 ステップ計 / 43 層平均** | 43 | **489** | **1,075.2 MB**（平均 25.00 MB/層） | **3.982 ms（92.6 µs/層）** |

床の幅: 264 GB/s → 4.073 ms、275 GB/s → 3.910 ms。
床は「各 rank が重みを 1 ステップに 1 回だけ読む」限界。活性（A / A_had / C）の DRAM traffic は含まない
（L2 32 MB に載る想定。m=16 では A_had slab が数 100 KB 級で誤差項、ESTIMATE）。

**この表が bandwidth 検分の 1,138.1 MB を置き換える**（同検分は compressor を 41 層すべて 1024 と読んだため
20 層分 ×3.164 MB = 63 MB を過大に積んでいた）。

### 1.4 いまの帯域効率（MEASURED byte ÷ MEASURED 時間）

- 全体: 1,075.2 MB ÷ 8.885 ms = **121 GB/s（270 の 45%）**。
  **v1 の「帯域効率 ≈ 6 割」は 3.156 MB クラス 1 本にだけ成り立つ話で、全体では 45%。**
- クラス別（時間の帰属は tile からの ESTIMATE）: 3.156 MB 級 ≈175 GB/s（65%）、
  6.310 MB 級 ≈202 GB/s（75%）、1.582 MB 級 ≈88 GB/s（33%）、**0.795 MB 級 ≈47 GB/s（18%）**。
- **束ねの主効果は帯域効率の改善ではなく、小行列の launch 固定費と SM 飢餓の解消**である。
  0.795 MB 級は n=256（TILE_N=128 なら n タイル 2 個）で 34 SM を全く埋められない。
  byte の 12% しか占めないのに時間の 33% を食っている。ここが最大の泉。

### 1.5 上限（MEASURED から導く ESTIMATE）

- 密セグメント上限 = 8.885 / 3.982 = **2.23×**。
- 密が step の 24.7% なので、床まで行った場合の step 短縮 = 0.753 + 0.247/2.23 → **−13.6%**。
- **これは到達不能な上限**であり、合否基準に使わない。F3 の実測（`f1/REPORT-F3.md` 門 2）が示すとおり、
  メモリ経路はカードの天井に張り付いても trellis decode が天井を作る。
  ただし K=6 は 0.75 B/weight で、K=2（0.25 B/weight）に比べ 1 byte あたりの decode 仕事が 1/3 なので、
  **K=6 の密は K=2 の routed より帯域律速に近い**（3.156 MB 級が既に 175 GB/s を出していることが傍証）。

---

## 2. 現実的な利得（ESTIMATE。ここが合否の土台）

### 2.1 束ねられる組（同一入力）

| 組 | 構成員 | 層数 | launch | rank 局所 byte | 現状の実測時間（ESTIMATE 帰属） | 融合後の床 |
|---|---|---:|---|---:|---:|---:|
| **(a) 偶数層 h-fan** | wq_a + wkv + comp.wkv + comp.wgate + idx.comp.wkv + idx.comp.wgate | 21 | 6→1 | 12.640 MB | ≈105 µs/層 | 46.8 µs |
| **(a) 奇数層 h-fan** | wq_a + wkv + comp.wkv + comp.wgate | 20 | 4→1 | 7.902 MB | ≈72 µs/層 | 29.3 µs |
| **(a) 層 0,1 h-fan** | wq_a + wkv | 2 | 2→1 | 4.738 MB | ≈36 µs/層 | 17.5 µs |
| **(b) h′-fan** | shared w1 + w3 | 43 | 2→1 | 1.590 MB | ≈33.5 µs/層 | 5.9 µs |
| **(c) q_a-fan** | wq_b + indexer.wq_b | 21 | 2→1 | 9.466 MB | ≈49 µs/層 | 35.1 µs |

**(c) は v1 に無かった組**（indexer が表から欠けていたため）。両者とも q_a の出力を食う。上流もこれを束ねている（§3.1）。
残る単独 launch は wo_a・wo_b・w2 で、同入力の相手がいないので束ね対象外。

launch 数の見通し（ESTIMATE）:

- 現状 **489/step（11.4/層）**
- (a)+(b) 後: 489 − 167 − 43 = **279/step（6.5/層）**
- (a)+(b)+(c) 後: **258/step（6.0/層）**
- **v1 の「6/層 → 2〜3/層」は到達不能**。2〜3/層は依存対の融合（wq_a→wq_b、wo_a→wo_b、w1/w3→act→w2）を
  cooperative の grid.sync で 2 段にする「追加最適化」まで行って初めて意味を持つ数で、本発注の scope 外。
  **正しい目標は 489 → 260 前後（−47%）**。

### 2.2 step 利得の見積り（ESTIMATE、幅で書く）

融合後の 1 launch が出せる帯域を、現状の最良クラス（202 GB/s、n が広い場合）から
狭い n の悲観値（≈100 GB/s）までの幅で置いて積む。

| 組 | 節減/step（悲観〜楽観） |
|---|---:|
| (a) 偶数 21 層 | 0.70 〜 1.01 ms |
| (a) 奇数 20 層 | 0.38 〜 0.64 ms |
| (b) 43 層 | 0.76 〜 0.98 ms |
| (c) 21 層 | 0.04 〜 0.17 ms |
| **計** | **1.9 〜 2.8 ms** |

- 密セグメント 8.885 → **6.1〜7.0 ms、1.27〜1.46×**
- step 短縮 = 0.247 × (1.9〜2.8)/8.885 → **−5.3% 〜 −7.8%**
- **加えて aten 系の削減**: 層頭で h→fp16 の cast が 3 回、cat が 3 回走っている（MEASURED、
  integration 検分 §1.2 の層内トレース、各 0.8〜2.1 µs）。束ねれば cast 1 回・cat 0 回。
  ESTIMATE で step の **−1〜1.5%** 追加。
- bandwidth 検分は同じ scope を −3〜6% と見た。軍師の幅と重ねた**採用値は step −4% 〜 −8%（ESTIMATE）**。

**正直に書く**: これはステップの数 % の話である。1 流 decode 62 tok/s は −6% で ≈66 tok/s、
4 流 136 tok/s は ≈145 tok/s。倍にはならない。それでも取りに行く理由は §2.3。

### 2.3 それでも取りに行く理由

1. **工数が小さい**。§3 のとおり新規カーネルが要らない。plugin の wiring と門が本体。
2. **既定 OFF の flag で撤退が汚くない**。落ちた道として REPORT に残せば疵にならない。
3. **NCCL の粒揃え**は v1 が「効く」と書いたが、これは**仮説であって計測項目に格下げする**。
   h-fan の重みは全 rank 同形の複製なので rank 間分散が減る方向ではあるが、量の裏付けは無い。
   8 rank の P50/P95 と最遅 rank 差で判定する（門 5）。
4. F1/F3 で作った計測の型（同条件 A/B・T_floor・pressure）がそのまま使える。

---

## 3. 設計

### 3.1 **新規 CUDA カーネルは書かない**（v1 の基準案 3 の撤回）

v1 は `csrc/lna_dense_group.cu(.cuh)` を成果物にしていた。**これは不要である。**

家の ext には既に `exl3_mgemm_kernel` + host `exl3_mgemm_gr` + binding `exl3_mgemm`
（`exllamav3_ext/bindings.cpp:161`）があり、v1 の基準案 3 が要求したもの——
「K=6 復号をそのまま使い、複数の (trellis, suh, svh, out) の表を受けて CTA を行列×出力タイルに割る」——
を丸ごと実装している:

- B/suh/svh は int64 のポインタ表、`bszm_in==1` で A を全行列へ broadcast（`exl3_gemm_kernel.cuh:171`）。
- 行列ごとの出力幅と出力先は `size_n_list` + `c_ptrs`（`exl3_gemm.cu:464-477`、kernel 側 `:194-204`）。
- 行列ごとに suh/svh を独立適用（入力 Had `:165-183`、出力 Had `:227-259`）。**ABI 正典（`REVIEW-F1-sol.md` §1.2）と整合。**
- sm_120 では per-z の sense 反転 `group_barrier`（`ptx.cuh:319-351`）で行列間は直列化しない。
- K=6/mcg のインスタンスはコンパイル済み（`comp_units/exl3_comp_unit_6_cb1.cu`）。
- K=6 に GEMV 経路は存在しない（`exl3_gemv.cu:49` が K<2||K>4 で弾く）ので、束ねによる経路分岐のリスクは無い。

しかも**上流 exllamav3 の DSV4 実装が、まさにグループ (a) と (c) を同じ API で動かしている**
（軍師が実読みで確認）:

```
exllamav3-src/exllamav3/modules/dsv4.py:914-957   _build_x_fan()
  "q_a / wkv / comp wkv+wgate / idx wkv+wgate as ONE per-matrix-N exl3_mgemm"
  self.q_fan = mk_fan([self.q_b, self.idx_wq_b])        ← まさに組 (c)
exllamav3-src/exllamav3/modules/dsv4.py:1445-1470  rows <= 32 でゲートして ext.exl3_mgemm を呼ぶ
```

`mk_fan` が課している不変条件は 3 つで、そのまま我々の登録時検査になる:
- `len({(i.K, i.mcg, i.mul1) for i in inner}) != 1` → グループ内で K/mcg/mul1 が一律（`dsv4.py:930`）
- `max(ns) != ns[0]` → **先頭の出力が最幅**（dtype/幅のキャリア、`dsv4.py:933-935`）
- `l.out_features == l.out_features_unpadded` → padding 無し

**上流が (b)（shared w1/w3）を束ねていない**のがうちの取り分である。

したがって F2 の実装は次のどちらか:
- (i) plugin から binding `exl3_mgemm` を直叩き（呼び出し形は `dsv4.py:1456-1464` が手本）、または
- (ii) `csrc/lna_dense_group.cu` は**薄い wrapper 数十行**（§3.4 の検査と autotune の塩のためだけ）。

新規カーネルを書く選択は、REPORT で数字（mgemm の A_had slab traffic・grid.z の不均衡・lock 区画）を
示して正当化できたときだけ。**それが出るまで書かない。**

### 3.2 段階投入（integration 検分の T0/T1 を採用）

| 段 | 内容 | 状態 | launch/step | 危険 |
|---|---|---|---:|---|
| **T0** | `apply` / patch helper 内の**同一 call 内 shard 融合のみ**: fused_wqa_wkv 2→1、compressor 2→1、indexer.compressor 2→1、shared gate_up 2→1 | 無状態・順序仮定なし・単一 stream で自明 | 489 → **341** | ほぼ無し |
| **T1** | T0 ＋ load 時に静的登録した層単位 h-fan（最大 6 行列）と q_a-fan（2 行列）＋ owner trigger の遅延束ね | 遅延束ねの安全形（§3.3） | → **258** | 中（eager 試験で抑える） |
| T2 | model patch で単一 Module 化 | 本発注の枠外。長期形として記録のみ | 同 258 | 低 |

**T0 を先に、単独で出す。** T0 だけで launch −148/step（−30%）が状態を一切持たずに取れる。
そして (b) は `exl3.py:1490-1509` の shard loop の内側なので T0 に含まれる——
**v1 が (b) を「遅延束ね」扱いしていたのは誤り。(b) に遅延束ねは要らない。**
遅延束ねが要るのは h-fan（(a)）と q_a-fan（(c)）だけで、これらは vLLM 側で別 module にまたがる。

### 3.3 遅延束ねと graph capture の安全条件（必須）

capture 中は plugin の Python が一度だけ走って launch 列が焼かれ、replay は kernel だけを同じ順で再実行する。
よって「最初に来たメンバーがグループ全体を計算し、残りはキャッシュを返す」は capture/replay の両方で正しい。
**本当の危険面は graph ではなく eager 側**である（capture sizes `[1,2,4]` に対し 4 流は m=8..16 で毎ステップ eager）。

条件（全部を門に落とす）:

1. **キャッシュの有効範囲は 1 forward 内**。キーは (層 id, グループ id, `x.data_ptr()`, m) ＋ 世代カウンタ。
   forward をまたいだ参照は禁止。`data_ptr()` 単独をキーにしない（caching allocator が同一アドレスを再利用する）。
   kernel 検分の推奨どおり `id(x)` ＋弱参照を併用してよい。
2. **最初に来たメンバーが全量を計算する**（呼び出し順に依存しない）。キー不一致・メンバー欠落
   （indexer は 21 層だけ、vision/mhc 経路、draft 層）は**そのメンバーの solo 計算に fallback**。
3. **出力・A_had は load 時に確保した恒久バッファ**。`run_alloc` は毎回 `at::empty`
   （`libtorch/linear.cpp:56-70`）で capture プールに載る。`_FUSED_TEMP_CACHE`（`exl3.py:57-59`）と
   `g_tensor_cache`（`dsv4.py:979-995`）が手本。A_had は `bszm × m × k` 要素必須（`exl3_gemm.cu:477-481`、不足は無言 OOB）。
4. **ポインタ表は load 時に 1 回**（`exl3.py:287-330` が手本）。capture 中に H2D も host 同期も出さない。
5. **stream 同一性を trigger 時に検査**し、不一致は solo fallback か明示 error。
   `exl3_devctx` の lock/barrier は device 大域の共有バッファで、2 stream 同時走行は永久 spin の実績あり
   （`README.md:36-38`、`docs/PLAN.md:68`）。`LNA_DSV4_AUX_STREAMS=0` を実行条件として記録する。
6. **m > 16 は束ねない**。mgemm は m>16 を 16 行チャンクの barrier 直列で回す（`exl3_gemm_kernel.cuh:202-224`）ので
   大 m では個別並行のほうが速い可能性が高い（ESTIMATE）。上流も `rows <= 32` でゲートしている（`dsv4.py:1448`）。
   145 行超は reconstruct+hgemm の別経路（`modules/quant/exl3.py:10` の `AUTO_RECONSTRUCT_THRESHOLD = 144`）。
   **本発注は m ≤ 16 に限定し、超過は現行の個別 forward に素通しする。**
7. **prewarm をグループ呼び出しに拡張**。mgemm は capture 中 autotune を**無警告で**スキップして
   静的ヒューリスティックに落ちる（`exl3_gemm.cu:578-583`。単発 gemm 側は警告付き `:59-68,262-268`）。
   `LNA_EXL3_PREWARM_ROWS`（`exl3.py:1417-1429`）を全 m bucket × 全グループに拡張しないと、
   正しいが遅い設定が graph に焼き付く。

### 3.4 登録時の検査（TORCH_CHECK / assert。全部必須）

1. グループ内で **K / mcg / mul1 が一律**（`dsv4.py:930` と同じ）。
2. **先頭の出力が最幅**（`ns[0] == max(ns)`）。mgemm は `size_n_list` があっても `C.size(2)` を検査せず、
   破ると z スライス間で lock 範囲が重なり**無言の race** になる（`exl3_gemm.cu:464-477`、lock 範囲 `exl3_gemm_kernel.cuh:207`）。
3. 全 `n_j % tile_n == 0`（今回の幅 256/512/1024/4096 はすべて満たすが、形状選択の確認は要る）。
4. **入力が同一 tensor**（data_ptr・shape・dtype・contiguous）。不一致は fallback ＋ 1 回だけ loud log。
5. `bf16_shards` を持つモジュールは束ね対象から除外（`exl3.py:1073-1084`）。
6. **C は fp32**。mgemm を C=half で呼ぶと `write_sum_gl` に fp16 中間丸めが入り
   （`exl3_gemm_inner.cuh:544-549`）、epilogue も half 演算境界になる。
   FP32 境界の正典（`REVIEW-F1-sol.md` §1.5）を密でもそのまま守る。

### 3.5 dtype 境界は **call-site 単位**で保存する

v1 の「出力 dtype は既存と同じ（fp32 → bf16 の境界を動かさない）」は正しいが、密経路は 3 つの
呼び出し元を持ち、それぞれ境界が違う（MEASURED、integration 検分 §1.1, §3.6）:

- `Exl3LinearMethod.apply`: fp32 → `x.dtype`（bf16）へ cast（`exl3.py:1517-1518`）
- `_lna_kv_score`（compressor / indexer.compressor）: **fp32 のまま**（`patch_dsv4_attention_compressor_exl3.py:13-16`）
- `_o_proj`（wo_a / wo_b）: fp32 → bf16 → reshape（`patch_dsv4_dense_exl3.py:101-107`）

**compressor と wo_a は `apply` を通らない。** 束ね機構は plugin 側の runner に置き、
`apply` と両 patch helper の 3 か所から呼ぶ形にする（patch script の最小差し替えで「vLLM 層コード不改変」は維持できる）。

### 3.6 torch.compile 耐性（任意だが推奨）

現行給仕に compile 由来の activity は無い（MEASURED）。将来に備え、グループ GEMM は
`torch.ops` の custom op として登録し「1 op = 1 launch」の静的意味にする。遅延束ねの状態遷移は op の外（Python）に置く。

---

## 4. 門（順に。速さは最後）

**Gate 0（コード前・机上／席の 1 ステップ計測のみ）— 帳尻合わせ。**

- **eager トレースで launch → module/形状の帰属表を作る**。graph ON の replay 内 kernel は
  CPU op との correlation が切れており `prof-shapes.py` が属性付けできない（MEASURED、integration 検分 §1.4）。
  帰属は eager 1 step ＋ `record_shapes` で採る。`prof/` の 09-04 一式は消失しているので再採取。
- §0.1 の形表・§1.3 の byte 表・§1.1 の 24 ステップを**席で再確認**し、REPORT-F2 の冒頭に置く。
- capture bucket の m 全集合を確認（`[1,2,4]` の想定、`serve-dsv4-tp8.sh:17`）。
- **ここが揃わないと以降の速度の数字は無意味。**

**Gate 1（机上 parity）— 最強の捕獲器。**

- 参照は行列ごとの `LinearEXL3.forward` を個別に呼んだ結果。F1 の「エラーなしの誤り」
  （|out|/|ref| ≈ 1/√2、`REPORT-F1-opus.md:41-42`）はこの比較で抜けない。
- m ∈ {1,2,3,4,8,16}（3 は `threadblock_reduce` の size_m≤8 分岐境界、`exl3_gemm_inner.cuh:375-398`）
  ＋ capture bucket 全部。m=17 と m=145 は「束ねずに素通しする」ことの確認として走らせる。
- 実 tensor は **偶数層（6 本 fan）・奇数層（4 本 fan）・層 0 か 1（2 本 fan）**の 3 系統 ＋ 合成 tensor。
  グループ (b)(c) も同様。
- **主判定は fp32 段階で per-row rel ≤ 2e-3**、bf16 変換後は別枠（2e-3 は bf16 の丸めに埋もれる）。
  段階別（入力 Had 後・GEMM 生和・最終出力）の突き合わせを入れる。max abs と finite 一致も。
- **同一入力の二度走りで bitwise 一致**（固定順・atomics なし構造なので要求してよい）。
- **norm 比 canary**（|out|/|ref| が 1.0 から離れたら即赤）。F1 で実際に効いた捕獲器。
- **呼び出し順の入れ替え**（wkv を wq_a より先に）で fallback が正しく出ること。
- **eager キャッシュ試験**: m ∈ {1,2,4,8,16} の eager で hit と再計算の両方を踏む。
  NaN poison ＋ 入れ替え入力で 1000 回以上 replay し、出力が最後の入力に追従すること（stale 検出）。
  bucket を交互に replay しても混線しないこと。capture 前後の allocator 統計で per-replay の追加確保が無いこと。
- **統合 parity**: `apply`・`_lna_kv_score`・`_o_proj` の 3 経路それぞれで group ON/OFF の
  end-to-end 出力差分（dtype 境界込み、§3.5）。
- `compute-sanitizer`（memcheck ＋ **racecheck** ＋ synccheck ＋ initcheck）。
  ただし F1 時点で像に入っていない（MEASURED、`REPORT-F1-opus.md:134-138`）。
  **像に入れられるかを職人が最初に見極め、無理なら代替（lock 範囲の静的検査＋多重 replay）を REPORT に書く。**
- **untuned tripwire**: mgemm の捕捉中フォールバックは無警告（`exl3_gemm.cu:578`）なので、
  prewarm の網羅を検査する仕掛けを自分で入れる。

**Gate 2（同条件 baseline の A/B）— 速度の門の土台。ここが v1 に無かった。**

- 同一ビルド・同一 pack・同一席構成で `LNA_EXL3_DENSE_GROUP` **OFF→ON→OFF の交互、各 3 回以上**。
  **09-04 の過去数値と比べない**（.so 積み替えと plugin 変更で条件が変わる。baseline は当日再測）。
- **flag OFF の経路無変更確認**: OFF の kernel 名・件数が baseline と完全一致（回帰ゼロの直接証明）。
- launch の数え方: graph ON は profiler 表の kernel 名で数える。対象は `exl3_gemm_kernel<6` 全 variant ＋
  `exl3_mgemm_kernel<6` ＋ **plugin が増やした補助カーネル（cat/copy/elementwise）**。
  F1 正典の「補助 kernel が層の外に増えていない」（`REVIEW-F1-sol.md:214`）を F2 にも課す。
- 判定: 密 kernel 合計時間・launch 総数・step 時間の 3 つを実測値で報告。

**Gate 3（圧 = pressure）— グループごとの帯域を出す。**

- 融合後の各 group launch について `bytes ÷ 時間 ÷ 270 GB/s` を出し、**融合前の和と比べる**。
  **1 つでも「融合前の和より悪い」グループがあれば、そのグループは早期に切る**（REPORT に per-group 前後を必須）。
- F3 と同じく、届かない場合は compute ceiling（decode+MMA を外した同一 load 経路）を実測で示す。

**Gate 4（1.25×T_floor）— 絶対値。グループごとに。**

| 組 | T_floor（270 GB/s） | 1.25×T_floor | 現状（ESTIMATE 帰属） |
|---|---:|---:|---:|
| (a) 偶数層 12.640 MB | 46.8 µs | **58.5 µs** | ≈105 µs |
| (a) 奇数層 7.902 MB | 29.3 µs | **36.6 µs** | ≈72 µs |
| (b) 1.590 MB | 5.9 µs | **7.4 µs** | ≈33.5 µs |
| (c) 9.466 MB | 35.1 µs | **43.8 µs** | ≈49 µs |

- **(b) は 1.25×T_floor を満たさない見込みである**（n 合計 512、TILE_N=128 なら n タイル 4 個で
  34 SM を埋められない）。F3 でも門 3 の絶対値は FAIL だった。
  **(b) の拘束力ある門は Gate 2 の同条件 A/B であり、Gate 4 は届かなくても FAIL 扱いにしない。
  届かなかったら理由を数字で書く**（F3 と同じ作法）。
- (a)(c) は満たしうる。満たさないときも同じく理由を数字で。

**Gate 5（席内）— 最後。**

- `LNA_EXL3_DENSE_GROUP=1/0` を**同一プロセスで** A/B。
- graph ON/OFF で貪欲突き合わせ（基準＝同カーネル二度走りの揺れ幅 49〜118 字）、
  作文 7 本 finish=stop、ppl（4k・spec off）**6.72 ± 0.02**、受理長 ≈ 2.0/1.8/2.2。
- **eager でも落ちない**の測定式（ケンの完成の定義）:
  1. graph を切って `bench-streams.py <port> 256 1,4` を OFF/ON 交互 3 回。
     agg tok/s が OFF のばらつき帯（交互 OFF run の max−min）を下回らないこと。
  2. eager トレース（op 帰属あり）で密経路 kernel 時間の総和・launch 総数が減り、層あたりの補助 kernel が増えていないこと。
  3. **gap の報告**: (graph ON の tok/s − graph OFF の tok/s) を flag OFF と ON の両方で出し、
     **gap が縮んだ/消えたことを数字で示す**。これがフェラーリの完成の定義。
- **8 rank 全部**の P50/P95 と最遅 rank 差を 3 回以上。NCCL 粒揃え仮説はここで判定する（効果でなく計測項目）。
- in-seat selfcheck・engagement log・NCCL byte・untuned tripwire。
- 実行条件として `LNA_DSV4_AUX_STREAMS=0`・`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` を記録。

---

## 5. 数値目標と KILL LINE

すべて Gate 2 の当日 baseline に対する相対値。

| 段 | 指標 | KILL | PASS | stretch |
|---|---|---|---|---|
| **T0（shard 融合のみ）** | launch/step | — | 489 → **≤ 345** | — |
| | 密セグメント | **< 1.05×（= step −1.3% 未満）→ T1 に進まず終了** | **≥ 1.08×** | ≥ 1.15× |
| **T1（+ h-fan / q_a-fan）** | launch/step | — | → **≤ 265** | ≤ 258 |
| | 密セグメント 8.885 ms | **< 1.15×（7.73 ms 超）→ 中止、flag OFF で出荷** | **≥ 1.20×（≤ 7.40 ms）** | ≥ 1.35×（≤ 6.58 ms） |
| | step 時間 | **−3% 未満 → 中止** | **−4%** | **−7%** |
| | 4 流 agg tok/s（136 基準） | 非劣化 | **≥ 142** | ≥ 147 |
| | eager gap | 非劣化 | 縮小 | 消失 |

**KILL LINE（最重要）**: **T0 の実測で密セグメントが 1.05× に届かなければ、T1（遅延束ね）に進まない。**
T0 は状態を持たない安全な変更で、ここで効かないなら遅延束ねの危険を負う理由が無い。
所見を `REPORT-F2.md` に「落ちた道」として残して終了する。既定 OFF なので疵は残らない。

**上限として扱い、合否に使わない数**: 密 2.23×・step −13.6%（§1.5）。

---

## 6. 成果物

- `src/vllm_exl3/exl3.py`: グループ runner（`apply` と 2 つの patch helper の 3 か所から呼ぶ）、
  load 時のポインタ表構築、登録時検査（§3.4）、遅延束ねのキャッシュ（§3.3）、
  env `LNA_EXL3_DENSE_GROUP=1`（**既定 OFF**）。
- `recipe-lna/patch_dsv4_attention_compressor_exl3.py` / `patch_dsv4_dense_exl3.py`: runner を呼ぶ最小差し替え。
- **グループ×bucket 別の恒久 scratch**: A_had（≥ `bszm·m·k`）と出力バッファ。model move/reload で再構築＋graph 再 capture。
- **prewarm 拡張**: `LNA_EXL3_PREWARM_ROWS` を全 m bucket × 全グループへ。
- 机上 parity テスト一式（§4 Gate 1）。
- `csrc/lna_dense_group.cu(.cuh)`: **書くなら薄い wrapper のみ**（§3.4 の検査と autotune の塩）。
  新規カーネルを書くなら、mgemm を使わない理由を数字で REPORT に書くこと（**これが無ければ書かない**）。
- `REPORT-F2.md`: 冒頭に Gate 0 の帰属表と byte 家計簿、per-group の前後帯域、落ちた道、
  module 別 launch 数内訳（graph ON と eager、m 別）、既存 mgemm との差分報告。

## 7. 読むべきもの

**まずこれ**（本発注の核心が既に書かれている）:
- `exllamav3-src/exllamav3/modules/dsv4.py:914-957`（`_build_x_fan` = グループ (a)(c) の実装）、`:1445-1470`（呼び出し・`rows<=32` ゲート）、`:979-995`（`g_tensor_cache` のポインタ安定化）
- `exllamav3_ext/quant/exl3_gemm.cu:369-410`（mgemm ABI の doc）、`:464-481`（per-matrix 幅・A_had 検査）、`:578-612`（capture 中の無警告フォールバック）、`:59-68`
- `exllamav3_ext/quant/exl3_gemm_kernel.cuh:89-310`（mgemm kernel）、`exl3_gemm_inner.cuh`、`exl3_kernel_map.cuh`、`exl3_devctx.cuh`、`ptx.cuh:319-351`、`bindings.cpp:161`

**plugin と patch**:
- `vllm-exl3/src/vllm_exl3/exl3.py`: `apply` 1454-1532、prewarm 1417-1429、ポインタ表の手本 287-330、shard loop 1490-1509、disable_tp/Replicated 78-85、bf16_shards 1073-1084
- `recipe-lna/patch_dsv4_dense_exl3.py:19-31,63-129`、`patch_dsv4_attention_compressor_exl3.py:9-34`、`recipe/scripts/patch_dsv4_loader.py:60-63`

**正典**:
- `orders/REVIEW-F1-sol.md` §1（ABI 正典: 9 pointer・128 要素 Hadamard・suh/svh の独立・FP32 境界。密でも同じ）、§5（cooperative/stream/graph）
- `orders/F2-SYNTHESIS.md`（5 本の検分の突き合わせと、どこで誰が誤ったか）
- `f1/REPORT-F3.md` §0, §4（帯域の天井は decode 側にあるという実測。K=6 では事情が違う理由も本書 §1.5）
- `Lna-Factory/veins/dsv4-carve/PLAN.md`（家計簿・門の作法）
- `serve-dsv4-tp8.sh:11-17`、`README.md:36-38`、`docs/PLAN.md:61,66,68`

**GPU 実行の作法**: `vllm-exl3-v030-port/f1/PROTOCOL.md` と同じ型で `f2/` を使う（**GPU 11 のみ**。GPU10 は F1/F3 の職人）。
**席 :8899 は触らない。** 席内の門は監督が段取りする。
