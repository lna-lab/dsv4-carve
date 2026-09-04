# F2 密 EXL3（K=6）launch 融合 — 設計検分（vLLM/plugin 統合レンズ）

監督: GLM-L（統合担当）。2026-09-04。対象: `orders/F2-dense-launch-fusion.md`。
本検分の縛り: ビルド・GPU・席 :8899 は使わず、リポジトリ内コード・pack 設定実読み・
既存 profiler トレース（`prof-graph-0904/`、`prof/`）の CPU 解析のみ。数値には実測 / ESTIMATE を付す。
なお `prof/` のトレースは本検分の読み取り直後にディレクトリごと空になった（実測。数値は読み取り時点のもの）。

## 結論

方向（同入力の密線形を 1 launch に束ねる）は統合側から見て成立する。ただし発注書には
前提の誤りが 1 件、グループ定義の過不足が 1 件、遅延束ねの安全条件の明文化不足がある。

1. **発注書の基準線 255 launch/step（≈6/層）は両トレースと矛盾する（実測）**。
   `prof-graph-0904/` rank0 で `exl3_gemm_kernel<6,…>` 11,715 発、MoE カーネル 1,104 発
   ＝ 43 層 × 25.7 ステップ相当 → **≈456 発/step（≈10.6/層）**。`prof/`（読み取り時点）
   でも 11,232 発 → **≈437 発/step**。コード＋pack 設定からの算出（§1.3）は最大 489/step
   で、実測と整合する。255 と整合する構成は本リポジトリからは再現できない。
   **「割に合うか」の判定は再計測まで保留**。ただし修正後の基準線では利得の見込みは
   発注書の見積り（ステップの 8%）より大きい（§4）。時間側の数字（gemm<6> が 24.7%、
   平均 17.97/16.75/31.30 µs、MoE 31%、NCCL 17.6%）は発注書と一致（実測、§1.4）。
2. **同入力グループは 4 本ではない**。pack 設定（実読み）では `attn.indexer.compressor.
   fused_wkv_wgate`（21 層）も同じ層入力 h を食い、h 系は最大 **7 行列**（wq_a, wkv,
   comp.wkv, comp.wgate, indexer.comp.wkv, indexer.comp.wgate, indexer.wq_b）。逆に
   shared `w1/w3` は「別々の vLLM forward」ではなく **同一 module call 内の shard loop**
   （`vllm-exl3/src/vllm_exl3/exl3.py:1490-1509`）で、**遅延束ねなしで 2→1 にできる**。
   発注書の (a) と (b) の「どちらも遅延束ね」扱いは半分だけ正しい。
3. **「遅延束ね」は CUDA graph capture とは整合し得る**。capture 時に plugin の Python が
   一度走り、replay は録画済み kernel 列を再実行するだけなので、replay 中にキャッシュ整合
   問題は原理的に起きない（今の給仕がこの形で動いている、実測 §2.1）。ただし安全条件が
   4 つある（§2.1）。
4. **本当の危険面は graph ではなく eager 側**。給仕の capture sizes は `[1,2,4]`
   （`serve-dsv4-tp8.sh:17`、実測値）なので、4 流 decode の m=8..16 は**毎ステップ eager**
   で Python キャッシュが走る。ここでの stale hit は無言の誤値になる（§2.2）。
5. **代替案（推奨）は段階融合**。T0: module 内 shard 融合のみ（無状態・順序仮定なし）で
   148 発/step 減。T1: load 時静的登録の h-group ＋ owner trigger（遅延束ねの安全形）。
   T2: model patch で単一 Module 化（発注書の「vLLM 層コードを書き換えない」の枠外だが
   長期形、§4）。
6. **ストリーム**: 現行 production はこれら密線形がすべて単一メインストリーム（実測 §2.4、
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 既定 `serve-dsv4-tp8.sh:11`）。exllamav3 の
   per-device 共有 lock バッファ（`exl3_devctx.cuh:33-42`）と `__device__` 大域
   （`exl3_gemm_kernel.cuh:84-86`）を使う協調カーネルは 2 stream 同時走行で永久 spin の
   実績（`README.md:36-38`、`docs/PLAN.md:68`）。新カーネルは locks 不使用・`grid.sync` のみ
   （または非協調）構成を第一候補にし、グループ員の stream 単一性を assert せよ。

## 1. 呼び出し経路の地図（コード照合 + トレース実測）

### 1.1 密 EXL3 モジュールと vLLM からの呼ばれ方

pack `/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6/config.json`
の `quantization_config.non_routed_exl3.layers` を実読み（層数はエントリ数、bits=6、mcg）:

| module（mapper 後 prefix） | 層数 | vLLM 側の呼び出し元 | plugin 内の経路 | 入力 | rank あたり launch |
|---|---:|---|---|---|---:|
| `attn.fused_wqa_wkv`（wq_a+wkv、disable_tp 複製） | 43 | attention forward の QKV 投影 | `Exl3LinearMethod.apply`（`exl3.py:1454-1532`） | h | 2 |
| `attn.compressor.fused_wkv_wgate`（wkv+wgate、disable_tp） | 41 | attention.py の kv_score（patch helper） | `_lna_kv_score`（`recipe-lna/patch_dsv4_attention_compressor_exl3.py:9-18`）— **apply を通らない** | h | 2 |
| `attn.indexer.compressor.fused_wkv_wgate` | 21 | attention.py の indexer kv_score（同上） | 同上 | h | 2 |
| `attn.indexer.wq_b`（ReplicatedLinear） | 21 | indexer forward | `Exl3LinearMethod.apply` | h（要確認、ESTIMATE） | 1 |
| `attn.wq_b` | 43 | q_a 正規化後 | `Exl3LinearMethod.apply` | q_a（依存） | 1 |
| `attn.wo_a`（rank 局所 1 group） | 43 | `_o_proj`（`recipe-lna/patch_dsv4_dense_exl3.py:96-107`）— **apply を通らない** | `exl3_linears[0].forward(...)` 直叩き | inv-RoPE(o)（依存） | 1 |
| `attn.wo_b` | 43 | `_o_proj` の後段 | `Exl3LinearMethod.apply` | z（依存） | 1 |
| `ffn.shared_experts.gate_up_proj`（w1+w3） | 43 | SharedExperts runner | `Exl3LinearMethod.apply`（shard loop 1490-1509） | h′（注意後） | 2 |
| `ffn.shared_experts.down_proj`（w2） | 43 | 同上 | 同上 | act（依存） | 1 |

呼び出し元の根拠: compressor の quant_config は patch で通している
（`recipe-lna/patch_dsv4_dense_exl3.py:19-31`）、wo_a/wo_b は `_o_proj` の EXL3 分岐
（同 63-129。`z = z.to(torch.bfloat16)` が 106 行目、`return self.wo_b(z.flatten(1))` が 107 行目）、
kv_score の 2 個所（attention compressor / indexer compressor）は
`recipe-lna/patch_dsv4_attention_compressor_exl3.py:21-34` の anchor。
wq_a/wkv が vLLM 側で既に 1 module（`fused_wqa_wkv`）に融合されていることは
`recipe/scripts/patch_dsv4_loader.py:60-63` のリマップ表でも確認できる。

### 1.2 層内の実行順序（実測、`prof-graph-0904/` rank0、compressor+indexer 層の capture 時窓）

stream 上の kernel 列（gemm<6> と主要 kernel のみ、m≈16 の capture 時):

```
[前層の NCCL-AR]
mhc_fused 21µs, mhc_pre_big_fuse_with_norm 3.8µs        ← compressor 前段（bf16/tilelang）
copy(h→fp16) 2.1µs
GEMM6[t128] 21.9µs + GEMM6[t128] 18.3µs                  ← fused_wqa_wkv（wq_a, wkv）: 1 cast × 2 launch
cat 1.0µs, copy(fp32→bf16) 0.9µs
copy(h→fp16) 1.8µs
GEMM6[t128] 22.2µs + GEMM6[t128] 21.4µs                  ← compressor.fused_wkv_wgate（wkv, wgate）: もう 1 cast
cat, copy
bf16 GEMM 4.1µs + splitK 1.3µs                           ← bf16 のままの小線形（weights_proj 系、ESTIMATE）
copy(h→fp16) 1.9µs
GEMM6[t128] 17.4µs + GEMM6[t128] 17.9µs                  ← indexer.compressor（wkv, wgate）: 3 個目の cast
cat, copy
_fused_q_kv_rmsnorm → copy → GEMM6[t256] 21.4µs          ← wq_b
[sparse_mla attention 一式] → _inverse_rope_gptj
copy(o→fp16) → GEMM6[t128] 22.4µs                        ← wo_a（_o_proj 直叩き経路）
copy → GEMM6[t256] 21.6µs                                ← wo_b
NCCL-AR
mhc_fused, mhc_pre, bf16 GEMM+splitK
copy(h′→fp16) → GEMM6[t128] 17.7 + GEMM6[t128] 17.6      ← shared w1, w3（1 module call × 2 shard）
cat, copy, act_and_mul, copy
GEMM6[t128] 10.2µs                                       ← shared w2
topk 一式 → exl3_moe_kernel → add → NCCL-AR
```

統合上の要点（実測）:
- **h 系の 3 module call は層頭で連続しており、間に依存 kernel が無い**。束ねた 1 launch を
  `fused_wqa_wkv` の位置に置いても実行タイミングを後ろにずらす必要がない。
- **同一 h の fp16 変換が module call ごとに 3 回走っている**（各 ~2µs）+ cat 3 回（各 ~1µs）。
  グループ化で 1 cast ・ 0 cat にできる。
- **shared experts は routed MoE の前にメインストリームで直列実行**（別ストリーム無効の確認）。
- wo_a→wo_b、wq_a→wq_b、w1/w3→act→w2 の依存対は発注書 2 のとおり直列（実測とも一致）。

### 1.3 launch 数の算術（コード + pack 設定から、ESTIMATE だが実測と整合）

- 基本（全 43 層）: fused_wqa_wkv 2 + wq_b 1 + wo_a 1 + wo_b 1 + gate_up 2 + down 1 = 8 → 344
- compressor 41 層 × 2 = 82
- indexer 21 層 × 3 = 63
- 計 **489/step**（indexer 全部が走る場合）。tile 別実測（tile128 335/tile256 127/tile512 20 per step）
  は wqa+comp+wo_a+shared が tile128、wq_b+wo_b+w2 が tile256、indexer.wq_b が tile512、
  と整合する（ESTIMATE の帰属）。
- 実測 437–456/step は「基本 + compressor（426）」と「+indexer 全部（489）」の間 →
  indexer 系は一部のステップ/層のみ（tile512 が ~20/step ≈ 21 層 × 大半のステップ、ESTIMATE）。
- **発注書の ≈255（≈6/層）はどの組合わせでも再現できない**（43×5+41×2=297、
  43×4+41×2=254 に近いが wq_b か wo_b のどちらかを EXL3 外にする必要があり pack 設定と矛盾）。

### 1.4 時間側の実測（`prof-graph-0904/profiler_out_0.txt`、rank0、graph ON）

- `exl3_gemm_kernel<6>`: tile128 8,179 発 × 17.97 µs、tile256 3,053 発 × 16.75 µs、
  tile512 483 発 × 31.30 µs。合計 213.2 ms = **self-CUDA の 24.7%**（発注書の 25% と一致）。
- `exl3_moe_kernel` 31.0%、NCCL（AR+AG）17.6%（発注書の 31%/18% と一致）。
- decode 1 step = `execute_context_0` 46 回 × 平均 27.42 ms（実測）。
- **graph モードでは replay 内 kernel に CPU op の correlation が無く `prof-shapes.py` は
  属性付けできない**（実測: gemm 行がすべて "?"）。launch 数の再計測は capture 時の列か
  eager 1 step + record_shapes で行うこと（修正要求 1）。
- torch.compile / inductor 由来の kernel 名・frame はトレースに存在しない（実測）。
  給仕の `--compilation-config` は capture sizes の指定のみ（`serve-dsv4-tp8.sh:17`）。

## 2. 「遅延束ね」と capture / compile / 複数 stream の整合

### 2.1 CUDA graph capture — 整合する。ただし条件が 4 つ

機構: vLLM の capture で plugin の Python は普通に走り、出力 tensor は capture 用プールから
割り当てられ、そのアドレスが下流 node に焼き付く。replay は Python を実行しないので、
「最初の呼び出しで束ねて残りはキャッシュ」のキャッシュが replay 中に参照されることはない。
今の給仕がまさにこの形（eager な plugin Python を capture に通している）で動作している（実測 §1.4）。

条件:
1. **キャッシュの有効範囲は 1 forward 内**。キーは (x.data_ptr(), m) + 層ごとの group 状態。
   capture はサイズごと（[1,2,4]）に行われ、別 capture で作られた tensor への古い参照を
   返すと誤アドレスを焼き付ける。forward を跨いだ参照は禁止（メンバーの solo 計算へ fallback）。
   h は層 forward 中ずっと生存しているので、層内の data_ptr 一致はエイリアスに対して健全。
2. **capture 中の host 同期・H2D・autotune を出さない**。pointer 表は load 時に一度
   （`build_exl3_fused_state` 型、`exl3.py:287-330` が手本）。捕集中に未 tune の形で
   autotuner に入ると全 rank 死亡の実績（`docs/PLAN.md:66`、`exl3_gemm.cu:59-64,262-268` の
   capture guard は既存）。**グループ形状の prewarm** を `LNA_EXL3_PREWARM_ROWS` 機構
   （`exl3.py:1417-1428`）に追加しないと、capture が静的ヒューリスティックに落ちる
   （正しさは保たれるが遅い）。
3. **出力バッファは capture プールに per-call 割り当て、または load 時に確保した
   グループ専用の定置バッファ**。`run_alloc` は毎回 `at::empty`（`linear.cpp:56-70`）、
   m>1 はさらに `at::empty_like(x)` の A_had を毎回（`linear.cpp:43-45`）。グループ scratch は
   「行列 × m × k」の slab を要する（`exl3_gemm.cu:477-481` の `A_had` 検査が手本）。
4. **メンバー欠落への寛容**: indexer 有無（21 層のみ）、vision/mhc 経路、draft（層 43-45 は
   dense 非 EXL3、`non_routed_dtype_policy: bf16_as_stored`）で呼び出し集合が変わっても、
   trigger は「グループ最初の呼び出し」に限定せず最初に来たメンバーが全量計算、
   キー不一致のメンバーは solo 計算、で破綻しない。

### 2.2 eager — 本来の危険面

- capture sizes `[1,2,4]`（実測）に対し 4 流 decode は m=8..16 → **毎ステップ eager**。
  発注書の門 3「graph OFF（eager）でも落ちない」は希望ではなく 4 流の常態。
- eager ではキャッシュ論理が毎ステップ走る。§2.1-1 の forward 内キーと solo fallback に
  加え、**テストは m ∈ {1,2,4,8,16} の eager でキャッシュ命中・再計算の両方を踏む**こと
  （同一 forward 内で 2 度呼ばれるメンバーは無いので、hit は「2 番目以降の module call」のみ）。
- 17..144 行は gemm 経路のまま（`exllamav3-src/exllamav3/modules/quant/exl3.py:10,135-139` の
  `AUTO_RECONSTRUCT_THRESHOLD = 144`）、145 行超は reconstruct+hgemm に別経路。グループ化は
  **m ≤ 16 に限定**し、17 以上は現状個別のままにするのが安全（発注書に明記を要求）。

### 2.3 torch.compile

現行給仕では compile 由来の activity が無い（実測 §1.4）。将来 `-O3` 等で有効化した場合、
plugin の Python（apply / patch helper）は今でも compile 領域の外（graph break 扱い）であり、
遅延束ねはそれを悪化させないが、この領域の将来の compile を妨げる。**グループ GEMM を
`torch.ops` の custom op として登録し、適用は 1 op = 1 launch の静的意味にする**設計が
compile 耐性のある形。遅延束ねの状態遷移は custom op の外（Python）に置くこと。

### 2.4 複数 stream

- 実績: 共有 per-device lock バッファ（`exl3_devctx.cuh:33-42`、`BARRIER_LOCKS_OFFSET` 9 行目、
  MoE scheduler も同バッファ 14-16 行目）を 2 stream から同時に叩くと永久 spin
  （`README.md:36-38`、真因特定 `docs/PLAN.md:68`）。`mgemm` は `__device__` 大域
  `v_indices/v_weights/bszm_sync`（`exl3_gemm_kernel.cuh:84-86`）も持つ。
  さらに「per-linear scratch は共有キャッシュだと stream 間 race」との修正が既に入っている
  （`exllamav3-src/exllamav3/modules/quant/exl3.py:89` の LNA-LAB コメント）。
- 遅延束ねはこの上に**新しい形の危険**を足す: 最初の呼び出し側の stream で計算し、別 stream の
  メンバーが依存边なしでキャッシュを受け取ると race。生産では単一 stream（実測 §1.2、
  `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 既定）だが、`LNA_DSV4_AUX_STREAMS` /
  `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`（`serve-dsv4-tp8.sh:12`、`docs/PLAN.md:61`）で
  再有効化できる。**グループ trigger 時に `torch.cuda.current_stream()` の同一性を検査し、
  不一致は solo fallback（または明示 error）**にすること。
- カーネル設計としては、locks 不使用・`grid.sync()` のみ（単発 m≤16 なら入力 Hadamard を
  CTA 内で済ませる非協調構成も可。`exl3_gemm_inner.cuh:58` の `TILESIZE_M == 16` 前提と
  `exl3_gemm_kernel.cuh:29,49` の同期位置が参照形）を推奨する。F1 検分
  （`orders/REVIEW-F1-sol.md` §5）の cooperative/graph の注意点は密でもそのまま成立。

## 3. 危険点（門に反映すべきもの）

1. **基準線の不一致**（§1.3）: 255 vs 実済 437-456。利得見積りの分母が違う。
2. **eager キャッシュ**（§2.2）: m=8..16 が常時 eager。stale hit は無言の誤値。
   per-forward キー + solo fallback + eager m 全数テストを必須に。
3. **capture 間のキャッシュ無効化**（§2.1-1）: forward を跨ぐ tensor 参照の禁止。
4. **グループ形状の prewarm**（§2.1-2）: 無いと capture が静的ヒューリスティックに落ちる。
   `LNA_EXL3_PREWARM_ROWS` 機構への追加と、`EXLLAMAV3_TUNE_CACHE`（`serve-dsv4-tp8.sh:12`）
   の初回 tuning 1 回で済むことの確認。
5. **適用外の 2 経路**（§1.1）: compressor（`_lna_kv_score`）と wo_a（`_o_proj`）は
   `Exl3LinearMethod.apply` を通らない。束ね機構は plugin 側の runner に置き、apply と
   両 patch helper から呼ぶ形にする（patch script の最小差し替えで「層コード不改変」は維持可）。
   wo_a 自身は同入力の相手がいないので束ね対象外（発注書の表と同じ）。
6. **dtype 境界は呼び出し元ごとに保存**: apply は fp32 → x.dtype（bf16）キャスト
   （`exl3.py:1517-1518`）、kv_score は fp32 のまま（`patch_dsv4_attention_compressor_exl3.py:13-16`）、
   `_o_proj` は fp32 → bf16 → reshape → wo_b（`patch_dsv4_dense_exl3.py:101-107`）。
   発注書 5「出力 dtype は既存と同じ」は **call-site 単位**で要件化すること
   （REVIEW-F1-sol §1.5 の FP32 境界の正典を密に引き継ぐ）。
7. **ポインタ表と lifetime**: load 時 1 回構築・`LinearEXL3` の tensor 参照を保持
   （`process_weights_after_loading` が元 param を削除するため、`exl3.py:1449-1457` 以降は
   inner が唯一の所有者）。graph 再構築・reload 後の stale も REVIEW-F1-sol §5 と同様に。
8. **byte 下限の見積りに indexer が抜けている**: 発注書の 19 MB/層（ESTIMATE、
   0.75 byte/weight で wq_a 3.1 + wkv 1.6 + comp 3.2 + wq_b 3.1 + wo_a 3.1 + wo_b 3.1 +
   shared 2.4 ≈ 19.6 MB、妥当）は indexer 系（21 層）を含まない。indexer.wq_b は
   Replicated（全幅）なので 21 層の下限はこれより上がる（ESTIMATE、形状要確認）。
9. **NCCL「粒揃え」効果は ESTIMATE のまま**: h-group は disable_tp 複製重みで全 rank 同形
   （rank 間分散が消える）ため分散低下の方向は妥当だが、量の裏付けは今回取れていない。

## 4. 代替案 — 段階融合（推奨）

| 段 | 内容 | 状態 | launch/step（ESTIMATE、§1.3 の算術に基づく） | 危険度 |
|---|---|---|---|---|
| 現状 | shard 別個別 launch + module 別 cast/cat | — | ≈437–456（実測） | — |
| **T0** | `apply` 内 shard 融合のみ: fused_wqa_wkv 2→1、comp 2→1、indexer comp 2→1、gate_up 2→1。kernel は既存 `exl3_mgemm` の broadcast（1 入力・多出力・行列別 suh/svh、`exl3_gemm.cu:386`、`exl3_gemm_kernel.cuh:170-181`）か新カーネル | 無状態・順序仮定なし・stream 1 本で自明 | −148（43+41+21+43）→ ≈300 付近 | ほぼ無し |
| **T1** | T0 + load 時静的登録の層単位 h-group（最大 7 行列）+ owner trigger。キャッシュは forward 内キー、solo fallback 付き | 遅延束ねの安全形（§2.1-2.2 の条件を門に） | 43×6 = **258** | 中（eager 試験で抑える） |
| **T2** | model patch で `fused_h_proj` 的な単一 Module に（vLLM 側 1 call・plugin 1 apply） | 発注書の「層コード不改変」枠外。長期形。既に patch script 群で model には触っているので、将来の整理先として記録 | 同 258（構造がより単純） | 低（静的） |

T1 の 258/step は「h-group 1 + wq_b + wo_a + wo_b + gate_up + down」の 6/層。発注書の
「6/層 → 2〜3/層」は依存対融合（wq_a→wq_b 等）まで必要で、発注書自身が後回しにしている
追加最適化と整合する。aten 系（cast 3→1、cat 3→0、bf16 copy 数本/層、実測各 0.8-2µs）の
削減は graph モードで ≈0.4 ms/step（ESTIMATE、27.42 ms/step に対し ~1.5%）の追加利得。
m=4 replay 窓での実測 kernel 時間からする利得の本命はやはり gemm 本体の効率（帯域 6 割）
で、これはカーネル職人のレンズの仕事。統合レンズの判定は「T0 は即着手可、T1 は条件付き Go、
基準線再計測後に監督が割に合うかを判定」。

## 5. 発注書への修正要求

1. 基準線を再計測して書き直すこと。方法を明記（graph 内 kernel は prof-shapes.py で属性付け
   できない、実測）。修正後の基準線（≈440-490/step）で利得を見積もること。
2. グループ定義の修正: (a) を「h-group = fused_wqa_wkv（2）+ compressor（2）+
   indexer.compressor（2）+ indexer.wq_b（1、入力同一か要確認）」に拡張検討。
   (b) は「遅延束ね不要・apply 内 2→1」と分離して書くこと。
3. 遅延束ねを採る場合、§2.1-2.2 の条件（forward 内キー / capture 間無効化 / prewarm /
   stream assert / solo fallback / eager m∈{1,2,4,8,16} 試験 / m>16 は 17-144 と >144 の
   経路分け）を門に追加すること。
4. 門 1 に統合 parity を追加: kernel 単体ではなく、`Exl3LinearMethod.apply`・
   `_lna_kv_score`・`_o_proj` の各経路で group ON/OFF の end-to-end 出力差分
   （dtype 境界込み）。
5. 環境変数 `LNA_EXL3_DENSE_GROUP`（既定 OFF）は発注書どおり。追加で:
   OFF 時のコードパスが現状と bitwise 同一（既存 serve との回帰保証）と、
   グループ prewarm の追記を成果物に含めること。
6. REPORT-F2 の必須項目に「module 別 launch 数内訳（graph ON と eager、m 別）」を加えること。

## 参照

- 発注書: `orders/F2-dense-launch-fusion.md`、前検分: `orders/REVIEW-F1-sol.md`
- plugin: `vllm-exl3/src/vllm_exl3/exl3.py`（apply 1454-1532、prewarm 1417-1428、
  pointer 表手本 287-330、param 削除 1449-1457、wo_a TP8 1056-1062）
- 家の exllamav3: `exllamav3-src/exllamav3/modules/quant/exl3.py:10,89,135-139`、
  `exllamav3_ext/libtorch/linear.cpp:34-70`、`exllamav3_ext/quant/exl3_gemm.cu`
  （capture guard 59-64, 262-268、mgemm broadcast 386、per-matrix widths 制限 470-473、
  A_had slab 477-481）、`exl3_gemm_kernel.cuh`（単発 8-79、大域 84-86、broadcast 170-181、
  per-matrix 出力 194-198）、`exl3_gemm_inner.cuh:58,88`、`exl3_devctx.cuh:9,33-42`
- model patch: `recipe-lna/patch_dsv4_dense_exl3.py`（19-31, 63-129, 133-153）、
  `recipe-lna/patch_dsv4_attention_compressor_exl3.py:9-34`、
  `recipe/scripts/patch_dsv4_loader.py:60-63`
- 給仕: `serve-dsv4-tp8.sh:11-17`、`README.md:36-38`、`docs/PLAN.md:61,66,68`
- pack 設定（実読み）: `/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6/config.json`
  の `quantization_config.non_routed_exl3.layers`（43/41/21/21/43/43/43/43/43 層、bits 6、mcg、
  `layer_bits {13,22,28: 3bit, 43,44,45: 2bit}`、`non_routed_dtype_policy: bf16_as_stored`）
- トレース（実測）: `prof-graph-0904/profiler_out_0.txt` ならびに同 rank0 trace
  （gemm<6> 11,715 発 / moe 1,104 発 / cudaGraphLaunch 47 回 / inductor 由来なし）、
  `prof/`（読み取り時点: gemm<6> 11,232 発）
- 実行作法: `/run/media/tonoken3/DATA1/vllm-exl3-v030-port/f1/PROTOCOL.md`（F2 は `f2/` を同型で）
