# REPORT F2 T0 — 密 EXL3 shard 融合スパイク（同一 call 内 2→1）

職人: GLM-F。2026-09-04。発注: `orders/F2-dense-launch-fusion-v2.md` §3.2 の **T0 のみ**
（T1 の遅延束ね・h-fan / q_a-fan には進んでいない。状態は持たない）。
対象は本番 plugin `vllm-exl3/src/vllm_exl3/exl3.py` のみ（`vllm-exl3-v030-port` は触っていない）。
GPU 実行は `f2/` 中継（GPU 11、`f2/PROTOCOL.md` の型）で 1 回、`f2/RUN.log` ＝ `f2/history/run-1.log`
（rc=0、**PARITY_T0_PASS**）。席 :8899・GPU 0〜9 は触っていない。pkill -f は使っていない。

---

## 0. 結論

- **机上 parity は全項目 PASS**。本番 tensor（層 5・層 40）× 全 4 種グループ ＋ 合成 3 種、
  m ∈ {1,2,3,4,8,16} の fp32 per-row rel は **最悪 8.5e-07（門 2e-3 の約 2400 倍の余裕）**。
- **launch 数の見込み（ESTIMATE、pack 構成からの算術）: 489 → 341/step（−148、−30%）**。
  発注 §5 の T0 門「≤345」を満たす算術。実測は席の profiler（Gate 2）で確定する。
- **変更は既定 OFF**（`LNA_EXL3_DENSE_GROUP=1` で ON）。OFF 経路はビット等価で無変更
  （全グループで solo 直接呼び出しと bitwise 一致を実測で確認）。
- 新規 CUDA カーネルは**書いていない**（発注 §3.1 のとおり。`exl3_mgemm` の wiring のみ）。
- compute-sanitizer は像に無し（§4.4）。代替検査を実施し REPORT に記録した。

## 1. 環境（MEASURED、run-1 冒頭）

| 項目 | 値 |
|---|---|
| GPU | RTX PRO 2000 Blackwell（中継 GPU 11 = 容器 cuda:0）、34 SM |
| torch / CUDA | 2.13.0+cu130 |
| exllamav3 | 1.4.5（像の `exllamav3_ext` に `exl3_mgemm` binding 有り確認） |
| vllm / vllm-exl3 | 0.28.1rc1.dev337+g27a94d1ce / 0.2.3 |
| plugin | 実行中モジュール = `/lab/vllm-exl3/src/vllm_exl3/exl3.py`（編集側で確認済） |
| pack | `/model/model-dense-exl3.safetensors` |

## 2. 変更点

### 2.1 `vllm-exl3/src/vllm_exl3/exl3.py`（本体）

1. **runner 一式**（モジュール尾部、~170 行）:
   - `dense_group_enabled()`: `LNA_EXL3_DENSE_GROUP=1` のみ ON。**env は毎呼び出しで読む**
     （席の門 5「同一プロセス A/B」のため。run-1 で ON/OFF 同一プロセス切替を实测済）。
   - `_dense_group_tables(linears)`: 登録時検査（発注 §3.4 を T0 分だけ適用）。
     K / mcg / mul1 一律、bias 無し、**先頭 shard が最幅**（`ns[0]==max(ns)`、違反は
     lock 範囲重なりの無言 race を避けるため融合しない）、k%16==0・n%128==0（mgemm 制限）、
     出力 dtype は fp32/fp16 のみ。ポインタ表（trellis/suh/svh/n/idx）は load 時に 1 回だけ
     device tensor 化（`dsv4.py` `_build_x_fan` と同じ形）。
   - `build_exl3_dense_group(module)`: load 後に 1 回。センチネルは
     無キー=未検査 / `None`=検査済で拒否 / dict=融合可能。bf16 shard 混在・shard 数 <2 は最初から対象外。
   - `_dense_group_scratch(...)`: A_had（行列数 × m × k、不足は無言 OOB なので
     `bszm·m·k` 必須を shape で保証）＋ shard ごとの出力（m, n_j）＋ 出力ポインタ表。
     **恒久バッファ**（`_fan_outs`/`_FUSED_TEMP_CACHE` と同型。層間は単一 stream 直列で共有、
     その旨のコードコメント有り）。
   - `_exl3_group_mgemm(...)`: `ext.exl3_mgemm(A=(1,m,k), B=trellis表, C=先頭出力,
     suh, A_had, svh, idx, None, K, -1, mcg, mul1, -1, -1, 0, 1, size_n_list, c_ptrs)`。
     呼び出し形は上流 `dsv4.py:1456-1464` を写し。**bszm_in=1 で A を全行列 broadcast、
     suh/svh は行列ごとに独立適用（カーネル ABI、REVIEW-F1-sol §1.4 の順序をカーネル側で保証）**。
     C は fp32（既存の solo `out_dtype=torch.float32` と同じ FP32 境界）。
   - `run_exl3_group(module, x_fp16, out_dtype)`: 3 つの人差し指（apply / patch helper）から
     呼ぶ単一入口。融合不成立の全ケース（flag OFF・m>16・入力非適合・検査拒否・capture 中に
     表が無い・例外）で **現行の solo forward にそのまま落ちる**。fallback は理由ごとに 1 回だけ
     WARNING。capture 中の表構築（H2D）は禁止しているため capture 内で表未構築なら solo。
2. **`Exl3LinearMethod.apply`**: shard ループの手前で「bf16 shard 無し かつ shard 数 >1」のとき
   `run_exl3_group(layer, x_fp16, torch.float32)` に分岐。それ以外（bf16 混在・単 shard）は
   従来のループを文字通り残す。OFF 時の動作は **bitwise 等価**（§3.3 実測）。
3. **prewarm 拡張**（`process_weights_after_loading`）: `LNA_EXL3_PREWARM_ROWS` の各行数で
   solo 全 shard に加え **融合グループでも 1 回ずつ先走り**する。mgemm は capture 中
   autotune を黙ってスキップして静的ヒューリスティックに落ちるため（`exl3_gemm.cu:578` 側、
   run-1 の tripwire で挙動確認）で、capture 前に全 m bucket × グループ形の autotune key を
   掘らせる。flag OFF 时候は従来通り solo のみ（無変更）。
4. capture 安全: ポインタ表・バッファ・cptr はすべて capture 前に確立
   （prewarm が m=1,2,4,8,16 を掘る。vLLM capture bucket [1,2,4] は全域被覆）。

### 2.2 `recipe-lna/patch_dsv4_attention_compressor_exl3.py`（compressor / indexer.compressor の消費口）

`_lna_kv_score` ヘルパーだけ最小差し替え: `run_exl3_group(module, x, torch.float32)` を呼び、
plugin 旧版で ImportError の場合は従来の solo ループに落ちる（後方互換）。
**注意: idempotency マーカは `_lna_kv_score` の有無のままなので、すでに旧ヘルパーで
patch 済みの attention.py は「already patched」となり差し替わらない。席に置くときは
`attention.py.orig-lna2` から戻してから再 patch する（§5.6）。**
`_o_proj`（wo_a）と `patch_dsv4_dense_exl3.py` は T0 の対象外（wo_a は単独 linear、融合相手が無い）。

### 2.3 新規（門の道具）

- `f2/parity_t0.py`: 机上 parity ハーネス（`--slim` は sanitizer 用縮小モード）。
- `f2/RUN.sh`: 中継実行スクリプト。

## 3. 門の結果（MEASURED、`f2/RUN.log` / `f2/history/run-1.log`、rc=0）

### 3.1 pack 形の再確認（発注 Gate 0 項目のうち sandbox で可能な範囲）

`/model/model-dense-exl3.safetensors` から層 5（奇数）・層 40（偶数）を実読みし、
発注 §0.1 の形表と完全一致を確認（K=6・k=4096 も実測）:

| グループ | 実測 ns | 発注 §0.1 |
|---|---|---|
| 層 5 wqa_wkv | [1024, 512] | 一致 |
| 層 5 compressor | [512, 512] | 一致（奇数 512） |
| 層 5 shared gate_up（rank 局所 256 に狭幅） | [256, 256] | 一致 |
| 層 40 wqa_wkv | [1024, 512] | 一致 |
| 層 40 compressor | [1024, 1024] | 一致（偶数 1024） |
| 層 40 indexer.compressor | [256, 256] | 一致（偶数のみ存在） |
| 層 40 shared gate_up | [256, 256] | 一致 |

### 3.2 parity（参照 = shard ごとの `LinearEXL3.forward` solo、fp32 段階）

**per-row rel の定義**: 行ごと `max|f−r| / max(|r|_row, 1e-3)`。門 = 2e-3。

| グループ（本番 tensor） | m | per-row rel max | max abs | canary \|ratio−1\| | 二度走り bitwise |
|---|---|---|---|---|---|
| L5.wqa_wkv | 1..16 | 6.39e-07（m8） | 2.86e-06 | 1.79e-07 | PASS |
| L5.compressor | 1..16 | **8.48e-07（m16、全体最悪）** | 1.14e-05（m4） | 3.58e-07 | PASS |
| L5.shared_gate_up | 1..16 | 0（bitwise 一致） | 0 | 0 | PASS |
| L40.wqa_wkv | 1..16 | 6.47e-07（m16） | 2.86e-06 | 1.79e-07 | PASS |
| L40.compressor | 1..16 | 6.94e-07（m16） | 8.58e-06 | 2.38e-07 | PASS |
| L40.shared_gate_up | 1..16 | 0 | 0 | 0 | PASS |
| L40.indexer_compressor | 1..16 | 0 | 0 | 0 | PASS |
| SYN.wide_first（合成 512/256、suh/svh を 16× 分離） | 1..16 | 5.30e-07 | 1.76e-02 | 2.38e-07 | PASS |
| SYN.equal（合成 256/256、8× 分離） | 1..16 | 4.05e-07 | 5.86e-03 | 2.38e-07 | PASS |

- 全行 finite。norm 比 canary は最悪 3.6e-07（F1 の 1/√2 署名は出ていない）。
- **n=256 同幅グループ（shared・indexer）は solo と bitwise 完全一致**、幅が違う
  （wqa_wkv・compressor）は tile 配置が変わるため ~1e-6 の fp32 差。どちらも門内。
- **合成で suh/svh を 16×/8× 分離したのは、束ね側が suh/svh を共有するバグを弾くため**。
  rel 1e-07 台で通った=行列ごとの Hadamard は独立に効いている（REVIEW-F1-sol §1.4）。

### 3.3 経路・安全系の実測

| 項目 | 結果 |
|---|---|
| flag OFF（同一プロセスで 1→0） | 全グループで solo 直接呼び出しと **bitwise 一致**（OFF 経路無変更の直接証明） |
| 順序入れ替え（[wkv, wq_a] 等の逆順） | 「先頭最幅」検査で**融合を拒否**（INFO ログ 1 回）し solo で bitwise 正しい |
| m=17 / m=33 | 束ねず素通し、solo と bitwise 一致（fallback WARNING は 1 回のみ） |
| `Exl3LinearMethod.apply` 経路（stub layer, bf16 入力） | OFF=参照 solo+cat+cast と bitwise 一致。ON vs OFF は bf16 変換後に per-row rel **1.69e-03**（band 0.02、bf16 丸め 2^-9 が支配。fp32 段の門 2e-3 は §3.2 で充足）。出力 dtype 変らず bf16 |
| graph capture | m=1 と m=4 の 2 本を capture 成功 |
| stale / 混線 | scratch を NaN で毒しながら m=1/4 を交互 120 replay、全ステップ solo 参照と一致（最終入力に追従、bucket 混線無し） |
| capture 中の追加確保 | replay 120 回で allocator delta **1 B**（恒久バッファが効いている） |
| untuned tripwire | eager 29.2 µs/iter vs graph replay **28.4 µs/iter**（m=4, L40.compressor）。replay が eager を上回らない=静的ヒューリスティックが焼いていないことの間接信号（ヒューリスティックが偶々同速の可能性は排除しきれない。直取得 API は無い） |

### 3.4 発注 Gate 1 から落とした・できなかった項目（正直に）

- **compute-sanitizer（memcheck/racecheck/synccheck/initcheck）: 像に入っていなかった**
  （`compute-sanitizer: NOT IN IMAGE`、RUN.log 310 行目付近）。発注が許す代替として
  「solo との per-run parity ＋ 二度走り bitwise ＋ 毒し込み 120 交互 replay ＋ allocator 差分 1 B」
  で置いた。**racecheck 相当の検査は実施できていない**。lock 範囲の静的検査としては
  「先頭最幅」検査（発注 §3.4-2 の race 回避条件そのもの）を登録時に課した。
- 段階別（入力 Had 後・GEMM 生和）の突き合わせは未実施（Python からカーネル内部を
  観測する口が無い）。最終出力（fp32）での比較のみ。
- 統合 parity の `apply` は stub layer による。`_lna_kv_score`・`_o_proj` の実 vLLM 経路は
  席で費やす（§5）。中継 1 回に `--slim` の sanitizer を組み込んだが像に無かったため
  slim 実走行は無し。

## 4. launch 数の見こみ（ESTIMATE。pack 実測形 × §1.1 の 24 step 構成からの算術）

削減は「同一 module call 内の solo gemm 2 本 → `exl3_mgemm` 1 本」のみ。層種別:

| 層種（数） | 現状 launch/層 | T0 後 | 内訳 |
|---|---:|---:|---|
| 層 0,1（2） | 8 | **6** | wqa_wkv −1、shared −1 |
| 奇数層 3..41（20） | 10 | **7** | ＋compressor −1 |
| 偶数層 2..42（21） | 13 | **9** | ＋compressor −1、indexer.compressor −1 |
| **計/step** | **489** | **341（−148、−30%）** | |

- 発注 §5 の T0 門「489 → ≤345」を満たす算術。**実測は席で**（Gate 2 の counting rule:
  `exl3_gemm_kernel<6` 全 variant ＋ 新たに出る `exl3_mgemm_kernel<6` ＋ 補助 kernel）。
- cat / cast などの補助 kernel は **T0 では減らない**（出力は shard ごとバッファ→cat のまま。
  cat 消滅は T1 の fan 束ねの取り分）。
- prewarm 追加コスト（ESTIMATE）: load 時にグループ持ち層で行数 bucket 分（既定 5）の
  融合呼び出しを追加。autotune key の distinct は k=4096 × n_max∈{1024,512,256} × m 5 本
  ≈ 15 key（device 全体で 1 度ずつ）。

## 5. 席の門に残すもの

1. **Gate 2（同条件 A/B）**: `LNA_EXL3_DENSE_GROUP` OFF→ON→OFF 交互 ≥3 回、同一ビルド・同一 pack。
   flag OFF の kernel 名・件数が baseline と完全一致すること（OFF 無変更は bitwise でも
   実証済だが、profiler 表での件数一致が直接証明になる）。対象 kernel は
   `exl3_gemm_kernel<6` 全 variant ＋ `exl3_mgemm_kernel<6` ＋ cat/cast 等の補助。
2. **KILL LINE の当て方**: 密セグメント = `exl3_gemm_kernel<6>` 合計時間 ＋
   `exl3_mgemm_kernel<6>` 合計時間（ON 側）。ON を 8.885 ms の baseline と比べ
   **<1.05× なら T1 に進まず終了**（発注 §5）。
3. **Gate 3/4**: グループ別は `bytes ÷ 時間` を mgemm launch ごとに（発注 §4 の表をそのまま）。
   (b) shared（n 合計 512）は 1.25×T_floor を FAIL 扱いにしない（同条件 A/B を拘束門に）。
4. **Gate 5**: env は同一プロセスで効く（run-1 実測）。実行条件の記録を忘れないこと:
   `LNA_DSV4_AUX_STREAMS=0`・`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`。
   m=8..16 は capture bucket 外で毎ステップ eager のまま（T0 は状態を持たないので
   遅延束ねの stale リスクは無い。run-1 の 120 交互 replay が参考）。
5. **patch の再適用**: compressor ヘルパーを変えたので、席の attention.py が旧ヘルパーの
   ままだと compressor・indexer.compressor の −62 launch が効かない
   （`attention.py.orig-lna2` から復元 → `recipe-lna/patch_dsv4_attention_compressor_exl3.py` を再実行）。
   効いていれば eager トレースで compressor 層に `exl3_mgemm_kernel` が現れる。
6. **untuned tripwire の持ち方**: `f2/parity_t0.py` の `prewarm.tripwire` と同じ
   「eager vs graph replay の µs/iter 比較」を席でも 1 回（replay が有意に遅ければ
   capture が静的ヒューリスティックに落ちている。prewarm は実装済み）。
7. **sanitizer**: 像に無い。入れられるなら `python3 f2/parity_t0.py --slim` を
   4 tool で回すだけで良い（RUN.sh に雛形あり）。入れられないなら §3.4 の代替で
   裁定し、その旨を REPORT-F2 本編に書く。

## 6. 触っていないもの（明示）

- `vllm-exl3-v030-port/`（別職人の routed experts 作業場）、`csrc/lna_moe_*`、
  `exl3.py` の routed 経路、`f1/`、席 :8899、GPU 0〜9、seats の中継（f2 を使った）。
- 新規 CUDA カーネル・`csrc/` は 0 行。
- `exllamav3-src/` は読みのみ（ビルド無し。像の .so に `exl3_mgemm` を確認して使用）。

## 7. 出典

- 実行ログ: `f2/RUN.log`（= `f2/history/run-1.log`、rc=0、PARITY_T0_PASS）
- 実装: `vllm-exl3/src/vllm_exl3/exl3.py`（runner 一式・apply 分岐・prewarm 拡張）、
  `recipe-lna/patch_dsv4_attention_compressor_exl3.py`（ヘルパー差し替え）
- 手本: `exllamav3-src/exllamav3/modules/dsv4.py:914-957, 1445-1470, 979-995`、
  `exllamav3_ext/quant/exl3_gemm.cu:369-410, 464-481, 578-612`、`bindings.cpp:161`
- 正典: `orders/REVIEW-F1-sol.md` §1.4（suh/svh/Hadamard の独立）・§1.5（FP32 境界）、
  `orders/F2-SYNTHESIS.md`
