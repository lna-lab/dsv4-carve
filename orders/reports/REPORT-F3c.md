# REPORT F3c — lna2/lna を本番の木へ移植（一本化）

職人: ユキ（Fable 5.1）／入力: `f1/REPORT-F3-seat2.md`（席の門）／作業: run-62..71 ＋ f2 relay run-8/9、2026-09-04
機械: GPU10/11 の容器のみ。席（:8899, GPU0-9）には一切触れていない。
すべての数字に **MEASURED / ESTIMATE / 未測定** を付す。

---

## 0. 結論

| 門 | 結果 |
|---|---|
| 1. 本番の木での机上 parity（lna2） | **PASS**。合成 159 件（m 1..24、K2/K3）＋本番テンソル L0/L13/L22/L28、最悪 rel **1.64e-3**（門 2e-3） |
| 2. lna2 が実際に走ることの証明・strict | **PASS**。`gate_active` 18/18、＋本番固有の継ぎ目を叩く `gate_dispatch_prod` 11/11 |
| 3. T0/T1 が無傷 | **PASS**。f2 relay で `parity_t0` `parity_t1` とも rc=0、**323 PASS / 0 FAIL**（run-9、私の最終書き込み後に採り直し） |
| 4. diff レビュー | **§5 に全ハンク**。本番ファイルへの追加は **3 か所・583 行**、削除 0 行 |

**移植は成った。** 本番の木（`vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py`）に lna2 と lna が入り、
**T0/T1 は無傷**、**既定は現行 exllamav3 のまま**。席は今後 plugin の木を一本だけ mount すればよい。

**VRAM の追い込みも成った（MEASURED）**: lna2 の scratch を **6.470 → 4.853 MiB** にし、
かつ **stream ごとでなく device ごと 1 個**にした。席が UTIL=0.97 で足りなかった **約 10.7 MiB** に対し、
**1.6〜8.1 MiB を戻した**（§4）。**0.97 で立つかは席でしか決まらない。**

---

## 1. なぜ移植が要ったか（席の §4 の再掲）

`PLUGIN_SRC` は**ディレクトリ丸ごとの mount** なので、lna2 を載せた瞬間 T0/T1 のコードが席から消え、
`attention.py` のヘルパーが `except ImportError` で solo ループに落ちて**何も言わなかった**。
席は「併用」を測ったつもりで **lna2 単体**を測っていた。木を一本にするのが唯一の直し方である。

---

## 2. 本番の木の継ぎ目（先に読んだ）

本番の routed 経路は port より**ずっと素直**で、backend の分岐が**そもそも無い**:

| 場所 | 行（移植前） | 役 |
|---|--:|---|
| `build_exl3_fused_state` | **827** | `_exl3_ptrs` と fused temps を作る。ここが preflight の継ぎ目 |
| `apply_exl3_fused_moe` | **873** | incumbent `exllamav3_ext.exl3_moe` の呼び出し |
| `apply_exl3_experts` | **983** | **唯一の dispatch 点**。`use_fused` を決めて fused か python loop を呼ぶ |
| `Exl3MoEMethod.apply` | **1533** | vLLM からの入口。`apply_exl3_experts(...)` を `fused` 無しで呼ぶ（＝`fused=None`） |

port 側にあって本番に**無い**もの: `get_moe_kernel_backend`、`native_*`、`_load_lna*`。
port 側にあって本番にも**在る**もの（そのまま使えた）: `MCG_MARKER_SIGNED_INT32`、`map_topk_to_local`、
`pin_exl3_expert_map`、`apply_exl3_python_loop`、`make_linear_exl3`、`_FUSED_TEMP_CACHE`、`logger`、`os`、`importlib`。

---

## 3. 移植したもの／しなかったもの

移植は **AST で関数単位に抜き出して貼った**（手写しをしていない。`/tmp/lna_block.py` 生成スクリプト）。
**21 シンボル全部**が本番に在ることを検証済み（MEASURED, run-71 直前の検査）:

`get_moe_kernel_backend`(本番用に新規)・`moe_kernel_strict`・`moe_kernel_call_counts`・
`reset_moe_kernel_call_counts`・`moe_kernel_declined_reasons`・`_lna2_log_active`・`_lna1_log_reject`・
`_lna2_log_reject`・`_load_lna2_exl3_ext`・`lna2_moe_kernel_available`・`lna2_max_rows`・`_apply_lna2_moe`・
`_lna2_concurrent_streams`・`_lna2_scratch_for`・`lna2_scratch_bytes_resident`・`_load_lna_exl3_ext`・
`lna_moe_kernel_available`・`_validate_lna_pointer_contract`（**`mcg_tensor` 修正込み**）・
`_lna_scratch_for`・`_prepare_lna_state`（**backend 別**）・`_apply_lna_moe`

**移植しなかった（発注どおり。不在を検証済み、MEASURED）**:
`_apply_native_fused_moe`・`native_moe_kernel_available`・`p2b_fused_moe`・`_load_native_exl3_ext`、
および 0.3.0 の EP loader 変更・その他一切。

**`get_moe_kernel_backend` は本番用に書き換えた（意図的な差）**: この木に native p2b は無いので、
`auto` は **`exllamav3`** に落ち、`native` は受け付けない値にした。
→ **`VLLM_EXL3_MOE_KERNEL` 未設定＝これまでどおり incumbent**（MEASURED, run-65）。

**native 拡張は port の `.so` のまま**。本番プラグインは `importlib.import_module("vllm_exl3_c")` で
同じように掴む（`_load_lna2_exl3_ext`）。**この木でのビルドは何も変わらない。**
解決を実測で確認（run-62）: `vllm_exl3.exl3 = /lab/vllm-exl3/src/…`、`vllm_exl3_c = /port/src/…so`。

---

## 4. VRAM の追い込み（発注「10 MiB をどこかで削れないか」）

席の不足は **1.83 − 1.82 = 0.01 GiB ≒ 10.7 MiB/GPU**。二つ見つけて両方直した。

### 4.1 stream ごとの scratch が二重になっていた（MEASURED, run-67）
`_lna2_scratch_for` は `(device, stream)` キーだった。**二つ目の CUDA stream は 20.00 MiB かかる**
（実測）— 自分の scratch 6.47 MiB ＋ caching allocator の新しいセグメント。
そして **vLLM は decode graph を側 stream で capture する**ので、席は 2 本持っていた可能性が高い。

→ `LNA2_CONCURRENT_STREAMS == 1`（既定、かつ席の `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 運用）のときは
**device ごとに 1 個**にした。同時に走る lna2 は 1 本だけで、buffer は launch を跨いで何も持たないので安全。
`>1` のときだけ stream をキーに戻す（§F3b §4 の deadlock 対策と整合）。
実測（run-68）: 2 本目の stream を使っても **buffers=1、scratch resident 変わらず**。
副次効果として「capture 中に確保できない」問題も消える（load 時に既定 stream で作ったものを使う）。

### 4.2 `LNA2_MAX_M` 32 → **24**（MEASURED）
scratch は `MAX_PAIRS = MAX_M × topk6` に比例する。

| MAX_M | MAX_PAIRS | scratch |
|--:|--:|--:|
| 32 | 192 | 6,784,384 B = **6.470 MiB** |
| **24** | **144** | **5,088,640 B = 4.853 MiB** |
| 16（席が踏んだ天井） | 96 | 3,392,896 B = 3.236 MiB |

制式は 4 seq × DSpark3 = **16 行**。24 なら **6 seq ぶん（50% の余裕）**で、
席が踏んだ「16 ちょうどで一つ増やすと黙って戻る」崖は消えたままである。
`-DLNA2_MAX_M=32` で 8 seq に戻せる（ビルド旗）。

### 4.3 戻した量（MEASURED ＋ ESTIMATE）

| 席の F3b ビルド | F3c |
|---|---|
| 6.470 MiB × （1 or 2 stream） = **6.47〜12.94 MiB** | **4.853 MiB**（device ごと 1 個で確定） |

**戻した量 = 1.62 MiB（席が 1 本だった場合）〜 8.09 MiB（2 本だった場合）**、
さらに 2 本目だった場合は allocator セグメントぶん（実測 20 MiB のうち scratch 以外の ~13.5 MiB）も
消えるので、**上限は 21.6 MiB**（ESTIMATE）。**不足の 10.7 MiB を跨ぐかどうかは席でしか決まらない。**

**ケンのブートで見てほしい行**（UTIL=**0.97** で）:
- `available KV cache memory (X GiB)` — 席の F3b は 1.82。F3c で **1.83 以上**になれば立つ
- 立ったら `GPU KV cache size: N tokens`（制式 incumbent は 396,656、F3b@0.972 は 393,483）

### 4.4 探して「無かった」もの（正直に）
- **F1(`lna`) の確保**: 既に `backend=="lna2"` では**一切確保しない**（F3b で対処済み、run-71 で再確認）。ここに追加の余地は無い。
- **prewarm buffer**: `LNA_EXL3_PREWARM_ROWS` の `x` はローカルで即解放される。持ち越し無し。
- **`_FUSED_TEMP_CACHE`**（≈142 MiB）: プリフィルが incumbent に落ちる以上必要で、**incumbent も同額払う**。差分ではない。
- **`vllm_exl3_c` の module load**: 測ろうとしたが `mem_get_info` の粒度（2 MiB）と caching allocator に埋もれて
  **分離できなかった＝未測定**。ここに何十 MiB かある可能性は否定できない（席の残差 ≈48 MiB/GPU の説明がまだ付いていない）。
  切りたいなら **lna2 だけの slim `.so`**（p2b / exl3_gemm / exl3_gemv / lna を外す）が次の手（未実施、ESTIMATE）。

---

## 5. 門 4 — diff レビュー（本番ファイルに私が足した全ハンク）

`diff -u`（移植直前の写し `/tmp/exl3_prod_backup.py` 対 現在）: **+583 / −0 行、3 ハンク**。

| # | 位置 | 行数 | 目的 |
|--:|---|--:|---|
| **1** | `build_exl3_fused_state` の直前（`@@ -824 +839,519`） | +519 | **移植本体**。§3 の 21 シンボル＋見出しコメント。既定では 1 行も実行されない |
| **2** | `build_exl3_fused_state` 末尾（`@@ -868 +1396,14`） | +8 | **preflight**。`backend in {"lna","lna2"}` のときだけ `_prepare_lna_state(layer, inners, backend)`。これが `_exl3_lna_ready` を立てる。無いと lna2 は毎回 `None` を返して黙って落ちる（席 §2 の傷そのもの） |
| **3** | `apply_exl3_experts` の `pin_exl3_expert_map` 直後（`@@ -998 +1534,43`） | +37 | **dispatch**。`lna2` / `lna` を試し、成功なら `_exl3_last_apply` を立てて即 return。`_lna2_log_active()` で 1 回だけ ACTIVE を出す。例外は **strict なら再送出**（fallback ハンドラに握り潰させない）。`fused is not False` を守るので `fused=False`（＝python loop を明示要求）のときは手を出さない |

**削除行は 0。** 既存の関数の中身は一つも書き換えていない（ハンク 2・3 は末尾/直後への挿入のみ）。

### ⚠️ 同時編集の事故りかけ（報告義務）
diff を取ったら、**T0/T1 の職人が同じファイルを私の作業中に編集していた**
（`_DENSE_GROUP_MODE_ACTIVE_LOGGED` / `dense_group_call_counts` / 辞退を毎回ログにする変更 ＝
席が §4 で求めた「握り潰しをやめる」対応）。**私の読み書きで消していないことを確認済み**（現在のファイルに全部在る、
かつ §6 の gate 3 を**私の最終書き込みの後に採り直して** 323 PASS/0 FAIL）。
**が、これは運任せだった** — 私の編集は read-modify-write なので、相手の書き込みが私の read と write の間に
入っていたら黙って消えていた。**一本の木を二人で触る間は、編集の前に必ず diff を取ること。**

---

## 6. 門の数字（全部 MEASURED）

### 門 1 — 本番の木での机上 parity（run-70）
| 対象 | 件数 | 最悪 rel(FP32) | 判定 |
|---|--:|--:|---|
| lna2 合成（m ∈ 1..8,15,16,17,20,23,24、K2/K3、route 6 種×入力 7 種×weight 3 種） | **159** | 1.64e-3 | PASS |
| lna2 本番テンソル L0(K2)/L13,22,28(K3)、m ∈ {4,16,24} | 36 | ≤1.2e-3 | PASS |
| lna2 NaN poison replay | 48 | 9.8e-4 | PASS |
| `[m,6]` vs flat regression | — | ABI が拒否／転置で出力が変わる | PASS |
| **lna（F1）合成＋本番** | 110 | 1.70e-3 | PASS |

**テストファイルは port のものを一文字も変えずに使った。** `PYTHONPATH=/lab/vllm-exl3/src:/port/src` の
順序だけで本番プラグインに向けている（run-62 で解決先を assert）。
＝**同じ 159 件が両方の木で同じ答えを出す**ことの証明でもある。

### 門 2 — 走ったことの証明（run-69, run-71）
`gate_active.py` **18/18 PASS**（本番の木で）。要点:
- 実 `LinearEXL3` の `mcg_tensor` で ABI 検証が通る（席の傷 #2 の回帰）
- `backend=lna2` は F1 の scratch を**確保しない**（F1 entries=0）
- 呼び出しカウンタが動く／`max_rows=24`／m=16,20,24 受理・32 辞退・辞退は理由つきでログ
- STRICT=1 は形の辞退で落ちない（プリフィルが死なない）／STRICT=2 は落ちる／STRICT=1 は未 preflight で落ちる
- 8 層足しても scratch は増えない（5,088,640 → 5,088,640）

`gate_dispatch_prod.py` **11/11 PASS**（新規。**本番固有のハンク 3 を叩く唯一のテスト**）:
- 既定 env では backend=exllamav3、**どちらの LNA も呼ばれない**（counts {0,0}）
- `fused=False` は backend 指定があっても手を出さない
- `backend=lna2` / `lna` で実際にカーネルへ到達し、`_exl3_last_apply` が立ち、参照と一致し、ACTIVE ログが出る

### 門 3 — T0/T1 が無傷（f2 relay run-9、私の最終書き込み後）
```
plugin: /lab/vllm-exl3/src/vllm_exl3/exl3.py
T0/T1 present: True True True     lna2 ported in: True True
default MoE backend: exllamav3
PARITY_T0_PASS   parity_t0 rc=0
PARITY_T1_PASS   parity_t1 rc=0
323 PASS / 0 FAIL
```

### 参考 — kernel 単独 latency は不変（run-70、本番の木経由）
K2 m=4 **178.2 µs**（同条件 incumbent 比 **1.89×**）、K2 m=16 623.7（1.82×）、
K3 m=4 223.3（1.63×）、K3 m=16 787.6（1.54×）。`MAX_M` を 32→24 にしても差は雑音の中。

---

## 7. ケンの決裁に要る二つの事実（席報告 §1/§7 より）

1. **UTIL**: 席の F3b は **UTIL=0.97 で立たず、0.972 で立った**（KV **393,483 tok**、
   制式 incumbent の 396,656 に対し **−0.8%**）。これは**制式レシピの書き換え**であり、私が決めることではない。
   → F3c で **1.6〜8.1 MiB/GPU（上限 21.6、ESTIMATE）** を戻したので、**0.97 で立つ可能性がある**。
   立てばこの決裁事項そのものが消える。**要ブート確認**（§4.3 に見る行）。
2. **速さ**（席実測、F3b）: 単流 **1.20–1.22×**（code 103.2 tok/s）、routed kernel **2.23×**、
   166k 針 **正答**・TTFT 123.5 s（制式記録 167 s）、作文 7/7 stop、受理長 不変、ACTIVE 8/8、strict clean。
   ケンの 120 tok/s には **未達**（86%）。T0 併用が成っても ESTIMATE 約 111 で届かない。

---

## 8. 席に残すもの

1. **UTIL=0.97 での再ブート**（§4.3）。これが今回の唯一の未解決の可否。
2. **一本化した木での ppl と 8 rank P50/P95**（席 §7 の宿題。F3c でカーネルもプラグインも変わったので測り直し）。
3. **T0/T1 と lna2 の“本当の併用”**。木は一本になったので、`LNA_EXL3_DENSE_GROUP=1` と
   `VLLM_EXL3_MOE_KERNEL=lna2` を**同時に**立てて、家計簿に `exl3_mgemm_kernel` と `lna2_kernel<2>` が
   **両方**出ることを確認する（今回は机上までしか出来ていない）。
4. **compute-sanitizer 4 種** — 像に無い（変わらず）。device atomics と team barrier があるので racecheck が要る。
5. **slim `.so`**（§4.4）: 席の残差 ≈48 MiB/GPU の説明がまだ付いていない。分離を測るところから。
6. 掟: **一本の木を二人で触る間は、編集前に diff**（§5）。

---

## 9. 変えたファイル

| ファイル | 変更 |
|---|---|
| `vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py` | **本番**。3 ハンク +583/−0（§5） |
| `vllm-exl3-v030-port/csrc/lna_moe_ticket.cu` | `LNA2_MAX_M` 既定 32 → **24** |
| `vllm-exl3-v030-port/src/vllm_exl3/exl3.py` | scratch を device ごと 1 個に（本番と同一コード） |
| `vllm-exl3-v030-port/f1/gate_dispatch_prod.py` | **新規**。本番の dispatch ハンクを叩く |
| `vllm-exl3-v030-port/f1/gate_active.py` | 天井 24 に追随 |
| `vllm-exl3-v030-port/f1/parity_native.py` | m の上限をカーネルに問い合わせる（`_lna2_ceiling`） |
| `vllm-exl3-lab/f2/RUN.sh` | 門 3（T0/T1 parity）用。`history/f3c-gate3.log` に写し |
