# REPORT F3b — 席の指摘三点の修理（lna2）

職人: ユキ（Fable 5.1）／入力: `f1/REPORT-F3-seat.md`（席の門、ユキ検分）／作業: run-49..60、2026-09-04
機械: RTX PRO 2000 Blackwell 34 SM（GPU10/11 の容器）。席（:8899, GPU0-9）には一切触れていない。
すべての数字に **MEASURED / ESTIMATE / 未測定** を付す。

---

## 0. 結論

| 席の指摘 | 状態 |
|---|---|
| (1) 成果物が席で一度も走っていなかった（黙って現行に落ちる） | **修理済み**。席の 3 修正を採用・レビューし、その上に **大声の fallback**・**strict 二段**・**「実際に走った」を証明する回帰テスト**を足した |
| (2) VRAM: 6.556 MiB/層/stream が制式の窓を 389,120 → 263,424 に削る | **修理済み（MEASURED）**。**層あたり 0 になった**。lna2 の scratch は **全層で 1 個**（6.470 MiB／device・stream）、F1 の scratch は lna2 のとき**確保しない**。実測: 86 層を prepare して増分 **6,784,512 B** |
| (3) 16 行天井に余裕が無く、`--max-num-seqs` を 1 増やすと黙って戻る | **修理済み**。行天井を **32** へ（parity 込み）。加えて辞退は必ずログに出る。天井はプラグインが**カーネルに問い合わせる**（ハードコード廃止） |

**追加で見つけて直した傷が一つ**: 二つの lna2 カーネルが別 stream で重なると **team barrier が deadlock する**（§4）。
席の現運用（`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`）では起きないが、黙って固まるのが一番いけないので
`LNA2_CONCURRENT_STREAMS` で明示的に席を空ける形にし、既定は単一 stream であることを明記した。

**⚠️ 制式 389,120 でのブートは私には検証できない**（TP8 は GPU10/11 では立たない）。
発注どおり **scratch の算数を §3 に置いた**。ケンのブートで確認してほしい実際の数字も §3.3 に書いた。

---

## 1. 席の 3 修正のレビュー（採用、そのまま維持）

`src/vllm_exl3/exl3.py` に入っていた席の修正を全部読み、**三つとも正しい**と確認した。維持する。

| # | 修正 | 検証（MEASURED, run-60 `f1/gate_active.py`） |
|---|---|---|
| 1 | `build_exl3_fused_state` の `backend == "lna"` → `backend in {"lna","lna2"}` | `_prepare_lna_state(backend=lna2)` が `_exl3_lna_ready=True` を立てる |
| 2 | ABI 検証が `linear.mcg`（**bool**）でなく `linear.mcg_tensor`（**Tensor**）を読む | 実測でも `mcg=bool / mcg_tensor=Tensor`。実 `LinearEXL3` で `_validate_lna_pointer_contract` が通る |
| 3 | ACTIVE / declined の一発ログ | 残置。下の (a)(b)(c) を上乗せ |

`mcg` の件は **F1（`lna`）にも同じく効く**（席の推測どおり）。F1 の parity は変わらず PASS（§5）。

---

## 2. (1) 二度と黙って落ちないようにした

### (a) 辞退は必ず大声で出る
`_lna2_log_reject()` は理由ごとに一度 **warning** を出す（`ptr table incomplete` の抜けも塞いだ）。
F1 側も同様に `_lna1_log_reject()` を新設した — **F1 は今まで完全に無言で落ちていた**。

### (b) strict は二段（ここは席の案から意図的に変えた）
`VLLM_EXL3_MOE_STRICT` を **bool でなく段**にした。理由: **プリフィル（512 行）は設計どおり辞退する**ので、
「あらゆる辞退で落ちる」strict は本番で使えない＝結局誰も入れず、同じ穴が残る。

| 値 | 意味 |
|---|---|
| `0`（既定） | 従来どおり fallback。ただしログは出る |
| `1` | **kernel が ready でない**（拡張が無い／preflight 失敗／pointer table 欠落）ときに**起動を落とす**。形の辞退では落ちない。**本番に入れたまま安全**で、席が踏んだ級のバグ（`mcg`）はこれで必ず露見する |
| `2` | 形の辞退でも落ちる。**門専用** — 「MoE の全呼び出しがカーネルを通った」ことを保証するので、うっかり現行を測ることが起こり得ない |

`_prepare_lna_state` の `except` も、握り潰す代わりに **warning を出し、strict≥1 なら再送出**する
（`mcg` バグが隠れたのはまさにこの except）。fallback ハンドラ側でも strict は再送出する（§コード）。

### (c) 「実際に走った」を機械が証明する
`_MOE_CALLS` カウンタと `moe_kernel_call_counts()` / `reset_moe_kernel_call_counts()` /
`moe_kernel_declined_reasons()` / `lna2_scratch_bytes_resident()` を追加。
回帰テスト **`f1/gate_active.py`** が以下を主張する（MEASURED、run-60、全 PASS）:

```
PASS  ABI validation accepts a real exllamav3 LinearEXL3 (mcg_tensor)
PASS  _prepare_lna_state(backend=lna)  sets _exl3_lna_ready
PASS  _prepare_lna_state(backend=lna2) sets _exl3_lna_ready
PASS  backend=lna2 allocates only its own scratch (F1 entries=0)
PASS  _apply_lna2_moe returned a tensor
PASS  lna2 call counter incremented: {'lna': 0, 'lna2': 1}
PASS  row ceiling >= 32 so 制式 (16 rows) has headroom, got 32
PASS  lna2 accepts m=16 / m=20 / m=32
PASS  lna2 declines m=40 (> ceiling)
PASS  the decline is LOGGED with a reason
PASS  STRICT=1 does NOT raise on a shape decline (prefill must survive)
PASS  STRICT=2 raises on a shape decline (gate-only)
PASS  STRICT=1 raises when the kernel was never prepared
PASS  8 more layers add no lna2 scratch (6784384 -> 6784384 bytes)
```

**席の門で使う手順**: `VLLM_EXL3_MOE_STRICT=1` を付けて起動する。ready でなければ**起動時に落ちる**ので、
「走っていないのに門が通る」ことは構造的に起こらない。加えて `docker logs | grep 'F3 LNA2 MoE kernel ACTIVE'`
が 8 rank ぶん出ることを確認する。

---

## 3. (2) VRAM — 層あたりのコストを 0 にした

### 3.1 何が悪かったか
`_prepare_lna_state` は **backend に関係なく F1 の scratch を確保**し、lna2 の scratch を
**層ごとに**確保していた。1 層 1 stream あたり **3.320 MiB（F1）＋ 3.236 MiB（F3）＝ 6.556 MiB**。

### 3.2 直し方（二つ）
1. **backend 別**: `_prepare_lna_state(layer, inners, backend)` にして、`lna` は F1 の scratch だけ、
   `lna2` は F3 の scratch だけを確保する。lna2 のとき F1 の分は**一切確保しない**。
2. **層で共有**: lna2 の scratch は **`(device, stream)` ごとに 1 個**にした（`_LNA2_SCRATCH`）。
   根拠: この buffer は launch を跨いで何も保持しない — Hadamard/GEMV の staging は一時的で、
   scheduler 制御領域は**最後に退いた team が自分で 0 に戻す**。MoE 層は自分の stream 上で逐次に走るので
   共有して衝突しない。stream をキーに含めてあるので、shared-experts の aux stream を将来戻しても別 buffer になる。

### 3.3 算数（全部 MEASURED、run-60 `f1/gate_vram.py`）

| | bytes | MiB |
|---|--:|--:|
| `lna_moe_scratch_bytes()`（F1） | 3,481,616 | 3.320 |
| `lna2_moe_scratch_bytes()`（F3b、MAX_M=32 / MAX_PAIRS=192） | 6,784,384 | 6.470 |
| **席の版**: 1 層 1 stream（F1 3,481,616 ＋ F3@16行 3,392,896） | **6,874,512** | **6.556** |
| **F3b**: 86 層を prepare した実測増分 | **6,784,512** | **6.470** |
| F3b の層あたり償却 | — | **0.0752** |

パックの実測: **routed-expert 層 = 43**、**shared-expert 層 = 43**（`layers.N.ffn.experts.*` / `…shared_expert*` を索引から数えた）。
lna2 を通る層インスタンス数 `n` は席側でしか確定できないので、三通り並べる:

| `n` | 席の版が食っていた量 | F3b が解放する量 |
|--:|--:|--:|
| 43（routed のみ） | 281.9 MiB = 0.275 GiB | 275.4 MiB = 0.269 GiB |
| **72（席の不足 0.46 GiB から逆算）** | **472.0 MiB = 0.461 GiB** | **465.5 MiB = 0.455 GiB** |
| 86（routed＋shared） | 563.8 MiB = 0.551 GiB | 557.3 MiB = 0.544 GiB |

**逆算が効いている**: 席の不足は `1.83 − 1.37 = 0.46 GiB = 471 MiB`。
`471 / 6.556 = 71.9` — つまり **席では約 72 個の層インスタンスが scratch を持っていた**。
この一致は、原因の同定が正しいことの裏づけである（ESTIMATE の輪を実測が閉じた）。

### 3.4 ブートの見込み（正直に）

| `n` | F3b 後の KV 空き（ESTIMATE） | 389,120 に要る 1.83 GiB に対して |
|--:|--:|---|
| 72 | 1.37 + 0.455 = **1.825 GiB** | **約 5 MiB 足りない＝可否は五分**。席の 2 桁 GiB 表示は ±5 MiB 丸められているので、この差は丸めの中 |
| 86 | 1.37 + 0.544 = **1.914 GiB** | **約 86 MiB の余裕で立つ** |

**私には決められない。** TP8 は GPU10/11 では立たないので、発注どおり算数だけ置く。
**ケンのブート時に見てほしい行**:
- `available KV cache memory (X GiB)` — F3b では **1.37 → 1.82〜1.91 GiB** に増えるはず
- 立ったら `GPU KV cache size: N tokens` の **N**（これが「exact KV tokens number」。制式は ≥389,120 が要る）
- `docker logs dsv4 | grep -c 'F3 LNA2 MoE kernel ACTIVE'` が rank 数ぶん

**なお足りなかったときの手札**（順に安い）:
1. `--gpu-memory-utilization` を 0.97 → 0.972（+約 0.03 GiB）
2. `-DLNA2_MAX_M=16` で焼く → scratch 6.470 → 3.236 MiB（**さらに 3.2 MiB**）。ただし §4 の 16 行天井が戻るので、
   これは (3) の修理を捨てる取引になる。**推奨しない。**
3. scratch を実 max rows で runtime 可変にする（未実装。効くのは残り 3 MiB だけなので、いまは割に合わない）

---

## 4. (3) 16 行天井 — 32 へ上げた

`LNA2_MAX_M` を導入し既定 **32**（`MAX_PAIRS = 32 × topk6 = 192`）。
制式は 4 seq × DSpark3(1+3) = **16 行ちょうど**で天井に張り付いていた。32 なら **8 seq** まで余裕がある。

- 行ビットマスクは 1 expert あたり 32 bit 語 1 本のままなので構造の変更は無い（`unsigned` 化のみ）。
- `run_proj` は `ROWS_CAP=4` の行ブロックで回るので 32 行もそのまま通る。
- **プラグインは天井をカーネルに問い合わせる**（`lna2_moe_max_rows()`）。`16` のハードコードを消した — これが
  「一つ増やすと黙って戻る」の再発源だった。
- 天井を超えた辞退は **理由つきで必ずログに出る**（`geometry x=(40,4096) max_rows=32 …`）。

コスト（MEASURED, run-60）: static shared 15,176 → **21,128 B**、resident は **2 のまま**、
registers 128（spill 24/24 B）、**latency は雑音の中**（§5）。scratch は 3.236 → 6.470 MiB（§3、共有なので総額）。

### 新たに見つけた傷: 二 stream 同時実行で deadlock
gate4 の 2-stream stress が**固まった**（MEASURED, run-55/57）。原因は設計上の制約:
team barrier は **launch 全 CTA の同時常駐**を要求する（incumbent も同じ、`exl3_moe.cu:203-222`）。
既定は 8 team × 8 CTA = 64 CTA で、34 SM × resident 2 = 68 枠をほぼ使い切る。
二本重なると 128 CTA を要求して**片方が常駐できず、待ち合わせが永久に成立しない**。

対処（黙って固まらせない）:
- `LNA2_CONCURRENT_STREAMS`（既定 **1**）で grid を割る。`=2` で team 8 → 4（MEASURED）。
- その設定で 2-stream stress は **PASS**（worst_rel 7.97e-4、MEASURED run-59）。既定では当該テストは
  **明示的に SKIP と表示**して理由を出す（黙って通さない）。
- **席の現運用は `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` の単一 stream なので既定 1 で正しい。**
  aux stream を戻すなら `LNA2_CONCURRENT_STREAMS=2` が必須。

---

## 5. 門（この回）

### 5.1 parity — 変わらず全 PASS（MEASURED, run-53/60）

| 対象 | 件数 | 最悪 rel(FP32) | 判定 |
|---|--:|--:|---|
| lna2 合成（m ∈ 1..8,15,16,**17,20,24,31,32**） | **171**（旧 110） | 1.64e-3 | PASS |
| lna2 本番テンソル L0(K2)/L13,22,28(K3)、m ∈ {4,16,**32**} | 36 | ≤1.2e-3 | PASS |
| lna2 NaN poison replay | 48 | 9.8e-4 | PASS |
| lna2 `[m,6]` vs flat regression | — | ABI が拒否／転置で出力が変わる | PASS |
| **lna（F1）合成＋本番** | 110 | 1.70e-3 | PASS（**変化なし**） |

### 5.2 kernel-only latency — 雑音の中（MEASURED, run-51/60）

| K/m | F3（MAX_M=16） | **F3b（MAX_M=32）** | 同条件 incumbent 比 |
|---|--:|--:|--:|
| K2 m=4 | 192.6 | **178.2 – 190.5** | 1.76–1.89× |
| K2 m=16 | 622.7 | **624.7 – 626.8** | 1.78–1.82× |
| K3 m=4 | 241.7 | **225.3 – 239.7** | 1.53–1.61× |
| K3 m=16 | 784.5 | **788.5 – 789.6** | 1.52–1.54× |
| **K2 m=32（新規に走れる）** | 辞退 | **998.5** | **1.83×** |
| **K3 m=32（新規に走れる）** | 辞退 | **1276.0** | **1.53×** |

走行間の幅は熱とクロック（2490–2580 MHz）。**MAX_M=32 による劣化は見えない。**

### 5.3 scheduler（gate4）— 全 PASS（MEASURED, run-59）
active expert 数 1/2/7/8/9/16/96、ticket wrap 120 launch、NaN poison 24、
scaling 1.5 番兵（zero=0 / once=5.31785 / twice=7.97677、**比 1.500000**）、per-device prepared、
multi-stream（`LNA2_CONCURRENT_STREAMS=2` で PASS、既定では明示 SKIP）。

⚠️ gate4 の途中で**自分の計器で自分を殺した**: `lna2_moe_info` に 2 項目足したのに poison テストが
`[-2]` で ctrl 境界を取っていたため、**scheduler の barrier counter を NaN で塗って deadlock した**（run-54/55）。
名前引き（`INFO_FIELDS`）に直した。**負の添字で ABI を読むな**、というだけの話だが、実際に 15 分溶かした。

---

## 6. 変えたもの

| ファイル | 変更 |
|---|---|
| `csrc/lna_moe_ticket.cu` | `LNA2_MAX_M`（既定 32）、行マスク unsigned 化、`LNA2_CONCURRENT_STREAMS`、`lna2_moe_max_rows()`、`info` に max_m / max_pairs / concurrent_streams |
| `csrc/lna_moe_ticket.cuh`, `csrc/bindings.cpp` | `lna2_moe_max_rows` の口 |
| `src/vllm_exl3/exl3.py` | 席の 3 修正を維持。＋ `moe_kernel_strict()` 二段、`_lna1_log_reject`、`_MOE_CALLS` と 4 つの accessor、`_prepare_lna_state(..., backend)`、`_LNA2_SCRATCH` を層横断で共有、`lna2_max_rows()`（ハードコード 16 の廃止）、preflight 失敗の warning ＋ strict 再送出 |
| `f1/gate_active.py` | **新規**。黙って落ちる回帰を封じるテスト |
| `f1/gate_vram.py` | **新規**。scratch の算数を実測する |
| `f1/gate4.py` | INFO の名前引き、参照計算の間引き（120 launch は全部走らせ、参照は 15 回）、multi-stream の明示 SKIP |
| `f1/parity_native.py` | lna2 のとき m を 17/20/24/31/32 まで伸ばす（F1 は 16 のまま） |
| `f1/REPORT-F3.md` | §8 の危険 2（aux stream）に §4 の実測を追記 |

---

## 7. 席に残すもの（私には測れない）

1. **制式 389,120 でのブート**（§3.4）。見る行と手札は書いた。**これが唯一の未解決の可否。**
2. **`VLLM_EXL3_MOE_STRICT=1` を付けた席の門一式**（strict=1 は本番でも安全な設計にしてある）
3. **compute-sanitizer 4 種** — 像に無い（変わらず）。device atomics と team barrier があるので racecheck が要る
4. **8 rank の P50/P95・最遅 rank**、graph ON/OFF parity、NCCL bytes、166k 針、Vision、thermal steady
5. **台帳**: 制式 ppl 6.7159 を出した `ppl-vllm.py` の引数（席報告 §6 の宿題）
6. **掟**: 派生モデルは素の DL 先へ symlink を張る。素を消すと派生が全部死ぬ（席報告 §7）
