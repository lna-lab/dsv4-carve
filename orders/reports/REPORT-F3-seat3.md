# REPORT F3-seat3 — 一本化した木での席の門（lna2 ＋ T1 併用の可否）

検分: ユキ（Fable 5.1）／2026-09-04／席 = オオタニ（DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6、TP8 GPU 0,1,2,3,5,7,8,9、:8899）
対象: 一本化した本番の木 `vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py`（F3c 移植 ＋ Luna の T0/T1）
＋ port の `.so`（`vllm_exl3_c…so`、15:23 版）／入力: `f1/REPORT-F3c.md`・`orders/REPORT-F2-T1.md`
前回: `f1/REPORT-F3-seat2.md`。すべての数字に **MEASURED / ESTIMATE** を付す。
GPU10/11・`f2/` には触れていない。席は最後に制式へ戻してある（§5）。

---

## 0. 採決

| 門 | 結果 | 数字 |
|---|---|---|
| **lna2 単体が制式 UTIL=0.97 で立つか** | **PASS（本日の朗報）**（MEASURED） | **立った。** Available KV **1.86 GiB**、**GPU KV cache size 395,069 tokens**、ACTIVE 8/8、strict 発火なし |
| **lna2 ＋ T1（DENSE_GROUP=2）** | **FAIL — 起動しない**（MEASURED） | KV が **0.20 GiB**（0.97）／**0.23 GiB**（0.972）に潰れる。`estimated maximum model length` は **496／576** |
| lna2 ＋ T0（DENSE_GROUP=1、代替として実施） | **FAIL — 起動しない**（MEASURED） | KV **0.52 GiB**（0.97）／**0.55 GiB**（0.972）。最大長 29,184／37,632 |
| **ppl（この木・spec OFF・対照つき）** | **PASS**（MEASURED） | lna2 **4.7647** vs incumbent 対照 **4.7630**、**Δ +0.0017**（門 ±0.02） |
| 制式復帰 | **PASS**（MEASURED） | KV 396,656 tok・health 200・mount と env を数えて確認・warm 済 |
| bench / essays / needle（併用形） | **測定不能** | 候補席が起動しないため |

**結論: 今日の一本化で「lna2 単体」は制式の窓を UTIL=0.97 のまま保って立つようになった。
ここは大きい。一方 T1 も T0 も、いまの木では席に載らない — 速さ以前に GPU あたり 1.3〜1.7 GiB を食う。
原因は特定できていて、直し方も小さい（§3）。**

---

## 1. 門 1 — lna2 単体（一本化した木、MEASURED）

構成: `PLUGIN_SRC=vllm-exl3-lab/vllm-exl3/src/vllm_exl3`（本番）、
`NATIVE_SO=vllm-exl3-v030-port/src/vllm_exl3_c…so`、`MOE_KERNEL=lna2`、`-e VLLM_EXL3_MOE_STRICT=1`、
制式引数（`MAXLEN=389120 SEQS=4 BT=512 DSpark3`）。

| UTIL | Available KV cache memory | GPU KV cache size | 起動 |
|---|--:|--:|---|
| **0.97（制式のまま）** | **1.86 GiB** | **395,069 tokens** | **PASS** |
| 参考: incumbent 制式 @0.97 | — | 396,656 tokens | — |
| 参考: F3b（前回）@0.97 | 1.82 GiB | — | 不起動 |
| 参考: F3b @0.972 | 1.85 GiB | 393,483 tokens | 起動 |

**F3c の VRAM 追い込みは線を跨いだ。** 1.82 → **1.86 GiB**（+0.04）。
前回「0.01 GiB 足りない」で落ちていたものが、**UTIL を動かさずに立つ**ようになった。
KV は 396,656 → **395,069 tok（−0.4%）**。前回 0.972 で得た 393,483（−0.8%）より良い。

**→ 前回の報告で「ケンの決裁事項」として挙げた『UTIL を 0.97→0.972 に動かすか』という問いは、消えた。**

- `F3 LNA2 MoE kernel ACTIVE` = **8/8 rank**
- `VLLM_EXL3_MOE_STRICT=1` で **Traceback 0**（＝全 rank で kernel が ready）
- 辞退は **プリフィルの `x=(512,4096)` のみ**、`max_rows=24` を報告（F3c の MAX_M=24）。
  制式の decode は 16 行なので **50% の余裕**。decode の辞退は 0。

ケンの指示により、この段の bench / numerics は**実施していない**（単体は 103.2 tok/s で測定済のため）。

---

## 2. 門 2 — lna2 ＋ T1 は席に載らない（MEASURED）

`-e LNA_EXL3_DENSE_GROUP=2 -e LNA_EXL3_DENSE_STRICT=1` ＋ **現行 recipe から作り直した** `attention.py`
（Luna の版つきマーカ `_lna_kv_score_v2` を確認。像の素 `attention.py.orig-lna2` から再生成、42,217 B、`compile()` 通過）。

**観測（MEASURED）**:

| 構成 | Available KV | 最大長 | lna2 ACTIVE | F2 DENSE ACTIVE | 起動 |
|---|--:|--:|--:|--:|---|
| lna2 単体 @0.97 | **1.86 GiB** | — | 8 | 0 | PASS |
| lna2 ＋ **T0**（mode 1）@0.97 | **0.52 GiB** | 29,184 | 8 | 8 | **FAIL** |
| lna2 ＋ T0 @0.972 | 0.55 GiB | 37,632 | 8 | 8 | **FAIL** |
| lna2 ＋ **T1**（mode 2）@0.97 | **0.20 GiB** | 496 | 8 | **16**（mode 1・2 の両方） | **FAIL** |
| lna2 ＋ T1 @0.972 | 0.23 GiB | 576 | 8 | 16 | **FAIL** |

**観測されたコスト（差分、MEASURED）**: T0 = **1.34 GiB/GPU**、T1 = **1.66 GiB/GPU**。

**良い知らせ**: Luna の可視化はちゃんと効いた。**`F2 DENSE_GROUP=1 ACTIVE` と `F2 DENSE_GROUP=2 ACTIVE` が
両方・全 rank で出た**。前回のような「黙って落ちる」は起きていない。
Traceback 3 件はすべて **KV 不足の同じ `ValueError` の連鎖**で、コードの不具合ではない。
つまり**融合そのものは噛んでいる。載らないのはメモリだけ**である。

---

## 3. 原因の特定（MEASURED ＋ コード）

`vllm-exl3/src/vllm_exl3/exl3.py`:

```python
def _prepare_dense_group(group: dict) -> None:
    """Allocate every decode bucket before graph capture or steady-state use."""
    for m in range(1, 17):
        _dense_group_scratch(group, m, torch.float32, allocate=True)
```

`_dense_group_scratch` は bucket ごとに **A_had `(n_mat, m, k)` fp16 ＋ 出力 `(m, n_j)` fp32** を
**恒久バッファ**として確保する。`m=1..16` を全部持つので、1 グループあたり
Σm = 136 倍のコストになる（算術、MEASURED 形状より）:

| グループの形 | A_had | 出力 | 計 |
|---|--:|--:|--:|
| n_mat=2（T0 の 2 shard） | 2.12 MiB | 0.80 MiB | **2.92 MiB** |
| n_mat=4（奇数層 h-fan） | 4.25 MiB | 1.33 MiB | **5.58 MiB** |
| n_mat=6（偶数層 h-fan） | 6.38 MiB | 1.59 MiB | **7.97 MiB** |

**ところが vLLM が capture する decode bucket は `[1,2,4]` だけ**（`serve-dsv4-tp8.sh` の
`cudagraph_capture_sizes`）。**bucket 3 と 5..16 は確保されるが graph には入らない。**
制式の decode 行数は spec 込みで 16 なので m=16 は要るが、**間の 11 本は要らない。**

**直し方（小さい）**: 確保を `capture_sizes ∪ {実 decode 行数}` に絞る、
または **max m の buffer を 1 本持って行方向に slice する**（形は `(n_mat, m, k)` の先頭 m 行なので
そのまま view で足りる）。前者だけで **136 → 1+2+4+16 = 23、約 5.9 分の 1**（ESTIMATE）。
T1 の 1.66 GiB は **約 0.28 GiB** になり、lna2 の 1.86 GiB から引いても **1.58 GiB** ——
まだ 1.83 に足りない（ESTIMATE）。**後者（single buffer + slice）まで行くと 136 → 16、
約 8.5 分の 1 で 0.20 GiB、残り 1.66 GiB でも足りない。**
→ **bucket 削減だけでは届かない見込み。** グループ数そのものか、A_had の持ち方
（層をまたいだ共有。lna2 が §F3c §4.1 でやったのと同じ手）が要る。

### ⚠️ これは今朝からの後退である（MEASURED）

**今朝の T0 の席の門では、incumbent ＋ `DENSE_GROUP=1` が `UTIL=0.97`・`MAXLEN=389120` で
問題なく起動していた**（`orders/REPORT-F2-T0-seat.md` の G2b/G2c、UP=1）。
incumbent は lna2 の scratch を持たないので、当時の T0 のコストは **0.04 GiB 未満**だったはずである。
**それが今日の木では 1.34 GiB。約 33 倍に膨らんでいる。**

私は**この木を編集していない**（Luna が作業中の共有ファイルであり、F3c §5 が警告した
同時編集の事故を繰り返さないため）。**どのハンクで膨らんだかは、書いた人が diff を見るのが速い。**
`_prepare_dense_group` の `range(1, 17)` が新しいのか、グループ数が増えたのか、その両方かは**未確定**。

---

## 4. 表 — incumbent / lna2 / lna2+T1（出所つき）

**⚠️ 出所が違う行を混ぜないよう明示する。** 本日ケンの指示で lna2 単体の bench は再測していないので、
lna2 の行は **F3b の port ビルド**（`REPORT-F3-seat2.md`、UTIL=0.972）の実測である。
一本化した木では **kernel 単独 latency は不変**（F3c §6、雑音の中）で **ppl も一致**（下）だが、
**席の tok/s はこの木で測り直していない。**

| 構成 | code1 | en1 | ja1 | code4 | en4 | ja4 | KV tokens | UTIL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **incumbent 制式**（本日 4 走の帯） | 81.3–87.2 | 59.0–64.6 | 54.2–56.6 | 125.4–142.6 | 94.9–105.0 | 81.1–85.9 | 396,656 | 0.97 |
| **lna2 単体**（F3b build 実測） | **103.2** | **73.1** | **67.3** | 142.4 | 107.5 | 96.6 | 393,483 | 0.972 |
| **lna2 単体**（本日の木・起動のみ確認） | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | 未測定 | **395,069** | **0.97** |
| **lna2 ＋ T1** | — | — | — | — | — | — | **起動せず** | 0.97 / 0.972 |
| 制式復帰の確認 | 87.2 | 59.5 | 56.4 | 129.1 | 96.1 | 83.3 | 396,656 | 0.97 |

**家計簿（prof-shapes）は候補席が起動しないため採れていない。**
T1 の目標「密 ≤265 launch/step（見込み 258）」は **未検証のまま**である。
比較用に持っている実測は、今朝の incumbent **489.0/step・8.8938 ms/step** と
T0 ON **341.0/step・7.3297 ms/step**（`REPORT-F2-T0-seat.md`）まで。

---

## 5. 品質の門

### 5.1 ppl — この木で初めて測った（MEASURED、対照つき）

`ppl-vllm.py 8899 wiki.test.raw 4096 8`、**spec OFF**、UTIL=0.97、同じ引数の incumbent 対照とセット。

| 構成 | PPL | tokens |
|---|--:|--:|
| **lna2（一本化した木）** | **4.7647** | 30,002 |
| **incumbent 対照（同じ引数）** | **4.7630** | 30,002 |
| **差** | **+0.0017** | — |

**門 ±0.02 に対し差はその 1/12。PASS。** 前回報告で「このビルドで ppl を測っていない」と挙げた
宿題は**これで消えた**。（絶対値が記録の 6.7159 と違うのは前回同様 `ppl-vllm.py` の引数差で、
判定に使えるのは対照との差だけ。6.7159 の引数を台帳に書く宿題は残っている。）

lna2 側の spec-OFF 席は Available KV **2.84 GiB / 603,512 tokens**（draft を持たないぶん余裕がある）、
ACTIVE 8/8、Traceback 0。

### 5.2 その他

bench / numerics / essays / 受理長 / 166k 針は、**候補（併用）席が起動しないため実施できていない**。
lna2 単体ぶんは前回ビルドで全部 PASS しているが、**一本化した木では未再測**（§4 の注記）。

---

## 6. 制式復帰（MEASURED）— 数えて確認した

前回 mount を数え忘れて像の plugin で上げてしまった反省から、**今回は数えた**:

| 確認項目 | 期待 | 実測 |
|---|--:|--:|
| plugin mount（`vllm-exl3/src/vllm_exl3`） | 1 | **1** |
| `attention.py` の mount | 0 | **0** |
| 像の `attention.py` の `run_exl3_group` | 0 | **0**（stock） |
| `MOE_KERNEL` / `DENSE_GROUP` / `MOE_STRICT` env | 0 | **0** |
| ログの `F3 LNA2` / `F2 DENSE_GROUP` 行 | 0 | **0** |
| `GPU KV cache size` | 396,656 | **396,656 tokens** |
| health | 200 | **200** |
| warm | 実行 | `prompt_tokens=12269 took=9.9s` |

復帰後 bench: code1 87.2 / en1 59.5 / ja1 56.4 / code4 129.1 / en4 96.1 / ja4 83.3 — incumbent の帯。

---

## 7. 「ケンの制式判断に出せるか」

### **lna2 単体 → ほぼ Yes。** 前回の No の理由は三つとも消えた。

| 前回の No の理由 | いま |
|---|---|
| ① UTIL を 0.97→0.972 に動かす必要 | **消えた**。0.97 のまま立つ（KV −0.4%） |
| ② T0 と併用できない（木が別） | **木は一本になった**。併用が起動しないのは別問題（§3） |
| ③ ppl 未測定 | **測った。Δ +0.0017 で PASS** |

**残って足りないもの（正直に）**:

1. **一本化した木での席の速さを測っていない。** ケンの指示で単体 bench を飛ばしたため、
   `103.2 tok/s` は **F3b の port ビルドの数字**である。移植は「既存関数を書き換えず 3 か所挿入」
   （F3c §5、+583/−0 行）で kernel latency も ppl も一致しているので**同じはず**だが、
   **ESTIMATE であって MEASURED ではない**。制式に据えるなら**この木で一度は測るべき**。
   同じ理由で **166k 針もこの木では未実施**。
2. **8 rank の P50/P95・最遅 rank**（NCCL の判定に必須）— 変わらず未測定。
3. **compute-sanitizer 4 種**（device atomics ＋ team barrier があるので racecheck）— 像に無い。
4. graph ON/OFF parity、Vision、thermal steady、`LNA2_CONCURRENT_STREAMS=2` の席確認。

### **lna2 ＋ T0/T1 → No。** 席に載らない。§3 の scratch を先に直すこと。

### 進め方の推奨

1. **dense-group scratch を直す**（§3）。bucket を capture サイズ＋実 decode 行に絞るのは簡単だが、
   **算術では届かない見込み**なので、**層をまたいだ共有**（lna2 が F3c §4.1 でやった手）まで要る。
   **今朝 0.04 GiB 未満だったものが 1.34 GiB になっている**ので、まず diff を見るのが速い。
2. **一本化した木で lna2 単体の bench と 166k 針を一度**（1 ブートで済む）。これで ① が消える。
3. そこまで来れば **lna2 単体はケンに出せる**。判断材料は「KV を 0.4% 削って単流 +20% を取るか」だけになる。
4. T1 の launch ≤265 の検証は、scratch が直ってからの宿題。

---

## 8. 使った物

| もの | 中身 |
|---|---|
| `.tmp/f3c-boot.sh` | 一本化した木用のブート共通部（KV 行・ACTIVE・辞退の抽出つき） |
| `.tmp/f3c-g1.sh` / `f3c-g2.sh` / `f3c-g2b.sh` / `f3c-g3.sh` | 各段 |
| `.tmp/f3c-vllmroot/models/deepseek_v4/attention.py` | **現行 recipe から作り直した** patched attention.py（`_lna_kv_score_v2`） |
| `.tmp/essays7.py` | 予算を足した作文 7 本 |
| `.tmp/f2-count.py` | 家計簿計量器（今回は候補が起動せず未使用） |
| `.tmp/bench3-restored-f3c.txt` | 復帰後 bench |
