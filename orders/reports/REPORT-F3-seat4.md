# REPORT F3-seat4 — lna2 ＋ T1 併用の席の門（一本化した木・scratch 修理後）

検分: ユキ（Fable 5.1）／2026-09-04／席 = オオタニ（DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6、TP8 GPU 0,1,2,3,5,7,8,9、:8899）
対象: 本番の木 `vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py`（16:32 版、Luna の shared-pool 修理入り）
＋ port の `.so`（15:23 版）／前回: `f1/REPORT-F3-seat3.md`
すべての数字に **MEASURED / ESTIMATE** を付す。席は最後に制式へ戻してある（§6）。GPU10/11・`f2/` は触っていない。

---

## 0. 採決

| 門 | 結果 | 数字 |
|---|---|---|
| **併用が制式 UTIL=0.97 で立つか** | **PASS**（MEASURED） | **立った。** KV **1.86 GiB / 394,474 tokens**、ACTIVE 両方 8/8、strict 発火なし、Traceback 0 |
| **VRAM（前回の 1.66 GiB 事故）** | **PASS — 直った**（MEASURED） | 前回 KV 0.20 GiB → 今回 **1.86 GiB**。lna2 単体（1.86）と同じ。**T1 の追加コストは実質ゼロ** |
| **(a) 最高速度＝単流 code** | **109.1 tok/s**（MEASURED） | ケンの的 120 に対し **91%** |
| **(b) 飛距離** | **PASS**（MEASURED） | **394,474 KV tokens @ UTIL=0.97** ＋ **166k 針 正答**（TTFT 119.6 s） |
| 数値・作文・受理長 | **PASS**（MEASURED） | 分岐 41–130（帯 15–136）、作文 **7/7 stop**、受理長 2.25/1.86/3.07 |
| **T1 の launch 削減（≤265）** | **FAIL**（MEASURED） | 密 **489.0/step**、`exl3_mgemm_kernel` **0 件**。融合は decode で一度も走っていない |
| **T1 の速度寄与** | **無し。4 流は劣化**（MEASURED） | 単流は lna2 単体と同値。**4 流は −17%**（161.6 vs 193.8） |
| 制式復帰 | **PASS**（MEASURED） | KV 396,656・health 200・mount/env を数えて確認・warm 済 |

**結論: VRAM 事故は直った。併用席は制式の窓のまま立ち、品質も飛距離も揃っている。
だが T1 の融合は decode で走っていない — 走らないだけでなく、4 流では足を引っ張っている。
いま「併用」を制式にするのは、lna2 単体より遅い席を選ぶことになる。**

---

## 1. 門 A — 起動（MEASURED）

構成: `PLUGIN_SRC=`本番の木、`NATIVE_SO=`port の `.so`、`MOE_KERNEL=lna2`、
`-e VLLM_EXL3_MOE_STRICT=1 -e LNA_EXL3_DENSE_GROUP=2 -e LNA_EXL3_DENSE_STRICT=1`、
**現行 patch script から作り直した** `attention.py`（`_lna_kv_score_v2`、42,209 B、`compile()` 通過）、
制式引数（`MAXLEN=389120 SEQS=4 BT=512 UTIL=0.97 DSpark3`）。

| 構成 | Available KV | GPU KV cache size | 起動 |
|---|--:|--:|---|
| **lna2 ＋ T1（本日）** | **1.86 GiB** | **394,474 tokens** | **PASS @0.97** |
| lna2 単体（本日、同じ木） | 1.86 GiB | 395,069 tokens | PASS @0.97 |
| incumbent 制式 | — | 396,656 tokens | PASS @0.97 |
| 参考: **前回の lna2＋T1** | **0.20 GiB** | — | **不起動** |

**Luna の shared-pool 修理は効いた。** 前回 T1 が食っていた **1.66 GiB/GPU がほぼ消え**、
lna2 単体と同じ 1.86 GiB。KV は 395,069 → 394,474（**−595 tok、−0.15%**）で、
**T1 を足したぶんの飛距離の損は誤差の域**。制式 396,656 に対しても **−0.55%**。

- `F3 LNA2 MoE kernel ACTIVE` **8/8**、`F2 DENSE_GROUP=1 ACTIVE` **8/8**、`F2 DENSE_GROUP=2 ACTIVE` **8/8**
- **Traceback 0**（`VLLM_EXL3_MOE_STRICT=1` と `LNA_EXL3_DENSE_STRICT=1` の両方を立てて）
- lna2 の辞退は形のみ（`x=(512/100/25, 4096)`、`max_rows=24`）。制式 decode の 16 行は辞退 0

---

## 2. 門 B — 速さ（MEASURED、bench-streams 3 回平均 ±半幅）

| 構成 | code1 | en1 | ja1 | code4 | en4 | ja4 | 出所 |
|---|--:|--:|--:|--:|--:|--:|---|
| **incumbent 制式**（本日の帯、5 走） | 81.3–87.2 | 59.0–64.6 | 54.2–57.8 | 125.4–178.6 | 94.9–132.9 | 81.1–111.3 | 本日 MEASURED |
| **lna2 単体（本番の木）** | **108.6** (±0.9) | **78.6** (±5.0) | **71.1** (±0.4) | **193.8** (±3.7) | **146.4** (±3.2) | **129.0** (±0.5) | **本日 MEASURED** |
| **lna2 ＋ T1** | **109.1** (±0.8) | **78.1** (±5.0) | **70.9** (±0.6) | **161.6** (±1.5) | **116.6** (±3.4) | **101.8** (±2.4) | **本日 MEASURED** |
| 参考: lna2 単体（F3b port build） | 103.2 | 73.1 | 67.3 | 142.4 | 107.5 | 96.6 | 前回 MEASURED |
| 制式復帰の確認 | 86.1 (±2.4) | 61.5 (±4.4) | 57.8 (±1.6) | 178.6 (±5.5) | 132.9 (±4.6) | 111.3 (±0.5) | 本日 MEASURED |

### 読み方（正直に）

1. **前回の宿題が消えた**: 一本化した木での lna2 単体を測った。**108.6 tok/s** で、
   F3b の port build（103.2）**より速い**。移植で失っていない — これは ESTIMATE でなく MEASURED になった。
2. **T1 は単流に何も足していない**: 109.1 vs 108.6、en 78.1 vs 78.6、ja 70.9 vs 71.1。**すべて雑音の中。**
3. **T1 は 4 流を明確に落としている**: code4 **161.6 vs 193.8（−17%）**、en4 116.6 vs 146.4（−20%）、
   ja4 101.8 vs 129.0（−21%）。**この二つは 8 分違いの連続ブート・同じクロック帯（平均 2495 / 2498 MHz）**なので、
   条件差ではなく **T1 を入れたことによる劣化**である。§4 に原因。
4. ⚠️ **incumbent の 4 流は本日 125.4〜178.6 と大きく振れた**（同じ席・同じ引数）。
   4 流の絶対比較は当てにならない。**上の 3 は同一セッションの連続 A/B なので信用できる**が、
   incumbent との 4 流比較は幅で読むこと。
5. **単流のクロックは候補側が不利**（2495–2498 MHz vs 復帰後 2561 MHz）。それでこの差である。

**ケンの (a) 最高速度 = 単流 code = 109.1 tok/s。的 120 まで残り 10%。**

---

## 3. 門 C — 家計簿（prof-shapes、rank0、graph ON、decode 23 step、MEASURED）

| bucket | incumbent（本日朝） | **lna2 ＋ T1** | 差 |
|---|--:|--:|--:|
| **密 `exl3_gemm_kernel<6>`** | 8.894 ms ／ **489.0**/step | 8.851 ms ／ **489.0**/step | **変化なし** |
| **`exl3_mgemm_kernel`（融合）** | — | **0 件** | **出ていない** |
| **MoE** | 10.002 ms ／ 43.0 | **4.474 ms** ／ 43.0 | −5.53 ms |
| **routed µs/launch** | **232.6** | **104.0** | **2.24×** |
| NCCL | 2.436 ms（8.9%） | 2.430 ms（**11.6%**） | 絶対値ほぼ同、割合は分母が縮んだぶん上昇 |
| other | 4.175 ／ 1050.5 | 3.892 ／ 878.5 | −172 launch/step |
| aux: elementwise | 1.813 ／ 1500.0 | 1.249 ／ 1027.0 | −473 launch/step |
| aux: cat | 0.141 ／ 149.0 | 0.138 ／ 149.0 | 同 |
| **decode 全 kernel** | **27.460 ms/step** | **21.034 ms/step** | **1.306×** |

**T1 の門「密 ≤265 launch/step（見込み 258）」は FAIL。実測 489.0 で、朝の incumbent と同じ。**
家計簿に現れている改善は**すべて lna2 のもの**（routed 2.24×、補助カーネル減）。
**T0 の 341/step（本日朝 MEASURED）にすら届いていない。**

---

## 4. なぜ T1 が走らないか（MEASURED — Luna の大声ログがそのまま答えになった）

容器ログの辞退理由（件数つき）:

| 件数 | 理由 |
|--:|---|
| **5,640** | `scratch: shared pool is busy with language_model.model.layers.2.attn:hfan` |
| **1,720** | `registration: T1 fan is not complete; using the registered local T0 group` |
| 64 | `input: new input arrived before the prior owner generation was complete (group=…layers.2.attn:hfan count=1..8)` |

読み方:

1. **VRAM を救った shared pool が、こんどは排他の首輪になっている。** プールは 1 個で
   **一度に一つのグループしか持てない**。ログはほぼ全部
   「layers.2 の h-fan がプールを握っていて空かない」で、**他の層の fan が全部弾かれている**。
   融合が decode で一度も成立しないのはこれ（§3 の 489/step と整合）。
2. **T1 の fan 登録が完結していない**（1,720 件）ので、T1 は T0 の局所グループに降りようとする。
   ところがその T0 も 1 の排他で弾かれる。**結果として全部 solo に落ちる。**
3. 4 流で悪化するのも同じ理屈で説明がつく: **流が増えるほどプールの取り合いが増え、
   毎回「busy」を確かめてから solo に落ちる**ぶんが純粋な足枷になる（§2 の −17〜−21%）。

**前回（メモリで落ちた版）と今回（メモリは足りるが排他で走らない版）で、
T1 はまだ一度も席で融合できていない。** ただし **ACTIVE ログと理由つき辞退のおかげで、
今回は「効いているつもりで測る」事故は起きなかった** — 可視化の投資はここで回収された。

**次の的（ESTIMATE）**: プールをグループ単位でなく **(device, bucket) 単位で複数持つ**か、
**層ごとに小さな専用領域を切って共有をやめる**か。前回の 1.66 GiB は
`m=1..16` の 136 倍が原因だったので、**bucket を capture サイズ＋16 に絞れば
共有をやめても 5.9 分の 1 で収まる**見込み（前回報告 §3 の算術）。**排他と容量を同時に満たす道はまだある。**

---

## 5. 門 D — 品質と飛距離（MEASURED）

| 門 | 結果 |
|---|---|
| **貪欲一致**（帯 15–136） | inc vs combo: en **130** / ja **54** / code **85** / mix **41**。inc2 vs combo: 107 / 15 / 86 / 41。**全部帯の中。PASS** |
| **作文 7 本** | **7/7 finish=stop**（予算を足した版）。空返事なし。JA essay 730 tok / CODE 591 tok ほか |
| **受理長** | en **2.25** / ja **1.86** / code **3.07** — incumbent の当日帯（2.04–2.34 / 1.79–2.02 / 2.93–3.17）の中。**変化なし** |
| **166k 針** | **正答 `sazanami-hotaru-2026`**、TTFT **119.6 s**（697,460 字）。lna2 単体も 119.7 s。制式記録は 167 s |
| **KV tokens @ UTIL=0.97** | **394,474**（制式 396,656 の **99.45%**） |

**ケンの (b) 飛距離は満点**: 制式の窓 389,120 を UTIL を動かさずに保ち、166k の針も通る。

---

## 6. 制式復帰（MEASURED）— 数えて確認

| 確認 | 期待 | 実測 |
|---|--:|--:|
| plugin mount | 1 | **1** |
| `attention.py` mount | 0 | **0** |
| 像の `attention.py` の `run_exl3_group` | 0 | **0** |
| `MOE_KERNEL`/`DENSE_GROUP`/`MOE_STRICT` env | 0 | **0** |
| ログの `F3 LNA2` / `F2 DENSE_GROUP` | 0 | **0** |
| `GPU KV cache size` | 396,656 | **396,656 tokens** |
| health | 200 | **200** |
| warm | 実行 | `prompt_tokens=12269 took=9.8s` |

---

## 7. 「ケンの制式判断に出せるか」

### 併用席（lna2 ＋ T1）→ **No。**

立つし、品質も飛距離も問題ない。**しかし T1 は仕事をしていない上に 4 流を 17〜21% 落とす。**
**併用を選ぶ理由が現時点で一つも無い**（単流は同値、4 流は劣化、launch は不変）。
T1 は §4 の排他を直してから、もう一度席に来るべきである。

### lna2 単体 → **Yes、出せる。**

前回の報告で残していた宿題が、今日ぜんぶ埋まった:

| 前回「足りない」と書いた項目 | いま |
|---|---|
| 一本化した木での席の速さが未測定（ESTIMATE） | **MEASURED: 単流 code 108.6 / en 78.6 / ja 71.1**。port build より速い |
| 一本化した木での 166k 針が未実施 | **MEASURED: 正答、TTFT 119.7 s** |
| UTIL を 0.97→0.972 に動かす必要 | **不要。0.97 のまま KV 395,069 tok** |
| ppl 未測定 | **前回 MEASURED: Δ +0.0017（門 ±0.02）** |

**ケンに差し出す判断材料はこれだけ**:
**「KV を 396,656 → 395,069（−0.4%）削るかわりに、単流を +26%（86.1 → 108.6 tok/s）取るか」。**
飛距離は 389,120 の窓を保ったまま、166k の針も通る。品質（ppl・受理長・作文・貪欲）はすべて不変。

**それでも残っている穴（ケンが呑むかを知った上で決めるべきもの）**:
1. **8 rank 全部の P50/P95 と最遅 rank** — 未測定。家計簿は rank0 のみ。NCCL の割合が
   11.6% に上がっているので、**遅い rank がいれば真っ先にここに出る**。
2. **compute-sanitizer 4 種**（racecheck）— 像に無い。device atomics ＋ team barrier を使う以上、
   これは「まだ見ていない」と正直に言うしかない。
3. graph ON/OFF parity、Vision、thermal steady、`LNA2_CONCURRENT_STREAMS=2` の席確認。
4. **4 流の測定が本日不安定**（incumbent で 125.4–178.6）。単流の分離は明確だが、
   **4 流の利得を数字で約束しない方がよい。**

**私の推薦: lna2 単体を制式候補としてケンに出す。T1 は席から一度下ろす。**

---

## 8. 使った物

| もの | 中身 |
|---|---|
| `.tmp/f3d-boot.sh` / `.tmp/f3d-run.sh` | 併用席のブートと門（A 起動＋家計簿 / B 速さ・品質 / C lna2 単体対照 / D 復帰） |
| `.tmp/f3d-vllmroot/models/deepseek_v4/attention.py` | 現行 patch script から作り直した patched attention.py |
| `.tmp/f2-count.py` | decode step 窓で区切る家計簿計量器 |
| `.tmp/f3c-trace-combo-rank0.json.gz` | 併用席 rank0 の生トレース |
| `.tmp/bench3-{combo-t1,lna2-prod,restored-f3d}.txt` | bench の生ログ |
| `.tmp/numerics-combo.json` / `.tmp/essays7.py` | 貪欲生成・作文 |
