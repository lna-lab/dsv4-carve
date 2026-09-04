# REPORT F3-seat2 — F3b 修理版 lna2 の席の門（＋ T0 との併用試験）

検分: ユキ（Fable 5.1）／2026-09-04／席 = オオタニ（DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6、TP8 GPU 0,1,2,3,5,7,8,9、:8899）
対象: `vllm-exl3-v030-port` の **F3b**（`REPORT-F3b.md`）／前回: `f1/REPORT-F3-seat.md`
すべての数字に **MEASURED / ESTIMATE** を付す。席は最後に制式へ戻してある（§6）。
GPU10/11・`f2/` には触れていない（Luna の T1 作業中）。

---

## 0. 採決

| 門 | 結果 | 一行 |
|---|---|---|
| A. 制式 389,120 でブート | **条件付き PASS**（MEASURED） | **UTIL=0.97 では 0.01 GiB 足りず不起動。UTIL=0.972 で起動**、KV **393,483 tok** |
| A. 三つの修理の実証 | **PASS**（MEASURED） | VRAM は 1.37→**1.82 GiB**（+0.45、職人 ESTIMATE 1.825 と一致）。ACTIVE 8/8、strict 発火なし、16 行天井は解消 |
| B. bench-streams | **PASS**（MEASURED） | 単流 **1.20–1.22×**、**帯が完全に分離** |
| B. 数値・受理長・作文・166k 針 | **PASS**（MEASURED） | 針 **正答**、TTFT 123.5 s（制式記録 167 s）。作文 7/7 stop |
| C. lna2 ＋ T0 の併用 | **⚠️ 測定できず（FAIL to combine）** | **二つの改修は別々の plugin 木にあり、同時に載らない**。§4 |
| C. ケンの 120 tok/s | **未達**（MEASURED） | 単流 code **103.2**（目標比 86%）。併用が成っても **ESTIMATE 約 111** で届かない |
| D. 制式復帰 | **PASS**（MEASURED） | KV **396,656 tok**・health 200・warm 済 |

**結論: F3b は席の指摘三点を全部直しきった。lna2 単体は数字も品質も揃っている。
ただし「制式にしてよいか」への答えは、いまは **No（あと一歩）**。
理由は速さでも品質でもなく、(1) 制式の UTIL を 0.97→0.972 に動かす必要がある、
(2) **T0 との併用が構造的にまだできない**、(3) ppl をこのビルドで測っていない、の三点（§7）。**

---

## 1. 門 A — 制式 389,120 でのブート（MEASURED）

構成 = 制式そのまま（`MAXLEN=389120 SEQS=4 BT=512 DSpark3`）＋ port の plugin/`.so` ＋
`MOE_KERNEL=lna2` ＋ `-e VLLM_EXL3_MOE_STRICT=1`。

| UTIL | Available KV cache memory | GPU KV cache size | 起動 |
|---|--:|--:|---|
| **0.97**（制式） | **1.82 GiB** | — | **不起動**。要 1.83 GiB、`estimated maximum model length 386,048` |
| **0.972** | **1.85 GiB** | **393,483 tokens**（389,120 に対し 1.01×） | **起動・health 200** |
| 参考: 制式 incumbent @0.97 | — | **396,656 tokens** | — |

**職人の算数は当たった。** 前回の席は 1.37 GiB、F3b で **1.82 GiB**（+0.45 GiB）。
`REPORT-F3b.md` §3.4 の `n=72` 予測 **1.825 GiB** と実測 1.82 が一致し、
「席では約 72 個の層インスタンスが scratch を持っていた」という逆算が裏づけられた。
**ただし予測どおり「五分」の側に落ちた** — 制式の窓 389,120 に必要な 1.83 GiB に **0.01 GiB 届かない**。

`UTIL=0.972` にすると立つが、KV は **396,656 → 393,483 tok（−0.8%）**。
389,120 の窓自体は保たれる（余裕 1.01× vs 制式 1.02×）。

### 三つの修理の実証（MEASURED）

| 指摘 | 実証 |
|---|---|
| (1) 黙って現行に落ちる | `F3 LNA2 MoE kernel ACTIVE` が **8 rank 全部**。`VLLM_EXL3_MOE_STRICT=1` を付けて **raise なし・Traceback 0** ＝ 全 rank で kernel が ready |
| (2) VRAM 6.556 MiB/層 | KV 空きが **1.37 → 1.82 GiB**。層あたり 0 になったことが席の数字で確認できた |
| (3) 16 行天井 | 辞退ログが **`max_rows=32`** を報告。制式の 16 行は天井の半分 |

**辞退はすべて設計どおりの形のみ**（MEASURED、`x=(512,…)` のプリフィルと、
長文・多流時の `x=(33/34/70/76/100, 4096)`）。**decode の 16 行は一度も辞退していない。**
`F1 LNA declined` は 0。strict らしき grep 一致 1 件は
`WARNING Unknown vLLM environment variable detected: VLLM_EXL3_MOE_STRICT`（vLLM が知らない env を報せただけ。
読むのはプラグイン側）で、**実際の raise ではない**。

---

## 2. 門 B — lna2 単体の席の数字（MEASURED）

### 2.1 bench-streams（3 回平均、±半幅）

| 構成 | code1 | en1 | ja1 | code4 | en4 | ja4 |
|---|--:|--:|--:|--:|--:|--:|
| incumbent 制式（当日 baseline） | 85.9 | 64.6 | 56.6 | 125.4 | 96.6 | 82.8 |
| incumbent 復帰後（F2 回） | 85.0 | 59.8 | 55.6 | 130.0 | 96.0 | 83.7 |
| incumbent 復帰後（本回） | 82.1 | 60.3 | 54.2 | 142.6 | 105.0 | 85.9 |
| **incumbent の帯（3 走）** | 82.1–85.9 | 59.8–64.6 | 54.2–56.6 | 125.4–142.6 | 96.0–105.0 | 82.8–85.9 |
| **lna2 @389k（UTIL 0.972）** | **103.2** | **73.1** | **67.3** | **142.4** | **107.5** | **96.6** |
| **比（対 incumbent 平均）** | **1.22×** | **1.20×** | **1.21×** | 1.07× | 1.08× | 1.15× |

**単流は三言語とも incumbent の帯の外に完全に出ている**（103.2 > 85.9、73.1 > 64.6、67.3 > 56.6）。
**前回の F3 検分（262k 窓、1.27–1.29×）よりやや低いのは、今回の incumbent 側が速い日だったため。**
4 流は incumbent の帯が広く（code4 125–143）、code4/en4 は分離しない。ja4 のみ分離（96.6 > 85.9）。

### 2.2 数値・受理長・作文・針（MEASURED）

- **貪欲一致**（自己雑音帯 15–136 字）: inc vs lna2b = en 130 / **ja 210** / code 85 / mix 266。
  inc2 vs lna2b = en 107 / ja 15 / code 86 / mix 136。**全部が帯の中か帯より長い側**。劣化の証拠なし。
- **受理長**: en 2.19 / ja 1.84 / code 2.93。incumbent の当日帯（2.04–2.34 / 1.79–2.02 / 2.93–3.17）と同じ。
- **作文 7/7 finish=stop**（MEASURED）。初回は JA business と CODE が `length` だったが、
  いずれも prompt 側の `max_tokens` 不足で、予算を足すと stop（JA business 235 tok/stop、CODE 631 tok/stop、
  JA essay 661 tok/stop）。空返事なし。
- **166k 針: 正答 `sazanami-hotaru-2026`、TTFT 123.5 s**（697,460 字）。
  制式の記録は 166k 針 TTFT 167 s なので、**長い窓は生きているどころか速い**。

---

## 3. 門 C — 家計簿（prof-shapes、rank0、graph ON、decode 23 step、MEASURED）

| bucket | incumbent | **lna2** | 差 |
|---|--:|--:|--:|
| **MoE**（ms/step ／ n/step） | **10.002** ／ 43.0 | **4.489** ／ 43.0 | **−5.513 ms** |
| **routed kernel µs/launch** | **232.6** | **104.4** | **2.23×** |
| 密 `exl3_gemm_kernel<6>` | 8.894 ／ 489.0 | 8.863 ／ 489.0 | 同 |
| NCCL | 2.436 ／ 87.0 | 3.190 ／ 87.0 | +0.754（§3.1） |
| other | 4.175 ／ 1050.5 | 3.900 ／ **878.5** | **−172 launch/step** |
| aux: elementwise | 1.813 ／ 1500.0 | 1.254 ／ **1027.0** | **−473 launch/step** |
| aux: cat | 0.141 ／ 149.0 | 0.140 ／ 149.0 | 同 |
| **decode 全 kernel** | **27.460** | **21.836** | **1.258×** |

**routed 232.6 → 104.4 µs/launch（2.23×）** は前回検分（228.8 → 102.1、2.24×）を再現した。
**補助カーネルも減っている**: incumbent が払う argsort・scatter_add・zero 初期化が消え、
elementwise が 1500 → 1027/step、other が 1050.5 → 878.5/step。**lna2 は MoE の外でも安い。**

### 3.1 NCCL share について（正直に）

NCCL の絶対値は incumbent 2.436 / lna2 3.190 / 「combo」2.497 ms/step。
**同一構成（lna2）で 3.190 と 2.497 の двух 走行が出ている**ので、この項は
**走行間ばらつきが大きく、share の増減を主張できない**。NCCL 時間は他 rank 待ちを吸うため、
**8 rank 全部を測らないと意味を持たない**（§7 の宿題）。launch 数 87.0/step は全構成で不変。

---

## 4. ⚠️ 門 C の本題 — lna2 と T0 は**併用できなかった**（MEASURED）

「combined」として起動した席は、**実際には lna2 単体だった**。

| 証拠（MEASURED） | |
|---|---|
| combo の家計簿 | 密 **489.0 launch/step**、`exl3_mgemm_kernel` **0 件** |
| combo vs lna2only | 密 8.8525 vs 8.8627 ms/step、MoE 4.487 vs 4.489 — **区別がつかない** |
| combo の bench | code1 102.8 / en1 73.5 / ja1 67.0 ＝ lna2 単体（103.2/73.1/67.3）と同じ |

### 原因（MEASURED、決定的）

**二つの改修は別々の plugin 木にあり、木ごと mount するので同時に載らない。**

```
vllm-exl3-v030-port/src/vllm_exl3/exl3.py   lna2: あり(3)   dense_group: 0 件
vllm-exl3-lab/vllm-exl3/src/vllm_exl3/exl3.py   lna2: 0 件   dense_group: あり(11)
```

`PLUGIN_SRC` は**ディレクトリ丸ごとの mount** なので、lna2 を載せた瞬間 T0 のコードは席から消える。
そして patched `attention.py` のヘルパーは

```python
try:
    from vllm_exl3.exl3 import run_exl3_group
    outs = run_exl3_group(module, x, torch.float32)
except ImportError:
    outs = [lin.forward(...) for lin in linears]   # ← ここに落ちた
```

と書いてあるため、**ImportError を握って solo ループに落ち、何も言わない**。
`LNA_EXL3_DENSE_GROUP=1` も無視される（読む側のコードが席に居ない）。

**これは前回私が報告した「黙って落ちる」と同じ形の傷である。** 今回は lna2 側に
strict と ACTIVE ログが入っていたおかげで lna2 の方は疑わずに済んだが、
**T0 側には同じ守りが無い**ので、私は家計簿を採るまで併用できていると誤認していた。

### 直し方（どちらか）

1. **木を一本にする**（推奨）: T0 を port の `exl3.py` に取り込む、または lna2 を production の `exl3.py` に取り込む。
   どのみち制式化には一本化が要る。
2. 暫定: ヘルパーの `except ImportError` を **strict のときは再送出**し、
   さらに T0 にも「ACTIVE を一度ログする」を入れる。**握り潰しをやめるのが先。**

---

## 5. ケンの 120 tok/s まで（MEASURED ＋ ESTIMATE）

| | 単流 code tok/s | 目標比 |
|---|--:|--:|
| incumbent 制式 | 84.3（3 走平均） | 70% |
| **lna2 単体（実測）** | **103.2** | **86%** |
| lna2 ＋ T0（**併用は未実現**、ESTIMATE） | **約 111** | 93% |
| ケンの的 | 120 | 100% |

**ESTIMATE の根拠**: T0 は密セグメントを 8.894 → 7.330 ms/step にする（前回検分 MEASURED）。
lna2 の step 21.836 ms からこれを引くと 20.272 ms/step、対 incumbent 1.355×。
実測の step 比と tok/s 比のずれ（1.258× の step に対し tok/s は 1.22×）を同率で当てると
**84.3 × 1.355 × (1.22/1.258) ≈ 111 tok/s**。

**併用が成っても 120 には届かない見込み。** 残る 8% をどこから取るかは、家計簿では
密 8.86（40%）・MoE 4.49（21%）・other 3.90（18%）・NCCL 3.19（15%）なので、
**T1（密のさらなる融合）と NCCL/other が次の的**である。

---

## 6. 門 D — 制式復帰（MEASURED）

- plugin = `vllm-exl3-lab/vllm-exl3/src/vllm_exl3`（incumbent）、`MOE_KERNEL` 未設定、`UTIL=0.97`
- `docker logs | grep -c 'F3 LNA2'` = **0**／`grep -c run_exl3_group`（像の attention.py）= **0**（stock）
- **`GPU KV cache size: 396,656 tokens`** — 制式の値ちょうど
- health **200**、`warm-ohtani-head.sh` 実行済（`prompt_tokens=12269 took=10.1s`）
- 復帰後 bench: code1 82.1 / en1 60.3 / ja1 54.2 / code4 142.6 / en4 105.0 / ja4 85.9

**lna2 も DENSE_GROUP も本番に残していない。**

### ⚠️ 自分の道具の傷（自分で見つけて直した）

最初の復帰ブートは **`PLUGIN_SRC` を export し忘れていた**（`.tmp/f3b-boot.sh` は `EXT_SO` しか
export せず、incumbent の plugin を変数に持っていただけだった）。その結果、席は
**ホストの制式 plugin ではなく像に焼かれた plugin** で上がっていた。
health も KV 396,656 も正常に見えたので、**mount を数えるまで気づかなかった**。

再ブートして修正済（MEASURED、最終状態）:

| 確認 | 値 |
|---|---|
| plugin mount | `vllm-exl3-lab/vllm-exl3/src/vllm_exl3 -> …/dist-packages/vllm_exl3` **有り** |
| `exllamav3_ext` mount | 有り |
| `GPU KV cache size` | **396,656 tokens**（1.02×） |
| `F3 LNA2` 行 / `run_exl3_group` / `MOE_KERNEL`・`DENSE_GROUP` env | すべて **0** |
| warm | 実行済（`prompt_tokens=12269 took=10.1s`） |
| bench | code1 81.3 / en1 59.0 / ja1 54.8 / code4 126.4 / en4 94.9 / ja4 81.1（incumbent の帯） |

**教訓は §4 と同じ**: 「health が 200 で数字がそれらしい」は**構成が正しい証拠にならない**。
復帰の確認は **mount と env と KV を数える**まで終わらない。

---

## 7. 「ケンの制式判断に出せるか」への答え

### **まだ No。あと一歩。** ただし残っているのは**速さでも品質でもない**。

**出せる材料（揃っている）**:
- 単流 **1.20–1.22×**、帯が完全分離（MEASURED）
- routed kernel **2.23×**、補助カーネルも減（MEASURED）
- 166k 針 正答・TTFT が制式記録より速い（MEASURED）
- 作文 7/7、受理長 不変、貪欲分岐は雑音帯の中（MEASURED）
- 黙って落ちない仕組み（strict=1・ACTIVE ログ・辞退ログ）が入り、**席で実際に効くことを確認した**（MEASURED）
- 16 行天井は 32 へ。制式の 16 行に倍の余裕（MEASURED）

**足りないもの（これが No の理由）**:

1. **制式の `UTIL=0.97` では立たない**（0.01 GiB 不足）。`0.972` に動かすとは
   **制式のレシピを書き換えること**で、KV も 396,656 → 393,483（−0.8%）になる。
   **これはケンの決裁事項であって、私が黙って呑む変更ではない。**
   （職人の手札 §3.4-3「scratch を実 max rows で可変」を実装すれば 3 MiB 戻り、0.97 で立つ可能性がある。ESTIMATE）
2. **T0 との併用が構造的にできない**（§4）。制式化の前に木を一本にする必要がある。
   いま併用を諦めて lna2 単体で制式にすると、**T0 を後から入れる道が塞がる**。
3. **ppl をこのビルドで測っていない。** 前回ビルドでは対照差 −0.0055 だったが、
   F3b は `MAX_M=32`・scratch 共有・strict と**カーネルもプラグインも変わっている**。
   **数値の門は測り直しが要る**（前回同様、記録の 6.7159 は引数不明なので incumbent 対照とセットで）。

**そのほか未測定（前回から変わらず）**: 8 rank の P50/P95 と最遅 rank（NCCL の判定に必須、§3.1）、
compute-sanitizer 4 種（像に無い。device atomics と team barrier があるので racecheck は要る）、
graph ON/OFF parity、Vision、thermal steady、`LNA2_CONCURRENT_STREAMS=2` の席での確認。

### 進め方の推奨（順に）

1. **木を一本にする**（§4）。ここが全部の前提。ついでに T0 側の握り潰しを塞ぐ。
2. **一本化した木で ppl と 8 rank P50/P95**。
3. **scratch の runtime 可変**を試し、`UTIL=0.97` のまま立つかを見る。立てば 1 の障害が消える。
4. そこまで揃えば **ケンに出せる**。数字は既に出ている（1.2×・針 OK・品質不変）ので、
   **判断材料としては「窓を 0.8% 削るか、UTIL を 0.002 動かすか」だけが残る。**

---

## 8. 使った物

| もの | 中身 |
|---|---|
| `.tmp/f3b-boot.sh` | 席のブート共通部（lna2 / T0 / 併用 / 復帰、KV 行の抽出つき） |
| `.tmp/f3b-gateA.sh` / `f3b-gateB.sh` / `f3b-gateC.sh` / `f3b-gateCD.sh` | 各門 |
| `.tmp/f2-count.py` | decode step 窓で区切る家計簿計量器（`gpu_user_annotation` で 1 step = 1 窓） |
| `.tmp/f3b-trace-{combo,lna2only}-rank0.json.gz` | rank0 の生トレース |
| `.tmp/bench3-{lna2b,combo,restored-f3b}.txt` | bench の生ログ |
| `.tmp/numerics-{lna2b,combo}.json` | 貪欲生成の本文 |
| `.tmp/sakubun-lna2b.txt` | 作文 |
| `.tmp/f3b-needle-combo*.log` | 166k 針 |
