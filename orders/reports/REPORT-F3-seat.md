# REPORT F3-seat — lna2 の席の門（オオタニ実走）

検分: ユキ（Fable 5.1）／2026-09-04／席 = オオタニ（DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6、TP8 GPU 0,1,2,3,5,7,8,9、:8899）
対象: `vllm-exl3-v030-port` の二号機 `lna2`（職人報告 `f1/REPORT-F3.md`）
すべての数字に **MEASURED / ESTIMATE** を付す。席は最後に制式へ戻してある（§7）。

---

## 0. 採決

| 門 | 結果 | 一行 |
|---|---|---|
| 1. incumbent baseline | **PASS**（MEASURED） | 席の `exl3_moe_kernel<2>` = **228.8 µs/launch**（記録の 229 と一致） |
| 2a. lna2 が実際に走ったか | **⚠️ 初回 FAIL → 三つの欠陥を直して PASS** | 素の成果物は**黙って exllamav3 に落ちていた**。§2 |
| 2b. 数値（貪欲一致） | **PASS**（MEASURED） | 分岐は同一カーネル二度走りの雑音帯の中。劣化の証拠なし |
| 2c. 作文 7 本 finish=stop | **PASS**（MEASURED） | 予算を足せば 7/7 stop（1 本は prompt 側の 600 tok 上限） |
| 2d. bench-streams 1/4 流 ×3 | **PASS**（MEASURED） | 単流 **1.27–1.29×**、4 流 **1.09–1.12×** |
| 2e. 家計簿（prof-shapes rank0） | **PASS**（MEASURED） | `lna2_kernel<2>` = **102.1 µs/launch**、席で **2.24×** |
| 2f. 受理長 | **PASS**（MEASURED） | en 2.08 / ja 1.86 / code 3.05 ＝ 現行と同じ（受理率は動いていない） |
| 3. ppl（wikitext 4k、spec OFF） | **PASS**（MEASURED、対照つき） | inc 4.7696 vs lna2 **4.7641**（Δ −0.0055、門 ±0.02） |
| 4. 制式復帰 | **PASS**（MEASURED） | health 200・incumbent プラグイン・warm 実行済 |
| — VRAM 予算 | **⚠️ FAIL** | lna2 は **制式の 389,120 窓を維持できない**（§5）。ここが採用の唯一の障害 |

**結論: 速さは本物（席で MoE カーネル 2.24×、実効 tok/s 単流 1.27–1.29×）、精度も現行と同等。
ただし今日の姿のままでは制式にできない** — 理由は速度でも数値でもなく **VRAM**（§5）。
推奨は「**もう一手要る（needs more work）**」。詳細と直す順は §8。

---

## 1. 門 1 — incumbent baseline（席そのまま、MEASURED）

席は開始時点で制式のまま上がっていた（`--max-model-len 389120 --gpu-memory-utilization 0.97 --max-num-batched-tokens 512 --max-num-seqs 4 --speculative-config dspark/3`、
plugin = `vllm-exl3-lab/vllm-exl3/src/vllm_exl3`、`VLLM_EXL3_MOE_KERNEL` 未設定）。

### 1.1 bench-streams（3 回平均、±は半幅。MEASURED）

| | code1 | en1 | ja1 | code4 | en4 | ja4 |
|---|--:|--:|--:|--:|--:|--:|
| incumbent @389,120 | 85.7 (±1.4) | 62.9 (±1.0) | 55.5 (±2.2) | 133.7 (±5.8) | 99.0 (±1.9) | 82.3 (±2.6) |

クロック 2565–2580 MHz（平均 2572）、電力 平均 51.2 W。制式の `-pl62 -lgc2600` の粒に乗っている。

### 1.2 受理長（言語ごとに 1 流 512 tok を単発で流し、その窓の SpecDecoding metrics を平均。MEASURED）

| 走 | en | ja | code |
|---|--:|--:|--:|
| inc r1 | 2.06 | 2.02 | 3.04 |
| inc r2 | 2.07 | 1.85 | 2.98 |
| inc r3 | 2.04 | 1.79 | 3.07 |

発注書の目安 2.0/1.8/2.2 に対し、**この席の code は実測 3.0 前後**。以後の比較はこの実測値を分母にした。

### 1.3 貪欲生成の自己雑音（同一カーネルを二度走らせた分岐。MEASURED）

`numerics-gate run inc` と `run inc2`（まったく同じ席・同じ入力）:

| | en | ja | code | mix |
|---|--:|--:|--:|--:|
| identical_prefix (chars) | 107 | **15** | 85 | 136 |

**発注書の「49–118 字」より下に外れる（ja=15）。** 席は spec decode ＋ TP ＋ APC で走行ごとに非決定的なので、
**この門は「文字単位の一致」では判定できない**。以下の判定は「雑音帯 15–136 字と比べて有意に早く割れていないこと」で行った。

### 1.4 家計簿（torch profiler `record_shapes`、rank0、decode 64 tok。MEASURED）

席を PROFILE つきで焼き直して採取。全カーネル時間 **954.7 ms**。

| カーネル | launch 数 | 合計 | **µs/launch** | 全体比 |
|---|--:|--:|--:|--:|
| `exl3_moe_kernel<2,256,1>`（routed） | 1035 | 236.8 ms | **228.8** | 24.8% |
| `exl3_moe_kernel<2,256,1>`（`moe_forward_shared`） | 40 | 23.4 ms | 585.0 | 2.5% |
| `exl3_moe_kernel<3,256,1>`（routed） | 72 | 16.4 ms | 227.8 | 1.7% |
| **MoE 合計** | **1147** | **276.6 ms** | 241.2 | **29.0%** |

**席の 228.8 µs は記録の「229 µs/launch」と一致した。** 職人が同条件で測り直した 355.9 µs とは別の数字で、
席のほうが速い（graph ON・固定クロック・実 route の L2 状態）。**以下の席の速度比はすべて 228.8 を分母にする。**

---

## 2. 門 2a — lna2 は最初、黙って現行に落ちていた（⚠️ 重要）

候補構成（`PLUGIN_SRC=…/v030-port/src/vllm_exl3`, `NATIVE_SO=…vllm_exl3_c…so`, `MOE_KERNEL=lna2`）で起動したところ、
`VLLM_EXL3_MOE_KERNEL=lna2` は容器に入り、`get_moe_kernel_backend()` は `"lna2"` を返し、
`lna2_moe_decode` も呼べる状態だったのに、**MoE は一度も lna2 を通っていなかった**。
`_apply_lna2_moe` が例外を投げず `None` を返し、**警告も出さずに** exllamav3 経路へ落ちる作りだったため、
ログにも計器にも何の痕跡も残らない。**成果物のままでは席で走らない。**

原因は三つ、いずれも**カーネルではなくプラグインの結線**である（MEASURED、いずれも実機で確認）:

| # | 欠陥 | 場所 | 直し方 |
|---|---|---|---|
| 1 | `_prepare_lna_state()` が `backend == "lna"` のときしか呼ばれない。lna2 では `_exl3_lna_ready` が **False のまま**で、`_apply_lna2_moe` は先頭のガードで即 `None` | `exl3.py` `build_exl3_fused_state` | `backend in {"lna", "lna2"}` に。同じファイルの `has_lna` は既に `{"lna","lna2"}` で、**ファイル内で不整合だった** |
| 2 | `_prepare_lna_state()` 自体が `AttributeError("'bool' object has no attribute 'numel'")` で落ち、自分の try/except に握り潰されて `_exl3_lna_ready=False` | `_validate_lna_pointer_contract` | ABI 検証が `linear.mcg` を marker テンソルとして読む。**exllamav3 の `LinearEXL3` は `self.mcg_tensor = mcg` / `self.mcg = mcg_tensor is not None`（bool）**（MEASURED: 容器内の `inspect.getsource`）。`mcg_tensor` を読むように |
| 3 | 落ちたことが**どこにも出ない** | 同上 | 検分用に「declined 理由」の一発ログと「ACTIVE」の一発ログを足した |

**この 3 点は F1 一号機（`lna`）も同じ経路を共有する。** つまり **F1 も席では一度も走っていない** はず（ESTIMATE、未検証）。

### 逸脱の明示

以降の全数字は **「成果物 ＋ 上記 3 点の修正」** に対するものであり、納品された姿そのままではない。
修正はいずれも**カーネル・数値境界・ABI の中身には触れていない**（結線と属性名と観測ログのみ）。
入っている印（`src/vllm_exl3/exl3.py` 内、`F3 seat gate 2026-09-04` のコメント）:
`_lna2_log_active` / `_lna2_log_reject` / `mcg_tensor` / `backend in {"lna","lna2"}`。

修正後、**8 rank すべてで `F3 LNA2 MoE kernel ACTIVE` を確認**（MEASURED）。

### 落ちる形（正常な設計どおり、MEASURED）

`F3 LNA2 declined` として観測された形は二つだけ:
- `x=(512,4096)` — プリフィル。lna2 は decode 専用（1..16 行）なので正しい辞退。
- `x=(20,4096)` — 20 行。**16 行の上限を超えると現行に落ちる。** 制式（SEQS=4 × DSpark3 = 16 行）は上限ちょうどで、
  **余裕がない**。`--max-num-seqs` を 5 以上にした瞬間、MoE は静かに現行へ戻る（ESTIMATE: 5×4=20 行）。

---

## 3. 門 2b–2d — 数値・作文・スループット（候補、MEASURED）

候補席は **`--max-model-len 262144`** で走らせた（389,120 は VRAM が足りず起動しない。§5）。
交絡を潰すため **incumbent の 262,144 対照**も別途採った（§3.3）。

### 3.1 貪欲生成の一致（MEASURED）

| 対 | en | ja | code | mix |
|---|--:|--:|--:|--:|
| inc vs inc2（**雑音の物差し**） | 107 | 15 | 85 | 136 |
| inc vs lna2 | 218 | 15 | 85 | 323 |
| inc2 vs lna2 | 107 | **276** | 86 | 136 |

**lna2 の分岐位置は雑音帯の中、むしろ長い側にある**（en 218・mix 323・ja 276 は inc 同士より長い）。
**数値の劣化を示す証拠はない。** ただし席が非決定的なので、これは「異常なし」であって「一致」ではない。

### 3.2 作文 7 本（`sakubun.py`、MEASURED）

| 題 | tok | tok/s | finish |
|---|--:|--:|---|
| JA essay | 666 | 65.4 | stop |
| JA business | 384 | 74.1 | stop |
| EN essay | 464 | 72.2 | stop |
| CODE | 600 | 118.2 | **length** |
| JA diary | 316 | 61.1 | stop |
| JA letter | 393 | 64.1 | stop |
| JA chat | 139 | 58.5 | stop |

CODE の `length` は**中身のある 2311 字**で、prompt 側の `max_tokens=600` が足りないだけ。
同じ prompt を `max_tokens=1500` で再走 → **567 tok・finish=stop・118.9 tok/s**（MEASURED）。
**空返事（thinking が予算を食う形）は一件も無い。7/7 PASS。**

### 3.3 bench-streams（3 回平均、MEASURED）

| 構成 | code1 | en1 | ja1 | code4 | en4 | ja4 |
|---|--:|--:|--:|--:|--:|--:|
| incumbent @389,120（制式） | 85.7 | 62.9 | 55.5 | 133.7 | 99.0 | 82.3 |
| **incumbent @262,144（対照）** | 80.8 | 62.0 | 54.1 | 133.3 | 99.7 | 86.3 |
| **lna2 @262,144** | **104.4** | **79.0** | **70.0** | **145.4** | **111.2** | **94.8** |
| **速度比（対 262k 対照）** | **1.29×** | **1.27×** | **1.29×** | **1.09×** | **1.12×** | **1.10×** |

- **窓の長さは交絡していない**: incumbent の 389k と 262k はほぼ同値（差は走行間ばらつきの内）。
- 3 回の分布は **重なっていない**（例 en1: inc 61.7/63.2/63.7 vs lna2 80.0/78.9/78.1）。
- 4 流の伸びが小さいのは、4 流では MoE 以外（NCCL AllReduce・attention）の比率が上がるため（§4 と整合）。

### 3.4 受理長（MEASURED）

| 構成 | en | ja | code |
|---|--:|--:|--:|
| incumbent（3 走） | 2.04–2.07 | 1.79–2.02 | 2.98–3.07 |
| lna2（2 走） | 2.08 / 2.58 | 1.86 / 1.86 | 3.05 / 3.05 |

**現行と同じ帯。** 速さは受理率の変化ではなく、**一手が安くなったことから来ている**。

---

## 4. 門 2e — 席の家計簿（prof-shapes rank0、MEASURED）

候補を PROFILE つきで焼き直し、同じ decode 64 tok を採取。全カーネル時間 **733.8 ms**（incumbent 954.7 ms）。

| カーネル | launch 数 | 合計 | **µs/launch** | 全体比 |
|---|--:|--:|--:|--:|
| `lna2::lna2_kernel<2>`（routed） | 992 | 101.3 ms | **102.1** | 13.8% |
| `lna2::lna2_kernel<2>`（`moe_forward_shared`） | 40 | 13.0 ms | 325.0 | 1.8% |
| `lna2::lna2_kernel<3>`（routed） | 69 | 8.8 ms | 127.5 | 1.2% |
| **MoE 合計** | **1101** | **123.1 ms** | 111.8 | **16.8%** |

### 席での速度比（MEASURED）

| 経路 | incumbent | lna2 | 比 |
|---|--:|--:|--:|
| routed K2 | 228.8 µs | **102.1 µs** | **2.24×** |
| routed K3 | 227.8 µs | 127.5 µs | 1.79× |
| shared expert | 585.0 µs | 325.0 µs | 1.80× |
| MoE が占める割合 | 29.0% | **16.8%** | — |
| 全カーネル時間 | 954.7 ms | **733.8 ms** | **1.30×** |

### カーネル専用 vs 席

| 物差し | K2 の比 |
|---|--:|
| 職人の同条件 kernel-only（m=4、355.9→192.6 µs） | 1.85× |
| **席（in-seat、228.8→102.1 µs）** | **2.24×** |
| 席の実効 tok/s（単流） | 1.27–1.29× |
| 席の全カーネル時間 | 1.30× |

**席のほうが kernel-only より良い。** 職人の §7 の ESTIMATE（「end-to-end は kernel-only 比より良くなるはず」）は
**MEASURED で裏が取れた**: incumbent の `bitonicSortKVInPlace`（argsort、**1107 launch・3.8 ms**）は
lna2 の trace から**完全に消えている**。補助カーネルが増えていないことも確認済み
（scatter_add・zero fill・memset の増加なし）。

---

## 5. ⚠️ VRAM — 制式の 389,120 窓が立たない（MEASURED）

候補を制式そのままの引数（`MAXLEN=389120 UTIL=0.97`）で起動すると **起動に失敗する**:

```
ValueError: To serve at least one request with the model's max seq len (389120),
1.83 GiB KV cache is needed, which is larger than the available KV cache memory (1.37 GiB).
Based on the available memory, the estimated maximum model length is 263424.
```

原因は scratch（MEASURED、容器内で実測）:

| | 1 層 1 stream あたり |
|---|--:|
| `lna2_moe_scratch_bytes()` | 3,392,896 B = **3.236 MiB** |
| `lna_moe_scratch_bytes()`（F1 の分） | 3,481,616 B = **3.320 MiB** |
| 合計 | **6.556 MiB** |

`_prepare_lna_state` は **backend が lna2 でも F1 の scratch を必ず一緒に確保する**。
MoE 層ぶん積むと **GPU あたり約 0.37–0.46 GiB**（ESTIMATE、層数 ≈58 として 0.37 GiB）で、
これが不足分 0.46 GiB とほぼ一致する。

**同じ原因で ppl の初回も落ちた**（MEASURED）: `UTIL=0.97` の候補席に 4k の `prompt_logprobs` を投げると
`logits_processor` の `all_gather` が **128 MiB の確保に失敗**して 500（`torch.OutOfMemoryError`、GPU6 の空き 113 MiB）。
`UTIL=0.93` に下げたら通った。**数値やカーネルの問題ではなく、純粋に予算の問題。**

**F1 の scratch を lna2 のとき確保しないだけで、超過の約半分（3.32 MiB/層）は消える**（ESTIMATE）。

---

## 6. 門 3 — ppl（wikitext、4k、spec OFF。MEASURED）

`ppl-vllm.py 8899 wiki.test.raw 4096 8`。

| 構成 | PPL | tokens |
|---|--:|--:|
| incumbent spec-OFF @262,144 UTIL=0.97（対照） | **4.7696** | 30,002 |
| **lna2 spec-OFF @131,072 UTIL=0.93** | **4.7641** | 30,002 |
| 差 | **−0.0055** | — |

**門 6.72 ± 0.02 に対し、差は ±0.02 の 1/4。PASS。**

⚠️ **記録の 6.7159 とは絶対値が合わない**（4.77 vs 6.72）。`ppl-vllm.py` の引数（ctx・chunk 数・窓の取り方）が
制式記録時と違うためで、**モデルが変わったのではない**。同じ引数で incumbent を焚いた対照が 4.7696 なので、
**判定に使えるのはこの対照との差だけ**である。制式記録の再現には、6.7159 を出した引数を台帳に書き足す必要がある（§8）。

---

## 7. 門 4 — 制式復帰（MEASURED）

```
serve …-D2-K2x3-Dense6 --tensor-parallel-size 8 --quantization exl3
  --max-model-len 389120 --max-num-seqs 4 --max-num-batched-tokens 512
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.97
  --compilation-config {"cudagraph_capture_sizes":[1,2,4]} --disable-custom-all-reduce
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4 --speculative-config {"method":"dspark","num_speculative_tokens":3}
```

- plugin = `vllm-exl3-lab/vllm-exl3/src/vllm_exl3`（incumbent）、`VLLM_EXL3_MOE_KERNEL` **未設定**
- `docker logs dsv4 | grep -c 'F3 LNA2'` = **0**
- health **HTTP 200**
- `warm-ohtani-head.sh` 実行済（`prompt_tokens=12269 took=10.2s`）
- 復帰後の bench（MEASURED）: code1 84.1 / en1 61.0 / ja1 56.7 / code4 129.6 / en4 95.2 / ja4 80.8 — 開始時の baseline と同じ帯

**lna2 は本番に残していない**（ケンの決裁事項）。

### ⚠️ 席と無関係に見つかった事故（先に直した）

門 1 の途中で席を焼き直したところ、**制式モデルが起動しなくなった**。
`/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK`（**素の DL 先**）が消えており、
派生の 4 ディレクトリすべてがそこへの symlink を持っていたため:

- `tokenizer.json` / `tokenizer_config.json` / `generation_config.json` / `config.json.bak` が全滅
- `Dense6` の safetensors のうち **shard 1 と 45 の 2 本**が死にリンク（残り 44 本は hardlink なので生きていた）

**上がっていた席は 3 時間前に読み終わっていたから動いていただけで、落とせば二度と上がらない状態だった。**
`vcruz305/DSV4-Flash-Vision-EXL3-MixedK` から小物一式と shard 1・45 を落として復旧済（MEASURED、以後の全ブートは正常）。
`config.json.bak` の死にリンクだけ残っているが、vLLM は読まないので無害。

**掟に足すべき**: 派生モデルは素の DL 先への symlink を持つ。**素を消すと派生が全部死ぬ**。
派生を作るときは小物を symlink でなく**実体でコピー**すること。

---

## 8. 判定と推奨

### 数字のまとめ

| 物差し | 値 |
|---|--:|
| kernel-only（職人、K2 m=4 同条件） | **1.85×** |
| **in-seat MoE カーネル（K2 routed）** | **2.24×**（228.8 → 102.1 µs/launch） |
| in-seat 全カーネル時間 | 1.30×（954.7 → 733.8 ms） |
| **in-seat 実効 tok/s（単流 en/ja/code）** | **1.27× / 1.29× / 1.29×** |
| in-seat 実効 tok/s（4 流） | 1.12× / 1.10× / 1.09× |
| ppl（対照との差） | −0.0055（門 ±0.02） |
| 受理長 | 変化なし |
| VRAM 追加 | **+約 0.4 GiB/GPU** → 制式 389k が立たない |

### 推奨: **needs more work**（採用も、棄却も、まだ早い）

理由:

1. **速さは本物で、しかも席のほうが良い。** 2.24× は kernel-only の 1.85× を上回り、
   単流で **+27〜29%** という、ケンが実食で分かる大きさの差になっている。ここは疑う余地がない。
2. **落とすほどの傷は数値側に無い。** ppl は対照と 0.0055 差、受理長は同じ、作文 7/7、貪欲分岐は雑音の中。
3. **だが今日の姿では制式にできない。** 制式の売りは **389,120 の窓**で、
   lna2 はそれを **263,424 まで削る**。速さのために窓を 32% 削る取引は、ユキが決めることではない。
4. **そして、素の成果物は席で一度も走っていなかった。** §2 の 3 点は私が直したものであり、
   職人の検分を通っていない。**この修正を職人と合流させないまま採用してはいけない。**

### 直す順（次の一手）

1. **§2 の 3 点を上流（`v030-port`）へ正式に取り込む。** とくに #2（`mcg_tensor`）は F1 にも効く。
   **併せて「黙って落ちない」ようにする** — `MOE_KERNEL` を明示指定したのに使われなかったら、
   警告ではなく**起動時に大声で落ちる**べき。
2. **VRAM を削る。** lna2 のときに F1 の scratch を確保しない（−3.32 MiB/層 ≈ 半減、ESTIMATE）。
   これで 389k が立つかを測る。**立てば、採否はケンの一言で決まる。**
3. **16 行の天井に余裕を作る。** いまの制式は 4 seq × DSpark3 = 16 行ちょうどで上限に張り付いている。
   `--max-num-seqs` を 1 でも増やせば MoE は静かに現行へ戻る（ESTIMATE）。**静かに戻るのが一番いけない。**
4. **compute-sanitizer**（職人 §6 の残り穴、racecheck）。像に入っていないので、これは像の仕事。
5. **台帳の穴を埋める**: 制式 ppl **6.7159 を出した `ppl-vllm.py` の引数**を `Models/RECIPES.md` に書く（§6）。
6. **§7 の symlink 事故**を掟に足す。

### やらなかったこと（正直に）

- **8 rank 全部の P50/P95・最遅 rank**: 家計簿は **rank0 のみ**。他 7 rank は未測定。
- **graph ON/OFF の parity、NCCL bytes、consumer dtype**: 未測定。
- **166k 針・Vision**: 未測定（候補は 262k までしか立たないので、166k 針は原理的には可能だが走らせていない）。
- **thermal steady の長時間走行**: 未測定。bench は各 3 回、クロックは 2490–2580 MHz の帯で走った
  （候補の走行で平均 2510 MHz と少し低い＝**候補に不利な側**の条件で上の比が出ている）。
- **記録 6.7159 の再現**: できていない（§6）。

---

## 9. 使った物

| もの | 中身 |
|---|---|
| `.tmp/f3-boot.sh` | 席のブート共通部（incumbent / candidate の切替） |
| `.tmp/f3-gate1b.sh` / `f3-gate2a.sh` / `f3-gate2b.sh` / `f3-gate2c.sh` | 各門のブートと採取 |
| `.tmp/f3-gate3.sh` / `f3-ppl2.sh` / `f3-ppl3.sh` | ppl と対照 |
| `.tmp/f3-restore.sh` | 制式復帰 ＋ warm |
| `.tmp/acc-lang.sh` | 言語ごとの受理長（SpecDecoding metrics を行数差で窓取り） |
| `.tmp/f3-profshapes-inc.txt` / `-lna2.txt` | 家計簿の出力 |
| `.tmp/f3-trace-inc-rank0.json.gz` / `-lna2-rank0.json.gz` | rank0 の生トレース |
| `.tmp/numerics-{inc,inc2,lna2}.json` | 貪欲生成の本文 |
| `.tmp/bench3-{inc,inc262,lna2,restored}.txt` | bench の生ログ |
| `.tmp/sakubun-lna2.txt` | 作文 7 本 |
