# REPORT F2 T0-seat — 密 EXL3 shard 融合の席の門（オオタニ実走）

検分: ユキ（Fable 5.1）／2026-09-04／席 = オオタニ（DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6、TP8 GPU 0,1,2,3,5,7,8,9、:8899）
対象: 本番 plugin `vllm-exl3/src/vllm_exl3/exl3.py` の T0（`LNA_EXL3_DENSE_GROUP=1`、既定 OFF）
＋ `recipe-lna/patch_dsv4_attention_compressor_exl3.py` の新ヘルパー
発注: `orders/F2-dense-launch-fusion-v2.md` §4/§5／職人報告: `orders/REPORT-F2-T0.md`
**lna2（F3）とは混ぜていない。全走行で routed experts は現行カーネル。**
すべての数字に **MEASURED / ESTIMATE** を付す。席は最後に制式へ戻してある（§6）。

---

## 0. 採決

| 門 | 結果 | 数字 |
|---|---|---|
| 1. baseline（制式そのまま、flag OFF） | **PASS**（MEASURED） | 密 **489.0 launch/step**・**8.8938 ms/step**。発注の 489／8.885 ms と一致 |
| 2a. 融合が実際に噛んだか | **PASS**（MEASURED） | `exl3_mgemm_kernel<6,…>` が **148 launch/step** で出現 |
| 2b. launch/step | **PASS**（MEASURED） | **489.0 → 341.0**（−148、−30.3%）。門 ≤345、職人 ESTIMATE 341 に**寸分違わず一致** |
| 2c. **KILL LINE（密セグメント ≥1.05×）** | **PASS**（MEASURED） | 8.8938 → **7.3297 ms/step = 1.2134×**。PASS 門 1.08× も **stretch 1.15× も超えた** |
| 2d. 補助カーネルが増えていないこと | **PASS**（MEASURED） | cat 149.0/step・elementwise 1500.0/step・NCCL 87.0/step・MoE 43.0/step、**全部 OFF と同一** |
| 2e. flag OFF の経路無変更 | **PASS**（MEASURED） | patch 適用済み＋flag OFF が baseline を **8.8937 vs 8.8938 ms/step・489.0 launch・kernel 内訳まで一致** |
| 2f. 数値（貪欲一致） | **PASS**（MEASURED） | 分岐 37–135 字、当日の自己雑音帯 15–136 の中 |
| 2g. 作文 7 本 finish=stop | **PASS**（MEASURED） | 予算を足せば 7/7 stop |
| 2h. 受理長 | **PASS**（MEASURED） | en 2.15 / ja 1.80 / code 3.15 ＝ OFF と同じ帯 |
| 2i. 実効 tok/s | **条件付き PASS**（MEASURED） | 単流 code +3.6% / ja +4.5% は**分布が分離**、en は分離せず。§4 |
| 3. 制式復帰 | **PASS**（MEASURED） | health 200・stock attention.py・DENSE_GROUP 無し・warm 実行済 |

**結論: T0 は席で効いた。KILL LINE を 1.05× に対して 1.21× で通過。launch は算術どおり 489→341。
OFF 経路は完全に無傷。T1 へ進んでよい。**

---

## 1. 前提 — vLLM 側 patch の再適用（発注 §5 項目 5）

像の `attention.py` は **旧ヘルパーのまま**だった（MEASURED）:

```
docker exec dsv4 grep -c "_lna_kv_score" …/vllm/models/deepseek_v4/attention.py  → 3（patch 済）
docker exec dsv4 grep -c "run_exl3_group" …                                       → 0（旧版）
```
中身は `outs = [lin.forward(x, {}, out_dtype=torch.float32) for lin in linears]` の solo ループ。
**このままでは compressor / indexer.compressor の −62 launch は一切効かない。**

**採った手段: hot-mount（新しい像タグは作らない掟のため）。**

1. 像から `attention.py.orig-lna2`（素、40,828 B）をホストへ取り出す
2. ホスト木 `/run/media/tonoken3/DATA1/.tmp/f2-vllmroot/models/deepseek_v4/attention.py` に置き、
   `recipe-lna/patch_dsv4_attention_compressor_exl3.py` を**その木に対して実行** → 41,548 B、`compile()` 通過
3. `serve-dsv4-tp8.sh` の `NCCL_EXTRA` に `-v <ホストの patched attention.py>:<像の同パス>` を足して被せる

**像は作っていない。** 席内での確認（MEASURED）: `grep -c run_exl3_group` = **2**（新ヘルパーが乗っている）。
`NCCL_EXTRA` は serve script 内で unquoted 展開なので `-e` と `-v` を同居させられる。

**掟に足すべき**: 像に焼いた patch は「patch 済み」判定で**二度と更新されない**。
`_lna_kv_score` の有無を idempotency マーカにしているので、ヘルパーの中身が変わっても素通しになる。
マーカを**版つき**（例 `_lna_kv_score_v2`）にすれば、この事故は構造的に消える。

---

## 2. 門 1 — baseline（制式そのまま、MEASURED）

席は制式のまま。plugin は **T0 のコードを含む当日版**（`exl3.py` mtime 12:04、席の起動 12:37）で、
`LNA_EXL3_DENSE_GROUP` は未設定＝既定 OFF。**発注の「当日再測」条件を満たしている。**

### 2.1 数え方（決着）

trace の `gpu_user_annotation` に **`execute_context_0(0)_generation_1(4)` が step ごとに 2 本**入る。
重なりを畳んで **1 step = 1 窓**にし、**その窓に入った GPU kernel だけ**を数えた（prefill を除外）。
これで per-step が確定する（`.tmp/f2-count.py`）。

**「255」の誤読は決着**: 正しく数えると **489.0/step**。発注 §1.2 の 489 と一致した。

### 2.2 baseline 家計簿（MEASURED、rank0、graph ON、decode 23 step）

| bucket | ms/step | n/step |
|---|--:|--:|
| MoE | 10.002 | 43.0 |
| **`exl3_gemm_kernel<6>`（密 solo）** | **8.894** | **489.0** |
| other | 4.175 | 1050.5 |
| NCCL | 2.436 | 87.0 |
| aux: elementwise | 1.813 | 1500.0 |
| aux: cat | 0.141 | 149.0 |
| **decode 全 kernel** | **27.460** | — |

密の内訳（MEASURED）: `<…,32,128,…>` 339.0/step ／ `<…,32,256,…>` 129.0/step ／ `<…,16,512,…>` 21.0/step。

**`*** 密セグメント = 8.8938 ms/step、489.0 launch/step ***`（発注の 8.885 ms と 0.1% 差）

### 2.3 baseline の声（MEASURED）

- bench-streams 3 回平均: code1 85.9 (±3.2) / en1 64.6 (±1.0) / ja1 56.6 (±1.7) / code4 125.4 (±10.5) / en4 96.6 (±2.9) / ja4 82.8 (±4.8)
- 受理長: en 2.34 / ja 1.93 / code 3.17
- `numerics-gate run t0off` 採取済

---

## 3. 門 2 — 融合 ON（MEASURED）

構成 = 制式 ＋ patched attention.py ＋ `-e LNA_EXL3_DENSE_GROUP=1`。
実行条件（発注 Gate 5 の記録義務）: `LNA_DSV4_AUX_STREAMS=0`・`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`・
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`・`VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION=1`・
graph capture `[1,2,4]`・SEQS=4・BT=512・DSpark3・UTIL=0.97・MAXLEN=389,120。

### 3.1 融合が噛んだ証拠（MEASURED）

trace に **`void exl3_mgemm_kernel<6, true, 1, 16, 32, 256, 4, 3>` = 64.0/step**、
**`<6, true, 1, 16, 32, 128, 4, 3>` = 84.0/step**、合計 **148.0/step** が新規に出現。
容器ログの fallback は **プリフィルのみ**（`rows=512 outside fused range 1..16`、8 rank 各 1 回）。
decode 域の fallback WARNING は無し。

### 3.2 ON の家計簿（MEASURED、rank0、graph ON、decode 22 step）

| bucket | ms/step | n/step | baseline 比 |
|---|--:|--:|---|
| MoE | 9.939 | 43.0 | 同 |
| other | 4.170 | 1050.5 | 同 |
| `exl3_gemm_kernel<6>`（残った solo） | 3.703 | 193.0 | −296.0 |
| **`exl3_mgemm_kernel`（融合）** | **3.626** | **148.0** | 新規 |
| NCCL | 2.423 | 87.0 | 同 |
| aux: elementwise | 1.836 | 1500.0 | **同（増えていない）** |
| aux: cat | 0.138 | 149.0 | **同（増えていない）** |
| **decode 全 kernel** | **25.836** | — | **−5.9%** |

### 3.3 判定表（MEASURED）

| 指標 | baseline | ON | 比 | 門 | 判定 |
|---|--:|--:|--:|---|---|
| 密 launch/step | 489.0 | **341.0** | −30.3% | ≤345 | **PASS**（ESTIMATE 341 と完全一致） |
| **密セグメント ms/step** | 8.8938 | **7.3297** | **1.2134×** | KILL <1.05×／PASS ≥1.08×／stretch ≥1.15× | **PASS（stretch 超え）** |
| decode 全 kernel ms/step | 27.460 | 25.836 | 1.0629× | — | −5.9% |
| 補助 kernel/step（cat・elementwise） | 149.0・1500.0 | 149.0・1500.0 | 1.000× | 増えないこと | **PASS** |

融合 1 本あたりの節約 = 1.5641 ms/step ÷ 148 launch = **10.6 µs/融合**（MEASURED からの算術）。

### 3.4 flag OFF の経路無変更（発注 Gate 2 の直接証明、MEASURED）

**patched attention.py を被せたまま flag を OFF** にした対照:

| | baseline（stock, OFF） | patched, OFF | 一致 |
|---|--:|--:|---|
| 密セグメント ms/step | 8.8938 | **8.8937** | 4 桁一致 |
| 密 launch/step | 489.0 | **489.0** | 一致 |
| `<32,128>` / `<32,256>` / `<16,512>` per step | 339.0 / 129.0 / 21.0 | 339.0 / 129.0 / 21.0 | 一致 |
| `exl3_mgemm` | 無 | **無** | 一致 |
| aux cat / elementwise per step | 149.0 / 1500.0 | 149.0 / 1500.0 | 一致 |

**回帰ゼロを kernel 名・件数・時間で直接証明した。既定 OFF は本当に無傷。**

### 3.5 数値・作文・受理長（MEASURED）

**貪欲一致** `t0off` vs `t0on`:

| | en | ja | code | mix |
|---|--:|--:|--:|--:|
| identical_prefix (chars) | 99 | 54 | 135 | 37 |

当日実測の自己雑音帯（同一カーネル二度走り）は **15–136 字**。**全部その中。劣化の証拠なし。**
（※発注の「49〜118 字」は当日測り直すと 15–136 に広がる。席は spec decode ＋ TP で非決定的なので、
この門は「一致」ではなく「雑音帯を外れないこと」でしか判定できない。）

**作文 7 本**: 初回は JA essay が `length`（prompt の 800 字要求に対し `max_tokens=700` が足りない）。
`max_tokens=1600` で再走 → **625 tok・finish=stop・935 字**（MEASURED）。JA chat も再確認 stop。
**空返事は一件も無い。7/7 PASS。**

**受理長**: OFF 2.34 / 1.93 / 3.17 → ON **2.15 / 1.80 / 3.15**。当日の incumbent の帯
（en 2.04–2.34・ja 1.79–2.02・code 2.98–3.17）の中。**変化なし。**

---

## 4. 実効 tok/s — 正直に（MEASURED）

bench-streams 3 回平均を、OFF 側 4 走・ON 側 2 走ぶん並べる:

| 走 | code1 | en1 | ja1 | code4 | en4 | ja4 |
|---|--:|--:|--:|--:|--:|--:|
| OFF baseline（stock） | 85.9 | 64.6 | 56.6 | 125.4 | 96.6 | 82.8 |
| OFF patched r1 | 82.5 | 58.7 | 54.8 | 133.1 | 98.1 | 90.5 |
| OFF patched r2 | 84.4 | 62.6 | 56.7 | 115.1 | 86.7 | 74.6 |
| OFF 復帰後（stock） | 85.0 | 59.8 | 55.6 | 130.0 | 96.0 | 83.7 |
| **OFF の帯** | 82.5–85.9 | 58.7–64.6 | 54.8–56.7 | 115.1–133.1 | 86.7–98.1 | 74.6–90.5 |
| **ON r1** | 87.8 | 61.3 | 58.2 | 138.1 | 103.8 | 89.3 |
| **ON r2** | 87.2 | 63.3 | 58.7 | 133.4 | 97.0 | 80.1 |
| 平均比 ON/OFF | **1.036×** | 1.014× | **1.045×** | 1.078× | 1.064× | 1.022× |

**分布で読む（これが正直な読み方）**:
- **code1: ON 2 走とも OFF の帯の上に完全に出ている**（87.2, 87.8 > 85.9）。**分離。**
- **ja1: 同じく完全に分離**（58.2, 58.7 > 56.7）。
- **en1: 分離しない**（ON 61.3/63.3 は OFF の 58.7–64.6 の中）。
- **4 流はどれも分離しない。OFF の帯が広すぎる**（code4 は 115–133 と 15% 幅）。

**家計簿と矛盾しない**: 密の節約は 1.564 ms/step、step 全体 27.460 ms の **−5.7%**。
つまり **tok/s の天井が +6% 程度**の変更であって、走行間ばらつき（単流 ±3〜7%、4 流 ±8〜15%）と
同じ桁である。**単流 code/ja で分離が見えたのは、天井いっぱいまで出ているということ。**
**KILL LINE は密セグメントに掛かっており、そこは 1.21× で明快に通っている。**
tok/s の分離を門にしたいなら、走行数を増やすか eager（graph OFF）で測る必要がある（§7）。

---

## 5. 触っていないもの

- `vllm-exl3-v030-port/`（lna2 / F3 の作業場）— **今回は一切使っていない。routed は全走行で現行カーネル。**
- GPU 4・6・10・11、`ashigaru-searxng`、:3080/:3081/:8021/:8017 — 触れていない
- `pkill -f` / `pgrep -f` — 使っていない
- 新しい像タグ — **作っていない**（patch は hot-mount、§1）
- `/run/media/tonoken3/DATA1/hibi`・DistKura の trade — 読んでいない
- `sed` の `#` 区切り — 使っていない（編集は python / heredoc）

---

## 6. 制式復帰（MEASURED）

```
serve …-D2-K2x3-Dense6 --tensor-parallel-size 8 --quantization exl3
  --max-model-len 389120 --max-num-seqs 4 --max-num-batched-tokens 512
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.97
  --compilation-config {"cudagraph_capture_sizes":[1,2,4]} --disable-custom-all-reduce
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4 --speculative-config {"method":"dspark","num_speculative_tokens":3}
```

- **attention.py は像の素（stock）** — `grep -c run_exl3_group` = **0**（hot-mount は外れている）
- **`LNA_EXL3_DENSE_GROUP` は容器 env に無い**（= 既定 OFF）
- health **HTTP 200**
- `warm-ohtani-head.sh` 実行済（`prompt_tokens=12269 took=9.9s`）
- 復帰後 bench（MEASURED）: code1 85.0 / en1 59.8 / ja1 55.6 / code4 130.0 / en4 96.0 / ja4 83.7 — baseline と同じ帯

---

## 7. 発注の門のうち、今回やっていないもの（正直に）

コーディネータの指示は発注 §4 より狭い範囲だった。**未実施はこれだけある**:

- **Gate 2 の「OFF→ON→OFF 交互 各 3 回以上」**: 実際は OFF 4 走 / ON 2 走の**ブート単位**の交互。
  **同一プロセス内の env A/B は席では不可能**（容器 env はプロセス起動時に固まる。
  職人が run-1 で示した「同一プロセス切替」は sandbox の直呼びだから可能だった）。
  席で真の同一プロセス A/B を採るなら、**env でなく HTTP か signal で切れる口**が要る。
- **Gate 3（圧・グループ別 bytes÷時間）**: 未測定。融合前後の per-group 帯域比較をしていない。
- **Gate 4（1.25×T_floor をグループ別）**: 未測定。
- **ppl（4k・spec off・6.72±0.02）**: 未測定。
- **eager gap（graph OFF の bench と、ON/OFF それぞれの graph ON−OFF 差）**: 未測定。
  **これはケンの「完成の定義」そのもの**（発注 Gate 5-3）なので、T1 の前に埋めたい。
- **8 rank 全部の P50/P95・最遅 rank**: 未測定。家計簿は **rank0 のみ**。
- **untuned tripwire（eager vs graph replay µs/iter）**: 席では未実施。
- **compute-sanitizer**: 像に無し（職人 §3.4 と同じ。像の仕事）。
- **NCCL byte・in-seat selfcheck・engagement log**: 未測定。

---

## 8. 推奨

### **T1 へ進んでよい（adopt T0 as the base for T1）。ただし制式化はまだ。**

理由:

1. **KILL LINE を明快に超えた。** 密セグメント **1.2134×** は KILL 1.05× の倍以上、
   stretch の 1.15× も超えている。**「効かないなら遅延束ねの危険を負う理由が無い」という
   KILL LINE の趣旨に照らして、危険を負う理由はできた。**
2. **launch が算術どおり。** 489 → 341 は職人の ESTIMATE と**寸分違わない**。
   形の理解が正しいことの強い証拠で、T1 の 265 見込みも同じ算術で信用してよい。
3. **OFF が完全に無傷。** 8.8937 vs 8.8938 ms/step・489.0 launch・kernel 内訳一致。
   **既定 OFF のまま出荷しても疵は付かない**（発注の想定どおり）。
4. **補助カーネルが一つも増えていない。** F1 正典の条件を満たす。
5. **数値・受理長・作文に傷が無い。**

**制式にしない理由**: 実効 tok/s の利得（+1〜5%）は走行ばらつきと同じ桁で、
**ケンが実食で分かる差ではない**。T0 単体で席を切り替える価値は薄い。
**T1（h-fan / q_a-fan、launch → ≤265、step −4〜7%）まで積んで初めて制式の話になる。**

### 次の一手（順に）

1. **patch の idempotency マーカを版つきにする**（§1）。**今回の事故は構造的に再発する。**
2. **eager gap を測る**（発注 Gate 5-3）。ケンの完成の定義がここにある。T1 の前に baseline を採っておく。
3. **席で同一プロセス A/B できる口**を作る（env でなく HTTP/signal）。Gate 2 の「交互 3 回」は
   いまの形では原理的に満たせない。
4. **Gate 3/4（グループ別の圧と T_floor）** — どのグループが効いていないかが分かると T1 の的が絞れる。
   いまは 148 融合の平均 10.6 µs しか分かっていない。
5. **8 rank の P50/P95**（NCCL 粒揃え仮説の判定）。
6. **ppl** を T1 の門の前に一度。

---

## 9. 使った物

| もの | 中身 |
|---|---|
| `.tmp/f2-boot.sh` | 席のブート共通部（attention.py の hot-mount と flag の切替） |
| `.tmp/f2-run.sh` / `.tmp/f2-run2.sh` | 門の走行（baseline → ON → OFF 対照 → 復帰） |
| `.tmp/f2-count.py` | **decode step 窓で区切る密セグメント計量器**（`gpu_user_annotation` で 1 step = 1 窓） |
| `.tmp/f2-vllmroot/models/deepseek_v4/attention.py` | 再 patch 済みの attention.py（hot-mount 元） |
| `.tmp/f2-trace-{base,on,offmount}-rank0.json.gz` | rank0 の生トレース 3 本 |
| `.tmp/bench3-{t0off,t0on,t0on-r2,t0off-mounted,t0off-mounted-r2,restored-f2}.txt` | bench の生ログ |
| `.tmp/numerics-{t0off,t0on}.json` | 貪欲生成の本文 |
| `.tmp/sakubun-t0on.txt` | 作文 7 本 |
| `.tmp/f2-run.log` / `.tmp/f2-run2.log` | 走行ログ |
