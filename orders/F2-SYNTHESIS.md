# F2 五本の検分の突き合わせ（軍師）

軍師: YUKI（Fable 5.1）／2026-09-04。対象: `orders/F2-dense-launch-fusion.md`（v1）と GLM-L の 5 本
（overall / bandwidth / kernel / integration / gates）。
本作業は読み取りと 2 ファイルの書き込みのみ。ビルド・GPU・席 :8899 は使っていない。
軍師自身が確かめた事実には**行を引いて**ある。

---

## 判定: **GO-SMALL**

**新規カーネルの発注としては NO-GO。plugin wiring の限定スパイクとしては GO。**

- 期待利得（ESTIMATE）: **decode step の −4% 〜 −8%**（密セグメント 1.20〜1.40×）。
  1 流 62 tok/s → 65〜67、4 流 136 tok/s → 142〜148。**倍にはならない。数 % の話である。**
- 上限（到達不能な参考値）: 密 2.23×、step −13.6%。
- **職人 1 名を丸ごと張る発注ではなくなった**。理由: 本発注が求めたカーネルは家に既にあり、
  上流 exllamav3 がグループ (a)(c) を実戦で使っている。残る仕事は plugin の wiring と門。
- スパイクの定義（境界を切る）:
  - **T0 のみ**を先に出す ＝ 同一 module call 内の shard 融合（`fused_wqa_wkv`・compressor・
    indexer.compressor・shared gate_up の各 2→1）。状態を持たず、順序仮定も無く、単一 stream で自明。
    launch 489 → 341/step。
  - **KILL LINE: T0 で密セグメントが 1.05× に届かなければ T1（遅延束ね）に進まない。**
    既定 OFF の flag なので撤退は汚くない。
  - T0 が通ったときだけ T1（層単位 h-fan / q_a-fan の遅延束ね）へ。T1 の KILL は密 1.15× 未満・step −3% 未満。

---

## 1. 五本が一致しているところ

1. **方向は正しい**。同一入力の密線形を 1 launch に束ねる grouped GEMV は成立する。5 本とも肯定。
2. **v1 の byte 会計と上限利得（1.5× / step 8%）の導出は誤り**。5 本とも指摘。理由は各様（後述）。
3. **家の `exl3_mgemm` が本発注の基準案そのもの**。overall・bandwidth・kernel・gates の 4 本が指摘。
   v1 だけがこれに触れていなかった——これが v1 最大の抜けである。
4. **遅延束ねは graph capture と両立する**（capture 中に Python が一度走り launch 列が焼かれる）。4 本一致。
5. **危険は capture ではなく eager 側**。給仕の capture sizes は `[1,2,4]` なので 4 流 decode の
   m=8..16 は毎ステップ eager で Python キャッシュが走る。stale hit は無言の誤値。
6. **aux stream との並行は禁止**。`exl3_devctx` の lock/barrier は device 大域の共有バッファで、
   2 stream 同時走行は永久 spin の実績あり。
7. **prewarm がないと capture が静的ヒューリスティックに落ちる**。しかも mgemm 経路のフォールバックは
   **無警告**（単発 gemm 側は警告付き）。
8. **NCCL 粒揃えの効果は仮説**。「効果」でなく「計測項目」に格下げすべき。
9. **v1 の門には同条件 baseline の定義が無い**。gates 検分の中心的な指摘で、他も同旨。

---

## 2. 食い違いと、軍師の裁定

### 2.1 【最重要】1 ステップの launch 数 — 255 か 489 か

- overall・kernel: **254.7/step**（11,715 ÷ **46** step）。46 は `execute_context_0(0)_generation_1(4)` の呼び出し回数。
- bandwidth: **489/step**（÷ **24** step）。24 は `exl3_moe_kernel<2>` 1,032 = 43×24 から。
- integration: 456/step（÷ 25.7）。moe kernel 1,104 を 43 で割った中間解。

**裁定: 489/step、24 step が正しい。**

軍師が profiler 表で検算した（`prof-graph-0904/profiler_out_0.txt`）:
- `:5` `exl3_moe_kernel<2,…>` **1,032 回**、`:18` `exl3_moe_kernel<3,…>` **72 回**。
  主モデル 43 層のうち K3 は 3 層。**1032 = 43 × 24、72 = 3 × 24**。
  gcd(1032, 72) = 24 であり、**46 は 1032 も 72 も割り切らない**。46 も 25.7 もステップ数ではありえない。
- 24 で割ると tile 別は 340.8 / 127.2 / 20.1 となり、integration 検分が層内トレースから読んだ
  tile 帰属（tile512 ≈ indexer.wq_b、21 層）と整合する。46 では 10.5/step の非整数になって帰属できない。
- `:4` の `execute_context_0(0)_generation_1(4)` は CUDA total 1.261 s と表示されるが、
  self CUDA の合計は 862.2 ms しかない。**重複計上される CPU 側 span であって step 数ではない。**
  ここが 3 本の検分が同じ石につまずいた場所である。
- 決定打: §0.1 の形表から構造的に数えた launch 数（2 層×8 + 20 層×10 + 21 層×13）が**ちょうど 489**、
  そして **11,715 = 24 × 489 − 21**（−21 は窓の端で indexer.wq_b が 1 ステップ分欠けたもの。
  tile512 の 483 = 21 × 23 がそれを裏書きする）。

結果として **1 ステップの密 EXL3 時間は 8.885 ms（206.6 µs/層）**であり、v1 の 107 µs/層は半分の見積りだった。

### 2.2 compressor の幅 — 512 か 1024 か

- v1: 4096→**512** ×2、43 層中 41。
- bandwidth: 4096→**1024** ×2（layers.10 の svh を実読み）。v1 を「誤り」と断じた。
- kernel/gates: 触れず。

**裁定: どちらも部分的に正しく、どちらも誤り。層によって違う。**

軍師が pack の safetensors ヘッダを全層走査した（`DSV4-…-Dense6/model-dense-exl3.safetensors`）:

```
attn.compressor.wgate  4096->1024 K=6   21 層 [2,4,6,8,10,12,...]   ← 偶数層
attn.compressor.wgate  4096->512  K=6   20 層 [3,5,7,9,11,13,...]   ← 奇数層
attn.indexer.compressor.wgate 4096->256  21 層（偶数層のみ）
attn.indexer.wq_b            1024->8192  21 層（偶数層のみ）
```

bandwidth 検分は layers.10（偶数）だけを見て 41 層すべてに 1024 を当てたため、byte を 63 MB/step 過大に積んだ
（1,138.1 MB → 正しくは **1,075.2 MB/step**）。v1 は逆に奇数層だけを見ていた。
**indexer 系が丸ごと表に無かった点は bandwidth・integration の指摘が正しく、v1 の明白な欠落である。**

### 2.3 上限利得 — 1.5× / 1.44× / 3.1× / 2.11×

- v1: 1.5×（step 8%）。overall: local 256 読みなら 3×、local 2048 読みなら 1.44×。
  bandwidth: 2.11×（step −13%）。kernel: 1.5〜1.7×（step 6〜8%）。

**裁定: 2.23×（step −13.6%）。ただし到達不能な上限であり、合否に使わない。**
分子は §2.1 の 8.885 ms、分母は §2.2 の 1,075.2 MB ÷ 270 GB/s = 3.982 ms。
v1 の 1.5× は分子（launch 過少）と分母（byte 過少）の誤りが相殺して偶然そう見えていた。

**現実的な利得は step −4〜−8%（ESTIMATE）**。bandwidth 検分は同じ scope を −3〜6%、
kernel 検分は 6〜8% と見た。軍師の積算（融合後 1 launch の帯域を 100〜202 GB/s の幅で置く）は
密 1.27〜1.46×・step −5.3〜−7.8%、これに cast/cat 削減の −1〜1.5% を足し、
悲観側を bandwidth 検分に寄せて **−4〜−8%** を採用値とした。

### 2.4 shared experts の中間幅 — local 256 か 2048 か

- overall: 「local 256 と仮定」と「2048 の字面通り」で場合分けし、上限が 3× か 1.44× か決まらないとした。
- bandwidth: 列分割 /8 で local 256。

**裁定: local 256。** 保存形 4096→2048、`moe_intermediate_size=2048`（config 実読み）、
`n_shared_experts=1`、gate_up は列分割 /8 → rank 局所 4096→**256**、0.795 MB/本。
F1 正典の routed 中間幅 local 256 とも一致する。overall 検分の場合分けは、これで閉じる。

### 2.5 新規カーネル `lna_dense_group.cu` は要るか

- v1: 要る（成果物の筆頭）。
- kernel 検分: **不要**。`exl3_mgemm` が既にそれで、上流 DSV4 が使っている。
- overall: S0（mgemm）→ S1（新カーネル）の二段。bandwidth: 書くなら差分理由を数字で。
- integration: locks 不使用の非協調構成を第一候補に、と新カーネル寄り。

**裁定: 新規カーネルは書かない。** 軍師が上流を実読みして確認した:

```
exllamav3-src/exllamav3/modules/dsv4.py:914-957
  "q_a / wkv / comp wkv+wgate / idx wkv+wgate as ONE per-matrix-N exl3_mgemm"   ← グループ (a) そのもの
  self.q_fan = mk_fan([self.q_b, self.idx_wq_b])                                ← グループ (c) そのもの
exllamav3-src/exllamav3/modules/dsv4.py:1445-1470   rows <= 32 でゲートして ext.exl3_mgemm を呼ぶ
exllamav3_ext/bindings.cpp:161                       m.def("exl3_mgemm", ...)
```

`mk_fan` が課す不変条件（K/mcg/mul1 一律、**先頭の出力が最幅**、padding 無し）は、そのまま我々の登録時検査になる。
**上流が束ねていないのは (b)（shared w1/w3）だけで、そこがうちの取り分。**
integration 検分の「locks 不使用の非協調構成」は魅力的だが、それを作る前に mgemm の実測が要る。
新規カーネルは、mgemm の実測が不足を示したときだけ。**理由を数字で書けなければ書かない。**

### 2.6 グループ (b) に遅延束ねは要るか

- v1: (a)(b) とも遅延束ね扱い。
- integration・kernel: **(b) は同一 module call 内の shard loop**（`exl3.py:1490-1509`）なので、
  遅延束ねなしで 2→1 にできる。

**裁定: integration/kernel が正しい。** (b) に状態は要らない。
これが T0（無状態の shard 融合だけ）を先に出せる根拠であり、KILL LINE を T0 に置ける理由でもある。
**遅延束ねが要るのは h-fan と q_a-fan だけ**で、これらは vLLM 側で別 module にまたがる。

### 2.7 「6/層 → 2〜3/層」は達成可能か

- v1・kernel: 可能（86 launch/step）。ただし kernel 検分の根拠は 255/step 前提。
- bandwidth・integration: 不可能。依存対の融合まで行かないと届かない。

**裁定: 不可能。** 正しい基準は 489/step（11.4/層）で、
(a)+(b) 後は 279/step（6.5/層）、(c) を足して 258/step（6.0/層）。**目標は 489 → 260 前後（−47%）**。
2〜3/層は wq_a→wq_b・wo_a→wo_b・w1/w3→act→w2 の依存融合まで含む数で、本発注の scope 外。
gates 検分の「参考値に格下げ」が正しい。

### 2.8 遅延束ねのキャッシュキー

- overall: `data_ptr` で同一性検証。
- kernel: **`data_ptr()` はキーに使えない**（caching allocator が同一アドレスを再利用する）。`id(x)` ＋弱参照。
- integration: (x.data_ptr(), m) ＋ **1 forward 内に限る**。

**裁定: 併用。** 有効範囲を 1 forward 内に閉じたうえで、キーは (層 id, グループ id, data_ptr, m) ＋世代カウンタ、
`id(x)` ＋弱参照を併用。forward をまたいだ参照は禁止し、キー不一致は solo 計算に fallback。
kernel 検分の「data_ptr は再利用される」という指摘は、integration の「forward 内に閉じる」という制約と
組めば無害化できる——両者は排他ではない。

### 2.9 1.25×T_floor をグループごとに課せるか

5 本とも触れていない（発注側の追加要件）。**裁定: 課すが、(b) は例外扱いにする。**
(b) は融合後も n 合計 512（TILE_N=128 で n タイル 4 個）しかなく、34 SM を埋められない。
F3 でも門 3 の絶対値は FAIL だった（`f1/REPORT-F3.md`）。
**(b) の拘束力ある門は同条件 A/B とし、1.25×T_floor は届かなくても FAIL にせず、理由を数字で書かせる。**

---

## 3. 推薦の理由（正直に）

**GO-SMALL にした理由**:

1. **利得が小さい**。step −4〜−8% は、F3 が routed で取った 1.85×（kernel-only）とは桁が違う。
   密は既にステップの 24.7% しか無く、その中の 3.156 MB 級は既に 175 GB/s（65%）出ている。
   束ねが本当に効くのは 0.795 MB 級——byte の 12% で時間の 33% を食っている層——だけで、
   そこは絶対量が小さい。
2. **しかし工数も小さい**。カーネルが要らない。上流の実装が手本としてある。
   plugin wiring と門で、職人 1 名の全期間ではなく、区切ったスパイクで済む。
3. **危険は無視できない**。遅延束ねは無言の誤値を生む新しい故障面で、しかも 4 流 decode は
   毎ステップ eager でその論理を踏む。だから**状態を持たない T0 を先に出し、そこで効かなければ止める**。
4. **ケンの完成の定義（graph を入れても速度が変わらない）には効く方向**。
   launch を 489 → 260 に減らすことは、graph ON/OFF の gap を縮める最も直接の手である。
   利得の数字が小さくても、この一点は測る価値がある。

**NO-GO にしなかった理由**: T0 だけなら状態も新カーネルも無く、既定 OFF の flag で撤退できる。
そのコストで step −1〜3% とラウンチ −30% が取れるなら、割に合う。

**職人 1 名を丸ごと張る GO にしなかった理由**: v1 が想定した仕事（新規カーネル）の大半が既に存在する。
存在するものを書き直させるのはフェラーリではなく、車庫にある車をもう一台作る仕事である。

---

## 4. 軍師が自分で確かめた事実（引用元）

| 事実 | 出所 |
|---|---|
| 24 step（1032 = 43×24、72 = 3×24、gcd=24） | `prof-graph-0904/profiler_out_0.txt:5,18` |
| 46 は CPU span 数（CUDA total 1.261 s > self 合計 862.2 ms） | 同 `:4,:106` |
| compressor は偶数層 4096→1024（21 層）・奇数層 4096→512（20 層） | pack `model-dense-exl3.safetensors` ヘッダ全層走査 |
| indexer.wq_b 1024→8192（21 層）・indexer.compressor 4096→256 ×2（21 層）が実在 | 同上 |
| shared w1/w3 保存形 4096→2048、`moe_intermediate_size=2048`、`n_shared_experts=1` | pack `config.json` |
| 上流 exllamav3 が (a)(c) を `exl3_mgemm` で束ねている | `exllamav3/modules/dsv4.py:914-957, 1445-1470` |
| `mk_fan` の不変条件（K/mcg/mul1 一律・先頭が最幅・padding 無し） | `dsv4.py:928-935` |
| binding の存在 | `exllamav3_ext/bindings.cpp:161` |
| SM 数 34（MEASURED）、L2 32 MB | `f1/REPORT-F3.md` |
| 帯域の天井は decode 側にある（K=2 の実測） | `f1/REPORT-F3.md` 門 2 |

改訂発注書: `orders/F2-dense-launch-fusion-v2.md`
