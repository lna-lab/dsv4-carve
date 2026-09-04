# F2 密 EXL3（K=6）launch 融合 — 門と試験の検分（GLM-L）

検分者: GLM-L（監督）。対象: `orders/F2-dense-launch-fusion.md`（2026-09-04 版）。
本検分のレンズは**門と試験のみ**。設計・速度見積りの妥当性は、門・試験・計測法に響く範囲でのみ扱う。
数字にはすべて MEASURED（実測）／ESTIMATE（推定）を付す。実測の出所は 09-04 の席トレース
（`prof-graph-0904/`、`prof/`）と F1 の報告・検分。ビルド・GPU・席 :8899 には触れていない（ファイル書き込みのみ）。

## 結論

三段の門（机上 parity → 席内 → 速度、速さは最後）という骨格と、参照を実行列ごとの
`LinearEXL3.forward` に置く判断（`orders/F2-dense-launch-fusion.md:36`）は正しい。この参照が最強の
捕獲器であり、F1 の「エラーなしの誤り」（|out|/|ref| ≈ 1/√2、`orders/REPORT-F1-opus.md:41-42`）も
この比較なら抜けない。**ただし現状の門には F1 で実際に効いた捕獲器の半分近くが落ちており、
速度の門は baseline の取り方が未定義のまま目標値（6/層 → 2〜3/層）だけが立っている。**
かつ、その目標の算術（層あたり launch 数）は発注書自身の数字と整合していない（§4.1）。

重大度順の要約:

1. **速度の門の baseline が未定義**（§4）。同条件 A/B の取り方、launch の数え方、eager の判定式、
   ノイズ帯の定義、8 rank 計測がすべてない。このままでは合否が言えない。
2. **launch 数の算術が内部矛盾**（§4.1）。表は 1 層 10 本の密線形（`orders/F2-dense-launch-fusion.md:12-21`）、
   実測は ≈6 launch/層、正典は「43 層 × 15〜20 本」（`docs/PLAN.md:57`）。どの 6 本が
   `exl3_gemm_kernel<6>` に当たるのか機上で分解しない限り、「2〜3/層」は検証不能。
3. **家に既に融合カーネルがある**（§0.3）。`exl3_mgemm`（1 入力 → 複数行列、行列ごと suh/svh/出力幅、
   1 launch）は K=6 でコンパイル済み・Python バインディングあり・家の DSV4 attention 経路で実戦済み。
   新規 `lna_dense_group.cu` の前に第一候補として測るべき。mgemm 経由は捕捉中の静的フォールバックが
   **無警告**（`exl3_gemm.cu:578`）なので、prewarm の網羅とその検査が門に必要。
4. **F1 の教訓を捕まえる試験が明示されていない**（§2）。parity はこの誤りクラスを捕まえるが、
   F1 で実際に傷を出したのは段階別計測器と bitwise 一致で、いずれも現発注書に無い。
   norm 比 canary・bitwise 一致・NaN poison replay・起動設定検査を加える。
5. **「遅延束ね」のキャッシュは新しい無言故障面で、専用の試験が無い**（§1.4）。
6. 机上 parity の m 範囲が {1,2,4,8,16} に狭まり、m>16（プリフィル）の dispatch 境界が無試験（§1.1）。
   比較段階（FP32 2e-3 vs bf16 別枠）も未指定（§1.2）。
7. `compute-sanitizer` は F1 時点で像に無い（MEASURED、`orders/REPORT-F1-opus.md:134-138`）。
   門に書くなら実行可能性の見極めと代替を発注書に書くこと（§1.5）。
8. 席内の門に、in-seat selfcheck・engagement log・NCCL byte・8 rank latency・untuned-tripwire が無い（§3）。

## 0. 実測確認と、検分で新たに確認した事実

### 0.1 発注書の「いまの数字」はトレースと一致する（MEASURED）

`prof-graph-0904/profiler_out_0.txt`（rank0、graph ON）:

| kernel | calls | avg | share | 行 |
|---|---:|---:|---:|---|
| `exl3_gemm_kernel<6,true,1,16,32,128,4,3>` | 8,179 | 17.97 µs | 17.05% | :7 |
| `exl3_gemm_kernel<6,true,1,16,32,256,4,3>` | 3,053 | 16.75 µs | 5.93% | :10 |
| `exl3_gemm_kernel<6,true,1,16,16,512,4,3>` | 483 | 31.30 µs | 1.75% | :19 |

計 11,715 launch・213.2 ms・**24.73%**。発注書の「≈255 launch・25%・17〜18 µs（tile 128/256）・
31 µs（tile 512）」（`orders/F2-dense-launch-fusion.md:24`）と一致する。
ステップ数 46（= 11,715/255）は ESTIMATE（私の逆算）。3 形とも `c_fp32=true, cb=1`、
すなわち現行の密経路は fp32 出力・mcg（`vllm-exl3/src/vllm_exl3/exl3.py:1509` の out_dtype=fp32 と整合）。

### 0.2 K=6 に GEMV 経路は無い（MEASURED、コード読み）

QTIP 型 GEMV は `if (K < 2 || K > 4) return false;`（`exl3_gemv.cu:111`）かつ
`static_assert(bits == 2 || bits == 3 || bits == 4)`（`exl3_gemv_kernel.cuh:142`）で K=6 を拒否。
トレースにも `exl3_gemv_kernel` は出ていない。**よって「`exl3_gemm_kernel<6>` の launch 数」は
今日の密 EXL3 GEMM の取りこぼし無い指標である**（速度の門の数え方の土台として重要）。

### 0.3 家の `exl3_mgemm` が既に本発注の基準案そのもの（MEASURED、コード読み）

- 1 入力を複数行列へ broadcast、行列ごとに suh/svh を別適用、出力も別（`exl3_gemm.cu:385-393`、
  入力 Hadamard の行列別 slab は `exl3_gemm_kernel.cuh:170-183`、出力 svh は :232-258）。
  「fused gate/up projections に使う」と明記（`exl3_gemm.cu:389-391`）。
- **出力幅が行列ごとに違う group (a)（{1024, 512, 512, 512}）も `size_n_list` + `c_ptrs` で対応済み**
  （`exl3_gemm.cu:455-476`。ただし `num_tokens==1 && min_index<0 && !weights` の制約 :468-470。密は満たす）。
- K=6 の mgemm インスタンスはコンパイル済み（`exl3_kernel_map.cuh:90-109` を
  `comp_units/exl3_comp_unit_6_cb1.cu:11` が展開）、Python から `exllamav3_ext.exl3_mgemm` で呼べる
  （`exllamav3-src/exllamav3/exllamav3_ext/bindings.cpp:161`）。
- 家の DSV4 実装は既にこれで融合している: compressor wkv+wgate を「ONE exl3_mgemm over 2 experts」
  （`libtorch/dsv4_compressor.cpp:33-57`）、fan 投影（`libtorch/dsv4_attn.cpp:96,385`、
  `libtorch/attention.cpp:391,450`）、wo_a（`libtorch/dsv4_attn.cpp:270`）。

つまり職人の仕事は「新カーネルを書く」より「plugin 側で pointer table を組み既存 mgemm を叩く」が
第一候補になり得る。成果物の `lna_dense_group.cu`（`orders/F2-dense-launch-fusion.md:41`）は
mgemm 案が速度の門を落としたときの第二案とすべき（発注書 :28 は職人の変更を認めている）。
門への影響: ① parity の oracle は変わらず行列ごと `LinearEXL3.forward`、② 追加の交差検査
（grouped vs `exl3_mgemm`、両方 vs 個別参照）がただで手に入る、③ 新規コードの試験面が縮む。

### 0.4 mgemm の捕捉中フォールバックは無警告（MEASURED、コード読み）

単発 gemm は捕捉中に未チューンだと警告を出して静的ヒューリスティックへ落ちる
（`exl3_gemm.cu:264-268`、09-03 20:45 の全 rank 死亡事故の対策、`docs/PLAN.md:66`）。
mgemm は `if (!graph && !lna_stream_is_capturing(stream))`（`exl3_gemm.cu:578`）で**黙って**静的選択に
落ちる。plugin の既存 prewarm（`vllm-exl3/src/vllm_exl3/exl3.py:1413-1429`）は行列単体の gemm だけを
暖めており、mgemm の autotune hash は bszm を含む（`exl3_gemm.cu:96-121`）ので**グループ形状は
prewarm 済みにならない**。F2 はグループ種 × 行数の prewarm と、「捕捉中に未チューンで落ちなかった」
ことの検査（gemm は警告文の grep、mgemm は info binding か警告の追加）を門に含めること。

### 0.5 その他の実測事実

- 像に `compute-sanitizer` 無し・PATH にも無し、かつ「新しい像タグは作らない決まり」
  （MEASURED、`orders/REPORT-F1-opus.md:134-138`）。
- ⚠ **資料の消失**: 検分中の 09-04 11:22 に、`prof/`（eager 側トレース一式、trace gz 8 本 +
  profiler_out 8 本）が外部プロセスによって空にされた（本検分はファイル書き込みしかしていない。
  `prof-graph-0904/` は 06:56 のまま無事）。空にされる直前に読み取った eager 側の事実:
  `cudaLaunchCooperativeKernel` 11,232 回 = 8,179+3,053（gemm kernel 呼び出し数と一致）。
  graph 側トレース（`prof-graph-0904/profiler_out_0.txt`）には同 API 行が無い=graph 内 kernel は
  ランタイム呼び出しと相関しない。**帰属分解（§4.1）には eager トレースの再採取が前提**になる。
  消えた経緯は家計簿に記すべき（PLAN の作法: 「消したものが無い」も見る、`docs/PLAN.md:53`）。
- greedy は実行ごとに不一致（`docs/PLAN.md:41`）なので、貪欲突き合わせの基準帯（49〜118 字）は
  発注書の書き方（同カーネル二度走り）を正として採る。

## 1. 門 1（机上 parity）の抜け

### 1.1 m の範囲と dispatch 境界

- 発注書は m ∈ {1,2,4,8,16}（`orders/F2-dense-launch-fusion.md:36`）。F1 の正典は「graph bucket と
  別に、正しさは 1 ≤ m ≤ 16 全部」（`orders/REVIEW-F1-sol.md:29`）。密も同様に {1,2,3,4,5,6,7,8,15,16}
  で採ること（自己検査ログに 2,3,5,6 行が実出した実績があり、DSpark3 の draft/target で奇数行が来る）。
- **m > 16（プリフィル、BT 512/1024）の dispatch 境界が無試験**。設計は「m ≤ 16 の行は同じ CTA が担う」
  （:31）で、m > 16 は既存経路のままのはずだが、境界（16/17、32、512）を flag ON のまま
  全経路で通す parity を 1 通りずつ入れる。プリフィルが新しい経路に誤って入ると
  正しさとプリフィル速度の両方が静かに壊れる。
- CUDA graph の bucket は席の実設定（単流 [1,2,4]、多流 16 等）と鏡像にする
  （F1 と同じ作法、`orders/REVIEW-F1-sol.md:215`）。

### 1.2 比較の段階と閾値を固定する

- 現行の密出力は fp32 で書いて最後に bf16 へ落とす（`vllm-exl3/src/vllm_exl3/exl3.py:1509,1512`）。
  **主判定は fp32 段階で per-row rel ≤ 2e-3、bf16 変換後は別枠**（F1 の実測では 2e-3 は bf16 の
  量子化床 1.95e-3 そのもので、失敗全件がちょうど bf16 1 ULP、`orders/REPORT-F1-opus.md:111-118`）。
  段階を書かないと、bf16 段で 2e-3 を課して「失敗」の偽陽性か、fp32 段で 6e-3 を許して感度低下の
  どちらかが起きる。
- 比較は**行列ごと**に行う（group (a) の 4 出力を連結したまま norm を見る形は不可）。
  連結比較は 1 行列の劣化を他で希釈する。

### 1.3 実テンソルの層選定とグループ網羅

- 層 5 と 40（:36）だけでは足りない。**compressor を持たない 2 層のうち少なくとも 1 層**を含めること
  （41/43 層が compressor 持ちなので、group (a) は 4 本形と 2 本形（wq_a+wkv）の 2 種類ある。
  2 本形は 43 層中 2 層で、無試験のまま出荷されうる）。harness は各層がどちらの形かを log して
  「両形を cover した」ことを表明させる。
- 網羅リスト（parity の対象）を発注書に列挙する: group (a) 4 本形 / 2 本形、group (b) w1+w3、
  非グループの wq_b（k=1024, n=4096/rank）・wo_a（rank 局所 4096→1024）・wo_b（k=1024/rank）・
  w2（k=256/rank）、および singles を grouped kernel に通す場合の 1 行列グループ。
  k=1024/256 は suh 長と tiles_k が変わり、group (a)/(b) の k=4096 とは別の経路を叩く。
- 合成テンソルは**行列ごとに意図的に異なる suh/svh**（例: 2^i 倍）で作ること。
  「同じ入力でも suh/svh は行列ごとに別」が ABI の正典（`orders/REVIEW-F1-sol.md:62-68`）で、
  表の取り違えは suh が偶然一致する合成では発覚しない。実テンソル（校正付き焼き）は本来的に異なるが、
  保険として番兵を置く。

### 1.4 「遅延束ね」キャッシュの専用試験（新規の無言故障面）

設計 4（:32）の「最初の呼び出しで束ねて計算し、残りはキャッシュから返す」は F2 固有の新しい状態で、
graph capture と eager の両方で stale を出しうる。最低 3 試験:

1. **eager 連続試験**: 同じ層・同 m で異なる x を 2 回通す。2 回とも個別参照と一致すること
   （キャッシュが pass トークン無しで効いていると 2 回目が 1 回目の値を返す）。
2. **NaN poison replay**: グループ出力と A_had 系 scratch を NaN で埋め、capture した graph を
   m バケット（1/2/4/8/16）と層を交互に変えて多数回 replay（F1 は 48 回でやった、
   `orders/REPORT-F1-opus.md:126`）。NaN 残留 = stale。A_had は mgemm では「行列ごとに 1 slab」で
   **不足すると無言の OOB 破壊**（実測で見つかった、`exl3_gemm.cu:477-479`）。
   発注書の bindings 素案（:41）に A_had/scratch 引数が無い。明示的に渡すか、capture 安定な
   内部確保であることを書き、この試験で担保する。
3. **graph 完全無効の通し**: graphs を切った席構成で greedy 突き合わせ（§3）。

### 1.5 `compute-sanitizer` の実行可能性

門 1 の memcheck（:36）は F1 時点で像に道具が無く実行不能（§0.5）。発注書に以下を書くこと:
host 側 toolkit からの bind-mount を最初に試す（新像タグは作らない決まり）、不可なら
memcheck/racecheck/synccheck/initcheck の 4 点を**席の最初の作業**として回す（F1 が残した順、
`orders/REPORT-F1-opus.md:249-251`）。試みと結果を REPORT に残す。黙って門から消すのは化粧合格。
なお発注書は memcheck のみで、F1 正典は 4 工具（`orders/REVIEW-F1-sol.md:200`）。
CTA 間 lock 還元や複数行列の scratch 共有がある以上 racecheck は省けない。

### 1.6 落ちていない F1 の捕獲器（門 1 に戻す）

- **bitwise 一致**（同一入力二回、tolerance とは別 gate）: `orders/REVIEW-F1-sol.md:206`。
  固定順還元ならただで通る。通らない実装ならその旨を REPORT に書いて bitwise 要求を外す判断を残す。
- **ABI 検証**: 全 pointer の dtype/shape/stride/device/alignment/lifetime/marker/K extent
  （`orders/REVIEW-F1-sol.md:208` の密版: trellis.size(-1)==96、suh.numel()==k、svh.numel()==n_i、
  mcg marker 0xCBAC1FED は host 検査 — plugin が既にやっている形、`vllm-exl3/src/vllm_exl3/exl3.py:1360-1387`）。
  登録時に k が異なる行列（wq_b 等）を group (a) へ混ぜたら loud fail する番兵も含める。
- **有限性**: 入力 finite 契約と debug finite flag（F1 正典、`orders/REVIEW-F1-sol.md:96`）。

## 2. F1 の教訓（256 thread 起動で k-tile 半分落ち）を捕まえる試験

F1 傷 2 の構造（`orders/REPORT-F1-opus.md:33-42`）: `TILESIZE_K=32` は 512 thread 必須
（`exl3_gemm_kernel.cuh:9` の `__launch_bounds__(256 * TILESIZE_K / 16)`、
`exl3_gemm_inner.cuh:75` の `sub_k = threadIdx.x / 256`）なのに 256 thread で起動し、
`sub_k` が常に 0 になって各 k タイルの半分が黙って消えた。**エラーも NaN も出ず、速度だけは出る。**

parity（rel ≤ 2e-3 vs 個別参照）なら 0.755 の乖離は捕まる。ただし F1 で実際に傷を切り分けたのは
段階別計測器（`f1/debug_stage.py`）と norm 比であり、単一の rel 数値は「どこが悪いか」を与えない。
以下を重ねること（いずれも安い）:

1. **norm 比 canary（必須）**: 全 group × 全 m × 全 variant で、行列ごとに
   0.99 ≤ ‖out‖_F / ‖ref‖_F ≤ 1.01 を別 assert として課す。1/√2 = 0.755 はこの不変量の
   シグネチャそのもので、参照の組み間違いや harness バグにも効く。
2. **起動設定検査（必須）**: 新カーネル（または mgemm 呼び出し側）の host コードに、
   block_dim と compile-time tile の一致式（block_dim == 256 × TILESIZE_K / 16）を
   prepare 時（eager、capture 前）に検査させて不一致なら fail。F1 は式だけが再発防止で、
   機械検査ではなかった（`orders/REPORT-F1-opus.md:245` が自分で「式だけ」と認めている）。
3. **段階別 probe**: `f1/debug_stage.py` 相当（行列別・必要なら 128 block 別の部分出力比較）を
   F2 版として作り、失敗時の切り分けを 1 走で済ませる。
4. **variant 網羅 parity**: autotuner が選びうる tile 形（shape 2/3/4、`exl3_kernel_map.cuh:54-60`）
   の全部と、実際に production が選んだ形（profiler の kernel 名テンプレート引数で同定、§0.1 の表の通り）
   の両方で parity を通す。**検査した構成と出荷構成が違う**ことは F1 の教訓の一般形である
   （capture 時の静的フォールバック、§0.4、がまさに構成が変わる経路）。
5. **席の強い捕獲器は ppl**: 密は全層を貫くので、この誤りクラスは ppl 6.72±0.02 を大きく割る。
   速度の数字だけで判断させない本発注の門順序（速さは最後）は正しい。

## 3. 門 2（席内）の抜け

発注書 :37 を土台に、以下を追加。

1. **貪欲突き合わせは 2×2 で**: 発注書は「graph ON/OFF」しか書いてない。比較は
   (flag ON, graph ON) vs (flag ON, graph OFF) に加えて **(flag ON, graph ON) vs (flag OFF, graph ON)**
   （= 変更そのものの回帰検査）も同帯（49〜118 字）で。基準テキストは同日・同一構成・flag OFF から採る。
2. **in-seat selfcheck**: F1 の `LNA_EXL3_NATIVE_SELFCHECK` 相当（`orders/REPORT-F1-opus.md:252-254`）を
   密版で。実活性で 1 層分の group を個別 `LinearEXL3` で再計算して比較、capture 中は
   `cudaStreamIsCapturing` で分岐して走らせない。mismatch は strict で fail。
   「エラーなしの誤り」に対する唯一の稼働中捕獲器。
3. **engagement log**: routed の "EXL3 MCG trellis engaged" 相当の一行（group 数・single 数・層数）を
   rank ごとに出し、43 層 × 両 group が全 rank で登録済みであることを数で表明。
   「一部の層が黙って旧経路に落ちた」ことをログで見えるようにする。
4. **NCCL byte・dtype 境界**: row-parallel（w2/wo_b）の出力が all-reduce へ入る byte 数と dtype が
   baseline と同じこと（F1 正典 `orders/REVIEW-F1-sol.md:216`）。設計 5（:33）で境界を動かさないと
   書いてあるので、それを計測で閉じる。
5. **untuned-during-capture tripwire**: serve log に `[exl3_gemm] LNA-LAB: untuned shape`
   （`exl3_gemm.cu:265`）が出ていないことを assert。mgemm 経由の場合は警告が無い（§0.4）ので、
   捕捉前に grouped prewarm が効いたことを info で出すか、mgemm 側に同じ警告を足す。
6. **8 rank 計測**: P50/P95 と最遅 rank、3 回以上、thermal steady（F1 正典 :217）。発注書の
   「NCCL の待ち（粒揃え）にも効く」（:26）の主張は、**rank 間の密セグメント時間の散らばり
   （max−min/step）の前後比較**を取らないと反証不能。この計測を門に入れるか、主張を REPORT の
   仮説に格下げすること。
7. **ppl 計器の条件を固定**: 6.72 ± 0.02 の出所は Dense6 の 6.7159（`docs/PLAN.md:69`）。
   計器は `ppl-vllm.py`・spec OFF・4k（`docs/PLAN.md:50`）。同一 pack・同一計器・同一順で
   再測した baseline と比較すること（過去の数値と直接比べない）。
8. 家の門の作法（`docs/PLAN.md:31`）に画像テストカードと三言語正気がある。vision 塔は bf16 で
   本変更の外側だが、1 回の画像テストカードは安い保険なので回すこと。KV トークン数の log 行も
   慣例として残す（scratch 追加分の健全性確認）。

## 4. 門 3（速度）と同条件 baseline の取り方

### 4.1 まず launch の帰属を分解する（机上で可能）

発注書内で 3 つの数字が噛み合っていない: 表は 1 層 10 本（compress 層、:12-21）、実測 ≈6 launch/層
（:24）、正典は「43 層 × 15〜20 本」（`docs/PLAN.md:57`）。10 本の行列がすべて
`exl3_gemm_kernel<6>` に当たるなら 430 launch/step オーダのはずで、実測 255 と合わない
（draft 層・indexer 由来の混在、あるいは表に無い行列の一部が別経路の可能性。ここは未確定）。
**eager トレース（op 帰属が取れる。09-04 の一式は消失したので席で再採取、§0.5）で
launch → モジュール/形状の対応表を先に作り、
どの行列が今なんの launch で走っているかを確定してから速度の門を組むこと。**
これをしないと「6/層 → 2〜3/層」は目標として検証不能で、group (a)+(b) の融合だけなら
6 → 4/層（−2 launch/層）で止まる可能性が arithmetically ある（ESTIMATE）。
合否の本体は step 時間・密 kernel 合計時間・launch 総数の実測値に置き、
「2〜3/層」は REPORT の参考値に格下げする。

### 4.2 A/B の取り方（同条件）

- **同一ビルド・同一 pack・同一席構成で `LNA_EXL3_DENSE_GROUP` の OFF/ON**（既定 OFF、:42）は
  このための正しい設計。OFF→ON→OFF の交互、各 3 回以上、で系統ドリフトを相殺。
  過去の 09-04 数値と比べない（.so の積み替えと plugin 変更で条件が変わる。baseline は当日再測）。
- **flag OFF の経路無変更確認**: flag OFF の kernel 名・件数が baseline と完全一致することを
  profiler 表で assert（回帰ゼロの直接証明）。
- **launch の数え方**: graph ON は profiler 表の kernel 名で数える（op 帰属は切れる、§0.5）。
  対象は `exl3_gemm_kernel<6` 全 variant ＋ 新 grouped kernel 名 ＋ **plugin 側が増やした補助
  カーネル（cat/copy/elementwise）**。F1 正典の「補助 kernel が層の外に増えていない」
  （`orders/REVIEW-F1-sol.md:214`）を F2 にも課す。GEMV は K=6 で存在しない（§0.2）ので
  kernel 名の網羅はこれで足りる。
- **帯域効率の前後**: 発注書の「帯域効率 ≈ 6 割」（:25、3.1 MB launch の ESTIMATE）は
  fusion 後の各 group launch についても同じ式（bytes/時間 ÷ 270 GB/s）で出す。
  group 単位で「融合前の和」を割っていなければ早期に切る（REPORT に per-group 前後必須）。

### 4.3 eager でも落ちないことの測り方

「graph OFF（eager）でも落ちない」（:38）を測定式にする:

1. 同一席構成で graphs を切り（serve 台本の eager 相当）、`bench-streams.py <port> 256 1,4` を
   flag OFF/ON 交互 3 回。**agg tok/s が OFF のばらつき帯（交互 OFF run の max−min）を下回らない**こと、
   かつ step 時間（metrics）で同じ判定をすること。
2. eager トレース（`prof/` 型、op 帰属あり）で flag ON/OFF を比較: 密経路 kernel 時間の総和が
   減っていること・launch 総数が減っていること・層あたりの補助 kernel が増えていないこと。
3. ケンの完成の定義「CUDA graph を入れても速度が変わらない」は gap の指標として測る:
   (graph ON の tok/s − graph OFF の tok/s) を flag OFF と flag ON の両方で報告し、
   gap が縮んだ/消えたことを数字で示す。

### 4.4 速度の門の合否基準

F1 の作法に倣い、絶対値と相対値の両方: ① 密 kernel 合計時間が baseline 比 x µs 以上の削減
（発注書の ESTIMATE はステップ 8% 前後、:26 — あくまで上限）、② bench-streams 1,4 の agg tok/s が
baseline を下回らない、③ eager も同じ（§4.3）。①の削減が出ない場合は REPORT に落ちた道として
残し、flag は OFF のまま出荷（既定 OFF なので疵は残らない）。

## 5. 「この発注は割に合うか」（発注書 :26 が監督に訊いているので）

**割に合う、ただし条件付き（ESTIMATE）**:

- 上限利得は発注書自身の ≈1.5×（25% → 17%、ステップ 8% 前後、:26）。内訳は
  launch 固定費 ≈6.5 µs/launch（3.1 MB launch の 18 µs − 帯域下限 11.5 µs、:25 の実測から）
  × 3〜4 launch/層 ≈ 20〜26 µs/層 + 帯域効率の改善分。launch 固定費の回収だけなら
  6→4/層（group 2 本化）で ≈13 µs/層に留まり、8% 主張は帯域効率改善に依存する（未証明。
  F1 の教訓: 並列度を上げても帯域は自動では出ない、`orders/REPORT-F1-opus.md:173-212`）。
- 工数は mgemm 既存（§0.3）なら plugin 側が主で小さく、既定 OFF の flag で撤退も汚くない。
- NCCL 粒揃えの主張は現在反証不能（§3.6 の計測を入れて初めて門にできる）。
- 結論: §4.1 の帰属分解と per-group 前後の早期チェック（§4.2 最下段）を門に付けた上で着手可。
  それ無しの着手は、目標数値が検証不能なまま職人を走らせることになる。

## 6. 発注書へ反映すべき修正（要約）

- [ ] 門 1: m を {1,2,3,4,5,6,7,8,15,16} + dispatch 境界（16/17/32/512、全経路）。
- [ ] 門 1: 比較段階の固定（fp32 2e-3 主判定・bf16 別枠）と行列ごと比較。
- [ ] 門 1: 層選定に非 compressor 層、group/single 網羅リスト、suh/svh 番兵（行列ごとに異なる値）。
- [ ] 門 1: bitwise 一致・ABI 検証（K=96・marker・stride・k 一致の loud fail）・finite 契約を復活。
- [ ] 門 1: 遅延束ねキャッシュの 3 試験（eager 連続・NaN poison replay・graphs 全断ち）。
      A_had/scratch を ABI に明示（`exl3_gemm.cu:477-479` の教訓）。
- [ ] 門 1: norm 比 canary（0.99〜1.01）と block_dim/tile 一致の機械検査（F1 傷 2 対策）。
- [ ] 門 1: sanitizer は 4 工具。実行可能性の確認手順と席 fallback を明記。
- [ ] 門 2: 貪欲 2×2（graph × flag）、in-seat selfcheck、engagement log、NCCL byte、
      untuned tripwire（mgemm の無警告フォールバック対策含む）、8 rank P50/P95・最遅 rank、
      rank 間散らばり計測、ppl 計器条件の固定、画像テストカード 1 回。
- [ ] 門 3: launch 帰属の事前分解（eager trace + `prof-shapes.py`）、A/B は同日同構成 OFF/ON 交互、
      flag OFF 経路無変更の assert、数え方の定義（kernel 名 + 補助 kernel）、per-group 前後、
      eager の判定式と gap 報告、合否基準の明記。「2〜3/層」は参考値に格下げ。
- [ ] 成果物: 既存 `exl3_mgemm`（K=6 コンパイル済み・`bindings.cpp:161`）を第一候補とし、
      `lna_dense_group.cu` はそれが門を落とした場合の第二案と明記。grouped prewarm
      （group 種 × 行数）を成果物に追加。
- [ ] REPORT-F2 に必須項目: launch 帰属表・per-group 前後・rank 散らばり・使用 variant と
      parity 済み variant の対応・MEASURED/ESTIMATE 区別（F1 と同じ作法）。

## 参照

- 発注書: `orders/F2-dense-launch-fusion.md`（門 :36-38、成果物 :40-43、数字 :24-26、設計 :28-33）
- F1 検分・報告: `orders/REVIEW-F1-sol.md`（:29, :62-68, :96, :199-208, :214-217）、
  `orders/REPORT-F1-opus.md`（:33-42, :111-118, :126, :134-138, :173-212, :245, :249-262）
- 家の exllamav3: `exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu`
  （:59-68, :96-121, :264-268, :385-393, :455-479, :578, :660-701）、`exl3_gemm_kernel.cuh`
  （:9, :170-183, :232-258）、`exl3_gemm_inner.cuh`（:75）、`exl3_kernel_map.cuh`（:54-60）、
  `exl3_gemv.cu`（:110-111）、`bindings.cpp`（:161）、`libtorch/dsv4_compressor.cpp`（:33-57）、
  `libtorch/dsv4_attn.cpp`（:96, :270, :385）、`libtorch/attention.cpp`（:391, :450）
- plugin: `vllm-exl3/src/vllm_exl3/exl3.py`（:172-183, :1360-1387, :1413-1429, :1460-1530）
- 計測: `prof-graph-0904/profiler_out_0.txt`（:7, :10, :19。eager 側 `prof/` 一式は検分中に消失、§0.5）、
  `prof-shapes.py`、`bench-streams.py`、`docs/PLAN.md`（:31, :41, :50, :53, :57, :66, :69）
