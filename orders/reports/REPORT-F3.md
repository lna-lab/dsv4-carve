# REPORT F3 — C+A 二号機（ticket scheduler × small-R inner）

職人: ユキ（Fable 5.1）／発注: `f1/F3-ticket-smallr-kernel.md`／検分: `f1/REVIEW-REPORT-F1-sol.md`
作業: run-36..46、2026-09-04。機械: RTX PRO 2000 Blackwell、**34 SM**、L2 32 MB、sm_120、GPU10/11。
正典: `/run/media/tonoken3/DATA1/vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/`

すべての数字に MEASURED / ESTIMATE / 未測定 を付す。実行は `f1/dock.sh`（GPU10,11 の容器）のみ。

---

## 0. 結論

| 門 | 結果 |
|---|---|
| 1. 同条件 baseline | **PASS**。全 8 bucket で現行を上回る。K2 m=4 **1.85×**、最良 K2 m=1 **2.16×**、最小 K3 m=2 1.38×。p95/p50 ≤ 1.066 |
| 2. 圧（pressure） | **条件付き PASS**。K2 m=4 の trellis 区間は 125 GB/s で 200 GB/s に**届かない**。ただし発注の代替条項どおり **compute ceiling を実測で示した**: 同じ load 経路から decode+MMA だけを外すと同区間は **194 GB/s**（K2 m=4）〜**270 GB/s**（K3 m=16）に達する。**メモリ経路はカードの実測天井に張り付く。天井は trellis decode 側にある** |
| 3. 性能 | **FAIL（絶対値）／PASS（相対値）**。cold p50 ≤ 1.25×T_floor は全 bucket 未達（K2 m=4 は 2.78×、192.6 µs vs 目標 86.5 µs）。同条件現行比は全 bucket 非劣化、K2 m=4 は 1.85× ≥ 1.15× |
| 4. scheduler / resource | **PASS**（compute-sanitizer を除く。像に無い） |
| 5. 数値 / graph | **PASS**。F1 の parity 一式 134 件を lna2 で再走。`bench_lna.py` の fail-open を塞いだ |

**採決の材料**: 二号機は正しさで一号機と同等（むしろ現行より高精度）、速さで**現行を全 bucket で 1.38〜2.16× 上回る**。
だが発注の絶対目標（1.25×T_floor）には届かない。門 3 の絶対値は、§4 の実測から
**現在の trellis decoder のままでは K2 m=4 では原理的に到達不能**である。

---

## 1. 設計（作ったもの）

`csrc/lna_moe_ticket.cu(.cuh)` + `csrc/lna_gemv_core.cuh`。一号機 `csrc/lna_moe_decode.cu` は消していない。
dispatch は `VLLM_EXL3_MOE_KERNEL=lna2`（既定は現行のまま。`lna` も従来どおり残す）。

### 外側 = A（常駐 expert-team ＋ 動的 ticket）
`exl3_moe_kernel.cuh:21-69, 261-282` と同じ形。**grid 全体の barrier は一つも無い**。

- grid = `dim3(W, 1, T)`、`W`=team 幅（既定 8 CTA）、`T`=team 数。`W*T ≤ SM 数 × 同時常駐数`。
  MEASURED: 34 SM × resident **2** → 68 CTA 枠、`W=8`, `T=8`（64 CTA 使用）。
- team は ticket を一つ取り、その expert を **入力 Hadamard → gate/up → activation/down-input →
  down → down 出力 Hadamard** まで通しで流し、`atomicAdd(&sched[0],1)` で次を取る。
  team 内 barrier は `group_barrier`（`ptx.cuh:319-345`）のみ、1 expert につき 4 本。
- **team は途中で半分に割れる**: `size_n=256` の gate/up は `256/COLS = 4` の n-group しか無く、
  8 CTA 全部を一方に当てると半分が遊ぶ。CTA 0..3 が gate、4..7 が up を**同時に**回す
  （Sol 検分「8-CTA team の半分で gate、半分で up」）。down は `4096/64 = 64` group を 8 CTA で分ける。
- **固定順 gather は 1 launch のまま**。expert 出力は pair 単位の FP32 scratch に置き、
  retired counter（`fetch_add` acq_rel）で**最後に退いた team が全 row を top-k slot 順で gather** する。
  atomics 無し・二本目の launch 無し・grid barrier 無し。
- route plan は**全 CTA が自分の shared に冗長に作る**。これで plan 用の grid barrier が消える。
  一号機で直した並列版（expert 別 16 bit 行ビットマスク＋256 要素 Hillis–Steele 排他スキャン 2 本）
  をそのまま使う。MEASURED: 全体の 2.7〜3.3%（`ticket` 行に含まれる）。

### 内側 = C（small-R 専用）
`exl3_gemv_kernel.cuh:140-380` の中核を device 関数に切り出した（`csrc/lna_gemv_core.cuh`）。
**新規に書き起こしたのではなく、検証済みの上流実装を再スケジュールした。** 変更点は三つだけ:

1. 上流の `[入力 Hadamard] grid.sync [core] grid.sync [出力 Hadamard]` から **grid.sync 二本を外した**
   （Hadamard は外側が team barrier の下で自前で回す）
2. `group` ループの範囲を `blockIdx.x / gridDim.x` から**引数 `group_begin / group_stride`** にした
   （team の CTA が n-tile を分け合うため）
3. 上流の `MMODE`（ROWS 1 か 8 の二択）を **`ROWS_CAP ∈ {1,2,4,8}`** に一般化した

数値に効く部分（per-lane の trellis 窓定数、`dq8_regs_{2,3}bits`、FOLD 周期の FP32 fold、
k-split の cross-warp reduction）は**一文字も変えていない**。

**なぜこれが正しい内側か（MEASURED）**: この pack の実 `R_e`（expert あたりの row 数）は
一様 routing で **m=4: 平均 1.04 / 最大 2、m=16: 平均 1.16 / 最大 3**。
canonical `exl3_gemm_kernel_inner` は `TILESIZE_M == 16` 固定（`exl3_gemm_inner.cuh:58`）なので
m16n8k16 MMA の 15/16 が padding で、生きた 1 row のために 16×N の FP32 C buffer を抱える。
`ROWS_CAP=1` はそれを一切持たない。

### 数値境界（F1 v2 発注書どおり。現行 incumbent とは違う）
x bf16 → カーネル内で fp16／gate/up の C は **FP32**／出力 Hadamard と `svh` は FP32／
`silu(min(g,L)) * clamp(u,-L,L)` を FP32／activation だけ fp16 にして down へ／
down の C と出力 Hadamard も FP32／expert 和は FP32 を **top-k slot の固定順**で（atomics 無し）→
最後に consumer dtype へ一度だけ変換。

**意図した逸脱が一つ**: 内側 MMA は FP16 累算で FOLD（=2）k-slice ごとに FP32 へ畳む
（`exl3_gemv_kernel.cuh:14`）。F1 検分は「全 adversarial test で余裕がある場合のみ可」としていた。
MEASURED（§5）: 全 134 件で最悪 1.18e-3、門は 2e-3。**現行 incumbent（1.35e-3）より良い**ので採用した。

### 設定（`-D` で切替可能。既定は実測で選んだ）
`LNA2_CFG=1`（256 thread、WK=8、WNT=4、COLS=64）、`LNA2_TEAM_W=8`、`LNA2_PFD=2`、
`LNA2_ROWS_CAP=4`、`LNA2_MIN_BLOCKS=2`。
MEASURED ptxas（run-41/43）: `-Xptxas -v` で **K2 119 / K3 205 registers（bound 無し）**、
`__launch_bounds__(256, 2)` を入れて **両方 128 registers・spill 16 B store / 12 B load**、
static smem **15,176 B**、`cudaOccupancyMaxActiveBlocksPerMultiprocessor` = **2**。

---

## 2. 門 1: 同条件 baseline（MEASURED、run-46）

同一 process・同一 layer・同一 `x`/`ids`/`weights`・同一 `U`/`R_e`・同一 FP32 出力・交互測定。
cold は毎回 96 MB を zero-fill（L2 32 MB を流す）。各 3 run × 40 iter、p50 は run 間の中央値、
p95 は run 間の最大。incumbent は `exllamav3_ext.exl3_moe`（`concurrency=-1`、本番と同じ呼び方）。
**両者とも毎 bucket で Python 参照と突き合わせてある**（`BASE_CHECK` 行）。

| K | m | U | R̄ | T_floor | incumbent cold p50 | lna2 cold p50 | lna2 cold p95 | lna2 warm p50 | p95/p50 | **速度比** | lna2 / T_floor |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 1 | 6 | 1.00 | 18.1 | 137.2 | **63.5** | 65.6 | 48.7 | 1.033 | **2.16×** | 3.51 |
| 2 | 2 | 12 | 1.00 | 36.1 | 196.0 | **125.0** | 131.1 | 89.5 | 1.049 | **1.57×** | 3.46 |
| 2 | 4 | 23 | 1.04 | 69.2 | 355.9 | **192.6** | 196.7 | 181.7 | 1.021 | **1.85×** | 2.78 |
| 2 | 8 | 41 | 1.17 | 123.4 | 619.5 | **327.7** | 331.8 | 318.6 | 1.012 | **1.89×** | 2.66 |
| 2 | 16 | 83 | 1.16 | 249.8 | 1141.8 | **622.7** | 627.2 | 588.2 | 1.007 | **1.83×** | 2.49 |
| 3 | 1 | 6 | 1.00 | 26.8 | 145.4 | **86.0** | 90.1 | 61.0 | 1.048 | **1.69×** | 3.21 |
| 3 | 2 | 12 | 1.00 | 53.6 | 206.9 | **150.0** | 159.8 | 120.2 | 1.066 | **1.38×** | 2.80 |
| 3 | 4 | 23 | 1.04 | 102.7 | 381.3 | **241.7** | 245.8 | 216.5 | 1.017 | **1.58×** | 2.35 |
| 3 | 8 | 41 | 1.17 | 183.1 | 662.5 | **422.0** | 426.7 | 388.2 | 1.011 | **1.57×** | 2.30 |
| 3 | 16 | 83 | 1.16 | 370.7 | 1208.4 | **784.5** | 790.6 | 749.0 | 1.008 | **1.54×** | 2.12 |

（µs。T_floor = U × unique bytes ÷ 実測 cold read 270 GB/s。K2 812,544 B/expert、K3 1,205,760 B。）

**⚠️ 229 µs は使えなかった。** F1 発注書の「現行 = 229 µs/launch」は席の値で、同一条件で測り直した
ものではない（Sol §5 の指摘どおり）。同条件で測ると **K2 m=4 の現行は 355.9 µs** である。
差の理由は未測定だが、席は graph ON・`-pl62 -lgc2600` 固定クロック・実モデルの L2 状態で回っている。
**本報告の速度比はすべてこの同条件値に対するもので、229 µs は一切分母に使っていない。**

**incumbent が余分に払っているもの（本報告の kernel-only 時間には含めていない）**:
`argsort` / `scatter_add` による expert 別ソートと `expert_count`、および出力 FP32 バッファの
zero 初期化。lna2 はどれも要らない（route plan を device 内で作り、gather は上書き）。
**plugin end-to-end は未測定**（§7）。

---

## 3. 門 2: 圧（pressure）— MEASURED、run-44

**`ncu` はこの像に無い**（MEASURED run-43: `/usr/local/cuda/bin` は nvcc ツールチェインのみ、
`ncu`/`nsys`/`compute-sanitizer` いずれも不在）。新しい像タグは作らない決まりなので、
発注の代替条項「取れる範囲＋自前カウンタ、無い計器は未測定と書く」に従った。

**測ったもの**: (a) 自前の phase clock（`clock64`、team 0 / CTA 0 / thread 0）で区間比率、
(b) logical unique bytes ÷ 区間時間、(c) **null-decode ビルド**（`-DLNA2_NULL_DECODE=1`）。
**未測定**: actual DRAM read bytes、L2 sector/hit、outstanding request、long-scoreboard、
barrier/membar stall、tensor/ALU utilization、achieved occupancy の実値（すべて ncu 必須）。
以下の GB/s は **logical unique bytes 基準**であり、実 DRAM bytes ではない（scale・x・scratch を含まない）。

### phase 内訳（MEASURED、run-40。team 0 の累計 cycle）

| phase | K2 m=4 | K2 m=16 | K3 m=4 |
|---|--:|--:|--:|
| 入力 Hadamard | 5.8% | 5.9% | 4.9% |
| **gate/up GEMV** | **49.6%** | **49.2%** | **55.9%** |
| activation + down 入力変換 | 6.4% | 6.2% | 5.2% |
| **down GEMV** | **32.9%** | **33.3%** | **29.5%** |
| down 出力 Hadamard | 2.0% | 2.2% | 1.7% |
| ticket（route plan 含む） | 3.3% | 3.2% | 2.8% |

trellis を読むのは GEMV 二本＝**時間の 82〜85%**。

### 圧の実測と compute ceiling（MEASURED、run-44）

`null_decode` は**同じ load 経路・同じ prefetch ring 深さ・同じ team scheduling** のまま
trellis decode と MMA だけを外したビルド（load は accumulate で生かしてある）。

| K | m | 版 | cold p50 | 全体 GB/s | GEMV 区間比 | **GEMV 区間 GB/s** |
|--:|--:|---|--:|--:|--:|--:|
| 2 | 4 | full | 182.3 | 103 | 0.822 | **125** |
| 2 | 4 | null-decode | 131.0 | 143 | 0.733 | **194** |
| 2 | 16 | full | 569.4 | 118 | 0.807 | **147** |
| 2 | 16 | null-decode | 367.6 | 183 | 0.747 | **245** |
| 3 | 4 | full | 221.2 | 125 | 0.847 | **148** |
| 3 | 4 | null-decode | 163.8 | 169 | 0.782 | **217** |
| 3 | 16 | full | 701.5 | 143 | 0.846 | **169** |
| 3 | 16 | null-decode | 475.1 | 211 | 0.779 | **270** |

読み方は三つ:

1. **メモリ経路はカードの天井に届いている。** K3 m=16 の null-decode で GEMV 区間 **270 GB/s** ＝
   Luna が別途測った cold read 帯域（264–275 GB/s）そのもの。**load の並べ方・ring 深さ・team の配り方は
   十分な圧を掛けられている**（ケンの的そのもの）。F1 で「in-flight 不足が有力仮説」と書いた点は、
   ここで**否定された**: 足りていないのは圧ではない。
2. **天井は trellis decode 側にある。** K2 m=4 で full 182.3 µs vs null-decode 131.0 µs、
   差 **51.3 µs = カーネルの 28%** が decode+MMA の純コスト。同区間は 194 → 125 GB/s へ落ちる。
   **K2 m=4 で 200 GB/s は、いまの decoder のままでは到達不能**（decode を全部消しても 194 GB/s）。
3. **K3 のほうが天井に近い**（full で 148〜169 GB/s、K2 は 125〜147）。K3 は 1 tile あたりの
   byte が 1.48× 多いのに decode 命令数はほぼ同じなので、byte/命令 比が良い。

`bits=2` は 1 word で 2 tile を運ぶ（`LOADS = WNT/2`）ため、K2 は同じ byte あたり
K3 の 1.5 倍の shuffle と decode を回している。これが K2 側の天井の正体である（MEASURED の
null-decode 差から逆算した ESTIMATE ではなく、上の表の直接比較）。

---

## 4. 門 3: 性能

- **cold p50 ≤ 1.25 × T_floor: 全 bucket FAIL。** K2 m=4 は 192.6 µs（目標 86.5）＝ **2.78×**。
  最良は K3 m=16 の 2.12×。
- **同条件 incumbent 比: 全 bucket 非劣化、K2 m=4 は 1.85×（要求 1.15×）で PASS。**
- **p95/p50 ≤ 1.10: 全 bucket PASS**（最大 1.066、K3 m=2）。
- 3 run 実施。**8 rank の最遅値は未測定**（席の門）。K と U と R̄ は全値に併記済み。

§3 から、1.25×T_floor までの残り 2.2× の内訳（MEASURED からの分解）:
K2 m=4 で 192.6 µs のうち **decode+MMA が約 51 µs（28%）**、**非 GEMV phase が約 33 µs（17%）**、
残り約 108 µs が GEMV の memory 区間。仮に decode を全部消しても 131 µs（1.89×T_floor）で、
**1.25× には届かない**。したがって門 3 の絶対値は、decoder の高速化と非 GEMV phase の削減の
両方が要る。片方だけでは足りない。

---

## 5. 門 5: 数値 / graph（MEASURED、run-45/46）

F1 の parity 一式を `F1_PARITY_BACKEND=lna2` で**そのまま二号機に流した**（テストは共有）。

| 項目 | 数 | 最悪 rel(FP32) | 判定 |
|---|--:|--:|---|
| 合成 K2/K3 × m∈{1,2,3,4,5,7,8,15,16} × route 6 種 × 入力 7 種 × weight 3 種 | 110 | 1.64e-3 | PASS |
| 本番テンソル L0(K2)/L13,22,28(K3) × m∈{4,16} × route 3 種 | 24 | **1.18e-3** | PASS |
| NaN poison → リプレイ（m と route を交互）× K2/K3 | 48 | 9.8e-4 | PASS |
| CUDA graph capture/replay、m∈{1,4,8,16} × K2/K3、各 8 replay | 8 | 9.7e-4 | PASS |
| bitwise 決定性（全ケース同一入力二回） | 134+ | — | PASS |
| `[m,6]` vs flat `[6m]` regression | — | flat は ABI が拒否／転置 id は出力が変わる | PASS |
| 追加 smoke（K2/K3 × m∈{1,2,3,4,8,16} × route 4 種） | 48 | 9.8e-4 | PASS |

**同条件の現行より精度が良い**: 同じ入力で incumbent は 1.28e-3〜1.48e-3（`BASE_CHECK` 行）。
incumbent は中間を fp16 で持ち回る（`exl3_moe_kernel.cuh:137` が `c_fp32=false`）のに対し、
二号機は F1 の FP32 境界を守っているため。**FP16 MMA + FOLD fold を採用しても現行より良い**、
というのが §1 の逸脱を認める根拠。

**`bench_lna.py` の fail-open を塞いだ**（発注 5）。旧 `graph_test` は
(a) capture 失敗を `continue` で読み飛ばし、(b) `ok=False` を print するだけだった。
どちらも `raise SystemExit(1)` にした（`f1/bench_lna.py:84, 103`）。

**pairwise coverage の明示**: 合成 110 件は `(m, route, input, weight)` の全直積ではない。
`m`(9) × `route`(6) を全網羅し、`input`(7) を `(mi+ri)%7`、`weight`(3) を `(mi+ri)%3` で
回した標本である。したがって **(m,route) は完全網羅、(m,input)・(route,input)・(m,weight)・
(route,weight) は pairwise 相当、(input,weight) は対角線上に固定**（`input` と `weight` は
同じ添字から回すので独立に組み合わない）。ここが現テストの穴で、次に直すならここ。

---

## 6. 門 4: scheduler / resource（MEASURED、run-45）

| stress | 結果 |
|---|---|
| active expert 数 = 1 / 2 / teams−1(7) / teams(8) / teams+1(9) / 2×teams(16) / 96 | 全 PASS（最悪 rel 8.8e-4）。※「0 active」は route が必ず 1 つ以上 expert を名指すので構成不能。1 で代替 |
| ticket wrap / 自己リセット: active 数を毎回変えて 120 連続 launch | PASS、最悪 rel 8.7e-4 |
| NaN poison（**データ領域のみ**、制御領域は自己リセット状態なので触らない）× 24 | PASS、最悪 9.2e-4 |
| routing scaling 1.5 番兵（zero / once / twice） | PASS。zero=0、once=5.31785、twice=7.97677、**比 1.500000** |
| 複数 stream（2 stream × 20 launch、stream ごとに別 scratch） | PASS、最悪 8.0e-4 |
| 最後の team による gather | 全 launch が経路として通る（上記すべて） |
| CUDA graph 連続 replay | §5 のとおり PASS |
| per-device prepared state | PASS。`prepared` を `(device, bits)` キーの `std::map` にした（F1 は process-global だった）。dev0/dev1 両方で問い合わせて同値を確認 |

**compute-sanitizer（memcheck/racecheck/synccheck/initcheck）は未実施。** 像に入っていない
（MEASURED run-43）。新しい像タグは作らない決まりなので**席の門に送る**。
このカーネルは device 全体の atomics（ticket・retired counter）と team barrier を使うので、
**racecheck は F1 のとき以上に重要**である。ここは正直に穴として残す。

---

## 7. 落ちた道（全部 MEASURED）

| 試したこと | K2 m=4 cold p50 | 判断 |
|---|--:|---|
| **既定**: CFG1 / W=8 / PFD=2 / ROWS_CAP=4 / MIN_BLOCKS=2 | **178–193** | 採用（run 間のばらつきは熱とクロック） |
| `PFD=4`（prefetch ring を深く＝圧を増やす） | 180（K3 は 221→256 と悪化） | **不採用**。register 圧が上がり K3 が 16% 悪化。圧はすでに足りていた（§3）ので深くしても得が無い |
| `MIN_BLOCKS=3`（resident 3 CTA/SM、teams 12） | 221 | 不採用。CTA を増やすと悪化。34 SM に 102 CTA は過積載 |
| `LNA2_CFG=0`（512 thread、WK=16、COLS=32） | 207 | 不採用。down は k=256 しかなく、WK=16 だと warp あたり k-slice が 1 になる |
| `TEAM_W=4`（team 17 個） | 190 | 不採用。U=23 に対し 2 wave になり二波目の tail が大きい |
| `TEAM_W=16`（team 4 個） | 238 | 不採用。team が太いほど expert あたりの並列度は上がるが、同時に走る expert が減る |
| `ROWS_CAP=2` | 179 | 中立。R̄=1.04 なので ROWS_CAP は 2 でも 4 でも実質 `ROWS_CAP=1` 経路が使われる |
| `__launch_bounds__` 無し | K3 が resident 1 | **不採用**。MEASURED: K3 205 registers → 1 CTA/SM。bound を入れて 128 registers・spill 16/12 B・resident 2 |

**やらなかった判断**: `gate` と `up` を別 ticket にする案は、task が 23→46 に増えて wave が
1→2 になるだけで利用率は変わらないと算数で判り、実装しなかった（ESTIMATE）。
代わりに team を半分に割る形（§1）を採り、これは効いた。

---

## 8. 残っている危険

1. **compute-sanitizer 未実施**（§6）。device atomics と team barrier を使うので racecheck は必須。
2. **team barrier は全 CTA の co-residency を前提にする。** `prepare_one` は
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` の実値から grid を決めているが、
   **他のカーネルと同時に走ると常駐が崩れて team barrier が deadlock し得る**。
   shared-experts の aux stream は現運用どおり `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 前提で、
   **明示 error にはまだしていない**（Sol の指摘が未消化。incumbent も同じ前提だが、
   incumbent はそれを host 側 comment でしか守っていない）。
3. **scratch の制御領域は「起動時に 0」に依存する。** `_lna2_scratch_for` が `torch.zeros` で
   確保し、以後はカーネルが自己リセットする。**scratch を `empty` で確保すると静かに壊れる。**
   graph capture 前に確保することも必須（capture 中の確保は明示 error にしてある）。
4. **route plan を全 CTA が冗長に作る。** ids が全 CTA から見えることに依存する。
   MEASURED で 3% 程度だが、m が大きくなるほど絶対コストは増える。
5. **R > ROWS_CAP は B を読み直す。** MEASURED の実 route では R̄=1.04〜1.16 なので効かないが、
   全 row が一 expert に集中する adversarial route では weight を `ceil(R/4)` 回読む。
   **速度も測った**（run-48、m=16 cold p50）:

   | route | U | Rmax | K2 | K3 |
   |---|--:|--:|--:|--:|
   | unique | 96 | 1 | 618.6 µs | 807.0 µs |
   | hot | 1 | 96 | 110.6 µs | 137.2 µs |
   | R_e16 | 16 | 6 | 137.3 µs | 165.9 µs |

   読み直しても U が小さいので絶対時間は unique より遥かに短い。**病理は無い。**
   ただし `hot` は 1 expert を 24 回読む（19.5 MB）ので、U=1 に対する理想（0.81 MB）からは遠い。
   実 route では起きないので直していない。
6. **`p95/p50` は run 内 40 iter の分布であって、8 rank・長時間・thermal steady ではない。**

---

## 9. 席の門に残すもの（ケンが回すぶん）

1. **compute-sanitizer 4 種**（像に入れられたら最初にこれ）
2. **plugin / layer end-to-end**: 本報告は kernel-only。`map_topk_to_local`・int32 化・
   FP32 contiguous 化・`empty_like` は測っていない。launch trace で
   **層あたり補助カーネルが増えていないこと**の証明が要る（incumbent は argsort と
   scatter_add と zeros を払っているので、end-to-end 比は kernel-only 比より**良くなる**はず＝ESTIMATE）
3. **graph ON/OFF の parity と latency、NCCL bytes、consumer dtype**
4. **8 rank 全部の P50/P95 と最遅 rank、3 回以上、thermal steady**
5. **spec OFF と DSpark3、1/4 流、作文 7 本 finish=stop、ppl 6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2**
6. **実 prompt の U と R_e ヒストグラムを全層で採る**（本報告は一様乱択のみ）

**判断の材料**: 門 3 の絶対値（1.25×T_floor）は未達だが、§3 でその理由は数字で確定した
——圧は足りている、天井は decoder にある。**現行から乗り換えるかどうかは、
「同条件で 1.38〜2.16× 速く、精度も良い」を採るか、「1.25×T_floor に届くまで待つ」かの判断**で、
後者を狙うなら次は decoder そのもの（K2 の 1 word 2 tile 展開の shuffle 削減）が的になる。

---

## 10. 成果物

| ファイル | 中身 |
|---|---|
| `csrc/lna_moe_ticket.cu` / `.cuh` | 二号機。ticket scheduler・team barrier・最終 team の固定順 gather |
| `csrc/lna_gemv_core.cuh` | small-R 内側（上流 `exl3_gemv_kernel.cuh:140-380` の中核を device 関数化、`ROWS_CAP` 一般化） |
| `csrc/bindings.cpp` | `lna2_moe_decode` / `_scratch_bytes` / `_prepare` / `_info` |
| `src/vllm_exl3/exl3.py` | `VLLM_EXL3_MOE_KERNEL=lna2` の dispatch、per-stream zeroed scratch、prewarm（既定は不変） |
| `setup.py` | `LNA_NVCC_EXTRA` でタイル/段/幅を再ビルド、`-Xptxas -v` で register/spill を毎回記録 |
| `f1/baseline.py` | **門 1**: 同条件 baseline（両者を毎 bucket で参照と突き合わせる `BASE_CHECK` つき） |
| `f1/pressure.py` | **門 2**: 区間スループットと null-decode ceiling |
| `f1/gate4.py` | **門 4**: ticket wrap・active 数境界・NaN poison・scaling 番兵・複数 stream・per-device |
| `f1/gate1_final.py` | 門 1 の最終表 ＋ 二号機の graph replay |
| `f1/phase2.py` | phase clock 読み出し |
| `f1/lna2.py` / `f1/lna2_smoke.py` | 呼び出しヘルパと smoke |
| `f1/parity_native.py` | `F1_PARITY_BACKEND=lna|lna2` で同じ 134 件を両機に流す |
| `f1/bench_lna.py` | fail-open を塞いだ（capture 失敗・replay 不一致は nonzero exit） |
| `f1/dock.sh` / `f1/RUN.sh` | host 側の実行口と標準 run |
