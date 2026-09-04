# REPORT F1 — オオタニ専用 routed-experts decode カーネル

職人: ユキ（Fable 5.1）／引き継ぎ元: Luna（run-1..17）／作業: run-18..35、2026-09-04
機械: RTX PRO 2000 Blackwell、**34 SM**（MEASURED、`multi_processor_count`）、L2 32 MB、sm_120、GPU10/11
正典: `/run/media/tonoken3/DATA1/vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/`

すべての数字に MEASURED / ESTIMATE を付す。実行はすべて `f1/dock.sh`（docker、GPU10,11）。
過去ログ `f1/history/run-N.log`。

---

## 1. 結論（先に）

- **門 D（机上 parity）: 合格。** 合成 110 件＋本番テンソル 24 件、FP32 出力での最悪 per-row
  相対誤差 **1.17e-3 ≤ 2e-3**（MEASURED, run-34）。同一入力の **bitwise 一致**、NaN poison
  48 回リプレイ、`[m,6]` vs flat の regression、CUDA graph capture/replay 6 バケット、すべて通過。
- **門 E（速度）: 不合格。** 最良 K2・m=4 で **cold 233.4 µs / warm 198.0 µs**（MEASURED, run-34）。
  現行 exllamav3 `exl3_moe_kernel` の **229 µs に対し 0.98×**（= わずかに遅い）。
  T_floor 比 **3.37×**（門は 1.25×）。m=8/16 はさらに悪い（0.48× / 0.32×）。
- Luna の設計の骨格（1 launch・cooperative persistent・expert 単位の CTA チーム・
  正典 inner GEMM の再利用）は**正しさでは完全に立つ**が、**速さでは現行に届かない**。
  理由は §6 に数字で書いた。**§6 の三案のうち一つを採らない限り、この形のままでは 229 µs を割れない**。

---

## 2. Luna の状態から直した傷（すべて実測で特定）

### 傷 1: pointer table の expert 数と route group 数の取り違え（run-17 の停止原因）
`lna_moe_decode_cuda` が `n_experts = gt.numel()`（= 256、rank-local expert id の定義域）を
`MAX_GROUPS = 96`（= 一回の route が名指しできる**相異なる** expert 数の上限 6m ≤ 96）と
突き合わせていた。本番レイヤは必ず 256 experts なので**全レイヤが弾かれる**。
→ `MAX_EXPERTS = 256` を別の定数として導入。**検査を緩めたのではなく、二つの別の量を分けた。**

### 傷 2（真因・run-19 で特定）: `TILESIZE_K=32` を 256 スレッドで呼んでいた
正典 `exl3_gemm_kernel`（`exl3_gemm_kernel.cuh:9`）と `exl3_moe_kernel`
（`exl3_moe_kernel.cuh:19`）はどちらも
`__launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)` で起動し、
inner は `sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS`（`exl3_gemm_inner.cuh:75`）で
スレッドを k 方向に二分する。`TILESIZE_K=32` は **512 スレッド必須**。
Luna は 256 スレッドで呼んでいたため `sub_k` は常に 0 で、**各 k タイルの半分が黙って落ちていた**。

MEASURED（run-19、stage-wise）: gate 相対誤差 0.67–0.78、`|kernel| / |ref| ≈ 0.755 ≈ 1/√2`。
半分の項が消えた乱数和の比そのもの。エラーも NaN も出ない。**速度だけ見て文面を見ない罠と同じ形**。

→ 最終形では `LNA_TILE_K=32` と **512 スレッド CTA** に統一（§3）。

### 傷 3: `gridDim.x > 1` に耐えない箇所が三つ
1 ブロック 1 expert（`gridDim.x=1`）でしか正しくないコードが残っていた。並列度を上げた途端に壊れる。
- `route_plan` の起動条件が `blockIdx.z == 0 && threadIdx.x == 0` → `gridDim.x>1` では
  複数ブロックが同じ `blockIdx.z` を持ち、**単スレッドの計画作成が自分自身と競合する**。
  → `block_linear() == 0` に変更。
- 要素ごとの phase（act / gather）の `tid`/`threads` が `blockIdx.z`/`gridDim.z` 基準
  → 重複実行と本数不足。→ 線形ブロック index に変更。
- lock 配列を wave ごとに memset していた → 同じ group の別スライスが既にその lock を
  spin している最中に踏む。正典の `barrier_release(..., reset=true)`（`ptx.cuh:131`）が
  自己リセットするので、**launch あたり一度だけ**ゼロ化する形に変更。

### 傷 4（最大の速度傷・run-27 で特定）: route plan が単スレッド × global メモリ
MEASURED（`f1/phase_probe.py`、clock64 を phase 境界で採取）:

| | m=4 | m=16 |
|---|---:|---:|
| route plan（旧） | 262,650 cyc = **29.8%** | 2,827,074 cyc = **66.9%** |
| route plan（新） | 5,025 cyc = **0.8%** | 5,630 cyc = **0.4%** |

旧実装は O(routes × groups × rows) の線形走査を **global メモリ上で** 1 スレッドが回していた。
新実装は 256 スレッド 1 ブロック、expert id ごとに 1 スレッド:
(1) expert ごとの 16 bit 行ビットマスクを shared atomic で作る、
(2) 256 要素の Hillis–Steele 排他スキャン 2 本で group index と pair base を出す、
(3) slot → pair は「自分より下の行の popcount」。
group 番号は「初出順」から「expert id 昇順」に変わったが、**gather は元の `[m,6]` slot 順の
まま**なので FP32 の加算順は不変（bitwise 一致は測定で維持、§4）。

---

## 3. 出来上がった設計（Luna 版からの差分）

1 layer 1 launch、cooperative persistent grid。**grid = `slices` × 1 × `groups`、512 threads/CTA**、
`slices * groups = SM 数 × 同時常駐数`（MEASURED: 34 × 1 = 34、`slices=1`、`groups=34`）。
`slices` は expert 1 個を担当するブロック数で、正典 inner の
`num_slices = gridDim.x` / `blockIdx.x` スライス機構（`exl3_gemm_inner.cuh:86-88`）に乗る。
`exl3_moe_kernel` が expert ごとに SM グループを割り当てるのと同じ仕組み。
`LNA_MOE_SLICES` 環境変数で上書きでき、これが §5 のスイープに使った口。

phase（各 phase の間で `grid.sync()`。仕事の無い phase でも early return しない）:
1. route plan（線形ブロック 0、上記の並列版）＋ lock 配列の一度きりのゼロ化
2. gate/up 入力変換: work = (expert, row, 128 入力ブロック)。`x` を一度 load し、
   gate 用 `suh` と up 用 `suh` を別々に掛けて 2 本の 128 要素 Hadamard を FP16 scratch へ
3. gate GEMM + up GEMM（expert グループ単位の wave ループ）
4. gate/up 出力 Hadamard + `svh`（FP32、`had_ff_r_128_inner<false,true>`）
5. clamp + SwiGLU（FP32）→ FP16。`silu(min(g,L)) * clamp(u,-L,L)`、L=10.0
6. down 入力 `suh` + Hadamard（`had_hf_r_128_inner<true,false>`）
7. down GEMM（wave ループ）
8. down 出力 Hadamard + `svh`（FP32）
9. 固定順 gather: 元の top-k slot 0..5 の順に FP32 weight を掛けて上書き加算（atomics 無し）、
   consumer dtype（bf16/fp16/fp32）へ同じ launch 内で変換して書き出し

**タイル形**（`csrc/lna_moe_decode.cu` のマクロ、`LNA_NVCC_EXTRA` で上書き可能）:
`LNA_TILE_K=32`（→ 512 threads）、gate/up は `TILESIZE_N=256`・stages 4/3、
down は `TILESIZE_N=512`・stages 4/3。MEASURED smem: K2 53,248 B / K3 61,440 B、常駐 1 CTA/SM。

**変えなかったもの**: trellis 復号と MMA は正典 `exl3_gemm_kernel_inner` をそのまま呼ぶ
（`cb=1` コンパイル時固定、`mcg` は host 側で全 expert 検査、`c_fp32=true`、
`shmem_out_had=false` にして出力 Hadamard は上記 phase 4/8 で FP32 のまま行う）。
Luna の primitive gate（K2/K3 の half bit 一致、run-1..14）はこの経路を変えていないので有効。

---

## 4. 門 D（机上 parity）— 全部 MEASURED、run-34

判定規則（`f1/parity_native.py`）:
- **主判定 = FP32 出力**での per-row 相対 L2 誤差 ≤ **2e-3**、かつ同一入力二回の **bitwise 一致**。
- bf16 consumer 出力は別枠 6e-3。理由: bf16 は仮数 8 bit で、丸めだけで 2^-9 = 1.95e-3 かかる。
  MEASURED（run-23）で「失敗」した全件の `max_abs` が **ちょうど bf16 1 ULP**（0.25 / 0.125 /
  0.000244141）だった。**2e-3 は bf16 の量子化床であって、カーネルの誤差ではない。**
- 参照行が退化している場合（全零 route、相殺 weight、FP16 subnormal 帯）は絶対誤差 1e-4 で判定。
  MEASURED: 相殺 route（w = +1,−1,+.25,−.25,…）の残差 1.2e-5 は**参照側 `index_add_` の
  FP32 丸め**であって、カーネル側ではない。

結果:

| 項目 | 数 | 最悪 rel(FP32) | 判定 |
|---|---:|---:|---|
| 合成 K2/K3 × m∈{1,2,3,4,5,7,8,15,16} × route 6 種 × 入力 7 種 × weight 3 種 | 110 | 1.70e-3 | PASS |
| 本番テンソル L0(K2) / L13,22,28(K3) × m∈{4,16} × route 3 種 | 24 | **1.17e-3** | PASS |
| NaN poison → 24 回リプレイ（m と route を交互に変える）× K2/K3 | 48 | 8.5e-4 | PASS |
| CUDA graph capture/replay、M_CAP 4/8/16 × K2/K3、各 8 replay | 6 | 1.05e-3 | PASS |
| `[m,6]` vs flat `[6m]` regression | — | flat は ABI が拒否、転置 id は出力が変わることを確認 | PASS |
| bitwise 決定性（全ケース） | 134 | — | PASS |

段階別の最大値も採った（本番 L22 K3 m=16 で `gate_max=6.51 up_max=7.46 act_max=48.5 down_max=6.54`、
全段 finite）。route は unique / hot / zipf / boundary(0,255) / within-row-dup / R_e=16、
入力は zero / real-bf16 / amplified / clamp 9.99, 10.0, 10.01 / large を含む。

**未実施（席で必要）**: `compute-sanitizer` の memcheck / racecheck / synccheck / initcheck。
**像 `lna-lab/vllm-exl3:dsv4-dense` に compute-sanitizer が入っていない**（MEASURED, run-35:
`/usr/local/cuda-13.0` 以下に無く PATH にも無い）。新しい像タグは作らない決まりなので、
ここは席側の門として残す。

---

## 5. 門 E（速度）— 全部 MEASURED、run-34、最終ビルド

計測法: kernel 単独。CUDA event を launch の直前直後に置き、cold は各回 96 MB バッファを
zero-fill して L2（32 MB）を流してから。50 回の p50/p95。routing は独立一様乱択。
T_floor = 実 unique bytes ÷ **264–275 GB/s**（Luna の `f1/cold_read_bw.cu` 実測、270 GB/s を使用）。
unique expert 1 個の byte: K2 812,544 / K3 1,205,760。

| 層 | K | m | U | cold p50 | cold p95 | warm p50 | T_floor | cold/T_floor | 229 µs 比 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| L0 | 2 | 1 | 6 | 208.3 | 209.9 | 185.9 | 18.1 | 11.53 | 1.10× |
| L0 | 2 | 2 | 12 | 208.9 | 211.3 | 188.0 | 36.1 | 5.78 | 1.10× |
| L0 | 2 | **4** | 23 | **233.4** | 235.6 | **198.0** | 69.2 | 3.37 | **0.98×** |
| L0 | 2 | 8 | 41 | 474.5 | 477.7 | 431.7 | 123.4 | 3.85 | 0.48× |
| L0 | 2 | **16** | 83 | **722.3** | 735.2 | 729.3 | 249.8 | 2.89 | **0.32×** |
| L13 | 3 | 1 | 6 | 218.5 | 221.6 | 194.1 | 26.8 | 8.15 | 1.05× |
| L13 | 3 | 2 | 12 | 222.2 | 223.3 | 196.2 | 53.6 | 4.15 | 1.03× |
| L13 | 3 | **4** | 23 | **268.7** | 270.8 | **224.8** | 102.7 | 2.62 | **0.85×** |
| L13 | 3 | 8 | 41 | 527.4 | 534.5 | 509.5 | 183.1 | 2.88 | 0.43× |
| L13 | 3 | **16** | 83 | **833.9** | 856.1 | 852.5 | 370.7 | 2.25 | **0.27×** |

U は一様乱択の実測値で、Sol の解析値（m=4 → 22.95、m=16 → 80.18）とよく一致した。

**cold ≈ warm**（233.4 / 198.0）が要点。**この カーネルは帯域律速ではない。**
m=4 K2 で 18.65 MB / 233 µs = **80 GB/s**（実測帯域 270 GB/s の 30%）。

phase 内訳（MEASURED、run-28、route 修正後、TILE_K=16 世代）:
m=4 で gate/up GEMM **69%**、down GEMM 23%、その他の phase 合計 8% 未満。
**時間はほぼ全部 GEMM の中**にある。

---

## 6. なぜ届かないか（数字で）

m=4 の全仕事量は タイル反復 で数えられる:
gate 512 + up 512 + down 128 = 1,152 タイル/expert × 23 expert = **26,496 タイル反復**。
34 SM に均等に配れば 1 SM あたり 780。MEASURED の 233 µs（2.6 GHz で 606k cycle）から
**1 タイル反復あたり ≈ 780 cycle**（TILE_K=16 世代では ≈1,700 cycle だった）。

1 タイル反復が運ぶ B は `TILEBLOCKS_K × TILEBLOCKS_N × 16 × K` uint16。
gate/up（TILE_K=32, N=256）で K2 なら 2 KB。`cp_async_wait<SH_STAGES-2>` により
実際に飛んでいるのは 2 ステージ分 ≈ 4 KB/CTA。34 CTA で **136 KB in flight**。
Little の法則で 270 GB/s × 800 ns ≈ **216 KB 必要**（ESTIMATE、レイテンシ 800 ns 仮定）。
つまり **in-flight が足りない**。そして in-flight を増やす道は smem に塞がれている:
K3・down（N=512）の 1 CTA が 61,440 B を使い、sm_120 の 100 KB/SM では **1 CTA/SM しか
常駐できない**（MEASURED: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` = 1）。
cooperative launch は全ブロック同時常駐を要求するので、**grid は 34 ブロックで頭打ち**。

ここが設計の袋小路。現行 `exl3_moe_kernel` が 229 µs を出せるのは、
**cooperative な全体バリアを持たず**、expert を動的チケットで配って
SM グループを走らせ続けるからで、常駐制約に縛られていない。
「1 layer 1 launch」を守ったまま速くするには、次のどれかが要る:

- **案 A（推奨・最小変更）**: phase 間の `grid.sync()` を捨て、`exl3_moe_kernel` と同じ
  **動的チケット式の expert スケジューラ**＋group 単位バリアにする。
  そうすれば grid は常駐上限に縛られず、tail も潰れる。gather だけは全 expert 完了後なので
  そこに 1 本だけ全体バリアが要る（= 2 段構成）。
- **案 B**: down の `TILESIZE_N` を 256 に落として smem を 32,768 B にする。
  ⚠️**訂正（Sol 検分・2026-09-04）**: 初版はここに「常駐 2〜3 CTA/SM に上がる」と書いたが、
  それは ESTIMATE で、**実測は反証している**。`f1/history/run-32.log:18-33` の occupancy 出力は
  K2 `[34,1,1,34,28672]`・K3 `[34,1,1,34,32768]`、すなわち**どちらも resident=1** である。
  512 スレッドとレジスタが同時に効いているので、smem だけ下げても 2 CTA/SM にはならない。
  MEASURED（run-33）では down N=256 は N=512 とほぼ同じ（240.0 vs 233.4 µs）で、
  **単体では効かなかった**。案 A と組み合わせないと意味が無い。
- **案 C（フェラーリらしい道・工数大）**: 正典 inner GEMM を使うのをやめ、
  この 1 台の形（H=4096, I=256, topk=6, m≤16）専用に
  「expert のタイルを一度復号してその expert の全 row に使い切る」内側ループを自分で書く。
  正典 inner は m≤16 でも 16 行パネル固定で、`R_e` が小さいときに MMA が空回りしている。

⚠️**§6 全体への訂正（Sol 検分・2026-09-04）**: 「incumbent は常駐制約に縛られない」は誤り。
incumbent も group barrier のため全 block を co-resident にしており（`exl3_moe.cu:203-222`）、
34 SM では 4 group × 8 CTA = 32 CTA 程度である。本当の差は block 数ではなく、
**8-CTA の expert team が ticket で一 expert を end-to-end に流し、global phase barrier を持たない**
こと。また in-flight の算数も m=4 では GEMM に入るのが 34 でなく 23 CTA なので約 92 KB が正しく、
800 ns のレイテンシも未測定の仮定である。**in-flight 不足は有力仮説であって確定診断ではなかった。**
→ F3（`f1/REPORT-F3.md`）でこの点は実測で決着した。

**m=16 が特に悪い理由**（0.32×）: U=83 > groups=34 なので wave が 3 回、
各 wave の後に全体 `grid.sync()` が入り、最後の wave は 83−68=15 expert しか無いので
**34 ブロック中 15 しか働かない**。案 A（動的チケット）はこの tail を直接消す。

---

## 7. 落ちた道（やって効かなかったこと。全部 MEASURED）

| 試したこと | 結果（K2 m=4 cold） |
|---|---|
| `slices`（expert あたりブロック数）1 → 2 → 4 → 8 → 17 のスイープ | TILE_K=16 世代で 349 → 275 → 262 → 262 → 304 µs。TILE_K=32 世代では **slices=1 が最良**（233 vs 260）。常駐が 1 に落ちるので分割の得が消える |
| gate/up の pipeline を深くする（stages 6/5、8/3） | 258 / 279 µs。**FRAG_STAGES=5 はレジスタ圧で常駐が 2→1 に落ち、深くした分を食い潰す** |
| down も stages 6/3 | 278 µs（悪化）。smem 54,272 B で常駐 1 固定 |
| down の `TILESIZE_N` を 512 → 256 | 240.0 µs（233.4 より僅かに悪化）。smem は 61,440 → 32,768 B に落ちて常駐は上がるが、cooperative grid は SM 数で頭打ちなので効かない |
| gate/up の `TILESIZE_N` 128 → 256 | 285 → 258 µs（TILE_K=16 世代）。**効いた**。タイルあたりの B が倍になり in-flight が増える |
| `TILESIZE_K` 16 → 32（512 スレッド CTA） | 258 → 233 µs。**最も効いた**。現行 `exl3_moe_kernel` と同じ形 |

**やらなかった判断**: gate と up を別タスクにして wave 内の並列度を上げる案は、
34 groups に対しタスクが 23 → 46 になり wave が 1 → 2 に増えるだけで
利用率（23/34 vs 46/68）は変わらないと机上で判り、実装しなかった（ESTIMATE）。

---

## 8. 残っている危険

1. **compute-sanitizer 未実施**（§4）。特に racecheck。新しい phase 並列化と
   `slices>1` の lock 共有は目で追ったが、機械に見せていない。
2. **`slices > 1` は本番既定ではない**が、コードには残っている。`LNA_MOE_SLICES` で
   有効化した場合、gate/up は `TILESIZE_N=256` で `tiles_n=1` になるため
   **スライスは k 方向に切れ、lock 連鎖の cross-block reduction が走る**。
   parity は slices=1 でしか通していない。**席で slices>1 を使うなら parity を取り直すこと。**
3. **shared experts の別ストリームとの重なりは未対応**。常駐 grid が GPU を占有するので、
   `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 前提のまま。明示エラーにはしていない（Sol §5 の指摘は未消化）。
4. **`prepared[][]` はプロセス大域**で、複数デバイスで別々の SM 数を持つ場合に最初の
   デバイスの値を使い回す。家は 8 枚同型なので実害は無い（ESTIMATE）が、正しくはない。
5. **scratch は stream ごと**に持つ形（`_exl3_lna_scratch[stream_key]`）にしてあるが、
   同時に複数の graph instance が同じ stream で replay される形は試していない。
6. **`LNA_TILE_K` を 16 に戻すと 256 スレッドになる**。マクロは通るが、
   §2 傷 2 の再発を防ぐのは `LNA_THREADS` の式だけ。書き換える人は `exl3_gemm_inner.cuh:75` を読むこと。

---

## 9. 席の門（ケンが回すぶん）に残っていること

門 D/E は机上で閉じた（E は不合格）。以下は席でしか測れない:

1. **`compute-sanitizer` 4 ツール**（像に無い。像を用意できたら最初にこれ）
2. **席内自己検査**: `VLLM_EXL3_MOE_KERNEL=lna` ＋ `LNA_EXL3_NATIVE_SELFCHECK=N`。
   実装済み（`_apply_lna_moe`、capture 中は `cudaStreamIsCapturing` 相当で分岐して走らせない）。
   steady decode の全バケットと K3 層（13/22/28）が通ったことの記録が要る。
3. **給仕の門**: graph ON/OFF の parity と latency、launch trace に補助カーネル
   （`.half()` / `.to()` / sort / zeroing）が層の外に増えていないこと、
   spec OFF と DSpark3、1/4 流、作文 7 本 finish=stop、ppl 6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2、
   scaling 1.5 が一度だけ、NCCL byte が基準と同じ。
4. **8 rank 全部**の P50/P95 と最遅 rank、3 回以上、thermal steady。各計測に K と U を併記。
5. **U のヒストグラム**を実 prompt・全層で採る（今回は一様乱択の U しか測っていない）。

**ただし 3〜5 に進む前に決めることがある**: 現状 m=4 で 0.98×、m=16 で 0.32× なので、
**このまま席に載せる意味は無い**。§6 の案 A（動的チケット式スケジューラ）を先にやるか、
F1 をここで畳んで現行 `exl3_moe_kernel` を続投させるかの判断が要る。
正しさの土台（parity 一式・phase 計測器・タイル形のスイープ機構）は全部残っているので、
案 A の作業は「phase 構造の差し替え」だけで、ABI も検証も再利用できる。

---

## 10. 成果物

| ファイル | 中身 |
|---|---|
| `csrc/lna_moe_decode.cu` / `.cuh` | カーネル本体。タイル形は `LNA_TILE_K` / `LNA_GU_*` / `LNA_DOWN_*` マクロ |
| `csrc/bindings.cpp` | `lna_moe_decode` / `lna_moe_scratch_bytes` / `lna_moe_prepare` / **`lna_moe_info`**（新: sms, resident, slices, groups, smem） |
| `setup.py` | `LNA_NVCC_EXTRA` でタイル形を再ビルドできる口（門 E のスイープ用） |
| `src/vllm_exl3/exl3.py` | dispatch・9 pointer 検証・prewarm・自己検査（Luna 版のまま） |
| `f1/dock.sh` | **host 側の実行口**（PROTOCOL.md の中継の代わり。`f1/history/run-N.log` に採番して記録） |
| `f1/RUN.sh` | 標準の門 D+E 一式 |
| `f1/parity_native.py` | 門 D。判定規則・shape regression・NaN poison replay |
| `f1/bench_lna.py` / `f1/bench_final.py` | 門 E。cold/warm latency と CUDA graph replay |
| `f1/phase_probe.py` | **phase ごとの clock64 計測器**（route plan の傷を出したのはこれ） |
| `f1/debug_stage.py` | 段階別（gate/up/act/down）の scratch 直読み比較。傷 2 を出した道具 |
| `f1/primitive_gate.cu` / `f1/cold_read_bw.cu` | Luna の門 1 と帯域実測（そのまま有効） |
