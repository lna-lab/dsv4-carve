# F1 専用 MoE decode カーネル設計検分

## 結論

方針そのもの、すなわち「routing を device 内で整理し、expert ごとに重みタイルを一度 decode して、その expert に来た全 row へ再利用する」ことは妥当である。ただし、現発注書のままでは正しさと性能目標の両方に重大な修正が要る。

- `suh` は row 共通でも expert 共通でもない。さらに同じ expert の gate と up でも別である。したがって「入力 Hadamard を row ごとに一度だけ計算し、全 expert で共有」は不可能である。共有できるのは、同じ `(expert, row)` の gate/up 変換を同じ CTA が担当するときの `x` の load までである。
- 「expert を一度だけ読む」は、expert 全体を on-chip に置く意味ではなく、各 trellis/scales タイルを、その expert に属する全 row に使い切ってから捨てる意味に限定する。0.81～1.21 MB の expert 全体は shared memory に載らない。
- 1 layer 1 launch は cooperative persistent kernel で実現可能性がある。ただし全 CTA の同時常駐、全 `grid.sync()` への同順序到達、graph capture 前の occupancy 決定、stream ごとに分離した scratch が必須である。
- `<= 105 us` は **K2・m=4 の stretch goal** としてのみ残せる。K3・m=4 は理論 peak 288 GB/s でも重みだけで約 96 us になり、実用的な余白がない。m=16 は一様 routing 想定で K2 でも約 226 us が peak 帯域下限なので、105 us は成立しない。
- 正確性の基準は Cruz の半精度中間や既存 fused kernel の演算順ではなく、plugin の Python reference と canonical EXL3 decoder/Hadamard とする。特に clamp の位置、FP32 accumulation、activation を FP16 に落とす位置を固定する。

この修正を受け入れるなら実装着手可、受け入れずに「全 m/K で 105 us」「row ごとに Hadamard 一回」「FP16 gate/up 中間」を必須とするなら発注仕様として不成立、と判定する。

## 1. 形状・ABI の検分

### 1.1 TP8 の形状

実モデルの config、safetensors、`F1-ref/vllm_exl3_exl3.py` の shard 規則を照合した結果、次の前提は正しい。

- hidden `H = 4096`
- TP-local intermediate `I = 2048 / 8 = 256`
- routed experts `E = 256`
- `top_k = 6`、従って 1 rank、1 layer の route slot 数は最大 `6m <= 96`
- gate と up は column shard、down は row shard。TP slice は連続する 256 要素で、128 要素 Hadamard block の境界にも整列する。

主モデル 43 layer のうち K3 は layer 13/22/28 の 3 layer、残る 40 layer は K2 である。後述の scale 込み実 byte 数で平均すると 1 expert/layer は約 0.840 MB なので、発注書の 0.86 MB は安全側の概算だが、K3 acceptance を代表しない。

一方、decode の `m` は 4 と 16 だけではない。selfcheck の過去ログにも 2、3、5、6 row が現れている。graph の主 bucket を 4/8/16 にしても、kernel の正しさは全 `1 <= m <= 16` を対象にする必要がある。

### 1.2 9 系統の EXL3 pointer ABI

1 expert 当たり、device kernel が必要とする実データは下記の 9 系統である。

| projection | trellis | `suh` | `svh` |
|---|---:|---:|---:|
| gate (`w1`) | `[256, 16, 16K]` int16 | `[4096]` fp16 | `[256]` fp16 |
| up (`w3`) | `[256, 16, 16K]` int16 | `[4096]` fp16 | `[256]` fp16 |
| down (`w2`) | `[16, 256, 16K]` int16 | `[256]` fp16 | `[4096]` fp16 |

表の先頭 2 次元は `[入力/16, 出力/16]` の 16×16 tile grid であり、最後の次元 `16K` int16 は tile 当たり `8K` uint32、すなわち `32K` byte/256 weights に対応する。実装 API は少なくとも `trellis/suh/svh × gate/up/down` の 9 pointer table と、その lifetime を明示すべきである。`w1` と `w3` は形が同じでもデータも `suh/svh` も別であり、alias を仮定してはいけない。

`mcg` は weight payload ではなく codebook marker である。実 tensor の値は signed int32 で `-877912083`、unsigned 表現で `0xCBAC1FED` だった。現行 plugin は host 側で全 expert の marker を検査し、device kernel には codebook 1 を選ばせており、kernel は `mcg` tensor を dereference しない。新 kernel も以下のどちらかに固定する。

- 推奨: load/build 時に全 projection・全 expert の marker を検証し、kernel は compile-time `cb=1` とする。
- 汎用化する場合だけ codebook ID を ABI に加える。ただし raw `mcg` pointer を毎回読む設計にはしない。

K は config と tensor extent の双方から検査し、`trellis.size(-1) == 16K` を全 expert、全 projection で保証する。K2 と K3 を同じ pointer table に混在させる仕様にはしない。

### 1.3 trellis decode の正典

- 16×16 tile の circular bitstream 開始点は `((t_offset + 257) * K - 16) mod (256K)` bit である。`257`、wrap、lane 内 bit order のいずれも ABI の一部として扱う。
- K2 は 256 weights 当たり 16 uint32 words。Cruz の paired-N-tile load と canonical `dq8_aligned_2bits`/`dq8_regs_2bits` は利用できる。
- K3 は 24 uint32 wordsで、load は lane 0..23、抽出は modulo 24 と追加 state `s2` を含む canonical `dq8_regs_3bits` に合わせる。
- MCG codebook は unsigned multiply `0xCBAC1FED`、`lop3 0x6a`、half 加算の丸めまで含めて同一にする。

`F1-ref/exl3_dequant.cuh` の scalar convenience helper は、最後の加算を float で返す経路があり、half の丸めまで canonical decoder と bit exact とは限らない。これを oracle にせず、`exllamav3-src/.../exl3_dq.cuh` と `codebook.cuh`、または Cruz の register decoder を正典にする。

最低限、all-zero、all-one、walking-bit、乱数 bitstream を用い、全 lane、全 `t_offset`、wrap 境界、K2/K3 について「decode 後の half bit pattern」を canonical device 実装と比較する primitive test が必要である。

### 1.4 Hadamard の扱い

EXL3 の Hadamard は長さ H/I 全体への一回の変換ではなく、**独立した 128 要素 block ごとの Walsh–Hadamard** である。各 projection について次の順序を守る。

1. 入力要素へ、その matrix 固有の `suh` を乗算する。
2. 各 128 要素 block に Hadamard と `1/sqrt(128)` normalization を適用する。
3. trellis から復号した matrix と積を取る。
4. 出力も各 128 要素 block に Hadamard と normalization を適用する。
5. その matrix 固有の `svh` を乗算する。

実 tensor を比較すると gate と up の `suh` は一致せず、expert 間でも一致しない。従って許される再利用は次の通りである。

- `(expert, row, 128-input-block)` を一つの task とし、同じ CTA が `x` を一度 load する。
- load した `x` から、gate 用 `suh` と up 用 `suh` をそれぞれ掛け、2 本の Hadamard 結果を作る。
- その 2 本は当該 expert と row にしか再利用しない。down は入力次元も `suh` も別なので別変換である。

発注書の「input Hadamard は row ごとに一回」は「同じ `(expert,row)` の gate/up で元の `x` load を共有する」に書き換えるべきである。

### 1.5 SwiGLU、dtype、出力境界

plugin reference の演算境界は次である。

```text
x(BF16) -> FP16
gate/up GEMM と出力 Hadamard: FP32
act = silu(min(gate, L)) * clamp(up, -L, L): FP32
act -> FP16
down GEMM と出力 Hadamard: FP32
routing weight 乗算と expert 和: FP32
最終 consumer dtype へ変換
```

従って以下を固定する。

- 正しい式は `silu(min(g, L)) * clamp(u, -L, L)` であり、`silu(g)` の後を clamp する既存 `had_hf_r_128_guad_inner` の順序はコピーしない。
- `fminf/fmaxf` は NaN を数値側へ潰す場合がある。reference と同じ NaN semantics を要求するか、入力 finite を契約にして debug finite flag で fail するかを明文化する。
- gate/up accumulation を FP16 scratch に落としてから clamp してはいけない。負側 overflow が `-inf` になれば SiLU の挙動も変わり得る。FP32 で clamp/SwiGLU まで行う。
- clamp 後の activation は概ね絶対値 100 以下なので FP16 化の意図が明確であり、ここが down の ABI 境界である。down accumulation と最終 Hadamard は FP32 を基本とする。
- Cruz の FP16 MMA accumulation + 定期 FP32 fold は optional optimization に留める。採用条件は最終許容値ぎりぎりではなく、全 adversarial test で十分な余裕を持つこととする。

真に 1 launch と数えるなら、BF16 入力の FP16 化、route ID の int32 化、route weight の FP32 化、sorting、出力 zeroing を外部の Torch kernel にしてはならない。kernel が実際の入力 dtype を受けて load 時に変換するか、caller ABI が初めから必要 dtype を供給する必要がある。

発注書は「FP32 を返して caller が AllReduce」としているが、現 plugin の最終値は `x.dtype`（通常 BF16）へ戻る。FP32 出力のまま既存 shared expert と加算すると型 promotion や NCCL payload 倍増を招き得る。内部和は FP32 としつつ、同一 launch の最後に既存 consumer dtype へ書く案を第一候補とし、実際の vLLM consumer と AllReduce byte 数で決める。

また routing scaling factor 1.5 が kernel 入力に既に含まれるのか、外側で掛かるのかを ABI に明記する。同じ固定入力で「省略」「一回」「二回」を識別できる sentinel test を置く。

## 2. 推奨する 1-launch CTA 構成

### 2.1 grid と常駐条件

- 基本形は cooperative persistent grid、`P = cudaDevAttrMultiProcessorCount` blocks、1 CTA = 256 threads（8 warps）とする。
- RTX PRO 2000 の 4352 CUDA cores から 34 SM と推測できるが、製品 SKU や runtime を推測値で固定せず、8 rank 全てで device attribute を記録して `P` を決める。
- 最終 register/shared-memory 使用量で `cudaOccupancyMaxActiveBlocksPerMultiprocessor` を取り、`P` blocks が同時常駐可能なことを launch 前に保証する。`2P` blocks や 512-thread CTA は bench variant とし、常駐性と tail を実測して選ぶ。
- `<K, M_CAP>` を compile-time variant とし、例えば M_CAP 4/8/16 を prewarm する。実 `m` は bucket 内 runtime 値として扱う。
- occupancy query、autotune、allocation は graph capture 前に一度だけ行い、結果を cache する。

### 2.2 phase と task 割当て

全 phase で persistent CTA が work queue を取り、全 CTA が同じ回数・同じ順序で `grid.sync()` に到達する。CTA が仕事を持たない phase でも early return させない。

1. **Route plan 作成**
   - CTA 0 が最大 96 個の `(expert, row, kpos, weight)` を固定順で整列する。
   - 同一 row 内に同じ expert が複数回あっても `(expert,row)` の計算は一度だけにする。一方、各 `kpos` と weight は合算せず保存し、最後に元の top-k 順で足す。これにより duplicate route でも reference の加算順を保つ。
   - active expert list、各 expert の row span、各 `kpos` から計算済み expert-row slot への対応を小さい global scratch に書く。
   - scratch counter/flag もここで世代管理または初期化し、別 zero kernel を不要にする。その後 `grid.sync()`。

2. **Gate/up 入力変換**
   - work item は `(expert, row, input-128-block)`。
   - `x` を一度 load し、別々の gate/up `suh` を掛けて 2 本の normalized Hadamard 結果を FP16 scratch に置く。その後 `grid.sync()`。

3. **Gate + up と activation**
   - 1 CTA が `(expert, intermediate-output-128-block)` を所有する。I=256 なので active expert 当たり 2 task。
   - gate と up の trellis は別々だが、それぞれの tile を一度 decode し、その expert に属する全 row へ使う。
   - FP32 accumulation、出力 Hadamard、`svh`、正しい preactivation clamp/SwiGLU を行う。
   - activation を FP16 にした直後、同じ CTA で down 用 `suh` と入力 Hadamardを行い、down-input scratch に置く。こうすれば activation と down-input の間の追加 grid barrier を避けられる。
   - m=4 の一様 routing なら active expert は期待値約 23 で、約 46 task となり、34 SM 級を一 wave 以上埋められる。
   - 全 task 後に `grid.sync()`。

4. **Down**
   - work item は `(expert, output-column-tile)` とし、N tile 256 と 512 を比較する。N=512 なら expert 当たり 8 task、m=4 の期待値で約 184 task ある。
   - 各 CTA は K=256 を sweep し、decode した B fragment をその expert の全 row fragment に適用してから捨てる。
   - `R_e=16` でも、二つ目の 8-row panel のために同じ B を再 load/decode してはならない。16-row MMA panel、または B register を保持したまま row panel を反復する構造が必要である。
   - FP32 accumulation と blockwise 出力 Hadamard/`svh` 後、`(row, expert-slot, output-tile)` の FP32 scratch へ書く。その後 `grid.sync()`。

5. **固定順 gather**
   - 1 warp を `(row, output-128-block)` に割り当て、元の top-k slot 0..5 の固定順に FP32 routing weight を掛けて加算する。
   - atomics を使わず、出力を上書きする。これで local kernel の反復 bitwise determinism と scratch の stale 値排除を両立する。
   - 必要なら同じ launch 内で consumer dtype に変換して書く。

この分解なら「同じ weight tile をその expert の row 数だけ再読する」問題を避けながら、m=4 でも expert 単位 CTA より並列 task 数を増やせる。m=1 は active expert が最大 6 なので gate/up phase の並列度が低い。m=1 も性能対象なら N=64、K-split、または既存 group GEMV への dispatch を別途比較するが、主目標 m=4/16 の設計を複雑化させない方がよい。

## 3. 帯域下限と 105 us の現実性

scale を含む 1 unique expert の正確な local weight byte 数は次である。

- K2: trellis `786,432 B` + scales `26,112 B` = `812,544 B`
- K3: trellis `1,179,648 B` + scales `26,112 B` = `1,205,760 B`

独立・一様な routing の参考値として、`U = 256 * (1 - (255/256)^(6m))` より、m=4 で `U=22.95`、m=16 で `U=80.18` となる。つまり random route での dedup 効果はそれぞれ約 1.05 倍、1.20 倍にすぎない。実 workload は row 間相関があり得るため、全 layer・実 prompt で U の histogram を採ることが必須である。

| 条件 | unique weight bytes | 288 GB/s 下限 | 200 GB/s 下限 | 127 GB/s 下限 |
|---|---:|---:|---:|---:|
| m=4, K2 | 18.65 MB | 64.8 us | 93.2 us | 146.8 us |
| m=4, K3 | 27.67 MB | 96.1 us | 138.4 us | 217.9 us |
| m=16, K2 | 65.15 MB | 226.2 us | 325.8 us | 513.0 us |
| m=16, K3 | 96.68 MB | 335.7 us | 483.4 us | 761.3 us |

これは scratch traffic、route plan、integer decode、MMA、Hadamard、grid barrier、tail を一切含まない下限である。従って次の目標に分けるべきである。

- K2・m=4: 105 us を stretch goal。達成には cold read で実効 200 GB/s 以上かつ compute/barrier の重なりが必要。
- K3・m=4: 固定 105 us を外し、`T <= 1.25 * T_floor` と既存比 speedup の複合基準にする。peak 下限から 9 us しか余らず、通常は不成立。
- m=16: U 実測依存の帯域下限で判定する。一様 routing では 105 us は不可能。同じ 6 expert に全 row が集中するような極端な相関時だけ別扱いにする。

発注書の D2D 127 GB/s は「copy payload byte/time」なら DRAM では read+write 約 254 GB/s 相当であり、read-only kernel の上限が 127 GB/s という意味ではない。重みと同じ access width/order、cold cache の read-only microbenchmark を別に作り、`T_floor = actual_unique_bytes / measured_read_bandwidth` を算出する。RTX PRO 2000 の公称 memory bandwidth は 288 GB/s である。

想定 bottleneck の優先順位は下記である。

1. m=4 K2 は cold trellis DRAM read。
2. K3 は 24-word circular extraction、integer multiply、`lop3`、shuffle と tensor work の比率が上がる。
3. m=16 は weight reuse が増えるため、MMA/reduction、register pressure、row panel 処理へ移り得る。
4. phase barrier、task tail、wave quantization。
5. Hadamard と global scratch traffic。

Nsight Compute では DRAM read bytes/bandwidth、L2 sector/hit、tensor/ALU utilization、achieved occupancy、long-scoreboard、barrier/membar stall、wave 数を採る。「論理 unique bytes/time」と「実 DRAM bytes/time」を分けて報告し、cold と warm も分ける。56～62 W は診断情報に留め、合否基準にはしない。高速な memory-bound kernel が常に board power 上限へ達するとは限らない。

## 4. 追加すべき検証ゲート

### 4.1 parity / primitive

現 `parity-native.py` の m=1、I=2048、限定 expert、cosine 主体の確認だけでは足りない。以下を追加する。

- TP 実形状 I=256、m は少なくとも `1,2,3,4,5,7,8,15,16`、K2/K3 の直積。
- production tensor は K2 layer と K3 の layer 13/22/28 を含める。合成 tensor も併用する。
- route は重複なし、全 row 同一 hot expert、hot/cold・Zipf、expert 0/255、同一 row 内重複、`R_e=16`、zero weight、相殺 weight。無効 ID `-1/256` を受けるならその仕様も試す。
- top-k ID の `[m,6]` と flat `[6m]` を取り違える regression test。
- zero、実 BF16 capture、増幅入力、NaN/Inf、clamp 境界 `9.99/10.0/10.01` と大値。gate は pre-SiLU clamp、up は対称 clamp であることを個別に検証する。
- gate/up FP32、activation FP32 と FP16 cast 後、down、route 後 final の stage-wise 比較。
- 最終判定は per-row relative error `<= 2e-3` に加え max absolute error と finite 一致を持つ。reference が零付近では denominator epsilon または absolute threshold を使う。
- scratch/output を事前に NaN poison し、graph を多数回 replay、K/m/route を交互に変えて stale state を検出する。
- `compute-sanitizer` の memcheck、racecheck、synccheck、initcheck を release 前 gate にする。

### 4.2 selfcheck

- Python reference は allocation/sync を含むため capture 中に走らせない。`cudaStreamIsCapturing` で明示的に分岐し、graph 前後の eager call で行う。
- mismatch は warning のみで継続せず、strict mode では fallback または起動失敗にする。再現用に x、routes、weights、K、limit、全 9 pointer table の識別情報を dump する。
- 同じ入力を二回実行し、推奨 fixed gather なら local kernel 出力の bitwise 一致も要求する。reference との数値 tolerance とは別 gate にする。
- 初回 N 回は `max |gate|, |up|, |act|, |down|` と finite flag を収集する。正しい limit=10 の後の act はおおむね 100 に収まるが、pre-clamp gate/up と down は FP16 range 内とは仮定しない。
- trellis だけでなく全 9 pointer、dtype、shape、stride、device、alignment、lifetime、marker、K extent を検証する。
- graph warmup の一形状だけでなく、steady decode の全 bucket と K3 layer が selfcheck を通ったことを記録する。

### 4.3 serving

- CUDA graph ON/OFF の parity と latency。M_CAP 4/8/16、K2/K3 を全て capture/replay する。
- graph node/launch trace で、`.half()`、`.to()`、sort、zeroing 等の補助 kernel が layer 外に増えていないことを確認する。
- speculative decode OFF と DSpark3、1/4 stream、既定 7 essays、perplexity baseline、accept length を確認する。
- routing scaling factor が一度だけ適用されること、shared+routed 加算後 dtype、NCCL payload byte 数が baseline と同じことを確認する。
- rank 0 だけでなく 8 rank 全ての P50/P95 と slowest rank、3 回以上、thermal steady state の長時間 run を記録する。各計測に K と active unique expert 数 U を併記する。
- layer/expert を回すか L2 flush を用いた cold-cache 計測を acceptance の主値とし、warm replay は別記する。
- shared expert auxiliary stream を無効にした production 条件と、意図的な overlap 条件を試す。後者を未対応とするなら silent race ではなく明示 error/fallback にする。

## 5. cooperative launch、CUDA graph、決定性の注意点

- cooperative launch は device の cooperative-launch attribute が必要で、grid size は最終 kernel の register/shared-memory 使用量から得る同時常駐上限以下でなければならない。超過は `cudaErrorCooperativeLaunchTooLarge` になる。
- CUDA graph への cooperative kernel capture 自体は現 stack の既存 EXL3 kernel でも使われており、禁止事項ではない。ただし device property/occupancy query、variant 選択、allocation、pointer-table 構築は capture 前に完了する。
- 全 CTA は全 `grid.sync()` に同順序で到達する。route が空、task がない、error flag が立った場合も block 単位 early return をしない。
- 常駐 grid は GPU を占有するため、shared expert の auxiliary stream と安全に overlap するとは仮定しない。現運用どおり overlap を無効化するか、別 kernel/明示的同期を設計する。
- scratch と route plan は layer 単位だけでなく、同時 replay し得る stream/graph instance ごとに分離する。既存 process-global lock/scratch の流用は避ける。
- graph 内 raw pointer は replay 間で安定でなければならない。model move/reload、EPLB/repack 後に stale にならないよう、graph は最終 load 後に構築し、tensor reference を保持する。
- graph は実 m/stride/pointer を capture する。bucket ごとに graph を持ち、未 prewarm shape が hot path で autotune/allocate しないようにする。
- expert 和の FP32 atomic は順序が不定である。top-k=6 なら fixed-order gather の費用は小さいため atomics を避ける。モデル全体の完全な決定性とは別に、同一 local kernel の同一入力については bitwise repeatability を要求できる。
- atomics を残す場合は bitwise 一致を要求せず、反復誤差分布を測って tolerance 内を gate とする。ただし再現性と selfcheck の切り分けは悪化する。

## 6. 発注書へ反映すべき具体的修正

- 「input Hadamard は row ごとに一度、expert 間共有」を削除し、「`(expert,row)` ごとに gate/up 固有の `suh` Hadamard を作る。同じ CTA 内で元 x の load のみ共有」とする。
- Hadamard は 128 要素独立 block、入出力の `1/sqrt(128)` normalization、`suh` 前・`svh` 後という順序を ABI に記す。
- gate/up/down それぞれの `trellis/suh/svh`、計 9 pointer table の形、dtype、stride、alignment、lifetime を明記する。
- `mcg` は host で全 expert の `0xCBAC1FED` marker を検査し、kernel は `cb=1` とする、と明記する。
- K2/K3 の circular decoder を canonical half-bit exact primitive test で gate し、scalar convenience helper を oracle にしない。
- m の correctness 範囲を `1..16` とし、graph/performance bucket と分離する。
- SwiGLU を `silu(min(g, limit)) * clamp(u, -limit, limit)` と明記し、NaN 方針も決める。
- gate/up、down、route 和を FP32、activation のみ clamp/SwiGLU 後 FP16 とする。FP16 accumulation variant は追加最適化扱いにする。
- kernel 入出力 dtype、特に BF16→FP16 と最終 FP32→consumer dtype を 1 launch 内に含めるか caller 契約にする。補助 cast launch を性能値から隠さない。
- routing weight の dtype と scaling factor 1.5 の適用場所を一意にする。
- 「expert 一回」を「各 projection の各 weight tile を、その expert の全 row に使用してから破棄し、DRAM/L2 transaction でも重複を計測する」と定義する。
- persistent cooperative grid `P blocks × 256 threads` を基準案とし、runtime SM 数・occupancy による上限確認を必須にする。
- route plan、gate/up input transform、paired gate/up+activation+down-input transform、tiled down、fixed-order gather の phase/task 構成を基準案にする。
- `R_e=16` でも down B fragment を row panel ごとに再 load/decode しないことを acceptance に入れる。
- 105 us を K2・m=4 の stretch goal に限定し、K3 と m=16 は実測 U と cold read bandwidth から求めた帯域下限倍率で判定する。
- acceptance report に logical unique bytes、実 DRAM bytes、cold/warm、U histogram、全 8 rank latency、K を必須項目として加える。
- parity/selfcheck/serving の追加 gate、sanitizer、NaN-poison graph replay、固定順 determinism を上記どおり追加する。
- shared expert aux stream と scratch/lock の同時利用方針を明文化し、未対応 overlap は明示 error または fallback にする。

## 参照

- `orders/F1-ref/PLAN.md`
- `orders/F1-ref/exl3_moe.cu`, `exl3_moe.cuh`, `exl3_moe_kernel.cuh`
- `orders/F1-ref/exl3_dequant.cuh`, `exl3_dq.cuh`, `exl3_gemm_kernel.cuh`
- `orders/F1-ref/p2b_moe.cu`, `p2b_moe.cuh`
- `orders/F1-ref/vllm_exl3_exl3.py`, `parity-native.py`
- `exllamav3-src/exllamav3/ext/quant/exl3_dq.cuh`, `codebook.cuh`, `hadamard_inner.cuh`, `exl3_gemv_kernel.cuh`, `exl3_gemm_inner.cuh`
- NVIDIA RTX PRO 2000 Blackwell: <https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-2000/>
- NVIDIA CUDA Runtime API, cooperative launch: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html>
- NVIDIA CUDA Programming Guide, CUDA Graphs: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
