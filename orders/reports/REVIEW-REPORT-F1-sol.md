# F1 納品報告の検分

## 結論

**採決は「incumbent を維持し、F1 は性能不合格で閉じる」**とする。`lna` は正しさの実験成果・次回の比較対象として残してよいが、本番既定にはしない。運用経路は明示的に exllamav3 `exl3_moe` を選ぶ。

机上 parity の実測結果は価値がある。しかし、速度は K2/m=4 cold 233.4 us、K3/m=4 cold 268.7 us、K2/m=16 cold 722.3 us で、発注の `1.25 * T_floor` を大きく外す（`f1/history/run-34.log:11-23`）。さらに sanitizer は未通過であり、plugin 全体の 1-launch 性も未確認なので、これは「正しさの土台ができた」であって release 合格ではない。

職人の診断は半分正しい。**全 grid の phase barrier と expert-wave 構造は確かに悪い。だが「incumbent は常駐制約を受けないから速い」「61 KB smem が単独の真因」「N=256 なら 2–3 CTA/SM」という説明はコードとログに一致しない。** incumbent も custom group barrier のため全 grid を同時常駐させる設計であり、32 CTA 程度に制限している。違いは、4 組前後の 8-CTA expert team が ticket を取り続け、expert ごとに gate/up から down まで完走するため、global phase barrier と 1-CTA/expert の薄い圧力を避けている点である。

したがって A 単独は「incumbent を別 ABI で作り直す」域を出ず、B 単独は既に反証されている。ケンの「帯域天井まで圧を掛ける」を狙うだけの余地があるのは C である。ただし再発注するなら、**C の small-R 専用 inner を、A の ticket scheduler / expert-local pipeline の中に最初から入れる**。順番は `A -> B -> C` ではなく **`C+A` を一つの設計として作り、resource 実測後に B を織り込む**。これは F1 の小改修ではなく、新しい F2 発注にすべきである。

ESTIMATE（cold、同一 harness、まだ profiler 裏付けなし）は次の通り。

| 案 | K2 m=4 | K3 m=4 | K2 m=16 | 判断 |
|---|---:|---:|---:|---|
| A だけ（canonical inner のまま） | 190–220 us | 225–265 us | 560–650 us | incumbent 近傍まで。F1 の門には届かない |
| C+A（small-R inner、十分な load concurrency） | 100–140 us | 145–190 us | 300–380 us | 唯一、帯域床へ近づく可能性あり |

A の見積りは、現行 m=4 の非-GEMM phase が 8% 未満という報告と、23/34 CTA の利用不足を直す上限から置いた。C+A の下限は測定済み `T_floor`（K2/m=4 約 69–71 us、K3/m=4 102.7 us、K2/m=16 249.8 us）に、decode、Hadamard、reduction、tail を足した工学見積りであり、約束値ではない。K2/m=4 の 105 us はなお stretch である。

## 診断の裏取り

### 1. delivered kernel の全体 barrier と 34-block cap

これは確認できた。

- kernel は `cg::this_grid()` を取り、route plan 後、input transform 後、gate/up の**各 wave 後**、activation/down-input の各段、down の**各 wave 後**、down-Hadamard 後、gather 後に `grid.sync()` する（`csrc/lna_moe_decode.cu:305-335`, `355-382`, `404-471`, `492-493`）。
- dynamic smem は gate/up と down の大きい方で kernel 全体に割り当てられる。現在の down `N=512, stages=4` が K2 53,248 B、K3 61,440 B になる式もコード通りである（`csrc/lna_moe_decode.cu:519-531`）。
- host はその最大 smem で occupancy を問い合わせ、`sms * resident` 個だけを cooperative launch する（`csrc/lna_moe_decode.cu:539-573`, `621-624`）。最終ログは 34 SM、resident=1、grid `1 x 1 x 34` を記録している（`f1/history/run-34.log:2`）。
- m=16/U=83 では gate と down の双方が 34, 34, 15 expert の 3 wave となり、そのたび全 grid が待つ。これは tail と phase skew を作る。

ただし m=4/U=23 では、GEMM 本体の `if (g < *group_count)` に入るのは 23 block だけであり、11 block は barrier 待ちである（`csrc/lna_moe_decode.cu:362-380`, `439-452`）。従って「34 CTA が weight load を飛ばす」という報告の計算は過大である。

### 2. incumbent との本当の差

「incumbent に grid-wide barrier がない」は正しいが、「常駐制約に縛られない」は誤りである。

- incumbent は `group_size = gridDim.x`、`num_groups = gridDim.z` とし、expert team ごとの barrier を gate/up input、GEMM、activation/down-input、down GEMM、ticket 更新で使う（`../vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh:21-45`, `111`, `162`, `185`, `234`, `261-266`）。
- host 側も「group barrier のため全 block が co-resident でなければならない」と明記し、`groups * width <= num_sms` になる grid を作る（`../vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe.cu:203-222`）。launch は cooperative API ではないが、標準 launch の grid 自体を resident 枠内に収めている（同 `290-297`）。
- production plugin は concurrency を `num_sms / 8` から作り、hot path では active 数を host sync せず `-1` を渡す（`../vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe.cu:14-18`; `src/vllm_exl3/exl3.py:871-910`）。34 SM なら基本は 4 group x 8 CTA = 32 CTA である。
- 各 group は初期 ticket の expert を処理し、完了時に `atomicAdd` で次の expert を取り、最後の group が scheduler を reset する（`exl3_moe_kernel.cuh:47-69`, `261-282`）。また一つの expert を input transform -> gate/up -> activation/down-input -> down -> scatter まで完走する（同 `82-259`）。異なる group が異なる phase に居られる。

従って incumbent の利点は「34 より多い block を飛ばせる」ことではない。**約 32 resident CTA を 8-way に一 expert へ当て、4 team を ticket で継続稼働させ、expert 間の global phase barrier を持たないこと**である。delivered kernel は m=4 の主要 GEMM で 23 CTA しか働かず、1 CTA が一 expert 全体を直列 sweep する。この差のほうがコードから直接説明できる。

### 3. bandwidth 診断の強さ

K2/m=4 の logical unique bytes / total time = 約 80 GB/s、cold 233 us 対 warm 198 us なので、**DRAM throughput ceiling に張り付いていない**ことは言える。しかし、それだけでは long-scoreboard、trellis decode の integer dependency、MMA、register pressure、barrier のどれが一次原因かは決められない。cold と warm にも約 15% の差があり、memory latency の寄与は残る。要求されていた actual DRAM bytes、achieved bandwidth、L2 hit、stall、tensor/ALU、register/spill の profiler 値が報告にない。

Little の法則の説明も仮説である。canonical inner は B を `cp.async` で shared memory に入れ（`exl3_gemm_inner.cuh:227-265`）、`cp_async_wait<SH_STAGES-2>` で進める（同 `630-655`）。LNA の gate/up K2 は stage 当たり B=2 KB なので「待機点で約 2 stage」は妥当な概算だが、m=4 で実際に GEMM を行うのは 34 でなく 23 CTA である。同じ数え方なら 136 KB でなく約 92 KB である。一方、prefill 時の transient group、A load、L2 transaction、同時 outstanding request 数はこの算数には入っていない。800 ns も未測定仮定である。**in-flight 不足は有力仮説だが、確定診断ではない。**

### 4. smem と canonical inner

K3/down の 61,440 B が current variant の resident=1 に寄与するのは事実だが、smem だけが制約とは言えない。

- report は down N=256 で 32,768 B、resident 2–3 と書くが、該当 sweep の occupancy 出力は K2 `[34,1,1,34,28672]`、K3 `[34,1,1,34,32768]`、つまり双方 resident=1 である（`f1/history/run-32.log:18-33`）。512 threads と register 使用も occupancy を止めている。N=256 の latency も K2/m=4 240.0 us で改善しなかった。
- inner の shared 使用量は A/B の stage buffer と FP32 C reduction bufferで決まり、C は tile N に比例する（`exl3_gemm_inner.cuh:36-48`, `67-71`）。単に B tile を小さくしても、thread/register/C-buffer の全てを下げなければ 2 CTA/SM にはならない。
- LNA は scheduling だけを変え、計算本体は canonical `exl3_gemm_kernel_inner` をそのまま呼ぶ（`csrc/lna_moe_decode.cu:264-271`）。inner は `TILESIZE_M == 16` 固定で、A load だけを `m < size_m` で predicate する（`exl3_gemm_inner.cuh:56-65`, `114-125`）一方、MMA loop 自体は 16-row fragment で回る（同 `637-648`）。`size_m <= 8` の特別扱いは reduction 書き戻しだけである（同 `375-398`）。m=4 の U=23 なら平均 `R_e` は約 1.04、m=16/U=83 でも約 1.16 なので、small-R 専用 inner には大きな余地がある。
- `M_CAP` は別 symbol を作るだけで、kernel 内では static assert 以外に使われていない（`csrc/lna_moe_decode.cu:281-305`）。4/8/16 bucket が現状 resource や inner shape を変えていない。

### 5. 比較と検証の留保

233 us cold と 229 us incumbent の比較は同一条件と証明されていない。LNA bench は毎回 96 MiB を zero-fill して cold 化する（`f1/bench_lna.py:26-56`）一方、229 us はこの run で同じ route/U、cold/warm、output dtype を測り直した値ではない。また m=16 の表で 229 us を分母にする根拠はない。m=16 incumbent を同じ harness で測るまで「0.32x」は比較値として採用しない。これは LNA の `T_floor` 不合格を救わないが、次回の改善率判定には必須である。

## 次の一手

本 F1 に追加工数は入れない。`VLLM_EXL3_MOE_KERNEL=lna` は experimental opt-in のままとし、実運用では incumbent を継続する。

再発注する場合の最小構成は次である。

1. **外側は A**: incumbent と同じ resident expert-team + dynamic ticket を使い、各 expert を gate/up input から down まで end-to-end に流す。全 phase の `grid.sync()` は撤去する。ただし grid を resident 枠より大きくすること自体を目的にしない。
2. **内側は C**: `R_e=1`, `2`, `3–4`, `5–8`, `9–16` の実 histogram に合わせた small-R variant を用意する。特に R=1/2 で M16 tensor tile と 16xN FP32 reduction bufferを払い、1 CTA が十分な独立 B request を持つ形にする。weight tile は同一 expert の全 row に使ってから捨てる。
3. **fixed-order gather は一 launch のまま可能**: 各 group の expert 出力を pair scratch に置き、incumbent の retired-group counter と acquire/release を流用する。最後に retire した group だけが全 row を top-k 順で gather する。これなら危険な non-resident grid barrier も二本目の launch も要らない。実装が複雑なら、2-launch compute+gather を graph 上で先に比較し、launch 数より総 latency を優先してよい。
4. **B は resource 目標として後から入れる**: N=256 化そのものではなく、実測で register/spill と dynamic smem を下げ、2 CTA/SM または同等の outstanding-request 数を得ることを意味する。occupancy が変わらなければ B は不成立である。

A-only spike をどうしても行うなら一回に限り、K2/m=4 cold <= 190 us かつ profiler で weight interval >= 150 GB/s を kill gate とする。届かなければ C へ進まず終了する。A のまま 105 us を追うのは見込みが薄い。

## 見落とし

### smem reduction で 2 CTA/SM

着眼は正しいが、current N=256 variant は実測 resident=1 である。必要なのは down N だけでなく、512-thread CTA、fragment registers、静的な 16xN FP32 C bufferを同時に小さくすること。`M_CAP=4` が未使用なのも機会で、m=4 variant だけ C scratch と row state を小さくできる。build 時には ptxas の registers/thread、spill load/store、occupancy 制約要因を必ず記録する。

### deeper cp.async per CTA

単純な `SH_STAGES` 増加は既に遅くなっており、canonical inner は shared stage と register fragment を一緒に増やすため逆効果になりやすい。次に試すなら、stage 数の再 sweep ではなく、producer warp と decode/MMA consumer の分離、複数 independent B stream、または対応が確認できる場合の TMA/bulk copy を small-R inner 内で比較する。合否は long-scoreboard と outstanding request の実測で決める。

### down と gate/up の overlap

同一 `(expert,row)` の down は activation 完成前には始められないが、**expert 間では overlap できる**。incumbent のように各 ticket group が end-to-end に進めば、group A の down と group B の gate/up が自然に重なる。さらに 8-CTA team の半分で gate、半分で up を同時実行し、group barrier 後に全 CTA を down へ回す案は、職人が退けた「gate/up を global task として二 wave にする」案とは別物である。

### m=16 専用設計

m=16 でも uniform route は 96 pair / 83 expert、平均約 1.16 row/expert であり、「m=16 だから M16 GEMM」が合っていない。expert を `R_e` bucket に分け、R<=4 は small-R kernel、hot expert/R>8 だけ canonical M16 を使う hybrid が自然である。task は expert 一個ではなく `(expert, projection, N-tile)` まで分け、dependency counter で down-ready を発行すれば 83 expert の末尾でも tile-level parallelism を保てる。K2/m=16 の最終目標は床 249.8 us に対し約 300–312 us、初回 C+A の見積りは 300–380 us とする。

### incumbent ticket + repeated-expert dedup

incumbent は `expert_count` と expert 順の token spanを使い、一 expert の全 row を一回の ticket で処理する（`exl3_moe_kernel.cuh:52-80`, `87-110`）。従って**row 間で同じ expert が再登場する場合の weight reuse は既にある**。LNA の追加利益は同じ row の top-k 内に同一 expert が重複した時に `(row,expert)` 計算を一回にする部分だけである（`csrc/lna_moe_decode.cu:185-231`）。通常の top-k は一 row 内で distinct なので、本番速度への寄与はほぼない。これは正しさ・adversarial robustness として残すが、帯域改善案には数えない。

### 一 launch の境界

plugin path は `map_topk_to_local`、int32 化、FP32/contiguous 化、`empty_like` を kernel 呼び出し前に行う（`src/vllm_exl3/exl3.py:544-595`）。expert map が不要な TP8 でも `torch.where` と dtype conversion が補助 CUDA kernel になり得る。kernel-alone 233 us はこの費用を含まない。ABI を実際の vLLM dtype に合わせるか、upstream の既存 tensor をそのまま受け、launch trace で層あたり一 launch を証明する必要がある。

## 門

次の iteration/F2 は以下を**順番に**通す。前段不合格なら serving 席へ進めない。

1. **同条件 baseline**
   - candidate と incumbent を同じ process、同じ x/ids/weights/U/`R_e`、同じ output dtype、同じ clock/thermal 条件で交互に測る。
   - K2/K3 x m=1/4/8/16 の cold p50/p95 と warm p50/p95 を各 3 run。m ごとに incumbent 値を持ち、229 us を全 m に流用しない。
   - kernel-only と plugin/layer end-to-end を分ける。graph ON/OFF、launch trace、NCCL bytes も記録する。

2. **原因確定・pressure gate**
   - Nsight Compute 相当で actual DRAM read bytes/throughput、L2 sector/hit、outstanding request、long-scoreboard、barrier/membar、tensor/ALU/issue utilization、achieved occupancy、register/thread、dynamic smem、spill を採る。
   - logical unique bytes/time と actual DRAM bytes/time を別記する。K2/m=4 の trellis-dominant intervalで少なくとも 200 GB/s、またはその未達を説明できる compute ceilingを示す。単なる SH_STAGES 算数は証拠にしない。
   - 全 resident CTA の active-time と expert/tile tail を timeline で示す。2 CTA/SM を主張する variant は occupancy API の実値 2 以上を必須にする。

3. **性能 gate**
   - primary は cold p50 `<= 1.25 * T_floor`。現測定を使う暫定値は K2/m=4 <= 86–88 us、K3/m=4 <= 128 us、K2/m=16 <= 312 us、K3/m=16 <= 463 us。帯域を再測定したら floor も同時更新する。
   - 同条件 incumbent に対し全 primary bucket で非劣化、主要 K2/m=4 は最低 1.15x の end-to-end 改善を要求する。warm 値で cold 不合格を代用しない。
   - p95/p50 <= 1.10、3 run と全 8 rank の最遅値を報告する。K、U、`R_e` histogram を各値に添える。

4. **scheduler / resource correctness**
   - ticket の初期化、wrap/reset、0 active、1 group、全 group、expert 数が group 数の前後、最後の group による gather、graph の連続 replay、複数 stream/graph instance を stress する。
   - K2/K3 と全 group width/resource variant で parity を通す。`prepared[2][3]` は device index を key に含める（現状は process-global: `csrc/lna_moe_decode.cu:497-517`）。shared-expert aux stream は明示同期、明示 fallback、または明示 error のいずれかにする。
   - compute-sanitizer の memcheck/racecheck/synccheck/initcheck を全て release gate とする。

5. **numerical / graph gate**
   - K2/K3、m=1..16 の**全整数**、production K3 layer 13/22/28、boundary/hot/Zipf/within-row duplicate/R_e=16、clamp、zero/cancel、FP32 rel <= 2e-3、fixed-order bitwise repeatability を通す。現テストの m 標本は `(1,2,3,4,5,7,8,15,16)` に限られる（`f1/parity_native.py:28-35`）。
   - 現テストの 110 synthetic cases は input と weight profile の全直積ではなく、`(m, route)` ごとに profile を rotate した標本である（`f1/parity_native.py:367-395`）。次回は少なくとも pairwise coverage を明示し、scheduler/resource variant を横断する。
   - graph capture failure または replay rel failure を必ず nonzero exit にする。現 `graph_test` は capture failure で `continue` し、`ok=False` でも print だけなので fail-open である（`f1/bench_lna.py:60-97`）。scratch と out を NaN poison した graph replay、routing scaling 1.5 の zero/once/twice sentinel も追加する。

6. **integration / serving gate**
   - layer の launch trace に hidden cast/map/sort/zero/copy kernel が増えていないこと。graph capture 前に allocation、pointer table、variant、scratch を確定する。
   - graph ON/OFF parity、spec OFF/DSpark3、1/4 stream、作文 7 本 finish=stop、ppl 6.72 +/- 0.02、受理長、scaling 一回、consumer dtype、NCCL bytes、8 rank thermal steady を従来どおり通す。
   - ここまで通って初めて incumbent から切り替える。途中で pressure gate または `1.25 * T_floor` を外した場合は、F1 と同じく実験成果を保存して incumbent を維持する。
