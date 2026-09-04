# F2 密 EXL3（K=6）decode launch 融合 設計検分（GLM 監督）

対象: `orders/F2-dense-launch-fusion.md`。参照: `orders/REVIEW-F1-sol.md`（ABI 正典）、`exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh` / `exl3_gemm.cu` / `exl3_gemm_inner.cuh`、`vllm-exl3/src/vllm_exl3/exl3.py`、`prof-graph-0904/profiler_out_0.txt`。実測／推定を【実測】【推定】で書き分ける。本検分は静的読みと 09-04 プロファイルの読み直しのみで、GPU・ビルド・席 :8899 は使っていない。

## 結論

方針（同入力の密線形を 1 launch に束ねる grouped GEMV）は成立する。ただし発注書の利得の導出に会計上の誤りがあり、門にも「計測を先行させる段」が足りない。修正を入れれば着手可。

- **利得の見積り（ESTIMATE 1.5×、ステップ 8%）は「上限」枠なら概ね妥当だが、導出の byte 集合が誤っている**。発注書の床計算（19 MB/層 → 70 µs、`orders/F2-dense-launch-fusion.md:23`）は束ね対象外の線形を含む（「割に合うか」参照）。束ね対象 6 本/層の実 byte は【推定】≈9.4 MB/層で、床は ≈35 µs/層。現状 107.8 µs/層【実測】と比べると上限は発注書の 1.5× より大きい（≈3×）が、このカードの EXL3 系カーネルの達成帯域の前例（80〜90 GB/s）を踏まえ、現実目標は 1.3〜1.5×（ステップ −5〜−8%）に置くべき。
- **grouped GEMV は家のカーネル構造で成立する**。しかも新カーネルを書く前に、既存 `exl3_mgemm`（B/suh/svh ポインタ表・1 入力複数出力 broadcast・行列別幅 `size_n_list`/`c_ptrs`・K=6/mcg はコンパイル済み・Python バインド済み）で計測プロトタイプが組める。発注書がこれに触れていないのが最大の抜け。
- **遅延束ねは graph capture と両立する**（capture 中に Python が一度走って launch 列が焼かれ、replay は kernel のみ → 「グループ計算→全消費」の順序が焼かれる）。ただし成立条件が 3 つある: キャッシュ出力/A_had の capture 前恒久確保、入力同一性（data_ptr）の実機検証、miss したメンバーが自ら全グループを計算する順非依存設計。
- 発注書の層テーブル（`orders/F2-dense-launch-fusion.md:15-21`）と実測 launch 数（6/層）が噛み合っていない。束ね対象は実測上 {wq_a, wkv, compressor.wgate, compressor.wkv, shared.w1, shared.w3} の 6 本/層（43×4 + 41×2 = 254/step と一致）で、テーブルの残り 4 本（wq_b, wo_a, wo_b, w2）は現行グラフでは `exl3_gemm_kernel<6>` を叩いていない。Gate 0 で確定するまで利得見積に含めないこと。

## 割に合うか

**条件付きで割に合う。コストを既存 mgemm プロトタイプに載せられるので最初の一歩は安く、期待利得はステップ −5〜−8%（上限 −8〜−17%）。ただし Kill 基準を先頭に置くこと。**

実測の確認（`prof-graph-0904/profiler_out_0.txt`、rank0、46 step、09-04）:

- `exl3_gemm_kernel<6,…>` 3 形状 合計 11,715 launch / 46 step = **254.7 launch/step**、合計 213.25 ms、CUDA 全体の **24.7%**（`profiler_out_0.txt:7,10,19`）。tile 128: 8,179 回・平均 17.97 µs、tile 256: 3,053 回・16.75 µs、tile 512: 483 回・31.30 µs。
- 254.7 = 43×4 + 41×2 と厳密に一致する。すなわち実測の 6 launch/層は {wq_a, wkv} + compressor 層の {wgate, wkv_c} + {w1, w3}。発注書の「≈255 launch（≈6/層）」（`orders/F2-dense-launch-fusion.md:12`）はこの口径【実測】。
- 1 ステップの密 EXL3 時間 = 213.25 ms / 46 = **4.64 ms**、43 層で割ると **107.8 µs/層**【実測】。routed 31%・NCCL 18% という内訳も MoE 行（250.6 ms→5.82 ms/step）・NCCL 行（147.1 ms→3.20 ms/step）と整合【実測】。

byte 集合の検算【推定】（0.75 B/weight = K=6 trellis 幅 96×2B/256 重み、`orders/F2-dense-launch-fusion.md:13` どおり）:

| 線形（rank 局所形） | byte |
|---|---:|
| wq_a 4096→1024 | 3.15 MB |
| wkv / wgate / wkv_c 4096→512 ×3 | 4.71 MB |
| w1 / w3 4096→256（F1 正典と同じく shared 中間幅も local 256 と仮定。`orders/F1-lna-moe-decode-kernel.md:12`） | 1.58 MB |
| **束ね対象 6 本合計** | **9.44 MB/層** |
| wq_b / wo_a / wo_b / w2（現行グラフでは exl3_gemm<6> に不在） | ≈9.4 MB/層 |

この読みでは、束ね対象 6 本の実効帯域は 400 MB/step ÷ 4.64 ms ≈ **86 GB/s**（ピーク 264〜275 GB/s の 3 割強）【推定】。発注書の「帯域効率 ≈ 6 割」は 3.1 MB の 1 本（≈175 GB/s）だけの話で、0.8 MB クラスは ≈45 GB/s まで落ちる。**束ねの主効果は帯域効率ではなく、小行列の launch 固定費と SM 飢餣の解消**である。発注書がこの点を 6 割効率の文脈で語っているのは誤り。

上限と現実目標【推定】:

- 床 = 9.44 MB ÷ 270 GB/s = **35 µs/層** → セグメント上限 ≈3.1×、ステップ上限 −17%。shared 中間幅が local 2048 だった場合（テーブルの字面通り）は 6 本で 20.4 MB/層・床 75 µs → 上限 1.44×・ステップ −8.9% で、発注書の 1.5×/8% に一致する。つまり **1.5× は shared 幅の読みが「2048 local」のときだけ成り立つ上限**であり、その場合は「0.8〜3.1 MB/launch」の記述と矛盾する（0.8 MB は local 256 のときしか現れない）。
- 一方、達成帯域の実績は慎重に: F1 の MoE カーネルは同カードで 80〜90 GB/s 実効（`orders/F1-lna-moe-decode-kernel.md:23`）、現 dense も平均 ≈86 GB/s【推定】。連結 1 launch で 150〜200 GB/s に乗れば 47〜63 µs/層 → **1.7〜2.3×、ステップ −7〜−12%**。安全側の計画値として **1.3×（ステップ −5%）を PASS、1.5×/−8% を stretch**、3× は到達不能な上限として扱わない、を推奨。
- NCCL「粒揃え」への効果（`orders/F2-dense-launch-fusion.md:24`）は仮説。8 rank が対称に速くなる分は効くが、待ち時間の短縮は最遅 rank 従属。門で計測項目にする（下記）。

## 設計

### (2) grouped GEMV の成立性と CTA 割り

成立する。exllamav3 の `exl3_gemm_kernel` は m≤16 を TILESIZE_M=16 の 1 タイルで扱い（`exl3_gemm_inner.cuh:59` の static_assert）、B タイルは一度 shared/registers に載せて全行に適用する構造なので「重みは一度だけ読む」は既存の inner をそのまま使えば自動的に成立する。K=6 復号（dq_dispatch）も無変更。

**S0（第一歩・新カーネルなし）: 既存 `exl3_mgemm` をそのまま使う。** 同一入力の複数行列はちょうどこのために存在する:

- B/suh/svh は int64 ポインタ表、`a_batches == 1` なら A[0] を各行列へ broadcast（`exl3_gemm.cu:369-384` の設計説明、kernel 側 `exl3_gemm_kernel.cuh:175`）。
- 行列別の出力幅と出力先は `size_n_list` + `c_ptrs`（`exl3_gemm.cu:466-476`、kernel 側 `exl3_gemm_kernel.cuh:191-206`）。制約 `num_tokens == 1 && min_index < 0 && !weights`（`exl3_gemm.cu:470-472`）は今回の使い方なら満たす。
- CTA 割りは grid = dim3(num_sms, 1, concurrency)、z = 行列、`concurrency = MIN(total_sms/num_sms, bszm)`（`exl3_gemm.cu:617-635`）。sm_120（`__CUDA_ARCH__ > 890`）では per-z の sense 反転バリア `group_barrier` を使うので行列間の直列化はない（`exl3_gemm_kernel.cuh:96-97,184-188,219-223`、実装 `ptx.cuh:319-345`）。
  - グループ (a)（4 行列、k=4096、max_n=1024）: shape 3（tile 32×256、`exl3_kernel_map.cuh:54-60`）で slices=512 → num_sms=8 × z=4 = 32 CTA ≤ 34 SM。
  - グループ (b)（2 行列、n=256×2）: num_sms=17 × z=2 = 34 CTA。
- 入力 Hadamard は行列ごとの suh で A_had スラブを 4 枚/2 枚書く（`exl3_gemm_kernel.cuh:167-181`）。正典どおり suh は行列別（`orders/REVIEW-F1-sol.md` 1.4 節）で、束ねてもこの仕事は減らない。A の再読は m×4096×2B×行列数 ≈ 128 KB×4（m=16）で重み 9.4 MB に対し無視できる【推定】。
- A_had は `bszm × m × k` 要素必須（`exl3_gemm.cu:478-481`。不足は黙って OOB と警告にある）。locks は z×max_n/128 個（`exl3_gemm_kernel.cuh:208`）で DevCtx の 1M ints + barrier 領域（`exl3_devctx.cuh:6-8`）に対し十分。
- K=6/mcg の mgemm インスタンスはコンパイル済み（`comp_units/exl3_comp_unit_6_cb1.cu:11`）、Python バインドもある（`bindings.cpp:161`）。plugin 側は MoE の pointer table 構築（`vllm-exl3/src/vllm_exl3/exl3.py:288-311`）と同じ型で済む。

**S1（第二歩・条件付き）: `lna_dense_group.cu` は連結 slice 空間の 1D work queue で。** S0 の弱点は z 静止割りの不均衡（(a) では wq_a の slices が他の 2 倍で、終端が wq_a の z を待つ）と、行列ごとの barrier 待ち。S1 では:

- grid = P blocks（P = SM 数、実測で取得。F1 正典 2.1 節と同じ作法）、blockDim は autotune。
- Phase 0: 全行列の入力 Hadamard（G 枚の A_had スラブ、warp stride）→ **1 回だけ** grid.sync。
- Phase 1: 全 (行列 j, k-tile, n-tile) を連結した T slice 空間を CTA b が [T·b/P, T·(b+1)/P)受け持ち。slice→(j,k,n) は 8 entry 以下の prefix 表で解決し、`exl3_gemm_kernel_inner<6,…>` に B_j/A_j/C_j/n_j/locks_j を渡す。locks_j = 先頭 + Σ_{i<j} n_i/128 で行列間非重複。
- m≤16 は 1 タイル、m>16 は既存どおり 16 行ずつの外ループ + sync（`exl3_gemm_kernel.cuh:40-49` 相当）。
- 導入条件は「S0 の実測で (a) の tail 不均衡・barrier 待ちが dense セグメントの数 % を占めたとき」のみ。最初から S1 を書かない。

### (3) 遅延束ねと graph capture

**両立する。** vLLM の capture 中は Python（= plugin の apply）が一度だけ走り、その時の launch 列が graph に焼かれる。replay では Python は走らず kernel のみが同じ順序で再実行される。よって「最初のメンバー apply でグループ全体を計算し、残りのメンバーはキャッシュ（= 焼き込まれた出力バッファの view）を返す」設計は、capture と replay の両方で「全出力の計算 → それぞれの消費」という順序を保つ限り正しい。

ただし以下を条件とする（詳細は落とし穴）: (i) キャッシュ出力・A_had は capture 前に恒久確保し graph のメモリプールに依存しない、(ii) 入力同一性は capture 時・eager 時の両方で data_ptr/shape/dtype で検証し、不一致は standalone fallback、(iii) キャッシュミスしたメンバーが自ら全グループを計算する（呼び出し順に依存しない）、(iv) shared experts の補助ストリームは無効前提（現運用 `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`、`orders/F1-lna-moe-decode-kernel.md:49`）。

なおグループ (b) は w1/w3 が同一モジュールの 2 shard なら（`vllm-exl3/src/vllm_exl3/exl3.py:1486-1509` の shard ループ）、apply 1 回の内側で束ねられるためキャッシュ機構不要。グループ (a) だけが vLLM 的に 2 モジュール（`attn.fused_wqa_wkv` と `compressor.fused_wkv_wgate`、`orders/T4-dense-partial-bake.md:29-30`）にまたがり、遅延キャッシュが必要。**(b) から先に出し、(a) を後に出す**ことで危険域を段階的に増やせる。w13 が 1 モジュールか 2 モジュールかは Gate 0 で確定させる。

## 落とし穴

1. **入力同一性の前提**。`compressor.fused_wkv_wgate` に渡る h が `attn.fused_wqa_wkv` と同一 tensor（同一 data_ptr）とは限らない（norm の別インスタンス、view、copy の可能性）。検証なしだと「別々の入力を同じと見なして束ねる」最悪のバグになる。→ data_ptr・shape・dtype・contiguous の一致を capture 時・eager 時の両方で検証。不一致は fallback + 1 回だけ loud log。
2. **キャッシュの寿命とアドレス**。`LinearEXL3.forward` は現状毎回 `run_alloc` で出力を確保し（`exllamav3-src/exllamav3/modules/quant/exl3.py:131-138`、`libtorch/linear.cpp:56-71`）、capture 中の確保は graph プールに載る。プール tensor は参照を握らないと capture 続行中に再利用され得る。→ 出力・A_had とも capture 前に恒久確保（MoE の `_FUSED_TEMP_CACHE` 型、`vllm-exl3/src/vllm_exl3/exl3.py:57-59` と同じ作法。F1 正典の「stream/graph instance ごとの scratch 分離」`orders/REVIEW-F1-sol.md` 5 節も同じ趣旨）。
3. **stale キャッシュ**。replay 間で入力バッファの中身は変わる。毎回の全グループ再計算で消えるはずだが、呼び出し順の変化（モデル実装更新、spec/MTP、overlay の部分呼び出し）で「再計算前に消費」が崩れない保証が要る。→ 任意メンバーの miss が自ら全グループ計算する設計 + 「入れ替え入力で 1000 回 replay し出力が入力に追従する」検査（門 1.5）。
4. **autotune と capture**。mgemm は capture 中 autotune を黙ってスキップして静的選択に落ちる（`exl3_gemm.cu:581-583`。gemm 側は警告付き、`exl3_gemm.cu:59-68,262-268`）。hash は bszm を混ぜるので (a)(b) は別々に tune される（`exl3_gemm.cu:99-121`）。→ `LNA_EXL3_PREWARM_ROWS` 相当（`vllm-exl3/src/vllm_exl3/exl3.py:1414-1429`）をグループ呼び出しに拡張し、全 m bucket × 両グループを capture 前に prewarm。ディスク cache（`exl3-tune-cache/coop_autotune_v1.bin`、`coop_autotune.cu:626-640` 経路）は capture 中も使える。
5. **m bucket の取りこぼし**。発注書は「m = 4〜16」と言うが、実測プロファイルは 64 tok（`orders/F2-dense-launch-fusion.md:11` の数字自体が 64 tok）。correctness は capture bucket 全て（64 を含む）で、門 1 の m 集合 {1,2,4,8,16} に 64（と 32 があるならそれ）を足す。m>16 は 16 行ずつの外ループで sync が増えるが正しくは動く。
6. **locks / barrier の共有**。k-split reduction の locks と per-z barrier カウンタは DevCtx の共有バッファ（`exl3_gemm_kernel.cuh:96-97`、`exl3_devctx.cuh:6-8`）。別 stream・別 graph instance が同じカーネルを並行 replay すると壊れる。現運用（単一 decode stream・層は逐次）では問題ないが、補助ストリームや 2 stream 同時 replay を許す前に明示 error にする（F1 正典 5 節と同じ）。
7. **n の整除**。mgemm の shape 互換は max_n で判定され、各メンバーの n_j が tile_n で割り切れる保証は呼び出し側にある（512/1024/256 は tile 128/256 で全て OK。tile 512 は n=256 を静的に落とすので shape 選択の確認要）。新バインドでも TORCH_CHECK で各メンバー n_j % tile_n == 0 を課す。
8. **bf16 shard 混在モジュール**。`bf16_shards`（`vllm-exl3/src/vllm_exl3/exl3.py:1073-1084,1489-1497`）を持つモジュールは束ね対象から除外。codebook marker と K はグループ内で一律（mcg/K=6）を登録時に検査（marker 検査の既存実装 `vllm-exl3/src/vllm_exl3/exl3.py:1359-1387` を流用）。
9. **wo_a / wq_b / wo_b / w2 の所在**。現行グラフでこれらが `exl3_gemm<6>` にいない事実（43×4+41×2=254 の一致）は、発注書テーブルの「すべて EXL3 K=6」と矛盾する。吸収 MLA（attention カーネル内）、bf16 shard、他カーネルのいずれか。Gate 0 で実路径を確定するまで束ね対象に加えない。
10. **出力 dtype 境界**。現行 apply は fp32 出力 → x.dtype へ cast、複数 shard は cat（`vllm-exl3/src/vllm_exl3/exl3.py:1486-1529`）。発注書 5 項のとおり境界は動かさない。cat を消すなら出力バッファを隣接配置して view 返しにする（最適化）。x→fp16 cast も 4 回→1 回に減る（`vllm-exl3/src/vllm_exl3/exl3.py:1479`）。これらの削減分は「launch 数」の実績には出ないが step 時間に効くので計測に含める。

## 門

発注書の三段（`orders/F2-dense-launch-fusion.md:37-39`）を土台に、手前に 2 段追加し、既存段を補強する。

**Gate 0（新設・コード前に・計測のみ）: 帳尻合わせ。**

- `prof-shapes.py` で現行 trace を解析し、密線形ごとの実路径（exl3_gemm<6> / bf16 / 吸収 / 他カーネル）を確定。wq_b・wo_a・wo_b・w2 の所在を宣言する。
- pack の config.json（`non_routed_exl3`）と実 tensor から、全密線形の rank 局所形・K・codebook・byte の表を作る。特に shared w1/w3 の local n（256 か 2048 か）を実測で確定し、本検分の 9.4 MB/層【推定】を置き換える。
- w13 が 1 モジュール（2 shard）か 2 モジュールか、capture bucket の m 全集合（64 を含むか）を確定。
- 成果物は REPORT-F2 の冒頭に表として載せる。ここが揃わないと速度門の数字が無意味。

**Gate 0.5（新設）: mgemm プロトタイプ実測（新カーネルなし）。**

- plugin から既存 `exl3_mgemm` で (a)(b) を束ね、launch 数・dense セグメント時間・step 時間を m=4 と 16 で 3 回平均。
- **Kill 基準**: dense セグメント < 1.1× なら `lna_dense_group.cu` の着手禁止。所見を REPORT-F2 に残して終了。この場合でも (b) の same-call 束ねと cast/cat 削減だけは残せる。

**Gate 1（机 parity・発注書どおり+補強）。**

- m ∈ {1,2,4,8,16,64}（capture bucket 全て）、実 tensor は層 5・40 に加えて **非 compress 層（43 中 2）を 1 層**（グループ (a) が 2 本縮退する系）と合成 tensor。per-row rel ≤ 2e-3 + max abs + finite 一致（零近傍は abs 閾値）。
- **呼び出し順入れ替え**（wkv を wq_a より先に呼ぶ、eager）で fallback が正しく出ること。
- 同一入力の二度走りで **bitwise 一致**（固定順・atomics なし構造なので要求できる）。
- `compute-sanitizer` memcheck に racecheck・synccheck・initcheck を追加（locks と per-z barrier を使うので racecheck は必須）。

**Gate 1.5（新設・graph 特化）。**

- 出力・キャッシュを NaN poison し、**入れ替え入力で 1000 回以上 replay**。出力が最後の入力に追従すること（stale 検出）。bucket を交互に replay しても混線しないこと。
- capture 前後の allocator 統計で per-replay の追加確保が無いこと。launch trace で 6→2〜3/層（≈86/step）になり、層の外に `.half()`/`.to()`/cat/zeroing が増えていないこと。
- 入力同一性検証を debug フラグで常時 ON にして 1 ステップ走らせ、(a) の 2 モジュールの h が同一 data_ptr であることを実機確認。

**Gate 2/3（席内・速度・発注書どおり+補強）。**

- 席内: graph ON/OFF の貪欲突き合わせ、作文 7 本 finish=stop、ppl 6.72 ± 0.02、受理長 ≈ 2.0/1.8/2.2 は発注書どおり。
- `LNA_EXL3_DENSE_GROUP=1/0` の **A/B を同一プロセスで**。eager（graph OFF）は「落ちない」に加え、同一入力で両モードの出力一致（parity）。
- **8 rank 全部**の P50/P95 と最遅 rank の差を 3 回以上。NCCL 待ち仮説はここで判定する（発注書本文からは「効果」ではなく「計測項目」に格下げ）。
- 速度の合格値を明文化: launch ≈86/step（41×2 + 2×2）、dense セグメント ≤ 3.57 ms（≥1.3×）を **PASS**、≤ 3.09 ms（1.5×）を **stretch**、ステップ −5% を PASS / −8% を stretch。1.5×/8% は上限扱い（発注書の ESTIMATE 表記も上限であることを明記）。
- prof-shapes.py の再計測で束ね後の launch 数・時間を REPORT-F2 に載せる（前: 254.7 launch・4.64 ms・24.7%）。

## 発注書への修正

1. **層テーブルの形表記を rank 局所形に統一**（`orders/F2-dense-launch-fusion.md:15-21`）。「保存形」といいつつ wq_b 32768・shared 2048 は global 口径の疑い。F1 正典（routed 中間幅 local 256、`orders/F1-lna-moe-decode-kernel.md:12`）と突き合わせ、shared w1/w3 が local 256 なら 0.75 B/weight で 0.79 MB/本（= 発注書の「0.8 MB」）となり、逆に「19 MB/層」は 6 本分として矛盾する。表を local 形で書き直し、byte 集合を「束ね対象 6 本 ≈9.4 MB/層」と「その他 4 本」に分離すること。
2. **「密線形 790 本はすべて EXL3 K=6」と実測 254.7 launch/step の矛盾を解消**。プロファイル（`prof-graph-0904/profiler_out_0.txt:7,10,19`）では 6 本/層のみが `exl3_gemm_kernel<6>`。残る 4 本の現路径を Gate 0 で確定するまで、利得見積・束ね設計から外す。
3. **ESTIMATE の導出を修正**（`orders/F2-dense-launch-fusion.md:23`）。床は「束ね対象 byte ÷ 実測 cold read 264〜275 GB/s」。local 256 読みなら ≈35 µs/層（上限 ≈3×）、local 2048 読みなら ≈75 µs/層（上限 ≈1.44×）。「帯域効率 ≈ 6 割」は最大行列 1 本だけの話で、小行列は ≈45 GB/s —— 束ねの主効果は小行列の launch 固定費・SM 飢餣解消であることを明記。目標は PASS 1.3×/−5%、stretch 1.5×/−8%（1.5×/8% は上限）。
4. **既存 `exl3_mgemm` の活用を基準案に追加**。S0（mgemm 直呼び・新カーネルなし）を必須の第一歩とし、`lna_dense_group.cu`（S1: 連結 slice 空間・1 回の grid.sync・行列別 locks prefix）は S0 の実測で z 割りの不均衡・barrier 待ちが効いたときの第二歩とする。Gate 0.5 と Kill 基準を発注書に書く。
5. **遅延束ねの成立条件を発注書に明記**（`orders/F2-dense-launch-fusion.md:34`）: (i) グループ (a) は vLLM 側で 2 モジュール（`attn.fused_wqa_wkv`・`compressor.fused_wkv_wgate`、`orders/T4-dense-partial-bake.md:29-30`）にまたがる、(ii) 入力同一性（data_ptr・shape・dtype・contiguous）検証と fallback、(iii) miss したメンバーが自ら全グループ計算、(iv) 出力・A_had の capture 前恒久確保、(v) 補助ストリーム有効時は明示 error。**(b) → (a) の順で段階投入**も明記（(b) は same-call 束ねの可能性があり危険が小さい）。
6. **prewarm をグループ呼び出しに拡張**。mgemm は capture 中 autotune を黙ってスキップする（`exl3_gemm.cu:581-583`）。`LNA_EXL3_PREWARM_ROWS` 相当を全 m bucket × 両グループに。
7. **成果物に恒久 scratch と pointer table を追加**（`orders/F2-dense-launch-fusion.md:42-44`）: グループ×bucket 別の A_had（≥ bszm·m·k、`exl3_gemm.cu:478-481`）と出力バッファ、load 後 1 回の pointer table（model move/reload で再構築・graph 再 capture）。
8. **m の範囲を capture bucket 全てに**。発注書は m=4〜16 だが、引用した実測は 64 tok。門 1 の m 集合に 64 を追加。
9. **locks・n 整除の検査項目を追加**: 行列別 lock 範囲の非重複、全メンバー n_j % tile_n == 0 の TORCH_CHECK、sanitizer に racecheck。
10. **NCCL 待ちへの効果を「仮説・計測項目」に格下げ**（`orders/F2-dense-launch-fusion.md:24`）。8 rank P95 と最遅 rank 差で判定。
11. **数値目標・Kill 基準を門 3 に明文化**（launch ≈86/step、dense ≥1.3× PASS / 1.5× stretch、step −5%/−8%、mgemm プロトタイプ <1.1× で中止）。
12. 小修正: 「≈255 launch」は 46 step 平均 254.7【実測】。「tile 128/256 で 17〜18 µs・tile 512 で 31 µs」は実測と一致（17.97/16.75/31.30 µs）。成果物パス `vllm-exl3-v030-port/`（f2/）は存在を確認済み。

## 参照

- `orders/F2-dense-launch-fusion.md`、`orders/REVIEW-F1-sol.md`、`orders/F1-lna-moe-decode-kernel.md`、`orders/T4-dense-partial-bake.md`、`orders/REPORT-F1-opus.md`
- `exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm.cu`、`exl3_gemm_kernel.cuh`、`exl3_gemm_inner.cuh`、`exl3_kernel_map.cuh`、`exl3_devctx.cuh`、`comp_units/exl3_comp_unit_6_cb1.cu`、`../ptx.cuh`、`../coop_autotune.cu`、`../bindings.cpp`
- `exllamav3-src/exllamav3/modules/quant/exl3.py`、`exllamav3-src/exllamav3/exllamav3_ext/libtorch/linear.cpp`
- `vllm-exl3/src/vllm_exl3/exl3.py`
- `prof-graph-0904/profiler_out_0.txt`（rank0、46 step、09-04）
