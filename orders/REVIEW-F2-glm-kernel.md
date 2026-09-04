# F2 密 EXL3（K=6）launch 融合 — カーネル構造レンズの設計検分

監督: GLM-L（レンズ: カーネル構造のみ）。対象: `orders/F2-dense-launch-fusion.md`。
参照: `exllamav3-src/exllamav3/exllamav3_ext/quant/{exl3_gemm_kernel.cuh, exl3_gemm.cu, exl3_gemm_inner.cuh, exl3_kernel_map.cuh/.cu, coop_autotune.cu/.cuh, exl3_devctx.cuh}`、`vllm-exl3/src/vllm_exl3/exl3.py`、`orders/REVIEW-F1-sol.md`、`prof-graph-0904/`。

## 結論

**発注は成立する。ただし「新規カーネル `lna_dense_group.cu(.cuh)` を書く」は不要である。** 発注書の基準案 3 が要求するもの — 「`exl3_gemm_kernel<6,…>` の K=6 復号をそのまま使い、複数の (trellis, suh, svh, out) の表を受けて CTA を行列×出力タイルに割る」— は、家の ext に**既にある `exl3_mgemm_kernel` + `exl3_mgemm_gr` + binding `exl3_mgemm`** がそのまま実装しており、後述の通り F2 の 2 グループ（(a) h → wq_a+wkv+comp.wgate+comp.wkv、(b) h' → w1+w3）の形状要件をすべて満たす。しかも exllamav3 自身の DSV4 実装が**まさにグループ (a) と同じ fan を同じ API で**動かしている（`exllamav3-src/exllamav3/modules/dsv4.py:914-957, 1448-1470`）。

したがって F2 の実装の本体は **plugin 側の束ね wiring** であり、csrc に書くとしても `exl3_mgemm_gr` を呼ぶ薄い wrapper（後述の size_n 検査と autotune 塩付けのための数十行）で十分である。新規 CUDA カーネルを書く選択は、レビュー対象の構造からは正当化できない。

一方、既存 `exl3_mgemm` を plugin から使うにあたり、正しさと性能の両面で**必ず入れなければならない保証が 6 件**ある（§7）。特に (1) `size_n_list`経路の lock 範囲は `C.size(2) >= max(size_n_list)` という現在は呼び出し側の作法にのみ委ねられた不変条件、(2) 出力 dtype を既存と同じ fp32→bf16 境界に保つには C は fp32 でなければならない、(3) `group_barrier` カウンタは `exl3_moe_kernel` と同じ device 大域バッファを共有するため aux stream との並行は禁止 — の 3 つは設計時点で固定すべきである。

## 1. 発注書の実測数値の検証（すべて実測、ファイル照合）

`prof-graph-0904/profiler_out_0.txt`（rank0、graph ON、46 step）と照合した:

| 項目 | 発注書の主張 | ファイル照合結果 |
|---|---|---|
| launch 数/step | ≈255 | `<6,true,1,16,32,128,4,3>` 8179 回 + `<6,true,1,16,32,256,4,3>` 3053 回 + `<6,true,1,16,16,512,4,3>` 483 回 = 11715 / 46 step = **254.7/step** ✓ |
| 平均時間 | 17〜18 µs（tile 128/256）、31 µs（512） | **17.971 µs / 16.753 µs / 31.296 µs** ✓ |
| ステップ時間比 25% | 25% | Self CUDA 比率 17.05+5.93+1.75 = **24.73%** ✓（分母は profiler 全 CUDA 時間） |

- すべての密線形が `exl3_gemm_kernel<6, true, 1, …>`（bits=6, c_fp32=true, cb=1=mcg）で走っている。**c_fp32=true は現行 plugin 経路の実態**（`vllm-exl3/src/vllm_exl3/exl3.py:1507` の `out_dtype=torch.float32`）。発注書の「fp32 → bf16 の境界を動かさない」（案 5）は、この実測と整合する。
- graph OFF 側（`prof/profiler_out_0.txt`、23 step）ではホスト側 `cudaLaunchCooperativeKernel` が 11232 回出現し、graph ON 側では **0 回**ながら同一カーネルが 11715 回実行されている。つまり **cooperative launch は vLLM の CUDA graph に capture され replay されている**（実測）。F2 の grouped launch も同じ機構に乗る。
- 【推定】254.7/step ÷ 43 層 ≈ 5.9 launch/層は、発注書の表（圧縮層で最大 10 本）より少ない。一部の射影が bf16 部分焼き（`aten::mm` 47.5/step、cutlass bf16群）や bmm 経路である可能性が高いが、**per-matrix の launch 属性分解は本レビューでは実施していない**。REPORT-F2 では prof-shapes.py での形状別帰属を必須にしてほしい。特に N=512 群が 10.5/step しかない帰属（wq_b は rank 局所 25.2 MB になるはずで、31.3 µs では DRAM から読み切れない。L2 滞留か、別行列かの切り分けが必要）は未解決である。

## 2. `exl3_gemm_kernel_inner` の構造（CTA / タイル / cp.async 段 / C buffer）

以下はすべてソース読解（`exl3_gemm_inner.cuh`）。テンプレート引数は `<bits, c_fp32, cb, TILE_M, TILE_K, TILE_N, SH_STAGES, FRAG_STAGES>`（`exl3_kernel_map.cuh:7-16`）。

**CTA と thread 構成**
- blockDim = `EXL3_GEMM_BASE_THREADS(256) × TILESIZE_K/16`（`exl3_gemm_kernel.cuh:9, 89` の launch_bounds、`exl3_kernel_map.cuh:53-58` の形状表）。形状 2/3（K=32）は 512 thread、形状 4（K=16, N=512）は 256 thread。
- thread は `t = threadIdx.x % 256`、`sub_k = threadIdx.x / 256` に分解され、sub_k は K タイル内の 16 ブロック方向の担当（`exl3_gemm_inner.cuh:73-76`）。cp.async の発行と C 読み書きは sub_k==0 系のみが行い（`225-228, 599-614`）、MMA は全 thread。

**タイル**
- A: `TILE_M(=16) × TILE_K` half、XOR swizzle 付き（swizzle 定数 `50-55`、load index 計算 `116-127`）。`static_assert(TILESIZE_M == 16)`（`58`）で m≤16 を前提とし、m>16 は外側の while が 16 行ずつ回す（`exl3_gemm_kernel.cuh:37-49`、mgemm では `202-224`）。
- B: 16×16 trellis タイル（256 重み）× bits を `TILEBLOCKS_K × TILEBLOCKS_N` 面分（`42, 126-137`）。K=6 では 256 重み = 192 byte = 96 uint16。
- 復号は `load_frags` 内の `dq_dispatch<bits, cb>`（`287-296`）。**これが K=6 復号の唯一の経路**であり、単一・mgemm で共通（§5）。

**cp.async 段**
- `SH_STAGES` 段の global→shared パイプライン。`async_load_gl`（`225-262`）が A/B タイルを `cp_async` で投げ、`wait_stage` の `cp_async_wait<SH_STAGES-2>`（`630-634`）で受領。実形状は SH_STAGES=4（形状 2/3/4、`exl3_kernel_map.cuh:54-57`）。
- register 側は `FRAG_STAGES`（=3）段の frag 2 重バッファで、`FSTAGE` マクロの主ループ（`666-735`）で load/dequant/MMA/reduce を interleave する。

**C buffer（sh_c）と出力削減**
- `sh_c = max(4×256×FRAGS_N_PER_WARP, TILE_N×16)` float（`44-47`）。用途は 2 つ: (i) sub_k 間の threadblock reduce スクラッチ（`316-398`、TILEBLOCKS_K=2/3/4 の固定パターン）、(ii) `shmem_out_had=true` のときの出力 Hadamard ステージング（`424-478`）。
- グリッドは (k タイル × n タイル) 空間を gridDim.x で**連続スライス分割**し、k を最速走査（`88-97`）。1 列（n タイル）を複数 CTA が持つとき、部分和を global C へ `write_sum_gl`/`read_sum_gl` で鎖足しし、`locks[]` と `barrier_acquire/release`（`ptx.cuh:103-131`）で順序を取る。**最後の CTA が** `write_sum_tile_sh` → `output_had_sh_gl`（出力 Hadamard + svh、`424-478`）を行う。この `shmem_out_had+post_scale(svh)` 経路は単一行列 wrapper（`exl3_gemm_kernel`）からのみ true で呼ばれ、mgemm からは false + post_scale=nullptr で呼ばれて svh は外側の epilogue で適用される（`exl3_gemm_kernel.cuh:209-215, 227-259`）。

## 3. 複数 (trellis, suh, svh, out) の 1 launch（grouped GEMV）は成立するか

**成立する。既に実装済みであり、生産でも使われている。** `exl3_mgemm_kernel`（`exl3_gemm_kernel.cuh:89-310`）と host 側 `exl3_mgemm_gr`（`exl3_gemm.cu:411-657`）、binding（`bindings.cpp:161`）が次を提供する:

- **ABI**: B/suh/svh は行列アドレスを並べた int64 表（`exl3_gemm.cu:369-407` の doc）。1 入力複数出力（`bszm_in==1` のブロードキャスト、`exl3_gemm_kernel.cuh:171`）、**行列ごとの suh/svh 独立適用**（入力 Had: `165-183`、出力 Had: `227-259`）、weights/indices なしの separate outputs（doc `403-405` の "Without weights, every active C[j] is a separate output"）。発注書の案 1・3 の要件と一致する。
- **行列ごとの幅と出力先**: `size_n_list`(int32) + `c_ptrs`(int64 表) で行列ごとの n と出力ポインタを指定できる（`exl3_gemm.cu:464-477`、kernel 側 `194-204`）。制約は `num_tokens==1 && min_index<0 && !weights`（`470-471`）で、密経路はこれを満たす。グループ (a) の幅混合 (1024, 512, 512, 512) はこのためにある機能。
- **家内の使用実績（実測のコード、性能実績は未検証）**: DSV4 attention の x-fan「q_a / wkv / comp wkv+wgate / idx wkv+wgate as ONE per-matrix-N exl3_mgemm」（`dsv4.py:914-957`）、その呼び出し（`dsv4.py:1448-1470`、`rows <= 32` でゲート）、compressor の 2 行列 fan（`libtorch/dsv4_compressor.cpp:34-56`）、q_b と idx_wq_b の fan（`dsv4.py:1488-1494`）。ポインタ表・scratch は `g_tensor_cache` で shape キーに永続化し pointer 表のアドレス安定性を保証する作法（`979-995`）は F2 plugin がそのまま真似すべきである。
- **sm_120 でのバリア**: `__CUDA_ARCH__ > 890` では `grid.sync()` の代わりに `locks + BARRIER_LOCKS_OFFSET` の sense 反転 `group_barrier`（`exl3_gemm_kernel.cuh:184-190, 219-223`、`ptx.cuh:319-351`）で z スライス（=行列）ごとに同期する。行列ごとの Had → GEMM → epilogue の 3 相はこの barrier で区切られる。空スライスの CTA も barrier には到達する（inner の early return は `if (B)` の内側、barrier は外側、`202-224`）のでデッドロックしない構造である。
- **m ≤ 16 の行が同じ CTA で担われ重みを一度だけ読む**: inner のスライスは常に `slice_m = 0`（`exl3_gemm_inner.cuh:94-99`）、A フラグメントは k タイルごとに 1 回 load され FRAGS_N_PER_WARP 個の n フラグメントと全 16 行に再利用される（`275-296`）。B タイルは stage ごとに 1 回 load/decode。発注書の要求どおり。

## 4. CTA を「行列 × 出力タイル」に割る具体案

**新規に設計せず、既存 mgemm の割当てをそのまま使う。** すなわち:

- grid = `dim3(num_sms, 1, concurrency)`（`exl3_gemm.cu:635`）。**blockIdx.z = 行列スロット**（`bszm > concurrency` なら `for (i; i += gridDim.z)` で複数行列を順処理、`exl3_gemm_kernel.cuh:147-159`）、**blockIdx.x = その行列の (tiles_k × tiles_n) 空間の連続スライス**（k 最速、`exl3_gemm_inner.cuh:88-97`）。これが「行列 × 出力タイル（× k 分割）」の割当てそのものである。
- k 分割の部分和結合は列ごとの lock 鎖（`locks[slice2_n]`、`exl3_gemm_inner.cuh:590-628`）。z スライス間の lock 範囲は `locks + blockIdx.z * size_n/128`（`exl3_gemm_kernel.cuh:207`）で、`size_n` は最大幅である限り重ならない（使用量は `n_j/TILESIZE_N` 個 ≤ `size_n/128`）。バッファは device あたり `MAX_TILES_C = 1M` int（`exl3_devctx.cuh:6-9`）で十分。
- 並列度の目安【推定、SM 数は実機確認要】: グループ (a) のスライス総数（形状 2, K=32/N=128）は wq_a 128×8 + 3 行列 ×128×4 = **2560**。34〜48 SM に対し CTA 16〜32 個で 1 wave 80〜160 スライス/CTA となり、wave 量子化の tail は autotuner が `num_sms×concurrency` を採選する（`coop_autotune.cu:505-534`）。4 行列を z に 1 個ずつ置けば行列間は完全並行で、直列化するのは同一 z 内の 3 相 barrier のみ（現行 4 launch にも各 1 回の grid.sync があるので、構造的な悪化はない）。
- 静的 path（capture 中の未 tune 時）の grid 決定は `exl3_gemm.cu:618-636`: `tiles` は**最大幅**で計算され、`num_sms×bszm > total_sms` なら `num_sms = total_sms/bszm`、`tiles/num_sms > 48` で倍増、`concurrency = min(total_sms/num_sms, bszm)`。グループ (a) では例えば num_sms=16, concurrency=2〜4 程度に落ち着く【推定】。
- 代替案（(行列, n 列) を 1 次元 work queue に平ら化する F1 型 persistent 構成）は、4 行列 × 5〜9 列で既に SM 数を上回る task があるため不要である。複雑化する利益がない。

## 5. K=6 復号の流用可否

**流用可能であり、bit 単位で同一の経路である。**

- mgemm は inner を単一 kernel と**同じテンプレート実引数**（bits=6, cb=1, 同一タイル形状）で呼ぶ（`exl3_gemm_kernel.cuh:209-215` vs `38-42`）。復号は `dq_dispatch<6, 1>`（`exl3_gemm_inner.cuh:287-296`）に一本化されており、mgemm 固有の復号実装は存在しない。
- mcg codebook は host で `cb=1` に固定され（`exl3_gemm.cu:186-189`）、K=6 のインスタンスは `comp_units/exl3_comp_unit_6.cuh`（`ALL_EXL3_KERNEL_EXTERNS(6)`）+ 表登録（`exl3_kernel_map.cu:116-129`）で形状 1〜4・cb 0/1/2 すべて揃う。実プロファイルも `cb=1` で走っている（§1）。
- QTIP 型 `exl3_gemv_kernel` は `K < 2 || K > 4` で不成立（`exl3_gemv.cu:49, 111-113`）。**K=6 では現行も group 後も GEMV 経路に分岐しない**ので、束ねによる経路変更の parity リスクはこの点ではゼロである。
- ただし mgemm はグループ内で **K/mcg/mul1 が均一**であることを要求する（家内 fan は `dsv4.py:930` で検査）。今回の 790 本は全て K=6・mcg なので成立するが、plugin 側でグループ登録時に検査を入れること（§7-6）。

## 6. autotune / capture guard との整合

既存の仕組みがそのまま覆う。整合条件は「**capture 前に prewarm すること**」のみである。

- autotune key は `mgemm_autotune_hash` で `size_m(→pow2 cap 16), size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out(cap 24)` を混ぜる（`exl3_gemm.cu:97-124, 568-576`）。**size_n_list の中身は key に入らない**。同じ (m, k, max n, bszm) の別グループが設定を共有する。F2 の 2 グループは幅パターンが異なるが max n が等しい組み合わせはない（(a) は 1024、(b) は 2048）ため実害なし。汎用化するなら塩を混ぜること（§7-6）。
- 実行順序: メモリキャッシュ `launch_locked`（同期なし、capture 中も即 launch、`coop_autotune.cu:577-607`）→ disk cache（`191-250`）→ `tune()` は event/sync を使うため capture 禁止。`exl3_mgemm_gr` は `!graph && !lna_stream_is_capturing(stream)` でのみ tune し、それ以外は静的 `select_gemm_shape` に落ちる（`exl3_gemm.cu:581-612`）。guard `lna_stream_is_capturing` は `exl3_gemm.cu:59-63`。**capture 中に未 tune でもクラッシュしないが静的 heuristic で graph に焼き付く**。plugin の既存 prewarm（`exl3.py:1417-1429`、rows 1,2,4,8,16）をグループ mgemm 呼び出し分に拡張すれば、capture は必ず tuned 設定で張られる。
- cooperative launch の graph capture は実績済み（§1: graph OFF 11232 回のホスト launch が graph ON では 0 回で同一カーネル実行)。mgemm も同じ `cudaLaunchCooperativeKernel`（`exl3_gemm.cu:644-655`）なので差分はない。
- グラフ内ポインタ: B/suh/svh 表・c_ptrs・size_n_list は**カーネル引数のアドレスが焼き付く**。表の中身（= 各重みの data_ptr）を読むのは device 側ランタイムなので、永続 tensor に載せ替えなければ replay 間で安定する。家内の `g_tensor_cache` 作法（`dsv4.py:979-995`）がまさにこれである。vLLM 側では `_FUSED_TEMP_CACHE`（`exl3.py:59`）と同じ要領で (グループ, m bucket) キーの永続バッファを切ること。

## 7. 発注書へ反映すべき修正・保証（6 件）

1. **`size_n >= max(size_n_list)` の明示検査を入れる。** 現状この不変条件は呼び出し側の作法のみ（家内は「最初の出力を最幅にする」`dsv4.py:933-935` で守る）。`exl3_mgemm_gr` は `size_n_list` があっても `C.size(2)` を検査しない（`exl3_gemm.cu:464-477`）。破ると z スライス間の lock 範囲が重なり、**無言の race** になる（§4 の式）。F2 の wrapper で `TORCH_CHECK` 相当を入れるか、C を最幅キャリアとして必ず幅順ソートで渡す。
2. **出力は c_fp32=true で通す。** mgemm を C=half で呼ぶと、inner の `write_sum_gl` が fp16 中間丸め（`exl3_gemm_inner.cuh:544-549`）を挟み、epilogue も `had_hf`（half 演算境界、`exl3_gemm_kernel.cuh:252-259`）になる。現行の密経路は fp32 出力→bf16 変換（`exl3.py:1507-1518`）で、単一 kernel は出力 Hadamard まで fp32（`had_ff`/`had_fh`、`exl3_gemm_inner.cuh:466-479`）。**案 5「境界を動かさない」を守るには C キャリアと c_ptrs 出力を fp32 にする**こと。出力 byte の増（グループ (a) で m=16 のとき約 196 KB/step）は重み 7.86 MB に対して無視できる【推定】。
3. **aux stream との並行禁止を明文化。** `group_barrier` カウンタ（`locks + BARRIER_LOCKS_OFFSET`）と lock 範囲は `exl3_moe_kernel`（`exl3_moe_kernel.cuh:39-45` の `group_idx * MAX(hidden, inter)/128` 分割）および単一 gemm と**同じ device 大域バッファを共有**する。家の patch 自身が "EXL3 dense GEMMs are cooperative kernels sharing one device lock buffer; running them concurrently on aux streams deadlocks" と明記している（`recipe-lna/patch_dsv4_aux_streams_env.py:2-4`）。F2 はグループ mgemm を主 stream に置き、`LNA_DSV4_AUX_STREAMS=0` を前提条件にする。REVIEW-F1 §5 と同旨。
4. **prefill/大 m では束ねを切る。** mgemm は m>16 を 16 行チャンクの barrier 直列で処理する（`exl3_gemm_kernel.cuh:202-224`）ため、大 m では 4 launch 並行の方が速い可能性が高い【推定】。家内 fan は `rows <= 32` でゲートしている（`dsv4.py:1448`）。F2 も decode m ≤ 16（または bench 上限 32）でゲートし、超過時は現行の個別 `LinearEXL3.forward` に fallback する。vision の長行もこれで素通りする。
5. **「遅延束ね」のキャッシュ無効化を仕様化する。** 同じ入力 tensor オブジェクトの間のみキャッシュを返し、step をまたいだ stale 参照を許さない。推奨はキャッシュ key を `id(x)`（Python オブジェクト identity）+ 弱参照で持ち、対象が死んだら再計算。`data_ptr()` は caching allocator の再利用で同一アドレスが再現するため key に使えない。graph capture 中は当該 Python 分岐は capture 時に 1 回しか走らず、replay は kernel の副作用のみを再現するので、永続出力バッファ（§6）と組み合わせれば整合する。eager（graph OFF）でも step ごとに必ず再計算されることを parity 门に含めること。
6. **グループ登録時の検査。** (i) グループ内 K/mcg/mul1 均一（§5）、(ii) 全 n_j が選択 tilesize で割り切れること（互換性は `exl3_gemm_shape_compat`、`exl3_kernel_map.cu:86-91`; 今回の幅はすべて 512 の倍数なので形状 1〜4 いずれも可）、(iii) 入力が同一 tensor であること（グループ (a) は h、(b) は h'）、(iv) A は [1, m, k] fp16 contiguous。加えて wrapper を作るなら autotune key へグループ識別の塩。

## 8. 「割に合うか」の判定材料（監督判断用）

- 【実測】密 EXL3 は 4.64 ms/step（213.2 ms/46 step）、profile 全 CUDA 時間の 24.7%、254.7 launch/step。
- 【推定】launch 構成の算術: 254 ≈ 41 圧縮層 ×（グループ (a) 4 本 + (b) 2 本）+ 2 非圧縮層 ×（(a) 2 本 + (b) 2 本）。つまり **EXL3 launch はほぼグループ (a)/(b) の構成員のみ**で、wq_b/wo_a/wo_b/w2 はこの build では EXL3 launch を出していない（bf16 部分焼き/bmm 経路と推定。発注書の表との食い違いは REPORT で帰属確認を要する）。この読みが正しければ束ね後は 41×2 + 2×2 = **86 launch/step（6/層 → 2/層）**で、発注書の「6 → 2〜3/層」は正しい。
- 【推定】バイト下限: グループ (a) の重みは `4096 × (1024+512×3) × 0.75 B = 7.86 MB` → 発注書の cold read 264〜275 GB/s で 28.6〜29.8 µs。グループ (b) は 12.58 MB → 45.8〜47.7 µs。現行の実効帯域は 872 MB/step ÷ 4.64 ms ≈ **188 GB/s**（271 GB/s 比 69%）で、~2 MB 級 1 launch あたりに割ると 1 launch ≈ 10 µs の固定費+tail と読める。束ねはこの固定費を (a) 3 回・(b) 1 回/層削るので、推定 **1.5〜1.7 ms/step（密時間の 30〜37%、ステップの 6〜8%）**が利得上限にあたる。発注書の「上限 1.5×/ステップ 8%」と整合する上端である。**下押し要因**: 31.3 µs の N=512 群（10.5/step、帰属未解決）が示唆する weight の L2 滞留が現状の実効帯域を既に押し上げているなら、束ね後の利得はこの分だけ縮む。実測のみ判別可能。
- 判定: 既存 `exl3_mgemm` 再利用なら実装コストは小さく（plugin wiring + 保証 6 件 + prewarm 拡張）、parity リスクも low（同一 inner・同一復号）ので、**割に合う可能性が十分あるので着手を認める**。ただし受理基準は「1.5×」のような倍率でなく、(i) launch 数 254.7→100 以下、(ii) 密 EXL3 合計 ms の実測減、(iii) graph OFF でも劣化なし、の実測 3 点に張り替えることを推奨する。

## 9. 門への追加

- 门 1（机上 parity）に加えて: 束ね vs 個別の**同一入力・同一重みでの stage-wise 比較**（入力 Had 後、GEMM 生和、最終出力）で fp32 経路の一致を見ること。m ∈ {1,2,4,8,16} に 3（奇数 row、`threadblock_reduce` の size_m<=8 分岐境界の 8/9 を跨ぐもの）を加えることを推奨（`exl3_gemm_inner.cuh:375-398` の分岐のため）。
- 门 1 に `compute-sanitizer` に加え **NaN poison + graph 多重 replay + m bucket 交互**（REVIEW-F1 §4.1 と同じ stale 検出。永続 scratch の入れ替わり漏れが主リスク）。
- 门 3 の profiler 集計は `exl3_gemm_kernel<6` と `exl3_mgemm_kernel<6` の**両方**を数えること。graph ON/OFF 両方で launch 数を報告すること（§1 の 11232 vs 0 の差が host 側の目安になる）。
- 门 2・3 に `LNA_DSV4_AUX_STREAMS=0` を明示的な実行条件として記録すること（§7-3）。

## 10. 成果物の再提案

発注書の成果物欄は次のように読み替えることを推奨する（職人は変えてよい。理由は REPORT へ）:

- `csrc/lna_dense_group.cu(.cuh)`: 新規カーネルではなく `exl3_mgemm_gr` 呼び出しの薄い wrapper（§7-1 の size_n 検査、§7-6 のグループ検査、autotune 塩）。もしくは wrapper なしで plugin から binding `exl3_mgemm`（`bindings.cpp:161`、呼び出し形は `dsv4.py:1456-1464` を参照）を直接叩く。
- `src/vllm_exl3/exl3.py`: グループ (a) は遅延束ね（§7-5）、グループ (b) は同一 layer の 2 shard（`_exl3_linears`、`exl3.py:1489-1512`）なので**レイヤー内完結**で束ねられる。c_ptrs を 1 本の [m, 4096] バッファの前後半に指せば `torch.cat` も削れる（行インターリーブが cat と一致するため zero-copy）。
- prewarm 拡張（§6）と env `LNA_EXL3_DENSE_GROUP=1` は発注書どおり。

## 参照

- `exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh`, `exl3_gemm.cu`, `exl3_gemm_inner.cuh`, `exl3_kernel_map.cuh/.cu`, `coop_autotune.cu/.cuh`, `exl3_devctx.cuh`, `exl3_gemv.cu`, `hadamard_inner.cuh`
- `exllamav3-src/exllamav3/exllamav3_ext/ptx.cuh`, `bindings.cpp`, `libtorch/dsv4_compressor.cpp`
- `exllamav3-src/exllamav3/modules/dsv4.py`
- `vllm-exl3/src/vllm_exl3/exl3.py`, `recipe-lna/patch_dsv4_aux_streams_env.py`
- `orders/REVIEW-F1-sol.md`（ABI・graph・stream 方針の正典）, `orders/F2-dense-launch-fusion.md`
- `prof-graph-0904/profiler_out_0.txt`, `prof/profiler_out_0.txt`
