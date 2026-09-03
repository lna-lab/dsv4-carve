# 鉱脈 dsv4-carve — DSV4-Flash-Vision EXL3 MixedK を TP=8 (16GB×8) で DSpark3 + 長文脈が同居する形に彫る
総監督: ユキ（雲）。開戦 2026-09-03。ケン「DSpark3 が入る隙間を削りで作る。-c128 ではなく（笑）」。
職人: GLM（Z.ai）／Luna（codex gpt-5.6-luna xhigh）。

## 北極星
TP=8 **単流**（ケン 09-03 18:10「単流は目をつぶる」）で **DSpark3 が乗り、文脈 128k**（ケン決定）、decode は素を割らない。
画像（vision）は落とさない。無検閲版（ablit）は素で型が固まってから同じ型で。

## 実測の基点（2026-09-03、像 lna-lab/vllm-exl3:dsv4 = vLLM dev337 + exllamav3 1.4.5 + vllm-exl3 0.2.3）
- 札 15.48 GiB、util 0.96 → 14.86 使用可。重み+非torch 13.18、活性の山 0.73（BT 512）、graph 0.04、KV 1.12 = 69k tok（32k×2.1）
- decode 単流 en/ja/code 40 tok/s（CUDA graph sizes [1,2,4]）。eager 12-16。プリフィル 1.3k tok/s。画像 OK。
- draft 3 層を足すと 15.2/枚 → KV 0 → OOM。NCCL_BUFFSIZE 絞りは効かない（13.18 のまま）。
- ディスク家計簿（/枚）: routed 8.67 / attention BF16 1.18 / draft experts fp8 1.20 / vision 0.87(複製疑い) / shared 0.25 / embed 0.12 / head 0.12 / draft他 0.12

## 職人と発注
- T2 draft overlay（Luna/codex）: DATA1/vllm-exl3-lab/orders/T2-mtp-overlay.md。出所 = wrldsuksgo2mars K2.2-D2（draft が EXL3 K2、shard 2/11 に集中、≈2.4GB を range-read）
- T3 attention 自家焼き（GLM、第二段）: 元 checkpoint の置き場が先

## 必要な隙間
64k+draft: 2.9 GiB／128k+draft: 4.7 GiB（KV fp8 ≈ 28 KB/token、1.82 GiB/64k）

## 彫る順（効き × 易しさ）
1. ~~vision tower を TP 分割~~ → dev337 で既に分割済（QKV/Row parallel）。家計簿の 0.87 は誤り、実は ≈0.11/枚。削れない
2. **draft experts fp8 → EXL3 2bit**（−0.9）— 出所: 自家焼き（exllamav3 convert -mb 2）or 外の EXL3 pack から overlay。plugin は mtp_experts="exl3" 既定で読める
3. **attention/shared BF16 → EXL3 4bit overlay**（−1.0）— 世界に EXL3 版 attention の pack は無い（D2 も BF16）→ **自家焼き**が要る（元 checkpoint ≈300GB、置き場: DATA1 115G/stripe 101G では足りず→NAS 経由か掃除）。exllamav3 の計測パスを非 expert 線形だけに絞る道具が要る。第二段。⚠ bf16_shards は TP>1 不可 → 融合線形は全 shard EXL3 に
4. 活性 BT 512→384（−0.2）※画像 1 枚 387 token が下限
5. 3bit 層 6 つ → 2bit（−0.4）— 最後。ppl の門つき
6. 1トークンの家計簿（段ごとの issue_ms）で decode 40 を守る／伸ばす

## 門
- 各段: 正気三言語 + 画像テストカード + decode ≥ 38 tok/s + KV トークン数（ログ行）を記録。
- 投機: spec-OFF と同一 greedy 出力 + 受理長（/metrics）。[[speculation-gate-doctrine]]
- 化粧合格禁止。実測と ESTIMATE を書き分ける。

## 台本
DATA1/vllm-exl3-lab/{Dockerfile, serve-dsv4-tp8.sh, bench-dsv4.py, recipe/, vllm-exl3/}。口 :8899。GPU 0,1,2,3,5,7,8,9。GPU4/6 は触らない。

## 09-03 18:10 中間実測
- 🎖️ DSpark3 が TP8 に乗った（D2 overlay + VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128、16k）: en 67/ja 55/code 70 tok/s、素 45。受理長 ≈2.1。画像 OK。
- 128k への残り: KV ≈2.1 GiB 必要、空き 0.51 → **+1.6 GiB/枚**。削り代: T3 3bit→K2 −0.4／attention EXL3 自家焼き −0.85〜1.0／BT・util −0.3／(未確認) fp4 KV で KV 半減。
- 決定性: この建て付けは greedy でも走行間不一致 → ビット一致は門に使えない。品質は ppl/実食。

## 一週間の段取り（ケン 09-03 18:15「一週間かけてもいい。精度を保ったまま実用域へ」「コードを読んで無駄時間を詰めるスタイル」）
北極星: **TP=8 単流 DSpark3 付き 128k、精度は素の pack と同等（ppl 門）、decode ≥ 素**。
- **段 A（〜09-04）: 32k**。T3 layer_overlay で 3bit 層 6 つ → K2（−0.4 GiB）。donor shard 7 本を DATA1/.tmp/k2v1-donor に落として local 経路で組む。門: ppl（vLLM prompt_logprobs で自前計器を作る、同一文・同一 ctx で pack 間比較）＋三言語正気＋画像。
- **段 B（09-05〜07）: attention/shared の EXL3 化（−0.85〜1.0）→ 64k〜128k**。世界に無いので自家焼き。元 = NAS models-cold/DeepSeek-V4-Flash-Vision-Exp（157G）。置き場は掃除待ち（ケン決裁）。方式は exllamav3 の量子化器（quantize_linears_single）を非 expert 線形だけに当て、前段の forward は D2 pack の EXL3 experts で回す「部分焼き」。⚠ plugin の non_routed_exl3 は TP>1 で bf16_shards 不可 → 融合線形は全 shard EXL3。lm_head は plugin 未対応（0.12 だけなので後回し）。
- **段 C（並行）: 1 トークンの家計簿**。vLLM の torch profiler で段ごとの時間（attention/indexer/MoE/all-reduce/lm_head/draft）を取り、CUDA graph の外に漏れている起動律速を潰す。目安: 素 45 → 50+、DSpark3 70 → 80+。
- **段 D: 128k 実走**。長文プロンプト（100k）で TTFT・decode・受理長。KV 2.1 GiB の確保を確認。
- 撤退線: ppl が素 pack より 2% 超悪化する削りは採らない（数字は段 A の計器で確定）。
- **ppl 計器（09-03 18:50 確定）**: `DATA1/vllm-exl3-lab/ppl-vllm.py <port> wiki.test.raw 512 16`、計測席 = draft 無し pack・spec OFF・util 0.95・max-len 4096・BT 1024（prompt_logprobs は語彙 129k の logits を rank0 に要求するので満杯席では OOM）。**基準 = 元 pack 6.6271**（7,215 tok）。撤退線 2% → 6.76。
- **家計簿（09-03 19:00、torch profiler、rank0、DSpark3 付き 64tok ≈29 手）**: 密 BF16 GEMM **51%**（cuBLAS が sm120 核を持たず sm80 WMMA 16×16 核に落ちる、11,686 発 × 43-50µs、一手 ≈21.8 ms、帯域計算なら ≈3 ms）／EXL3 expert GEMM 29.5%（exl3_moe_kernel 282µs × 1120）／NCCL 7.3%／indexer・norm・attention 各 ≤1%。→ **段 B（attention EXL3 化）は容量と速度の両方の本丸**。計器: prof-summary.py、席は PROFILE='{"profiler":"torch","torch_profiler_dir":"/lab/prof"}' で立て /start_profile /stop_profile。
- **段 A 実測（09-03 19:25）**: K2×6（3,13,21,22,28,41 を K2）= ppl **6.8845（+3.9%）** vs 元 6.6271 → 撤退線 2% 超、**丸ごとは不採用**。容量は 13.78→13.22 GiB/枚、KV 0.49→**1.05 GiB（64.5k tok）**、DSpark3 decode en 66/ja 58/code 69、受理長 2.2/1.8/2.3。→ 層ごとの感度掃引（t3-sens.sh、一層ずつ K2 → ppl）で痛くない層だけ採る。
- 道具の傷（直した）: layer_overlay の shard 書き換えに (名前,meta) のリストを渡していて何も落ちていなかった（verify も見逃し）。verify に「書き換え shard に落とした名前が残っていない」を追加。教訓: **verify は「残したものが同じ」だけでなく「消したものが無い」も見る**。
- 段 B 設計メモ: docs/T4-dense-partial-bake-design.md（VariantSafetensorsCollection で EXL3 experts を差し、密線形だけ校正付きで焼く）。
- 🎖️ **段 A 着地（09-03 20:45）**: 層別感度 3/13/21/22/28/41 = +0.41/+1.06/+0.67/+0.89/+1.17/+0.66%。**K2×3（3,21,41）= ppl 6.7041（+1.16%）**、重み 13.50 GiB、KV 0.77 GiB = 47k tok → **32k + DSpark3**（en 60/ja 59/code 68、受理長 2.1/1.9/2.2）。給仕 pack = `DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3`、計測 pack = `…-MixedK-K2x3`。六層版は不採用（+3.9%）。
- 128k までの残り（ESTIMATE）: KV 2.1 GiB 必要 vs 0.77 → +1.35 GiB。段 B の attention EXL3（−1.1、wo_a 込み）+ BT/util（−0.3）でちょうど。段 B は T4（Luna 製作中）+ plugin/vLLM 改変（docs/T5）。
- **段 B 着手（09-03 夜）**: 元 checkpoint を DATA1 に写し（157G）。T4 `tools/dense_bake.py`（Luna）= exllamav3 の校正付き量子化で密線形 790 本（43 層 × 15〜20）だけを 4bit EXL3 に。GPU 10,11 で焼き中（作業場 DATA1/.tmp/t4-dense-work、層ごとに ckpt、`--resume`）。私が直した傷: 層鍵の正規表現・target の順序比較・inference_mode。
- T6（Luna）: plugin に実効 TP（disable_tp/Replicated）と rank 局所 `wo_a.slice.{rank}`、vLLM 側は recipe-lna/patch_dsv4_dense_exl3.py（compressor quant_config・_o_proj EXL3 分岐）。像 **lna-lab/vllm-exl3:dsv4-dense**（旧 :dsv4 は温存）。給仕台本は IMAGE= で切替。
- 教訓（今日三度目）: `pgrep -f`/`pkill -f` のパターンが自分の bash -c に入る → 自殺（exit 144）。**pid で殺す**。docker build のタグは最初から別名で。
- 段 B 焼き（09-03 16:46〜）: 8 枚並列（10 枚は peer 上限で不可 → [[exllamav3-tp-peer-limit]]）、層 0→1 が 77 秒、1 層 66 MB、wo_a 4bit proxy_err ≈0.001。完了後は t4-after.sh が自動で merge（K2x3 計測 pack / D2-K2x3 給仕 pack）→ ppl → 64k/128k DSpark3 席まで回す（ログ DATA1/.tmp/t4-after.log）。
- **段 B 初点灯（09-03 18:30）**: 焼き 43 層 68 分（8 枚）。merge は 43 shard 書き直しで 66G（元 checkpoint は NAS へ退去して場所を作った）。**密 EXL3 で重み 13.50 → 10.56 GiB/枚（−2.9）**。vLLM 側で三つ直した: (1) config の prefix は `language_model.model.layers.N`／`ffn.shared_experts`（Vision クラス）、(2) wo_a EXL3 分岐の出力は fp32→bf16、(3) compressor の kv_score は `torch.mm(weight.T)` 直叩き → EXL3 shard 分岐（recipe-lna/patch_dsv4_attention_compressor_exl3.py）。CUDA graph + 多ストリーム有効だと warmup で無言ハング（GPU 100% spin）、eager + `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0` で正気 OK。切り分けと 64k/128k を t4-after4.sh で実施中。
- 段 B ppl（09-03 18:29、eager）: 密 4bit = **6.8789（基準 +3.8%、K2×3 比 +2.6%）** → 線越え。余裕 2.9 GiB があるので attention/shared を 6bit で焼き直す（元 checkpoint を NAS から再写し）。graph ON + 多ストリーム OFF は起動するが生成で 500（要 traceback）。
- **128k の席が立った（09-03 19:00、密 4bit pack、graph ON・多ストリーム OFF・DSpark3）**: 重み 11.51 GiB/枚（draft 込み）、**KV 396k〜420k トークン（128k の 3 倍）**。ただし生成が「無言→EngineCore TimeoutError」で落ちる（graph ON でも eager でも。12 トークンの補完は通る、思考付きの長い生成で落ちる）。多ストリーム ON は warmup で固まる。容疑: 小バッチ経路 `LinearEXL3.forward` → `ext.BC_LinearEXL3.run_alloc`（rows ≤ 144）— TP8 の 8 プロセス×多モジュールでの静的状態か同期。exllamav3_ext を読んで特定する。
- 6bit 焼き直し開始（09-03 19:05、8 枚、work=.tmp/t4-dense-work-b6）。4bit の ppl は 6.8789（+3.8%）なので不採用、6bit で線内を狙う。
- 6bit 焼き完了（09-03 20:22、43 層 RC=0、trellis 幅 96）。merge が config に bits=4 を書いていて（--bits 未指定＝既定）読み込みで形不一致 → 設定を 6 に直し、dense_bake merge は work/args.json の bits を既定にするよう修正（cd3b686）。席の連鎖は .tmp/t4-after6.sh / t4-after6.log で張り直し（20:33）。
- 6bit 席の二つ目の傷（09-03 20:45）: 密 EXL3 の小バッチ GEMM が CUDA graph 捕捉中に**未チューンの形**で自動チューナへ入り `coop_autotune.cu:464` "operation not permitted when stream is capturing" で全 rank 死亡。4bit で通ったのは密が BF16 だった頃のキャッシュ形と重なっていたため。直し=①exl3_gemm.cu: 捕捉中は `cudaStreamIsCapturing` を見てチューンせず静的ヒューリスティックへ（mgemm も同条件）②プラグイン: 重み読込直後に行数 1,2,4,8,16 で事前実行（LNA_EXL3_PREWARM_ROWS）③像は焼き直さず .so と plugin を DATA1 から差し込む（serve の EXT_SO / PLUGIN_SRC）。a131603。
- ⚠️ルート盤事故（09-03 20:50）: 落ちた席のコンテナ書込層が 27.6GB（コアダンプ/キャッシュ）でルート 0 バイト。docker data-root がルートに在る構造問題。対策=serve に --ulimit core=0・/root/.cache と .tilelang を DATA1 へ。**data-root の DATA1 移設はケン決裁待ち**。
- ★decode ハングの真因（09-03 21:40、py-spy＋CUDA_LAUNCH_BLOCKING=1 で確定）: CUDA_LAUNCH_BLOCKING=1 では止まらない＝同時走行の衝突。vLLM FusedMoE は ≤256 トークンで **shared experts を別ストリームに重ねる**（VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256）。密焼きで shared experts も EXL3 協調カーネルになり、routed の mgemm と同じ DevCtx ロック領域を別ストリームから同時に叩いて永久 spin（GPU 100%）。プリフィル(1024行)で平気・decode で止まる症状と一致。対策= VLLM_DISABLE_SHARED_EXPERTS_STREAM=1（serve 既定）。
- 🎖️**128k 単流 DSpark3 到達（09-03 21:30）**: pack=`DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`（68G）。KV **324,519 tok**（128k×2.48）、重み 12.0 GiB/枚。ppl **6.7159（+1.34%）**。投機なし 4k: en 42.1/ja 42.0/code 41.9。**DSpark3 128k: en 54-56 / ja 53 / code 62 tok/s**（受理 2.0/1.8/2.2）。32k BF16 密より約 1 割低い（shared 別ストリーム OFF＋密 EXL3 の小バッチ経路、要 profiler）。席の起動= serve-dsv4-tp8.sh（EXT_SO/PLUGIN_SRC 差し込み・AUX_STREAMS=0・SHARED_STREAM_OFF 既定・UTIL 0.97・MAXLEN 131072・BT 512・SEQS 1・SPEC dspark 3）。Dense4 と 4bit qtensors は削除。
- **-c 307,200 実測（09-03 21:45）**: 同 pack・同席で MAXLEN=307200 → KV **394,570 tok**（1.28×）、短文 DSpark3 en 58.6/ja 53.1/code 62.6。**針テスト 166k tok**（実測 prompt_tokens、84 万字、針は 37% 位置）: 正答 `sazanami-hotaru-2026`、TTFT **167 s**（プリフィル ≈1.0k tok/s、実測から算出）、その深さの decode **37.4 tok/s**。作文テスト 7 本（思考オフ）品質良、誤字 1（冼）。思考オンは max_tokens を大きく。Vision OK。
- **多流実測（09-03 21:55、-c 307,200・SEQS=4・DSpark3、bench-streams.py=usage 計数・metrics 突合一致）**: en 1流 62.4 / 2流 35×2=69 / 4流 35×4=**136** ・ ja 55 / 62 / **116** ・ code 86 / 96 / **186** tok/s。★2 流と 4 流の壁時計がほぼ同じ＝2 流の一手が 4 流と同コスト（graph 捕捉サイズと EXL3 小バッチの丸め。2 流は損、4 流が甘い）。⚠️SSE チャンク数で数えると投機で過小（1流 28 に見えた）。
- 🎖️**制式採用（09-03 22:00 ケン決裁「俺達のショウヘイオオタニがやってきた」）**: 制式構成 = -c 307,200・SEQS 4・DSpark3。レシピは Models/RECIPES.md。残: docker data-root 移設（先）→ 常設化（systemd/DSH）→ HF＋GitHub lna-lab 公開（private 先行）→ プリフィル彫刻・8 流。
- -c 389,120（380K）・SEQS 4 でも起動: KV **396,656 tok**（1.02×）、4 流 en 132 / code 141 合計（128 tok 短走）。制式は **-c 389,120・4 流**に更新（KV は共用プール、1 本 380k or 4 本 ≈99k ずつ）。
- DATA1: ds4 222G と焼き作業場 4 つ（ケン決裁）を削除、533G 空き。ds4 の履歴は DATA2/Lna-Lab/archive。
