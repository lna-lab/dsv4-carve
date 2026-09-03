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

## 次の彫刻（ケン 09-03 深夜「さらなる高速化改修、磨ききりたい」）
1. **家計簿の取り直し**: 密 EXL3 化後の decode 1 ステップを profiler で段ごとに（旧台帳: 密 BF16 51%・experts 29.5%・NCCL 7.3% は失効）。
2. **2 流の損**: 一手のコストが 2=4（graph 捕捉サイズ [1,2,4] と EXL3 小バッチの丸め）。捕捉サイズと形の桶を合わせて 2 流・4 流双方を伸ばす。8 流も測る。
3. **投機**: 受理長 en 2.0/ja 1.8/code 2.2（3 本中）。draft experts K2→K3 の精度、本数 2/3/4 の実測、r 表（限界費用）で可否。
4. **プリフィル**: 166k で 167 s（≈1.0k tok/s）。BT 512→1024/2048、VLLM_SPARSE_INDEXER_MAX_LOGITS_MB、KV 余裕（396k）との折り合い。
5. **NCCL 7%**: P2P 無しの制約下で all-reduce の代替（custom AR は不可）。
6. 常設化（systemd --user・head warm）と docker data-root 移設（ケン sudo）は前提作業。

- 🎖️**公開（09-03 22:52）**: HF `sakamakismile/DSV4-Flash-Vision-EXL3-MixedK-D2-K2x3-Dense6`（public、86GB、lna-lab/ に .so とレシピ同梱、上り 38 分）＋ GitHub `lna-lab/dsv4-carve`（public）。private は HF 無料枠の上限で不可だった。

## 夜のトライ 09-03 22:30〜（ケン許可、席の上げ下げ可）
- T1a 投機 **n=2**: 1流 en 60.7/ja 57.2/code 80.5、4流合計 125/105/153 → **n=3 が勝ち**（62/55/86、136/116/186）。
- T1b 投機 n=4・T2 BT 1024/2048・T3 SEQS 8 + 捕捉 [1,2,4,8]: **-c 389,120 では KV が足りず起動不可**（必要 3.0 GiB に対し 1.5 GiB 余り／T3 は 1.83 vs 1.80 で紙一重）。380K を守る限り、投機 3 本・BT 512・4 流が上限。8 流や広いプリフィルは -c を 300K 前後へ譲るときの選択肢。
- T4 **家計簿（密 EXL3 化後、rank0、64 tok DSpark3、self-CUDA 1.06 s）**:
  | 30.6% | NCCL all-reduce/gather（vllm:: wrapper 込みで二重計上あり。カーネル実体 ≈190 ms ≈18%） |
  | 24.3% | routed experts EXL3 mgemm |
  | 18.8% | 密 EXL3 gemm（attention/shared、K=6） |
  | 8.2% | torch の糊（elementwise/copy/sort/index） |
  | 7.1% | 残る BF16 cutlass（lm_head/embed/indexer weights_proj） |
  | 6.6% | DSV4 融合 op（mhc/shared/topk/indexer） |
  | 1.7% | sparse MLA |
  → 密の 51% は 19% に落ちた。**次の的は NCCL（P2P 無しの RING_LL）**と routed の mgemm。BF16 残り 7% は lm_head/embed の EXL3 化で削れる余地。

## 09-04 朝: 太いプリフィル・Cruz v0.3.0・Roy・EP
- **太いプリフィル（実測）**: BT 512 @307K: 166k 針 TTFT 155 s。**BT 1024/2048 は -c 307K でも 200K でも KV 不足で起動不可**（活性化と索引の予備が BT に比例、1024 で +0.7 GiB、2048 で +1.5 GiB）。BT 1024 は -c 200K で起動（KV 263k）したが 166k 要求が `KeyError choices`（エラー chunk、計器が握り潰し）→ 計器を直して再測（V3）。結論の候補: 長窓を守る限り BT 512、プリフィル高速化は BT でなくカーネル側。
- **Cruz vllm-exl3 v0.3.0**（09-04）: native `p2b_fused_moe`（K=2/3/4、レジスタ内 Trellis 復号）＋ native gemm/gemv（K=2/3/4/8）＋ prefill gemm カーネル（**給仕経路に未配線**、テストのみ）。fused MoE の門= hidden 4096・**expert 中間 2048**・行 ≤8 → **TP8 の列分割（中間 256）では通らず、EP（expert-parallel）が必要**。密 K=6 は exllamav3 経路のまま。Luna が我々の 18 hunk を移植（`DATA1/vllm-exl3-v030-port`、native 経路を壊す 7 hunk は不採用）、native .so は家の像で sm_120 ビルド済（NATIVE_OK）。
- **EP の壁**: 現行 plugin は experts を process-global TP で narrow するため EP で `dest (2048,) != loaded (256,)`。移植版に「MoE 層自身の tp 幾何（moe_config.tp_size/tp_rank）で narrow」を入れた（LNA-LAB）。V1/V2 で実測。
- **Roy（roycorp.net/sglang）**: GLM-5.3-flash NVFP4 + DFlash2 で 2×6000 1,493 tok/s（周期文の峰、散文 45.7→96）。うちに効く手筋= **Relay AllReduce**（pinned host shm 経由、281→202 µs）・n-gram 前段 proposer（draft 0.06 ms）・fused accept・presharded＋JIT 永続。式 E=(1−α^(W+1))/(1−α)、W*=ln(s/t_AR)/ln(α) は家の「幅の天井検定」と同型。
- **NCCL の代替候補（家の像に実在）**: vLLM `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`（PCIe 専用 IPC all-reduce、world 2/4/8）→ ただし像の flashinfer 0.6.18 に `PcieIpcAllReduceWorkspace` 無し（main 08-20 以降、nightly index `https://flashinfer.ai/whl/nightly/`）。**P2P 実測**: GPU10↔11（同一 PHB）d2d 25 GB/s・正常。2026-06 のハングは NODE 経路（root complex 越し）か NCCL 側の可能性——空いた時に NODE ペアで再測。
- **EP 実測（09-04 05:47）**: 移植版 plugin（MoE 層の tp 幾何で narrow＋expert_map の読み取り専用対策）で EP=1 起動。KV **466k**（TP の 396k より増）。EP＋従来カーネル: 1流 55.6/50.6/77.6（TP 比 −1 割）、4流 125/117/187。EP＋**native fused MoE**: 1流 **84.5/81.6/100.4**、4流は門（行 ≤8）に掛からず 129/112/178。
- ⚠️**native は数値が壊れていた**（門が効いた）: 貪欲突き合わせで 5〜31 字で分岐、「その独特な概念」無限反復・`**"**"` 連打、作文 7 本 finish=length。Cruz の parity テストは sm_120 で cosine 1.00000（カーネル自体は正しい）。**真因 = DSV4 の `swiglu_limit: 10.0`**（silu(min(g,L))·clamp(u,±L)）を native カーネルが持たず素の silu(g)·u（GLM 用）。exllamav3 の fused 経路は limit を渡している。直し= p2b_moe.cu Phase 3 に clamp、`swiglu_limit` 引数を kernel→launcher→host→bindings→Python に通す（既定 0 = 素の silu で Cruz 互換）。再ビルド後に再検定（v030e）。
- ★**native 崩れの真因（09-04 06:20、確定）**: swiglu_limit ではなかった（clamp を入れても出力不変）。カーネルは exllamav3 LinearEXL3 基準で K=2/3・E=1〜12・重複 id・zero 重みすべて cos 1.00000。席内の自己検査（同じ入力で Python 参照と比較）で「参照が大きい層は一致、小さい層は native が 0」→ 再実行で完全一致（競合でない）→ 事例をディスクに落として GPU10 で再生 → **`map_topk_to_local` が (T·K,) の平坦な id を返し、native の行ループが `safe_ids[row]` で 1 要素（スカラー）を取る＝各トークンに expert が 1 個だけ**。`reshape(ids.shape)` で再生 rel 0.0008。Cruz は TP・rows=1 でしか通していないので未発見（彼の TP 経路も rows=1 では同じ罠の疑い）。G8 で席の門を再検定。
- **native 直後の実測（09-04 06:25）**: 受理長 2.09/1.88/2.28（正常）、作文 7 本 finish=stop（健全）、しかし **1 流 en 23.5 / ja 20.3 / code 33 tok/s**（制式の 4 割）。原因=Cruz カーネルは 1 行 1 launch（≈350〜410 µs/launch @sm_120）で、DSpark3 の 4 行 × 43 層 = 172 launch/step ≈ 60 ms。exllamav3 mgemm は全行 1 launch。8 分割の細い expert では固定費で負ける。**native 不採用、制式（TP8 exllamav3）据え置き**。EP は KV 466k の利点のみ（速度 −1 割）。
- **貪欲突き合わせの基準（09-04 06:31）**: 同じ exllamav3 を二度走らせても一致 49〜118 字・類似度 0.15〜0.88 で分岐（この家の既知の非決定性）。直した native は exl2 に対し code 完全一致・他はこの揺れ幅の内側 → **数値の門は通過**。門は「同カーネル二度走りの揺れ幅」を基準に読む。
- ★**ケン方針「フェラーリ流」**（09-04）: この一台（DSV4×8枚 PRO2000）専用の形固定カーネルを職人 1 人 1 本で。→ [[ferrari-kernel-doctrine]]。最初の一本= routed experts 細幅多行 fused。先に形つき家計簿。
- **クロック固定の実測（09-04 06:40〜、56 W 上限据え置き＝ケン「56W のまま粒を合わせる」）**: 電力上限は 56 W（最小。既定 70 W だがケン: 戻すと tok/J が落ち電源が危うい）。固定前は 4 流で 56 W 張り付き・2,475〜2,647 MHz 散り。3 回平均（1流 en/ja/code、4流合計）:
  | 固定前(1回) | 58.5/58.0/89.2 | 134/119/180 |
  | 2450 | 61.4/55.1/84.3 | 133.5/112.6/183.4 | 4流中 2,415〜2,422（粒 ±4）、51.3 W |
  | 2600 | 63.4/55.9/83.9 | 135.9/114.1/183.5 | 2,385〜2,535（上限当たり）、56.1 W |
  | 2595 | 62.8/56.2/84.6 | 133.9/113.4/180.0 | 2,377〜2,565、54.8 W。長走 4流×1024: en 133/ja 121/**code 210** |
  → 速度は全て揺れ幅（±2〜6）の内側。固定の効き目は峰でなく**粒**（2450 で 8 枚 ±4 MHz に揃う）。ケンの像「レブリミット 8000 固定でトルクを絞り出す」「56 W に張り付けられるカーネルを目指す」= 電力＝仕事量の指標。粒を揃える値は 2450〜2500、電力を使い切る値は 2595。
- 🎖️**回転数縛りを制式に（09-04 06:55 ケン決裁）**: `-pl 62` + `-lgc 2600,2600` → 4 流中 2,550〜2,572 MHz（幅 22）・59〜60 W 安定。1 流 en 65.4/ja 56.7/code 86.6、4 流長走 131/122/212。ケンの発案「電力でなくクロックで縛る」。unit= Lna-Lab/house/lna-gpu-clocks.service（要 sudo 登録）。

## 09-04 07:05 形つき家計簿（フェラーリの設計図）
- graph ON（制式）、rank0、64 tok 要求、カーネル総 862 ms: routed EXL3 mgemm **31%**（992 launch × 229 µs）／密 EXL3 gemm K=6 **25%**（8,050 launch × 18 µs）／NCCL **18%**（2,256 launch）／BF16 cutlass 10%／糊 5%／DSV4 融合 4%／sparse MLA 2%。
- eager（形つき）: decode の行数は **4**（[4,4096]）、routed は `moe_forward_shared` 経由 1 layer 1 launch（920 回）。密は 1 step ≈ 390 launch（≈9/層: wq_a・wkv・wq_b・wo_a slices・compressor・indexer・shared w1/w3/w2）。router gate は BF16 mm [4,4096]×[4096,256]。**eager では NCCL が 64%**＝CPU 発行の隙間で rank 同士が待つ時間が NCCL カーネルに計上される → NCCL の正体は「待ち」。粒揃え（クロック規律）が効く理由。
- 一本目（routed）: 1 launch で rows 4 × topk 6 = 24 expert スロット、1 rank の expert 幅 256（w1/w3/w2 ≈ 0.86 MB @2.2 bpw）→ ≈21 MB/launch を 229 µs = **≈90 GB/s**。帯域効率は下（実測帯域は別途）。的は「読んだバイトあたりのトークン」。
- 二本目（密）: 8,050 launch を層ごとに束ねる（graph 不要の形へ）。ケン「CUDA graph を入れても速度が変わらない状態」＝完成の定義。
- 帯域実測（GPU10、PRO 2000）: d2d copy 往復 254 GB/s（片道 ≈127）。routed mgemm の ≈90 GB/s は帯域の 1/3 前後 → 一本目の伸びしろ ≈2×（ESTIMATE）。発注書= vllm-exl3-lab/orders/F1-lna-moe-decode-kernel.md（Sol 検分へ）。
