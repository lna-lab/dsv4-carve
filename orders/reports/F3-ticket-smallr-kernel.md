# 発注 F3: C+A — 常駐 expert-team＋ticket の外側に small-R 専用の内側を入れる（フェラーリ一本目・二号機）

発注者: YUKI／決裁: ケン 2026-09-04「C+A の新しい発注（F3）を書いて職人を再出動させてよい」
的（ケン）: 「ビッタリ張り付いて上限を上げろとプレッシャーをかけるくらいのカーネル」＝**圧（飛行中バイト数）で 270 GB/s を埋め切る**。
前提: F1 初号（`csrc/lna_moe_decode.cu`）は正しさ合格・速度不合格で閉じた。ABI・机上 parity・bench・primitive gate は流用する。判定と設計方針は **`f1/REVIEW-REPORT-F1-sol.md` を全文読む**（発注書の一部）。F1 v2 発注書（`f1/F1-lna-moe-decode-kernel.md`）の ABI/数値境界（9 pointer・128 要素 Hadamard・suh/svh 独立・FP32 境界・swiglu clamp・固定順 gather）は据え置き。

## 設計（Sol §次の一手。職人は理由を書けば変えてよい）
1. **外側 = A**: 現行 `exl3_moe_kernel.cuh` と同じ「常駐 expert-team ＋ 動的 ticket」。team（既定 8 CTA、変えてよい）が expert を一つ取り、gate/up 入力変換 → gate/up → activation/down-input → down → scatter まで**通しで**流し、終わったら atomicAdd で次の expert を取る。**全 grid の phase barrier は無し**。team 内 barrier のみ。grid は resident 枠内（現行と同じく occupancy から算出、cooperative launch 不要なら使わない）。
2. **内側 = C**: canonical `exl3_gemm_kernel_inner`（M=16 固定・16×N FP32 C buffer・512 thread）を small-R 用に置き換える。実 route の R_e 分布（m=4: 平均 1.04、m=16: 1.16）に合わせ、**R_e ∈ {1, 2, 3–4, 5–8}** の variant を用意。R=1/2 では M16 tensor tile と 16×N の C buffer を捨て、1 CTA が**十分な数の独立した B 要求**（cp.async の複数ストリーム、producer warp と decode/MMA consumer の分離）を常に飛ばす形にする。weight tile は同一 expert の全 row に使い切ってから捨てる。R_e > 8 の hot expert だけ canonical M16 経路。
3. **固定順 gather は 1 launch のまま**: 各 team の expert 出力を (row, slot) scratch に置き、現行の retired-group counter（acquire/release）を流用して**最後に退いた team が全 row を top-k 順で gather**。複雑なら compute+gather の 2 launch を graph 上で先に比較し、launch 数より総 latency を優先してよい。
4. **B は資源目標として後から**: 2 CTA/SM を主張するなら occupancy API の実値 ≥2 が必須。ptxas の registers/thread・spill・dynamic smem・占有の制約要因を build ごとに記録。
5. `prepared[2][3]` を device index を含む key に。shared-experts aux stream は明示 error か明示 fallback。

## 門（Sol §門。順番に。前段不合格なら次へ進まない）
1. **同条件 baseline**: 候補と現行 exl3_moe を同じ process・同じ x/ids/weights/U/R_e・同じ出力 dtype・同じクロックで交互に測る。K2/K3 × m=1/4/8/16、cold p50/p95 と warm p50/p95 各 3 run。**m ごとに現行値を持つ（229 を全 m に流用しない）**。kernel-only と plugin end-to-end を分ける。
2. **圧の計器（pressure gate）**: Nsight Compute（容器に `ncu` があれば。無ければ CUPTI/torch.profiler で取れる範囲＋自前カウンタで代替し、無い計器は「未測定」と書く）で actual DRAM read bytes/throughput・L2 hit・outstanding requests・long-scoreboard・barrier stall・achieved occupancy・registers/spill。**K2 m=4 の trellis 区間で ≥200 GB/s**、未達なら compute ceiling を数字で示す。全 CTA の active-time と tail の timeline。
3. **性能**: cold p50 ≤ 1.25 × T_floor（暫定: K2 m=4 ≤ 88 µs、K3 m=4 ≤ 128、K2 m=16 ≤ 312、K3 m=16 ≤ 463）。同条件現行に対し全 bucket 非劣化、K2 m=4 は ≥1.15×。p95/p50 ≤ 1.10。
4. **scheduler/resource correctness**: ticket の初期化・wrap/reset・0 active・1 team・全 team・expert 数が team 数前後・最後の team の gather・graph 連続 replay・複数 stream/graph instance の stress。全 variant で parity。compute-sanitizer 4 種（容器に無ければ理由を書き、席の門に送る）。
5. **数値/graph**: F1 の parity 一式を維持（K2/K3、m=1..16、本番 K3 層 13/22/28、意地悪 route、rel ≤ 2e-3、bitwise 再現）。`bench_lna.py` の **fail-open を直す**（capture 失敗・replay 不一致は nonzero exit）。scratch/out の NaN poison replay。scaling 1.5 の zero/once/twice 番兵。pairwise coverage を明示。
6. 席の門は私が回す（ここまで通ってから）。

## 成果物
- `csrc/lna_moe_decode.cu` の二号機（ファイル名は `lna_moe_ticket.cu(.cuh)` でも可。F1 初号は消さず残す）、bindings、`exl3.py` の dispatch（`VLLM_EXL3_MOE_KERNEL=lna2` など、既定は現行のまま）
- 同条件 baseline harness、圧の計器スクリプト、直した `bench_lna.py`
- `f1/REPORT-F3.md`: 設計・全門の数字・落ちた道・未測定の計器・席の門に残すもの

## 読むべきもの
- `f1/REVIEW-REPORT-F1-sol.md`（全文）、`f1/REPORT-F1.md`、`f1/REVIEW-F1-sol.md`
- 現行の正典: `/run/media/tonoken3/DATA1/vllm-exl3-lab/exllamav3-src/exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh`（team・ticket・retired counter）、`exl3_moe.cu`（grid 算出・launch）、`exl3_gemm_inner.cuh`（cp.async 段・M16 固定・C buffer）、`exl3_dq.cuh`、`codebook.cuh`
- GPU 実行: F1 と同じ（GPU10/11 の容器）。席（:8899、GPU0〜9）に触らない。
