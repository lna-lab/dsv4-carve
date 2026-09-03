# T5 読み — vLLM（dev337）+ vllm-exl3 0.2.3 が EXL3 化した密線形をどう受けるか（2026-09-03 Explore 調査）
file:line は image lna-lab/vllm-exl3:dsv4 の /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ と plugin exl3.py。

## 層内の密線形（mapper 後の prefix）と今日の可否（TP=8）
| module | class | 今日 EXL3 可? | 必要な checkpoint 名 |
|---|---|---|---|
| attn.fused_wqa_wkv | MergedColumnParallel **disable_tp（複製）** | ✗ plugin が disable_tp を無視して global TP で割る（exl3.py:1144-1152, 1215-1220） | layers.N.attn.wq_a.* (shard0) + attn.wkv.* (shard1) |
| attn.wq_b | ColumnParallel | ✓ | layers.N.attn.wq_b.* |
| attn.wo_a | ColumnParallel だが線形として呼ばれない（is_bmm、_o_proj の einsum: flashinfer_sparse.py:206-237/602-633） | ✗ model 改変要 | — |
| attn.wo_b | RowParallel | ✓ | layers.N.attn.wo_b.* |
| attn.compressor.fused_wkv_wgate | Merged, disable_tp, **quant_config=None**（compressor.py:270） | ✗ model 一行改変 | compressor.wkv.* + wgate.* |
| attn.indexer.wq_b | ReplicatedLinear | ✗ 複製経路が無い | indexer.wq_b.* |
| attn.indexer.compressor.fused_wkv_wgate / indexer.weights_proj | quant_config=None | ✗（weights_proj 0.26M は bf16 のままで可） | |
| ffn.shared_experts.gate_up_proj / down_proj | Merged / Row | ✓（SP off・MegaMoE shared fusion off 前提: model.py:365-373, 899-904） | shared_experts.w1/w3/w2.* |
| ffn.gate, hc_*, norm, attn_sink, ape | raw param | 対象外 | |
- 名前は **融合前の HF 名 + EXL3 接尾**（stacked_params_mapping が shard に振る: model.py:1476-1484、trace 1512-1529 で plugin の weight_loader(param, loaded, shard_id) に届く）。shard ごとに trellis/suh/svh + mcg|mul1 一つ。
- bf16_shards は TP>1 で例外（exl3.py:1018-1026）。lm_head/embed は plugin 対象外（VocabParallelEmbedding、quant_config 無し）。tie 無し。

## wo_a
- 一層 33.55M param（bf16 67MB、attention 密の ≈24%）。TP=8 では rank あたり 1 group → `hdr` einsum は 2-D GEMM に退化 → **rank ごとの LinearEXL3 [1024,4096] で足りる**（最安の本道）。_o_proj の `self.wo_a.weight.dtype` 参照（207/603）を getattr に。
- 零改変案 = load 時に EXL3→bf16 へ戻す（容量は減らない）。

## 「全密線形 EXL3 @TP=8」に要る最小改変
plugin: (1) disable_tp / ReplicatedLinear を尊重した実効 TP で shard（exl3.py:1150-1152）、(2) is_bmm(wo_a) 対応、(3) bf16_shards 制限を実効 TP で緩和（任意）。
model: (4) compressor.py:270 に quant_config を通す、(5) attention.py:904（任意）、(6) flashinfer_sparse.py:207/603 の dtype 参照を耐性化。
運用: VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION / SP off。16 タイル整列は rank 単位で OK（wq_b 4096, wo_a 1024, wo_b 1024）。

## 容量見積り（ESTIMATE）
attention 密 ≈138M/層 × 43 ≈ 5.9B → bf16 11.9GB（家計簿の attention 9.5 + shared 2.0 と整合）→ 4bit ≈3GB → **−8.9GB 全体 ≈ −1.1 GiB/枚**。wo_a を残すと −0.85 程度。
