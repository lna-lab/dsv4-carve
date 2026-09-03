# T4 設計メモ — attention/shared だけを EXL3 に部分焼きする（段 B）
出所: 2026-09-03 Explore 調査（exllamav3 v1.4.5 ソース、DATA1/.tmp/exl3src）。file:line は同ソース。

## 結論: 案(b) 「EXL3 experts を既存 pack から読み、密線形だけ校正付きで量子化」が最安
- 混在ロードは Linear 単位で素で対応: `Linear.load` は `load_exl3(key)` → `load_fp16(key)` の順（modules/linear.py:437-446）。
- 別ディレクトリの EXL3 experts を鍵 glob で差す仕組み = `VariantSafetensorsCollection`（loader/safetensors.py:854-895、使用例 model_init.py:246-252 / optimize_model.py:149-153）: `vstc.add_stc(["layers.*.ffn.experts.*"], SafetensorsCollection(<exl3 pack>))`。
- convert_model.py の主ループ（1130-1420）を写して二点だけ変える: (1) 量子化対象 `linears`（1272）を `quant_type != "exl3"` かつ密鍵に絞り、`assert all(LinearFP16)`（1276, 1453）を外す。(2) `q_tensors` を密鍵だけ集め `compile_model` を呼ばず module ごとに save_file（convert_mtp.py:127 の形）。`strategy = {k: 4 for k in dense_keys}` を自前で作る（1273-1275 の assert 対策）。
- 校正の流れは維持される: H は qmap ごとに `capture_H`（linear.py:564-586）、量子化後に `set_new_tensors` → reload → `advance_state_parallel`（1316-1345）で **前段は量子化済みで forward**。`calibration_all_experts` 強制（deepseek_v4.py:253）。
- 元 checkpoint の fp8/fp4 ブロックは load 時に dequant（linear.py:133-179, 198-221、no_defer）。bf16/fp8 どちらでも可、ただしメモリ重。
- 出力は密テンソルの `.trellis/.suh/.svh/.mul1(or mcg)` のみ（数 GB）。後処理: `util/add_quant_config.py` / `util/add_safetensors_index.py`。
- ⚠ 無校正の近道（convert_mtp.py 流用、`init_H_data(False)`=RTN 相当、linear.py:552）は 4bpw attention で品質を落とす。使わない。

## DSV4 の密線形（層 `layers.N`、dsv4.py / architecture/deepseek_v4.py）
- attn: `.wq_a`(qmap block.attn) / `.wq_b`(qmap {key}.q_b) / `.wkv` / **`.wo_a.slice.{0..7}`**（一つの checkpoint テンソル `wo_a.weight` の frange スライス、qmap {key}.o.{g}、8 枚は同じ K に揃える: dsv4.py:998-1006）/ `.wo_b`(qmap {key}.o_b, out float)
- compressor: `.attn.compressor.wkv/.wgate`；indexer: `.attn.indexer.compressor.wkv/.wgate`、`.attn.indexer.wq_b`；**`.attn.indexer.weights_proj` は qmap None＝量子化しない**
- ffn: `.ffn.shared_experts.w1/w2/w3`（qmap block.mlp；experts と input 側 H を共有）；`.ffn.gate*` は qmap None
- attention は `select_hq_bits=2`（--hq で +2bit）。`trim_padded_out=True` 多数。
- model 級: embed / head(head_bits) / hc_* / norm は今回対象外（vLLM plugin の non_routed_exl3 は lm_head 未対応）。

## vLLM 側の受け皿（vllm-exl3 plugin）
- `non_routed_exl3.layers` に vLLM の module prefix（mapper 後）で bits を書く。融合 module は一項目、全 shard 同 bits。**bf16_shards は TP>1 不可**。
- vLLM の DSV4 は `attn.fused_wqa_wkv`（wq_a+wkv 融合）、`compressor.fused_wkv_wgate` を使う（model.py stacked_params_mapping）→ 融合先の両 shard を同 bits の EXL3 に。wo_a は vLLM では `o_groups` 分割の einsum（flashinfer_sparse.py `_o_proj`、Cruz パッチで bf16 経路）→ **EXL3 化した wo_a を vLLM がどう食うかは別途コード読み**（最大の未知）。

## 資源
- 元 checkpoint: NAS models-cold/DeepSeek-V4-Flash-Vision-Exp（157G）→ DATA1 へ写す（mmap 不可のため）。GPU は 1-2 枚で足りる（層ごとに load/swap_cpu）。RAM 512G で余裕。
- 見積り: 層あたり密 13 テンソル → 量子化は数秒/層、壁時計は校正 forward ×2/層と fp8 dequant ロードが支配。数時間の桁（ESTIMATE）。
