from __future__ import annotations

import json
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools import dense_bake
from tools import layer_overlay


EXPERTS = Path("/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK-D2")
SOURCE = Path("/run/media/tonoken3/DATA1/DeepSeek-V4-Flash-Vision-Exp")


def write_fixture_shard(path: Path, entries: list[tuple[str, dict, bytes]], metadata=None):
    with path.open("wb") as stream:
        layer_overlay.common.write_safetensors_header(
            stream, [(name, meta) for name, meta, _ in entries], metadata
        )
        for _, _, payload in entries:
            stream.write(payload)


def meta(dtype: str, shape: list[int] | None = None) -> dict:
    return {"dtype": dtype, "shape": [] if shape is None else shape}


class DenseFilterTests(unittest.TestCase):
    def test_exact_dense_filter_and_exclusions(self):
        accepted = [
            "layers.0.attn.wq_a",
            "layers.0.attn.wq_b",
            "layers.0.attn.wkv",
            *[f"layers.0.attn.wo_a.slice.{i}" for i in range(8)],
            "layers.0.attn.wo_b",
            "layers.0.attn.compressor.wkv",
            "layers.0.attn.compressor.wgate",
            "layers.0.attn.indexer.compressor.wkv",
            "layers.0.attn.indexer.compressor.wgate",
            "layers.0.attn.indexer.wq_b",
            "layers.0.ffn.shared_experts.w1",
            "layers.0.ffn.shared_experts.w2",
            "layers.0.ffn.shared_experts.w3",
        ]
        rejected = [
            "layers.0.attn.indexer.weights_proj",
            "layers.0.ffn.gate.weight",
            "layers.0.ffn.experts.0.w1",
            "mtp.0.layers.0.attn.wkv",
            "embed.weight",
            "head.weight",
            "layers.0.hc_expand.weight",
        ]
        self.assertEqual(dense_bake.dense_keys_for_layer(0, accepted + rejected), sorted(accepted))
        for key in rejected:
            self.assertFalse(dense_bake.is_dense_key(key), key)

    def test_wo_a_requires_all_eight_and_bits_are_shared(self):
        with self.assertRaisesRegex(RuntimeError, "exactly slices 0..7"):
            dense_bake.dense_keys_for_layer(0, ["layers.0.attn.wo_a.slice.0"])
        keys = [f"layers.0.attn.wo_a.slice.{i}" for i in range(8)]
        self.assertEqual({dense_bake.bits_for_key(key, 4, 3, 5) for key in keys}, {3})
        self.assertEqual(dense_bake.bits_for_key("layers.0.ffn.shared_experts.w2", 4, 3, 5), 5)

    def test_vllm_fused_mapping_has_one_entry_per_fused_module(self):
        args = Namespace(vllm_root="model", vllm_shared_prefix="mlp.shared_experts",
                         bits=4, attn_bits=3, shared_bits=5)
        bases = [
            "layers.0.attn.wq_a", "layers.0.attn.wkv", "layers.0.attn.wq_b",
            "layers.0.attn.compressor.wkv", "layers.0.attn.compressor.wgate",
            "layers.0.ffn.shared_experts.w1", "layers.0.ffn.shared_experts.w3",
            "layers.0.ffn.shared_experts.w2", "layers.0.attn.wo_a.slice.0",
        ]
        items = [{"base": base, "target": f"{base}.trellis", "meta": meta("I16", [1])}
                 for base in bases]
        config = dense_bake._config_with_vllm_block({"quantization_config": {}}, items, args)
        layers = config["quantization_config"]["non_routed_exl3"]["layers"]
        self.assertEqual(layers["model.layers.0.attn.fused_wqa_wkv"], {"bits": 3})
        self.assertEqual(layers["model.layers.0.attn.compressor.fused_wkv_wgate"], {"bits": 3})
        self.assertEqual(layers["model.layers.0.mlp.shared_experts.gate_up_proj"], {"bits": 5})
        self.assertEqual(layers["model.layers.0.attn.wo_a"], {"bits": 3})
        self.assertEqual(
            config["quantization_config"]["non_routed_dtype_policy"],
            "bf16_as_stored",
        )


class LocalPackTests(unittest.TestCase):
    @unittest.skipUnless(EXPERTS.is_dir(), "local D2 pack is not mounted")
    def test_local_d2_headers_have_expert_overlay(self):
        headers, weight_map = layer_overlay.source_headers(EXPERTS)
        self.assertGreater(len(headers), 100000)
        self.assertIn("layers.0.ffn.experts.0.w1.trellis", weight_map)
        layer_overlay.validate_source_index(EXPERTS, weight_map)

    @unittest.skipUnless(EXPERTS.is_dir(), "local D2 pack is not mounted")
    def test_offline_model_layout_uses_all_language_layers(self):
        result = dense_bake.dry_run_layout(SOURCE, EXPERTS)
        self.assertEqual(len(result["plan"]), 43)
        self.assertEqual(len(result["plan"][0][2]), 15)
        self.assertEqual(len(result["plan"][-1][2]), 20)
        self.assertTrue(result["model"].calibration_all_experts)

    @unittest.skipUnless(EXPERTS.is_dir(), "local D2 pack is not mounted")
    def test_source_fp8_wo_a_dequantizes_before_slice(self):
        import torch
        result = dense_bake.dry_run_layout(SOURCE, EXPERTS)
        config, model = result["config"], result["model"]
        config.stc.main.load_method = "python"
        for _, _, stc in config.stc.stcs:
            stc.load_method = "python"
        linear = model.find_module("layers.0.attn.wo_a.slice.0")
        try:
            linear.load(torch.device("cpu"))
            self.assertEqual(linear.quant_type, "fp16")
            self.assertEqual(tuple(linear.inner.weight.shape), (4096, 1024))
            self.assertTrue(torch.isfinite(linear.inner.weight).all().item())
            self.assertGreater(linear.inner.weight.abs().mean().item(), 0)
        finally:
            linear.unload()
            config.stc.close()

    @unittest.skipUnless(EXPERTS.is_dir(), "local D2 pack is not mounted")
    def test_cli_dry_run_merge_planning_is_write_free(self):
        with tempfile.TemporaryDirectory(prefix="t4-dense-cli-") as temp:
            work = Path(temp) / "work"
            work_q = work / "qtensors"
            work_q.mkdir(parents=True)
            entries = [
                ("layers.0.attn.wq_b.trellis", meta("I16", [1]), b"\x03\x00"),
                ("layers.0.attn.wq_b.suh", meta("F16", [1]), b"\x04\x00"),
                ("layers.0.attn.wq_b.svh", meta("F16", [1]), b"\x05\x00"),
                ("layers.0.attn.wq_b.mcg", meta("I32"), struct.pack("<i", dense_bake.MCG_MARKER)),
            ]
            write_fixture_shard(work_q / "layers.0.safetensors", entries)
            output = Path(temp) / "must-not-be-created"
            rc = dense_bake.main([
                "--src", str(SOURCE), "--experts", str(EXPERTS),
                "--work", str(work), "--merge", str(output), "--dry-run",
            ])
            self.assertEqual(rc, 0)
            self.assertFalse(output.exists())


class MergeFixtureTests(unittest.TestCase):
    def test_merge_plan_and_streaming_verify(self):
        with tempfile.TemporaryDirectory(prefix="t4-dense-test-") as temp:
            root = Path(temp)
            experts = root / "experts"
            work = root / "work"
            output = root / "merged"
            experts.mkdir()
            (work / "qtensors").mkdir(parents=True)
            (experts / "tokenizer.json").write_text("fixture\n", encoding="utf-8")

            source_entries = [
                ("layers.0.attn.wq_b.weight", meta("BF16", [1]), b"\x01\x00"),
                ("layers.0.norm.weight", meta("BF16", [1]), b"\x02\x00"),
            ]
            source_shard = experts / "model-00001.safetensors"
            write_fixture_shard(source_shard, source_entries)
            (experts / "config.json").write_text(
                json.dumps({"quantization_config": {"bits": 2}}) + "\n", encoding="utf-8"
            )
            (experts / "model.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {"total_size": 4},
                    "weight_map": {name: source_shard.name for name, _, _ in source_entries},
                }) + "\n", encoding="utf-8"
            )
            work_entries = [
                ("layers.0.attn.wq_b.trellis", meta("I16", [1]), b"\x03\x00"),
                ("layers.0.attn.wq_b.suh", meta("F16", [1]), b"\x04\x00"),
                ("layers.0.attn.wq_b.svh", meta("F16", [1]), b"\x05\x00"),
                ("layers.0.attn.wq_b.mcg", meta("I32"), struct.pack("<i", dense_bake.MCG_MARKER)),
            ]
            write_fixture_shard(work / "qtensors" / "layers.0.safetensors", work_entries)
            args = Namespace(
                bits=4, attn_bits=None, shared_bits=None,
                vllm_root="model", vllm_shared_prefix="mlp.shared_experts",
                dry_run=False, merge=str(output), work=str(work), experts=str(experts),
            )
            plan = dense_bake.plan_merge(work, experts, args)
            self.assertEqual(plan["dropped"], {"layers.0.attn.wq_b.weight"})
            self.assertEqual(len(plan["new_plan"]), 4)
            self.assertEqual(
                plan["config"]["quantization_config"]["non_routed_exl3"]["layers"],
                {"model.layers.0.attn.wq_b": {"bits": 4}},
            )

            dense_bake.merge(args)
            merged_index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertNotIn("layers.0.attn.wq_b.weight", merged_index["weight_map"])
            self.assertIn("layers.0.attn.wq_b.trellis", merged_index["weight_map"])
            self.assertTrue((output / "tokenizer.json").is_symlink())
            output_header_len, output_header = layer_overlay.read_header(
                output / "model-00001.safetensors"
            )
            start, end = output_header["layers.0.norm.weight"]["data_offsets"]
            with (output / "model-00001.safetensors").open("rb") as stream:
                stream.seek(8 + output_header_len + start)
                self.assertEqual(stream.read(end - start), b"\x02\x00")


if __name__ == "__main__":
    unittest.main()
