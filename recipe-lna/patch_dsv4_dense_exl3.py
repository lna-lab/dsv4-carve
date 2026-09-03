#!/usr/bin/env python3
"""Patch the DSV4 vLLM fork to consume the dense EXL3 overlay at TP=8.

The image keeps vLLM's model files private, so this recipe is deliberately an
exact-anchor patch.  Every edit is idempotent, syntax-checked, and backed up as
``*.orig-lna`` before it is written.

Usage: python recipe-lna/patch_dsv4_dense_exl3.py [path/to/site-packages/vllm]
"""

from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path


# LNA-LAB: the common compressor class is instantiated for both attention and
# indexer compressors, so this one constructor anchor covers both instances.
COMPRESSOR_ANCHOR = (
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            quant_config=None,\n"
    "            disable_tp=True,\n"
)
COMPRESSOR_PATCH = (
    "            bias=False,\n"
    "            return_bias=False,\n"
    "            # LNA-LAB: dense EXL3 must receive the real quant config.\n"
    "            quant_config=vllm_config.quant_config,\n"
    "            disable_tp=True,\n"
)


OPROJ_SIGNATURE = (
    "    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor)"
    " -> torch.Tensor:\n"
)
OPROJ_BASELINE_ANCHOR = OPROJ_SIGNATURE + "        return deep_gemm_fp8_o_proj(\n"

# This is the exact prefix emitted by recipe/scripts/patch_dsv4_stock028.py;
# accepting it keeps this patch composable with the existing serving recipe.
OPROJ_STOCK028_ANCHOR = (
    OPROJ_SIGNATURE
    + "        if self.wo_a.weight.dtype != torch.float8_e4m3fn:\n"
    + "            # bf16 wo_a (packs that keep non-routed weights unquantized):\n"
    + "            # the fp8 einsum path needs block scales that do not exist, so\n"
    + "            # use the Triton inverse-RoPE + bf16 einsum reference instead.\n"
    + "            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (\n"
    + "                rocm_inv_rope_einsum,\n"
    + "            )\n"
    + "\n"
    + "            z = rocm_inv_rope_einsum(\n"
    + "                self.rotary_emb,\n"
    + "                o,\n"
    + "                positions,\n"
    + "                self.rope_head_dim,\n"
    + "                self.n_local_groups,\n"
    + "                self.o_lora_rank,\n"
    + "                self.wo_a,\n"
    + "            )\n"
    + "            return self.wo_b(z.flatten(1))\n"
    + "        return deep_gemm_fp8_o_proj(\n"
)


OPROJ_COMMON = (
    OPROJ_SIGNATURE
    # LNA-LAB: rank-local EXL3 wo_a is a regular one-group LinearEXL3 call.
    + "        exl3_linears = getattr(self.wo_a, \"_exl3_linears\", None)\n"
    + "        if exl3_linears is not None:\n"
    + "            if self.n_local_groups != 1:\n"
    + "                raise NotImplementedError(\n"
    + "                    \"EXL3 wo_a requires TP=8 (one local group); \"\n"
    + "                    f\"got n_local_groups={self.n_local_groups}\"\n"
    + "                )\n"
    + "            if len(exl3_linears) != 1 or exl3_linears[0] is None:\n"
    + "                raise RuntimeError(\n"
    + "                    \"EXL3 wo_a has no rank-local LinearEXL3 instance\"\n"
    + "                )\n"
    + "            # LNA-LAB: use the same inverse GPT-J RoPE convention as the\n"
    + "            # Triton rocm_inv_rope_einsum reference before the EXL3 GEMM.\n"
    + "            try:\n"
    + "                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (\n"
    + "                    _fused_inverse_rope_gptj,\n"
    + "                )\n"
    + "                o_ref = _fused_inverse_rope_gptj(\n"
    + "                    o, positions, self.rotary_emb.cos_sin_cache,\n"
    + "                    self.rope_head_dim,\n"
    + "                )\n"
    + "            except ImportError:\n"
    + "                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (\n"
    + "                    _apply_inv_rope_ref,\n"
    + "                )\n"
    + "                o_ref = _apply_inv_rope_ref(\n"
    + "                    self.rotary_emb, o, positions, self.rope_head_dim\n"
    + "                )\n"
    + "            b = o_ref.shape[0]\n"
    + "            exl3_linear = exl3_linears[0]\n"
    + "            z = exl3_linear.forward(\n"
    + "                o_ref.reshape(b, -1).contiguous().half(),\n"
    + "                {},\n"
    + "                out_dtype=torch.float32,\n"
    + "            )\n"
    + "            z = z.to(torch.bfloat16).reshape(b, 1, self.o_lora_rank)  # LNA-LAB: EXL3 GEMM emits fp32/fp16\n"
    + "            return self.wo_b(z.flatten(1))\n"
    + "        wo_a_weight = getattr(self.wo_a, \"weight\", None)\n"
    + "        if wo_a_weight is None:\n"
    + "            raise RuntimeError(\n"
    + "                \"DSV4 wo_a has neither dense weight nor EXL3 linears\"\n"
    + "            )\n"
    + "        if wo_a_weight.dtype != torch.float8_e4m3fn:\n"
    + "            # LNA-LAB: preserve the existing bf16 reference path.\n"
    + "            from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (\n"
    + "                rocm_inv_rope_einsum,\n"
    + "            )\n"
    + "\n"
    + "            z = rocm_inv_rope_einsum(\n"
    + "                self.rotary_emb,\n"
    + "                o,\n"
    + "                positions,\n"
    + "                self.rope_head_dim,\n"
    + "                self.n_local_groups,\n"
    + "                self.o_lora_rank,\n"
    + "                self.wo_a,\n"
    + "            )\n"
    + "            return self.wo_b(z.flatten(1))\n"
    + "        return deep_gemm_fp8_o_proj(\n"
)


# LNA-LAB: AutoWeightsLoader maps the pre-fusion bake names to wo_a.tensors;
# integer ids are then consumed by Exl3LinearMethod as the physical TP rank.
WO_A_MAPPING_ANCHOR = (
    '            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),\n'
    '            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),\n'
    "        ]\n"
)
WO_A_MAPPING_PATCH = (
    '            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),\n'
    '            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),\n'
    "            # LNA-LAB: each pre-fusion wo_a slice is one rank-local shard.\n"
    '            ("attn.wo_a", "attn.wo_a.slice.0", 0),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.1", 1),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.2", 2),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.3", 3),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.4", 4),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.5", 5),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.6", 6),\n'
    '            ("attn.wo_a", "attn.wo_a.slice.7", 7),\n'
    "        ]\n"
)


def patch_file(path: Path, anchor: str, patched: str, expect: int) -> str:
    text = path.read_text(encoding="utf-8")
    have = text.count(patched)
    if have == expect:
        return "already patched"
    if have:
        raise SystemExit(f"{path}: partially patched ({have}/{expect}); refusing to guess")
    count = text.count(anchor)
    if count != expect:
        raise SystemExit(f"{path}: expected {expect} anchor(s), found {count}")
    backup = path.with_name(path.name + ".orig-lna")
    if not backup.exists():
        shutil.copy2(path, backup)
    new_text = text.replace(anchor, patched)
    ast.parse(new_text)
    path.write_text(new_text, encoding="utf-8")
    return f"patched x{expect} (backup {backup.name})"


def patch_flashinfer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "exl3_linears = getattr(self.wo_a, \"_exl3_linears\", None)"
    if text.count(marker) == 2:
        return "already patched"
    if marker in text:
        raise SystemExit(f"{path}: partially patched; refusing to guess")
    baseline_count = text.count(OPROJ_BASELINE_ANCHOR)
    stock_count = text.count(OPROJ_STOCK028_ANCHOR)
    if baseline_count == 2 and stock_count == 0:
        new_text = text.replace(OPROJ_BASELINE_ANCHOR, OPROJ_COMMON)
        expected = "baseline"
    elif stock_count == 2 and baseline_count == 0:
        new_text = text.replace(OPROJ_STOCK028_ANCHOR, OPROJ_COMMON)
        expected = "stock028-patched"
    else:
        raise SystemExit(
            f"{path}: expected two matching _o_proj methods; "
            f"baseline={baseline_count}, stock028={stock_count}"
        )
    backup = path.with_name(path.name + ".orig-lna")
    if not backup.exists():
        shutil.copy2(path, backup)
    ast.parse(new_text)
    path.write_text(new_text, encoding="utf-8")
    return f"patched x2 ({expected}; backup {backup.name})"


def main() -> int:
    if len(sys.argv) > 1:
        root = Path(sys.argv[1]).expanduser().resolve()
    else:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    model_dir = root / "models" / "deepseek_v4"
    nvidia_dir = model_dir / "nvidia"
    targets = {
        "compressor.py": model_dir / "compressor.py",
        "flashinfer_sparse.py": nvidia_dir / "flashinfer_sparse.py",
        "model.py": nvidia_dir / "model.py",
    }
    missing = [str(path) for path in targets.values() if not path.is_file()]
    if missing:
        print("missing DSV4 vLLM file(s):", ", ".join(missing), file=sys.stderr)
        return 2

    try:
        print("compressor.py:", patch_file(
            targets["compressor.py"], COMPRESSOR_ANCHOR, COMPRESSOR_PATCH, 1
        ))
        print("flashinfer_sparse.py:", patch_flashinfer(targets["flashinfer_sparse.py"]))
        print("model.py:", patch_file(
            targets["model.py"], WO_A_MAPPING_ANCHOR, WO_A_MAPPING_PATCH, 1
        ))
    except (OSError, SystemExit) as exc:
        print(exc, file=sys.stderr)
        return 1
    print("DSV4_DENSE_EXL3_PATCH_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
