from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "recipe-lna" / "patch_dsv4_dense_exl3.py"


class DenseExl3RecipeTests(unittest.TestCase):
    def test_patch_is_exact_and_idempotent_on_dev337_shape(self):
        # Import only the recipe constants; the fixture mirrors the three
        # private image files without requiring Docker or a CUDA runtime.
        import importlib.util

        spec = importlib.util.spec_from_file_location("dense_recipe", SCRIPT)
        recipe = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(recipe)

        with tempfile.TemporaryDirectory(prefix="dense-exl3-recipe-") as temp:
            root = Path(temp) / "vllm"
            model_dir = root / "models" / "deepseek_v4"
            nvidia_dir = model_dir / "nvidia"
            nvidia_dir.mkdir(parents=True)
            (model_dir / "compressor.py").write_text(
                "class C:\n"
                "    def __init__(self, vllm_config):\n"
                "        self.fused = MergedColumnParallelLinear(\n"
                + recipe.COMPRESSOR_ANCHOR
                + "        )\n",
                encoding="utf-8",
            )
            flash = (
                "class A:\n"
                + recipe.OPROJ_BASELINE_ANCHOR
                + "            o,\n"
                + "        )\n"
                + "\n"
                "class B:\n"
                + recipe.OPROJ_BASELINE_ANCHOR
                + "            o,\n"
                + "        )\n"
            )
            (nvidia_dir / "flashinfer_sparse.py").write_text(flash, encoding="utf-8")
            (nvidia_dir / "model.py").write_text(
                "def mappings():\n"
                "    stacked_params_mapping = [\n"
                + recipe.WO_A_MAPPING_ANCHOR
                + "    return stacked_params_mapping\n",
                encoding="utf-8",
            )

            first = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("DSV4_DENSE_EXL3_PATCH_OK", first.stdout)
            for path in (
                model_dir / "compressor.py",
                nvidia_dir / "flashinfer_sparse.py",
                nvidia_dir / "model.py",
            ):
                self.assertTrue(path.with_name(path.name + ".orig-lna").is_file())
            self.assertEqual(
                (nvidia_dir / "flashinfer_sparse.py")
                .read_text(encoding="utf-8")
                .count("exl3_linears = getattr(self.wo_a"),
                2,
            )
            self.assertIn('"attn.wo_a.slice.7"', (nvidia_dir / "model.py").read_text())

            second = subprocess.run(
                [sys.executable, str(SCRIPT), str(root)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("already patched", second.stdout)


if __name__ == "__main__":
    unittest.main()
