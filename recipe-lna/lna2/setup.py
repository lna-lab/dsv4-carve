from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
ext_include = os.environ.get("EXL3_EXT_INCLUDE")
if ext_include:
    ext_include = Path(ext_include)
else:
    try:
        import exllamav3
        ext_include = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
    except Exception:
        ext_include = None
include_dirs = [str(ROOT / "csrc")]
if ext_include and ext_include.is_dir():
    include_dirs.append(str(ext_include))

setup(
    name="vllm-exl3-native",
    ext_modules=[
        CUDAExtension(
            name="vllm_exl3_c",
            sources=[str(ROOT / "csrc" / "bindings.cpp"),
                     str(ROOT / "csrc" / "exl3_gemv.cu"),
                     str(ROOT / "csrc" / "p2b_batched.cu"),
                     str(ROOT / "csrc" / "p2b_moe.cu"),
                     str(ROOT / "csrc" / "exl3_gemm.cu"),
                     str(ROOT / "csrc" / "lna_moe_decode.cu"),
                     str(ROOT / "csrc" / "lna_moe_ticket.cu")],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # LNA_NVCC_EXTRA lets the F1 gate-E sweep rebuild tile/stage
                # variants without editing the source between runs.
                "nvcc": ["-O3", "-std=c++17", "-Xptxas", "-v"] + os.environ.get(
                    "LNA_NVCC_EXTRA", "").split(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
