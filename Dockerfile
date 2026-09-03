# lna-lab/vllm-exl3:dsv4 — vLLM nightly dev337 (vision class) + exllamav3 1.4.5 (compiled ext, sm_120) + vllm-exl3 + recipe patches
FROM vllm/vllm-openai:nightly
ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
RUN pip install --no-deps "https://wheels.vllm.ai/27a94d1ce4e3fc100c4732439ccec10f8246a804/vllm-0.28.1rc1.dev337%2Bg27a94d1ce-cp38-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -2 \
 && python3 -c "import vllm,os;print(vllm.__version__); print(os.listdir(os.path.join(os.path.dirname(vllm.__file__),'models','deepseek_v4','nvidia')))"
COPY exllamav3-src /opt/exllamav3
RUN pip install ninja 2>&1 | tail -1 && pip install --no-build-isolation --no-deps -v /opt/exllamav3 2>&1 | grep -i -E "error|Successfully" | tail -5 \
 && python3 -c "import torch, exllamav3_ext; print('exllamav3_ext OK')"
COPY vllm-exl3 /opt/vllm-exl3
RUN pip install --no-deps /opt/vllm-exl3
COPY recipe/scripts /opt/recipe
RUN python3 /opt/recipe/patch_dsv4_stock028.py && python3 /opt/recipe/patch_dsv4_vl_stream_load.py && python3 /opt/recipe/patch_dsv4_vl_sm120_wide_swa.py
