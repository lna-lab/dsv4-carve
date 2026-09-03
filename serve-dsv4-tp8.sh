#!/bin/bash
# DSV4-Flash-Vision EXL3 MixedK on 8 experiment GPUs (TP=8) via lna-lab/vllm-exl3:dsv4. YUKI 2026-09-03
# env: GPUS (default 0,1,2,3,5,7,8,9) PORT(8899) MAXLEN(65536) SPEC ('' or json) UTIL(0.90) NAME(dsv4)
set -u
GPUS=${GPUS:-0,1,2,3,5,7,8,9}; PORT=${PORT:-8899}; MAXLEN=${MAXLEN:-65536}; UTIL=${UTIL:-0.90}; NAME=${NAME:-dsv4}
TP=$(echo $GPUS | tr ',' '\n' | wc -l)
M=${MODEL:-/run/media/tonoken3/DATA1/DSV4-Flash-Vision-EXL3-MixedK}
EXTRA=()
[[ -n "${SPEC:-}" ]] && EXTRA+=(--speculative-config "$SPEC")
docker rm -f $NAME >/dev/null 2>&1
docker run --ulimit core=0 --cap-add=SYS_PTRACE -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=${SHARED_STREAM_OFF:-1} -v /run/media/tonoken3/DATA1/.tmp/dsv4-cache/root-cache:/root/.cache -v /run/media/tonoken3/DATA1/.tmp/dsv4-cache/tilelang:/root/.tilelang ${EXT_SO:+-v $EXT_SO:/usr/local/lib/python3.12/dist-packages/$(basename ${EXT_SO:-x})} ${PLUGIN_SRC:+-v $PLUGIN_SRC:/usr/local/lib/python3.12/dist-packages/vllm_exl3} -d --name $NAME --gpus "\"device=$GPUS\"" --shm-size=16g --ipc=host \
  -e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 -e EXLLAMAV3_TUNE_CACHE=/lab/exl3-tune-cache ${AUX_STREAMS:+-e LNA_DSV4_AUX_STREAMS=$AUX_STREAMS} ${NCCL_EXTRA:-} -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  -p 127.0.0.1:$PORT:8000 -v /run/media/tonoken3/DATA1:/run/media/tonoken3/DATA1 -v /run/media/tonoken3/DATA1/vllm-exl3-lab:/lab \
  lna-lab/vllm-exl3:${IMAGE:-dsv4} \
  $M --served-model-name DSV4-Flash --tensor-parallel-size $TP --quantization exl3 \
  --max-model-len $MAXLEN --max-num-seqs ${SEQS:-4} --max-num-batched-tokens ${BT:-2048} \
  --kv-cache-dtype ${KVDT:-fp8} --gpu-memory-utilization $UTIL ${EAGER:+--enforce-eager} --compilation-config "${COMPILE:-{\"cudagraph_capture_sizes\":[1,2,4]\}}" --disable-custom-all-reduce \
  --no-enable-prefix-caching --trust-remote-code ${PROFILE:+--profiler-config "$PROFILE"} \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 "${EXTRA[@]}"
echo "container $NAME on :$PORT (TP=$TP, spec=${SPEC:-off}); docker logs -f $NAME"
