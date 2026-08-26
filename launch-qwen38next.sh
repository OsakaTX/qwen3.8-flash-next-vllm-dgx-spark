#!/bin/bash
# launch-qwen38next.sh <0|1>   0 = head node (serves :8010), 1 = worker (START WORKER FIRST)
# Qwen3.8-Flash-Next-NVFP4 on 2x DGX Spark GB10, vLLM TP=2.
# See README.md for stack assembly and TROUBLESHOOTING.md for every non-obvious line here.
set -e
RANK=${1:?usage: launch-qwen38next.sh <0|1>}

# ---- EDIT THESE FOR YOUR CLUSTER ----
HEAD_IP=10.100.100.2        # head node fabric IP
WORKER_IP=10.100.100.4      # worker node fabric IP
FABRIC_IF=enp1s0f1np1       # fabric interface name (ip addr)
FABRIC_HCA=rocep1s0f1       # RoCE device (ls /sys/class/infiniband)
GID_INDEX=3                 # RoCE GID index for the fabric subnet
MODEL_DIR=$HOME/models/Qwen38-Flash-Next-NVFP4
OVERLAY_DIR=$HOME/qwen38next-vllm          # PR clone + image binaries (README step 1)
QWEN4FIX_DIR=$HOME/qwen4fix                # rebuilt MTP kernel (optional, see below)
CACHE_DIR=$HOME/qwen38-compile-cache       # persistent torch.compile cache
TRITON_CACHE=$HOME/qwen38-triton-cache     # persistent runtime-JIT kernel cache
# -------------------------------------

if [ "$RANK" = "0" ]; then
  HOST_IP=$HEAD_IP
  MODE_ARGS="--host 0.0.0.0 --port 8010"
else
  HOST_IP=$WORKER_IP
  MODE_ARGS="--headless"
fi

# MANDATORY on unified memory: a warm page cache starves the GPU allocator
# mid-load and hard-locks the node (TROUBLESHOOTING.md).
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

mkdir -p "$CACHE_DIR" "$TRITON_CACHE"
sudo docker rm -f qwen38next 2>/dev/null || true
sudo docker run -d --name qwen38next \
  --network host --ipc host --gpus all --restart no \
  --device /dev/infiniband --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -e NCCL_DEBUG=WARN \
  -v "$OVERLAY_DIR":/overlay \
  -v "$MODEL_DIR":/models/qwen38 \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$TRITON_CACHE":/root/.triton \
  -v "$QWEN4FIX_DIR":/qwen4fix \
  -e QWEN4FIX_SO=/qwen4fix/qwen4fix.so \
  -e PYTHONPATH=/overlay \
  -e VLLM_HOST_IP=$HOST_IP \
  -e QWEN4_PLE_FORCE_FP8=1 \
  -e NCCL_SOCKET_IFNAME=$FABRIC_IF \
  -e GLOO_SOCKET_IFNAME=$FABRIC_IF \
  -e NCCL_IB_HCA=$FABRIC_HCA \
  -e NCCL_IB_GID_INDEX=$GID_INDEX \
  -e VLLM_GDN_DECODE_KERNEL=cuda \
  --entrypoint vllm \
  eugr/spark-vllm:latest serve /models/qwen38 \
  --served-model-name qwen3.8-flash-next \
  --tensor-parallel-size 2 --nnodes 2 --node-rank $RANK \
  --master-addr $HEAD_IP --master-port 29501 \
  --gpu-memory-utilization 0.77 \
  --compilation-config '{"inductor_compile_config":{"triton.autotune_at_compile_time":false,"combo_kernels":false,"benchmark_combo_kernel":false,"enable_auto_functionalized_v2":false}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --max-model-len 65536 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  $MODE_ARGS
echo "launched qwen38next rank $RANK"

# No rebuilt kernel? Skip the qwen4fix mount + QWEN4FIX_SO and set
# VLLM_GDN_DECODE_KERNEL=triton instead (same measured speed, zero build effort).
# NOT wanting MTP at all? Drop --speculative-config; then kernel=cuda is fine as-is.
# First boot compiles for a few minutes; later boots hit CACHE_DIR and start in ~45 s.
