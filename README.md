# Qwen3.8-Flash-Next (NVFP4) on 2x DGX Spark with vLLM

A working recipe for serving `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135 GB, qwen4_exp /
Qwen4ExpForConditionalGeneration) across two DGX Spark GB10 nodes with vLLM, tensor
parallel 2 over the ConnectX fabric, including MTP speculative decoding and full
torch.compile + CUDA graphs.

As of 2026-08-26 (the model's release day), vLLM support for this architecture exists
only as an open PR ([vllm#53896](https://github.com/vllm-project/vllm/pull/53896)) and
no released vLLM build can load the checkpoint. This recipe runs that PR's Python on
top of a current GB10 nightly image without a source build, plus two small local
patches the RadixArk checkpoint needs. 

For an SGLang route to the same checkpoint, see
[tonyd2wild/qwen3.8-flash-next-nvfp4-dgx-spark](https://github.com/tonyd2wild/qwen3.8-flash-next-nvfp4-dgx-spark),
whose unified-memory page-cache warning saved this project real pain.

## Why two nodes

The checkpoint is ~135 GB against a single Spark's ~121 GB unified pool, and the model
carries a ~48 GB FP8 n-gram lookup table (PLE) on top of the NVFP4 experts. TP=2
shards it to ~62 GB of weights per node (the PLE shards across ranks as a
VocabParallelEmbedding).

## The stack

| Layer | What | Why |
|---|---|---|
| Image | `eugr/spark-vllm:latest` (vLLM main, built for GB10/sm_121a) | current binaries incl. FlashInfer |
| Python | PR #53896 branch (`peakcrosser7/vllm@release/qwen38next`) via `PYTHONPATH` overlay | registers qwen4_exp |
| Patch 1 | `patches/` FP8-PLE gate | RadixArk is hybrid: modelopt-NVFP4 experts + FP8 PLE; the PR only enables its FP8 PLE loader for all-FP8 checkpoints |
| Patch 2 | `qwen4fix/` rebuilt GDN-MTP kernel | the PR changes one CUDA kernel (sigmoid output gate); nightly binaries predate it |

## Quick start

1. Pull the image and clone the PR branch on the head node; copy the image's compiled
   artifacts into the clone so the PR Python rides on the nightly binaries:

   ```bash
   docker pull eugr/spark-vllm:latest
   git clone --depth 1 -b release/qwen38next https://github.com/peakcrosser7/vllm ~/qwen38next-vllm
   docker create --name tmp eugr/spark-vllm:latest
   docker cp tmp:/usr/local/lib/python3.12/dist-packages/vllm ~/image-vllm
   docker rm tmp
   rsync -a --ignore-existing ~/image-vllm/ ~/qwen38next-vllm/vllm/
   ```

2. Apply the two Python patches: `python3 patches/apply_patches.py ~/qwen38next-vllm`

3. Build the sigmoid-gate MTP kernel (`qwen4fix/build_qwen4fix.py`) to enable
   the fused CUDA GDN kernel. Runs INSIDE the container (not on the host):

   ```bash
   docker run --rm --gpus all --entrypoint python3 \
     -v ~/qwen38next-vllm:/clone -v $(pwd)/qwen4fix:/qwen4fix \
     eugr/spark-vllm:latest /qwen4fix/build_qwen4fix.py
   ```

4. Download the checkpoint once, copy to the second node over your internal fabric,
   then launch with `launch-qwen38next.sh` - worker node first, then head.

## Results (2x GB10, TP=2, MTP n=3, 65K ctx, warmed, RDMA verified)

| concurrency | aggregate tok/s | per user |
|---|---|---|
| 1 | 41-44 | 41-44 |
| 4 | ~103 | ~26 |
| 8 | 162-166 | ~20 |

Those are whole-request wall-clock (prefill included) on prose prompts - the
conservative read. Measured decode-only on code/structured output (high MTP
acceptance), single-stream peaks at ~69 tok/s.

Three things these numbers depend on, all defaults in the launch script:

1. **RDMA device passthrough** (`--device /dev/infiniband` + memlock ulimit). The
   NCCL env vars alone do nothing - without the device nodes NCCL silently falls
   back to TCP sockets and every number above drops 40-45% (we ran a full day of
   benchmarks in that degraded mode before catching it via hardware counters; see
   TROUBLESHOOTING for the verification one-liner).
2. **FlashInfer runtime autotune ON** (never pass `--no-enable-flashinfer-autotune`).
   It tunes progressively - benchmark only after a few minutes of sustained load.
3. **MTP n=3.** Swept warm under RDMA: n=3 >= n=4 at every concurrency (n=4 only
   looked better under the TCP-degraded fabric, where longer speculation amortized
   comms latency); n=5 crashes at startup (see TROUBLESHOOTING).
- warm boot to serving: ~2.5 min weight load + ~44 s engine init (compile cached)
- temp-0 sanity, tool calling (`qwen3_coder` parser) and reasoning extraction
  (`qwen3` parser, `reasoning` field) all verified

## The one thing you must not skip

**`triton.autotune_at_compile_time: false`** (already in the launch script's
`--compilation-config`). Without it, inductor benchmarks Triton kernel configs on the
GPU from compile-worker subprocesses during torch.compile, and on GB10 unified memory
this hard-locks the machine - not a crash, a full freeze: pingable, ssh dead, no
kernel log, only a power pull recovers it. It cost three power pulls to isolate.
Details in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Credits

- [peakcrosser7](https://github.com/peakcrosser7) - the vLLM qwen4_exp implementation (PR #53896)
- [RadixArk](https://huggingface.co/RadixArk) - the NVFP4 quantization
- [eugr](https://github.com/eugr/spark-vllm-docker) - GB10 nightly images
- [tonyd2wild](https://github.com/tonyd2wild) - the page-cache warning and the SGLang reference numbers

MIT license. Not affiliated with any of the above.
