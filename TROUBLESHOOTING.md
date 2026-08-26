# Troubleshooting / field notes (DGX Spark GB10, 2026-08-26)

All of these were hit for real while bringing the stack up. Symptoms first, so you
can grep your way here.

## Throughput 40-45% lower than the README table (NCCL on TCP without telling you)

If the container is not launched with `--device /dev/infiniband` and an unlimited
memlock ulimit, NCCL cannot open the RDMA devices and SILENTLY falls back to TCP
sockets over the same interface - the NCCL_IB_* env vars alone change nothing, and
nothing errors. Per-layer TP allreduces then dominate decode. Verify RDMA is live
(run on the head node during load):

    c1=$(cat /sys/class/infiniband/<hca>/ports/1/counters/port_xmit_data); sleep 20
    c2=$(cat /sys/class/infiniband/<hca>/ports/1/counters/port_xmit_data)
    echo "$(( (c2-c1)*4/1048576 )) MB RDMA in 20s"

Zero during active inference = you are on sockets. The launch script passes the
devices and sets `NCCL_DEBUG=WARN` so a fallback at least prints a warning.

## Hard freeze during torch.compile (pingable, ssh dead, needs power pull)

**Symptom:** both nodes lock at the same instant, 1-3 minutes into the torch.compile
phase (shortly after `Not enough SMs to use max_autotune_gemm mode` appears in the
log). Machine answers ping, sshd times out on banner exchange, existing sessions
freeze, nothing in the previous boot's journal, softlockup watchdog never fires.
Physical power cycle is the only recovery.

**Cause:** inductor's compile-time autotuning benchmarks Triton kernel candidates on
the GPU from compile-worker subprocesses. Each worker has its own CUDA context, next
to a ~62 GB main-process context, on a unified-memory SoC. It reproduced with 20 and
with 4 compile workers, with warm and with cold page cache, at 0.70-0.80 GPU memory
utilization, with combo-kernel benchmarking on and off. Runtime autotuning in the
main process has never caused a problem.

**Fix:** defer autotune to first execution:

```
--compilation-config '{"inductor_compile_config":{"triton.autotune_at_compile_time":false}}'
```

With this one key, full compile plus FULL and PIECEWISE CUDA graph capture work
reliably. Side effect: `Triton kernel JIT compilation during inference` warnings on
the first request that hits each new shape/temperature/batch size - a one-time
latency spike per shape per boot. Mount `/root/.triton` to a host path to persist
those too.

Also persist the vLLM compile cache (`-v <host>/compile-cache:/root/.cache/vllm`) so
warm boots skip compilation entirely (~44 s engine init instead of ~2 min).

## Hard freeze ~15-20 min into weight load

Different freeze, same look. On unified memory a warm page cache (for example right
after downloading or rsyncing the 135 GB checkpoint) can starve the GPU allocator
partway through model load. Drop caches immediately before every launch:

```
sync && echo 3 > /proc/sys/vm/drop_caches
```

Credit to tonyd2wild's recipe for documenting this one. The launch script does it
automatically. Note this is hygiene, not the compile fix above - they are separate
failure modes and you want both mitigations.

## `ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'`

The RadixArk checkpoint is hybrid: modelopt-NVFP4 routed experts plus an FP8 PLE
(n-gram table) with one global scale. PR #53896 only activates its FP8 PLE loader
when the whole checkpoint quant config is FP8. The patch in `patches/` adds an env
gate: set `QWEN4_PLE_FORCE_FP8=1` (the launch script does).

## Garbled output or schema errors with MTP enabled

The PR modifies `fused_gdn_decode_post_conv_mtp` (adds a sigmoid output-gate variant;
qwen4_exp uses sigmoid where earlier Qwen GDN models used silu). Prebuilt nightly
binaries predate that change, so the compiled op is silu-only with the old signature.
Two options:

1. `VLLM_GDN_DECODE_KERNEL=triton` - vLLM's Triton GDN path has correct sigmoid
   gating. Zero build effort, measured identical single-stream throughput.
2. Build the PR's kernel standalone (`qwen4fix/`) and keep the fused CUDA path.
   Verified equivalent: draft acceptance 50.9% vs 48.4% on the Triton path, same
   per-position curve.

## Kernel build fails: `aoti_torch_get_current_cuda_stream is undefined`

The stable-ABI shim headers hide their CUDA declarations behind `#ifdef USE_CUDA`.
Add `-DUSE_CUDA` to both C++ and nvcc flags. Also needed:
`-DVLLM_ENABLE_FUSED_GDN_DECODE` and `-DVLLM_ENABLE_FUSED_KDA_DECODE` (the ops.h
declaration sits behind the KDA guard, not the GDN one). See
`qwen4fix/build_qwen4fix.py` for the complete working invocation.

## Crash at startup: `QSA ring capacity 12 must divide the attention block size`

You set `num_speculative_tokens` to 5 (or higher). The QSA circular-buffer capacity is
`compress_ratio * ceil((compress_ratio + n_spec) / compress_ratio)` - with this
model's compress ratio of 4, n=3 and n=4 give capacity 8 (fits the auto-computed
attention block size), but n=5 gives 12, which does not divide it, and the engine
refuses to start. Stick to n<=4. (Arguably a gap in the PR's block-size LCM logic;
reported upstream.)

Sweep results (warmed, identical protocol per arm, RDMA active): n=3 matches or
beats n=4 at every concurrency (c1 ~42 both, c8 164 vs 150 aggregate). Note: with
a DEGRADED fabric (NCCL on TCP sockets), the sweep inverts and n=4 wins single
stream - longer speculation amortizes comms latency. If your sweep says n=4, check
your RDMA before believing it. n=3 is the shipped default.

## Scheduler warning about max_num_scheduled_tokens

With speculative decoding on, add `--max-num-batched-tokens 8192` (or higher) so
prefill is not chunked at 2048 tokens. Decode is unaffected either way.

## Performance expectations

Two regimes, and it is easy to measure the wrong one:

- WITHOUT FlashInfer runtime autotune: single-stream decode plateaus around 23 tok/s
  and is invariant to eager/compiled/Triton-GDN/fused-GDN choices.
- WITH autotune (the launch script's default): the server tunes progressively over
  the first minutes of traffic and climbs to roughly 30 tok/s single-stream. Do not
  benchmark until repeated runs stop improving, and vary your prompts so prefix
  caching does not flatter the numbers.

Concurrency scales well (57-60 tok/s aggregate at 4 streams, warm). Beyond that,
the levers that matter are speculative token count and anything that reduces bytes
moved per token.
