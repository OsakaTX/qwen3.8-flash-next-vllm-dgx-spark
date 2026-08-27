# builds qwen4fix.so INSIDE the eugr/spark-vllm container on GB10 - do not run
# on the host. See README step 3 for the docker run invocation. Paths can be
# overridden with QWEN4FIX_DIR / CLONE_DIR env vars.
import os
import shutil
import sys

OUT_DIR = os.environ.get("QWEN4FIX_DIR", "/qwen4fix")
CLONE_DIR = os.environ.get("CLONE_DIR", "/clone")

if not (os.path.isdir(OUT_DIR) and os.path.isdir(os.path.join(CLONE_DIR, "csrc"))):
    sys.exit(
        "ERROR: expected container mounts not found.\n"
        f"  needs: {OUT_DIR} (this directory) and {CLONE_DIR} (the PR clone)\n"
        "This script runs INSIDE the vLLM container, not on the host. From the\n"
        "directory containing your PR clone and this qwen4fix folder:\n\n"
        "  docker run --rm --gpus all --entrypoint python3 \\\n"
        "    -v $(pwd)/qwen38next-vllm:/clone -v $(pwd)/qwen4fix:/qwen4fix \\\n"
        "    eugr/spark-vllm:latest /qwen4fix/build_qwen4fix.py\n"
    )

import torch
from torch.utils.cpp_extension import load

BUILD_DIR = os.path.join(OUT_DIR, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

load(
    name="qwen4fix",
    sources=[
        os.path.join(OUT_DIR, "qwen4fix_bindings.cpp"),
        os.path.join(CLONE_DIR, "csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu"),
    ],
    extra_include_paths=[
        os.path.join(CLONE_DIR, "csrc/libtorch_stable"),
    ],
    extra_cflags=["-O3", "-DVLLM_ENABLE_FUSED_GDN_DECODE", "-DVLLM_ENABLE_FUSED_KDA_DECODE", "-DUSE_CUDA"],
    extra_cuda_cflags=[
        "-O3",
        "-DVLLM_ENABLE_FUSED_GDN_DECODE", "-DVLLM_ENABLE_FUSED_KDA_DECODE",
        "-DUSE_CUDA",
        "--expt-relaxed-constexpr",
    ],
    build_directory=BUILD_DIR,
    is_python_module=False,
    verbose=True,
)

# verify registration and stage the artifact at a stable path
assert hasattr(torch.ops._qwen4fix, "fused_gdn_decode_post_conv_mtp"), "op not registered"
print("op registered:", torch.ops._qwen4fix.fused_gdn_decode_post_conv_mtp)
so = os.path.join(BUILD_DIR, "qwen4fix.so")
shutil.copy(so, os.path.join(OUT_DIR, "qwen4fix.so"))
print("staged:", os.path.join(OUT_DIR, "qwen4fix.so"))
