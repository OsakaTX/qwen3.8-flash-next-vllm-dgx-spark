# builds qwen4fix.so inside the eugr/spark-vllm container on GB10
# usage: python3 build_qwen4fix.py
import os
import shutil

import torch
from torch.utils.cpp_extension import load

BUILD_DIR = "/qwen4fix/build"
os.makedirs(BUILD_DIR, exist_ok=True)

load(
    name="qwen4fix",
    sources=[
        "/qwen4fix/qwen4fix_bindings.cpp",
        "/clone/csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu",
    ],
    extra_include_paths=[
        "/clone/csrc/libtorch_stable",
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
shutil.copy(so, "/qwen4fix/qwen4fix.so")
print("staged: /qwen4fix/qwen4fix.so")
