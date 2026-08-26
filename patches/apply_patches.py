# Applies the two Python patches this recipe needs to a PR#53896 checkout
# (peakcrosser7/vllm branch release/qwen38next, commit d4d0f73).
# usage: python3 apply_patches.py /path/to/qwen38next-vllm
import ast
import re
import sys

root = sys.argv[1].rstrip("/")

# --- patch 1: allow forcing the FP8 PLE loader on hybrid-quant checkpoints ---
# The RadixArk NVFP4 checkpoint stores the PLE (n-gram table) as FP8 shards with a
# global weight_scale, but its model-level quant config is ModelOpt, so the PR's
# Fp8Config-only gate never activates and loading fails on ngram_embedding.weight_scale.
p = root + "/vllm/models/qwen4_exp/nvidia/ple_layer.py"
src = open(p).read()
anchor = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):'''
patched = '''    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    # RECIPE PATCH: hybrid checkpoints (modelopt NVFP4 experts + FP8 PLE shards,
    # e.g. RadixArk/Qwen3.8-Flash-Next-NVFP4) need the FP8 PLE path even though
    # the model-level quant config is not Fp8Config. QWEN4_PLE_FORCE_FP8=1 forces it.
    import os
    if os.environ.get("QWEN4_PLE_FORCE_FP8") == "1":
        return Qwen4ExpPLEFp8EmbeddingMethod()

    if not isinstance(quant_config, Fp8Config):'''
if "QWEN4_PLE_FORCE_FP8" in src:
    print("patch 1: already applied")
else:
    assert src.count(anchor) == 1, "patch 1 anchor not found (wrong branch/commit?)"
    src = src.replace(anchor, patched)
    ast.parse(src)
    open(p, "w").write(src)
    print("patch 1: applied (ple_layer.py FP8 gate)")

# --- patch 2: dispatch the GDN MTP fused op to a locally rebuilt kernel ---
# Only needed for MTP + VLLM_GDN_DECODE_KERNEL=cuda. Prebuilt images predate the
# PR's sigmoid-gate kernel change; qwen4fix/ rebuilds it as torch.ops._qwen4fix.
p = root + "/vllm/_custom_ops.py"
src = open(p).read()
if "_qwen4fix" in src:
    print("patch 2: already applied")
else:
    if not re.search(r"^import os$", src, re.M):
        assert src.count("\nimport torch\n") == 1
        src = src.replace("\nimport torch\n", "\nimport os\n\nimport torch\n")
    anchor = """    torch.ops._C.fused_gdn_decode_post_conv_mtp(
        mixed_qkv,"""
    patched = """    # RECIPE PATCH: use the locally rebuilt PR#53896 kernel (sigmoid output
    # gate) when QWEN4FIX_SO is set; prebuilt _C op is the older silu-only build.
    if os.environ.get("QWEN4FIX_SO") and not getattr(
        fused_gdn_decode_post_conv_mtp, "_qwen4fix_loaded", False
    ):
        torch.ops.load_library(os.environ["QWEN4FIX_SO"])
        fused_gdn_decode_post_conv_mtp._qwen4fix_loaded = True
    _ops_ns = (
        torch.ops._qwen4fix
        if getattr(fused_gdn_decode_post_conv_mtp, "_qwen4fix_loaded", False)
        else torch.ops._C
    )
    _ops_ns.fused_gdn_decode_post_conv_mtp(
        mixed_qkv,"""
    assert src.count(anchor) == 1, "patch 2 anchor not found (wrong branch/commit?)"
    src = src.replace(anchor, patched)
    ast.parse(src)
    open(p, "w").write(src)
    print("patch 2: applied (_custom_ops.py qwen4fix dispatch)")

print("done")
