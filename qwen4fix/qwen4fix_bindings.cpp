// qwen4fix: standalone rebuild of the PR#53896 fused GDN MTP decode kernel
// (adds sigmoid output-gate support) registered under torch.ops._qwen4fix
// so it can coexist with the image's older silu-only _C registration.
// Sources: vllm PR#53896 csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu (unmodified)
//          + this binding file (schema copied verbatim from the PR's torch_bindings.cpp).

#include "ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_qwen4fix, ops) {
  ops.def(
      "fused_gdn_decode_post_conv_mtp("
      "Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
      "Tensor! state, Tensor output_gate, Tensor norm_weight, Tensor! out, "
      "float scale, float norm_eps=1e-5, "
      "str output_gate_activation='silu') -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_qwen4fix, CUDA, ops) {
  ops.impl("fused_gdn_decode_post_conv_mtp",
           TORCH_BOX(&fused_gdn_decode_post_conv_mtp));
}
