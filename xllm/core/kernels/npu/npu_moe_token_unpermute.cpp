/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {

torch::Tensor apply_npu_moe_finalize_routing(
    const torch::Tensor& expanded_tokens,
    const torch::Tensor& probabilities,
    const torch::Tensor& expanded_row_ids) {
  CHECK_EQ(expanded_tokens.dim(), 2);
  CHECK_EQ(probabilities.dim(), 2);
  CHECK_EQ(expanded_tokens.size(0), probabilities.numel());
  CHECK_EQ(expanded_row_ids.numel(), probabilities.numel());
  // Mode 3 consumes row-major gather indices and skips -1 routes. Its 3D
  // input uses flat row offsets; this view preserves the dropless FFN buffer.
  const auto expanded_view = expanded_tokens.view(
      {probabilities.size(1), probabilities.size(0), expanded_tokens.size(1)});
  return at_npu::native::custom_ops::npu_moe_finalize_routing(
      expanded_view,
      /*skip1=*/std::nullopt,
      /*skip2=*/std::nullopt,
      /*bias=*/std::nullopt,
      probabilities,
      expanded_row_ids,
      /*export_for_source_row=*/std::nullopt,
      /*drop_pad_mode=*/3);
}

torch::Tensor apply_npu_moe_token_unpermute(
    const torch::Tensor& permuted_tokens,
    const torch::Tensor& sorted_indices,
    const std::optional<torch::Tensor>& probes,
    bool padded_mode,
    c10::OptionalIntArrayRef restore_shape) {
  if (!padded_mode) {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probes.value());
  } else {
    return at_npu::native::custom_ops::npu_moe_token_unpermute(
        permuted_tokens,
        sorted_indices,
        probes.value(),
        padded_mode,
        restore_shape.value());
  }
}

}  // namespace xllm::kernel::npu
