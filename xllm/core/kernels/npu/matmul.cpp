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
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

namespace {
int64_t npu_format_of(const torch::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}
}  // namespace

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias) {
  if (b.defined() && !b.device().is_cpu() &&
      npu_format_of(b) == ACL_FORMAT_FRACTAL_NZ) {
    // Weight already carries [K, N] layout in FRACTAL_NZ format (see
    // maybe_cast_weight_nz in common/linear.cpp); no runtime transpose.
    CHECK_GE(a.dim(), 1);
    CHECK_EQ(a.size(-1), b.size(0));
    torch::Tensor output;
    if (a.dim() == 2) {
      output = torch::matmul(a, b);
    } else {
      std::vector<int64_t> output_shape = a.sizes().vec();
      output_shape.back() = b.size(1);
      output =
          torch::matmul(a.reshape({-1, a.size(-1)}), b).reshape(output_shape);
    }
    if (bias.has_value()) {
      output = output + bias.value();
    }
    return output;
  }
  if (!bias.has_value()) {
    return torch::nn::functional::linear(a, b);
  } else {
    return torch::nn::functional::linear(a, b, bias.value());
  }
}

}  // namespace xllm::kernel::npu
