/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <optional>

#if defined(USE_NPU)
#include "core/platform/npu/static_kernel_manager.h"
#endif

namespace xllm::static_kernel {

template <typename Repeats>
torch::Tensor repeat_interleave(
    const torch::Tensor& input,
    const Repeats& repeats,
    std::optional<int64_t> dim = std::nullopt,
    std::optional<int64_t> output_size = std::nullopt) {
#if defined(USE_NPU)
  return npu::repeat_interleave(input, repeats, dim, output_size);
#else
  return input.repeat_interleave(repeats, dim, output_size);
#endif
}

}  // namespace xllm::static_kernel
