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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace xllm::npu {

// During argument collection, excluded RepeatInterleave is evaluated on the
// host to avoid CANN serializing its device-resident value-dependent input.
// Outside collection these functions dispatch the original device operator.
torch::Tensor repeat_interleave(
    const torch::Tensor& input,
    int64_t repeats,
    std::optional<int64_t> dim = std::nullopt,
    std::optional<int64_t> output_size = std::nullopt);
torch::Tensor repeat_interleave(
    const torch::Tensor& input,
    const torch::Tensor& repeats,
    std::optional<int64_t> dim = std::nullopt,
    std::optional<int64_t> output_size = std::nullopt);

// CANN argument dumping and reselection are process-wide. Local workers share
// a dump session; remote processes collect their own actual operator sets.
class StaticKernelManager final {
 public:
  static StaticKernelManager& get_instance();

  bool is_collecting() const;
  std::string control(const std::string& command, const torch::Device& device);

 private:
  StaticKernelManager();
  ~StaticKernelManager();

  class Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> collecting_{false};
};

}  // namespace xllm::npu
