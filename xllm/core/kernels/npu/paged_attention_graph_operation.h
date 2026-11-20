/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace xllm::kernel::npu {

// Retains the ATB operation and scratch storage referenced by an updatable
// graph task. Setup regenerates tiling and launch dimensions for each replay.
class PagedAttentionGraphOperation final {
 public:
  PagedAttentionGraphOperation(torch::Tensor query,
                               torch::Tensor key,
                               torch::Tensor value,
                               torch::Tensor block_table,
                               torch::Tensor output,
                               double scale);
  ~PagedAttentionGraphOperation();

  void prepare(const std::vector<int32_t>& kv_seq_lens,
               torch::Tensor& shared_workspace);
  void execute();
  uint64_t workspace_size() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xllm::kernel::npu
