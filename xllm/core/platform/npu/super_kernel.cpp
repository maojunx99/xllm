/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/platform/npu/super_kernel.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>

#include <array>
#include <chrono>

namespace xllm::npu {
namespace {

constexpr const char* kModelScope = "xllm_model";
thread_local bool model_scope_active = false;

// Older TorchNPU headers expose the graph API but lack the cache-maintenance
// option used by CANN 9.1. Keep those builds usable with the feature disabled.
template <typename Option = aclskOption, typename Type = aclskOptionType>
void optimize_graph(c10_npu::NPUGraph& graph, bool auto_op_parallel) {
  if constexpr (requires(Option option) {
                  Type::DCCI_AFTER_KERNEL_END;
                  option.dcciAfterKernelEnd;
                }) {
    char pattern[] = ".*";
    char* patterns[] = {pattern};
    std::array<Option, 2> options{};
    options[0].optionType = Type::DCCI_AFTER_KERNEL_END;
    options[0].dcciAfterKernelEnd = {patterns, 1};
    uint32_t option_count = 1;
    if (auto_op_parallel) {
      if constexpr (requires(Option option) {
                      Type::AUTO_OP_PARALLEL;
                      option.autoOpParallel.enableAutoOpParallel;
                    }) {
        options[1].optionType = Type::AUTO_OP_PARALLEL;
        options[1].autoOpParallel.enableAutoOpParallel = 1;
        option_count = 2;
      } else {
        LOG(FATAL) << "SuperKernel automatic scheduling requires CANN/TorchNPU "
                      "headers providing AUTO_OP_PARALLEL";
      }
    }
    aclskOptions settings{options.data(), option_count};
    // Matches vLLM-Ascend's cache-flush policy. Do not enable aggressive
    // dependency bypasses: graph events and updatable tasks retain semantics.
    graph.super_kernel_optimize(&settings);
  } else {
    LOG(FATAL) << "SuperKernel requires CANN/TorchNPU headers providing "
                  "DCCI_AFTER_KERNEL_END (CANN 9.1 or newer)";
  }
}

}  // namespace

void super_kernel_scope_begin() {
  c10_npu::super_kernel_scope_begin(kModelScope);
  model_scope_active = true;
}

void super_kernel_scope_end() {
  c10_npu::super_kernel_scope_end(kModelScope);
  model_scope_active = false;
}

SuperKernelScopePause::SuperKernelScopePause(bool enabled)
    : resume_(enabled && model_scope_active) {
  if (resume_) {
    super_kernel_scope_end();
  }
}

SuperKernelScopePause::~SuperKernelScopePause() {
  if (resume_) {
    super_kernel_scope_begin();
  }
}

void optimize_super_kernel(c10_npu::NPUGraph& graph, bool auto_op_parallel) {
  const auto started = std::chrono::steady_clock::now();
  optimize_graph(graph, auto_op_parallel);
  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  LOG(INFO) << "SuperKernel graph optimization completed: scope=" << kModelScope
            << ", auto_op_parallel=" << static_cast<int32_t>(auto_op_parallel)
            << ", elapsed_seconds=" << elapsed_seconds;
}

}  // namespace xllm::npu
