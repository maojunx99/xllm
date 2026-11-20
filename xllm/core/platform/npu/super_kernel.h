/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace c10_npu {
struct NPUGraph;
}

namespace xllm::npu {

// Capture markers delimit the candidate range. CANN selects eligible regions
// inside it; unsupported kernels and task dependencies remain graph boundaries.
void super_kernel_scope_begin();
void super_kernel_scope_end();

// Pause only an active native capture scope. Nested pauses and eager forwards
// leave the enclosing scope unchanged.
class SuperKernelScopePause final {
 public:
  explicit SuperKernelScopePause(bool enabled);
  ~SuperKernelScopePause();
  SuperKernelScopePause(const SuperKernelScopePause&) = delete;
  SuperKernelScopePause& operator=(const SuperKernelScopePause&) = delete;

 private:
  bool resume_ = false;
};

void optimize_super_kernel(c10_npu::NPUGraph& graph,
                           bool auto_op_parallel = false);

}  // namespace xllm::npu
