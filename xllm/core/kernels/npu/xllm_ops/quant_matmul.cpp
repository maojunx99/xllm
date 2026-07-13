/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

at::Tensor quant_matmul(const at::Tensor& x1,
                        const at::Tensor& x2,
                        const bool transpose2,
                        const at::Tensor& scale,
                        const c10::optional<at::Tensor>& offset,
                        const c10::optional<at::Tensor>& pertoken_scale,
                        const c10::optional<at::Tensor>& bias,
                        c10::optional<at::ScalarType> output_dtype) {
  const at::ScalarType out_dtype = output_dtype.value_or(at::kChar);

  at::Tensor result = at_npu::native::custom_ops::npu_quant_matmul(
      x1,
      x2,
      scale,
      offset,
      pertoken_scale,
      bias,
      static_cast<int64_t>(out_dtype));

  return result;
}

}  // namespace xllm::kernel::npu
