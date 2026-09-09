/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <dlfcn.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/utils.h"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kMegaGdnChunkSize = 128;

struct MegaGdnPrefillMasks {
  torch::Tensor mask_lower;
  torch::Tensor mask_full;
  torch::Tensor minus_identity;
};

std::unordered_map<int32_t, MegaGdnPrefillMasks> g_prefill_mask_cache;
std::mutex g_prefill_mask_cache_mutex;

MegaGdnPrefillMasks get_or_create_prefill_masks(const torch::Device& device) {
  const int32_t device_index = static_cast<int32_t>(device.index());
  std::lock_guard<std::mutex> lock(g_prefill_mask_cache_mutex);
  auto it = g_prefill_mask_cache.find(device_index);
  if (it != g_prefill_mask_cache.end()) {
    return it->second;
  }

  const torch::TensorOptions fp32_options =
      torch::TensorOptions(device).dtype(torch::kFloat32);
  const torch::TensorOptions bf16_options =
      torch::TensorOptions(device).dtype(torch::kBFloat16);
  MegaGdnPrefillMasks masks;
  masks.mask_lower = torch::tril(
      torch::ones({kMegaGdnChunkSize, kMegaGdnChunkSize}, fp32_options),
      /*diagonal=*/-1);
  masks.mask_full = torch::tril(
      torch::ones({kMegaGdnChunkSize, kMegaGdnChunkSize}, fp32_options),
      /*diagonal=*/0);
  masks.minus_identity =
      torch::zeros({kMegaGdnChunkSize, kMegaGdnChunkSize}, bf16_options);
  masks.minus_identity.diagonal().fill_(-1);
  g_prefill_mask_cache.emplace(device_index, masks);
  return masks;
}

int64_t get_ffts_address() {
  if (is_ascend950()) {
    return 0;
  }

  using RtGetC2cCtrlAddr = int32_t (*)(uint64_t*, uint32_t*);
  static RtGetC2cCtrlAddr get_c2c_ctrl_addr = []() {
    void* runtime = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
    CHECK(runtime != nullptr) << "Failed to load libruntime.so: " << dlerror();
    void* symbol = dlsym(runtime, "rtGetC2cCtrlAddr");
    CHECK(symbol != nullptr)
        << "Failed to resolve rtGetC2cCtrlAddr: " << dlerror();
    return reinterpret_cast<RtGetC2cCtrlAddr>(symbol);
  }();

  uint64_t ffts_address = 0;
  uint32_t ffts_length = 0;
  CHECK_EQ(get_c2c_ctrl_addr(&ffts_address, &ffts_length), 0)
      << "rtGetC2cCtrlAddr failed";
  CHECK_GT(ffts_length, 0) << "rtGetC2cCtrlAddr returned an empty region";
  return static_cast<int64_t>(ffts_address);
}

}  // namespace

torch::Tensor npu_mega_gdn_prefill(
    const torch::Tensor& mixed_qkv,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& z,
    const torch::Tensor& conv_weight,
    torch::Tensor& conv_state,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    const torch::Tensor& conv_state_read_indices,
    const torch::Tensor& conv_state_write_indices,
    const torch::Tensor& ssm_state_read_indices,
    const torch::Tensor& ssm_state_write_indices,
    torch::Tensor& ssm_cache,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& norm_weight,
    int64_t num_matrices) {
  const MegaGdnPrefillMasks masks =
      get_or_create_prefill_masks(mixed_qkv.device());
  const int64_t ffts_address = get_ffts_address();
  torch::Tensor out = torch::empty_like(z);
  EXEC_NPU_CMD(aclnnMegaGdnPrefillOp,
               mixed_qkv,
               b,
               a,
               z,
               conv_weight,
               conv_state,
               A_log,
               dt_bias,
               conv_state_read_indices,
               conv_state_write_indices,
               ssm_state_read_indices,
               ssm_state_write_indices,
               ssm_cache,
               masks.mask_lower,
               masks.mask_full,
               masks.minus_identity,
               cu_seqlens,
               norm_weight,
               ffts_address,
               num_matrices,
               out,
               conv_state,
               ssm_cache);
  return out;
}

}  // namespace xllm::kernel::npu
