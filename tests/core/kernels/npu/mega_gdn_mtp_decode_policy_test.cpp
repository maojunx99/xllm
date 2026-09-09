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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/kernels/ops_api.h"

namespace xllm::kernel {
namespace {

MegaGdnMtpDecodeParams make_supported_params(int64_t batch_size = 4,
                                              int64_t sequence_length = 5) {
  constexpr int64_t kNumKHeads = 8;
  constexpr int64_t kNumVHeads = 16;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kCacheSlots = 4;
  const int64_t conv_dim =
      2 * kNumKHeads * kHeadDim + kNumVHeads * kHeadDim;
  const torch::TensorOptions bf16 = torch::dtype(torch::kBFloat16);
  const torch::TensorOptions fp32 = torch::dtype(torch::kFloat32);
  const torch::TensorOptions int32 = torch::dtype(torch::kInt32);
  MegaGdnMtpDecodeParams params;
  params.qkv = torch::empty({batch_size, sequence_length, conv_dim}, bf16);
  params.z = torch::empty(
      {batch_size, sequence_length, kNumVHeads, kHeadDim}, bf16);
  params.b = torch::empty({batch_size, sequence_length, kNumVHeads}, bf16);
  params.a = torch::empty({batch_size, sequence_length, kNumVHeads}, bf16);
  params.conv_weight = torch::empty({4, conv_dim}, bf16);
  params.conv_state =
      torch::empty({kCacheSlots, sequence_length + 2, conv_dim}, bf16);
  params.A_log = torch::empty({kNumVHeads}, fp32);
  params.dt_bias = torch::empty({kNumVHeads}, fp32);
  params.ssm_state = torch::empty(
      {kCacheSlots * sequence_length, kNumVHeads, kHeadDim, kHeadDim}, fp32);
  params.read_state_indices = torch::empty({batch_size}, int32);
  params.write_state_indices = torch::empty({batch_size}, int32);
  params.num_accepted_tokens = torch::empty({batch_size}, int32);
  params.norm_weight = torch::empty({kHeadDim}, bf16);
  return params;
}

MegaGdnPrefillParams make_supported_prefill_params(
    int64_t batch_size = 2,
    int64_t total_tokens = 256,
    int64_t checkpoint_stride = 5) {
  constexpr int64_t kNumKHeads = 8;
  constexpr int64_t kNumVHeads = 16;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kCacheSlots = 4;
  const int64_t conv_dim = 2 * kNumKHeads * kHeadDim + kNumVHeads * kHeadDim;
  const torch::TensorOptions bf16 = torch::dtype(torch::kBFloat16);
  const torch::TensorOptions fp32 = torch::dtype(torch::kFloat32);
  const torch::TensorOptions int32 = torch::dtype(torch::kInt32);
  MegaGdnPrefillParams params;
  params.mixed_qkv = torch::empty({total_tokens, conv_dim}, bf16);
  params.b = torch::empty({total_tokens, kNumVHeads}, bf16);
  params.a = torch::empty({total_tokens, kNumVHeads}, bf16);
  params.z = torch::empty({total_tokens, kNumVHeads, kHeadDim}, bf16);
  params.conv_weight = torch::empty({4, conv_dim}, bf16);
  params.conv_state =
      torch::empty({kCacheSlots, checkpoint_stride + 2, conv_dim}, bf16);
  params.A_log = torch::empty({kNumVHeads}, fp32);
  params.dt_bias = torch::empty({kNumVHeads}, fp32);
  params.conv_state_read_indices = torch::empty({batch_size}, int32);
  params.conv_state_write_indices = torch::empty({batch_size}, int32);
  params.ssm_state_read_indices = torch::empty({batch_size}, int32);
  params.ssm_state_write_indices = torch::empty({batch_size}, int32);
  params.ssm_cache = torch::empty(
      {kCacheSlots * checkpoint_stride, kNumVHeads, kHeadDim, kHeadDim}, fp32);
  params.cu_seqlens = torch::empty({batch_size + 1}, int32);
  params.norm_weight = torch::empty({kHeadDim}, bf16);
  params.num_matrices = 32;
  return params;
}

TEST(MegaGdnMtpDecodePolicyTest, AcceptsCurrentBf16Mtp4Shape) {
  EXPECT_TRUE(supports_mega_gdn_mtp_decode(make_supported_params()));
}

TEST(MegaGdnMtpDecodePolicyTest, RejectsUnsupportedBoundariesBeforeDispatch) {
  EXPECT_FALSE(supports_mega_gdn_mtp_decode(
      make_supported_params(/*batch_size=*/33, /*sequence_length=*/5)));
  EXPECT_FALSE(supports_mega_gdn_mtp_decode(
      make_supported_params(/*batch_size=*/4, /*sequence_length=*/18)));
  MegaGdnMtpDecodeParams fp16 = make_supported_params();
  fp16.qkv = fp16.qkv.to(torch::kFloat16);
  EXPECT_FALSE(supports_mega_gdn_mtp_decode(fp16));
}

TEST(MegaGdnMtpDecodePolicyTest, RejectsNonContiguousMutableCaches) {
  MegaGdnMtpDecodeParams non_contiguous_cache = make_supported_params();
  const auto cache_sizes = non_contiguous_cache.conv_state.sizes().vec();
  non_contiguous_cache.conv_state =
      torch::empty({cache_sizes[0], cache_sizes[1], cache_sizes[2] * 2},
                   torch::kBFloat16)
          .slice(/*dim=*/2, /*start=*/0, /*end=*/cache_sizes[2] * 2, /*step=*/2);
  ASSERT_EQ(non_contiguous_cache.conv_state.sizes(), cache_sizes);
  ASSERT_FALSE(non_contiguous_cache.conv_state.is_contiguous());
  EXPECT_FALSE(supports_mega_gdn_mtp_decode(non_contiguous_cache));

}

TEST(MegaGdnMtpDecodePolicyTest, AcceptsNonContiguousReadOnlyInputs) {
  MegaGdnMtpDecodeParams non_contiguous_indices = make_supported_params();
  const int64_t batch_size = non_contiguous_indices.read_state_indices.numel();
  non_contiguous_indices.read_state_indices =
      torch::empty({batch_size * 2}, torch::kInt32)
          .slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size * 2, /*step=*/2);
  ASSERT_FALSE(non_contiguous_indices.read_state_indices.is_contiguous());
  EXPECT_TRUE(supports_mega_gdn_mtp_decode(non_contiguous_indices));
}

TEST(MegaGdnPrefillPolicyTest, AcceptsPackedBf16Shape) {
  EXPECT_TRUE(supports_mega_gdn_prefill(make_supported_prefill_params()));
}

TEST(MegaGdnPrefillPolicyTest, RejectsInvalidStateStrideAndDtype) {
  MegaGdnPrefillParams invalid_stride = make_supported_prefill_params();
  invalid_stride.conv_state =
      torch::empty({4, 6, invalid_stride.mixed_qkv.size(1)}, torch::kBFloat16);
  EXPECT_FALSE(supports_mega_gdn_prefill(invalid_stride));
  MegaGdnPrefillParams fp16 = make_supported_prefill_params();
  fp16.mixed_qkv = fp16.mixed_qkv.to(torch::kFloat16);
  EXPECT_FALSE(supports_mega_gdn_prefill(fp16));
}

TEST(MegaGdnPrefillPolicyTest, RejectsInvalidMatrixCount) {
  MegaGdnPrefillParams non_multiple = make_supported_prefill_params();
  non_multiple.num_matrices = 31;
  EXPECT_FALSE(supports_mega_gdn_prefill(non_multiple));

  MegaGdnPrefillParams too_large = make_supported_prefill_params();
  too_large.num_matrices =
      too_large.mixed_qkv.size(0) * too_large.z.size(1) + too_large.z.size(1);
  EXPECT_FALSE(supports_mega_gdn_prefill(too_large));
}

}  // namespace
}  // namespace xllm::kernel
