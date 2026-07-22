/* Copyright 2026 The xLLM Authors.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "core/runtime/mtp_async_state.h"

namespace xllm::kernel::npu {
namespace {

class MtpPrepareNextDraftTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(MtpPrepareNextDraftTest, MatchesTorchReferenceForMixedAcceptance) {
  constexpr int32_t kBlockSize = 4;
  constexpr int64_t kHiddenSize = 16;
  const torch::Tensor accepted_tokens_cpu = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}},
      torch::kLong);
  const torch::Tensor accepted_embeddings_cpu =
      torch::arange(3 * 4 * kHiddenSize, torch::kFloat)
          .reshape({3, 4, kHiddenSize})
          .to(torch::kBFloat16);
  const torch::Tensor placeholder_cpu =
      torch::full({kHiddenSize}, -7.0, torch::kBFloat16);
  const torch::Tensor base_positions_cpu =
      torch::tensor({4, 8, 12}, torch::kInt);
  const torch::Tensor base_kv_seq_lens_cpu =
      torch::tensor({5, 9, 13}, torch::kInt);
  const torch::Tensor block_tables_cpu = torch::tensor(
      {{10, 11, 12, 13, 14},
       {20, 21, 22, 23, 24},
       {30, 31, 32, 33, 34}},
      torch::kInt);
  const torch::Device npu_device("npu:0");

  const auto output = try_mtp_prepare_next_draft(
      accepted_tokens_cpu.to(npu_device),
      accepted_embeddings_cpu.to(npu_device),
      placeholder_cpu.to(npu_device),
      base_positions_cpu.to(npu_device),
      base_kv_seq_lens_cpu.to(npu_device),
      block_tables_cpu.to(npu_device),
      kBlockSize);
  ASSERT_TRUE(output.has_value());

  const mtp_async::AcceptedState state = mtp_async::build_accepted_state(
      accepted_tokens_cpu,
      accepted_embeddings_cpu,
      placeholder_cpu,
      base_positions_cpu,
      base_kv_seq_lens_cpu);
  const torch::Tensor offsets = torch::tensor({-1, 0}, torch::kLong);
  const torch::Tensor positions = mtp_async::make_row_positions(state, offsets);
  const torch::Tensor expected_tokens =
      torch::stack({state.previous_tokens, state.last_tokens}, /*dim=*/1)
          .to(torch::kInt)
          .flatten();
  const torch::Tensor expected_embeddings =
      torch::stack({state.previous_embeddings, state.last_embeddings},
                   /*dim=*/1)
          .flatten(/*start_dim=*/0, /*end_dim=*/1);
  const torch::Tensor cache_positions = torch::stack(
      {mtp_async::make_repair_cache_positions(state), state.base_positions},
      /*dim=*/1);
  const torch::Tensor expected_slots = mtp_async::map_positions_to_cache_slots(
      block_tables_cpu, cache_positions, kBlockSize);

  EXPECT_TRUE(torch::equal(output->token_ids.cpu(), expected_tokens));
  EXPECT_TRUE(torch::equal(output->embeddings.cpu(), expected_embeddings));
  EXPECT_TRUE(torch::equal(output->positions.cpu(),
                           positions.to(torch::kInt).flatten()));
  EXPECT_TRUE(torch::equal(output->kv_seq_lens.cpu(),
                           state.base_kv_seq_lens.to(torch::kInt)));
  EXPECT_TRUE(torch::equal(output->cache_slots.cpu(), expected_slots));
}

TEST_F(MtpPrepareNextDraftTest, RejectsUnsupportedHostInputs) {
  const torch::Tensor tokens = torch::tensor({{1, 2}}, torch::kLong);
  const torch::Tensor embeddings =
      torch::zeros({1, 2, 16}, torch::kBFloat16);
  const torch::Tensor placeholder = torch::zeros({16}, torch::kBFloat16);
  const torch::Tensor positions = torch::tensor({1}, torch::kInt);
  const torch::Tensor kv_seq_lens = torch::tensor({2}, torch::kInt);
  const torch::Tensor block_tables = torch::tensor({{0, 1}}, torch::kInt);

  EXPECT_FALSE(try_mtp_prepare_next_draft(tokens,
                                          embeddings,
                                          placeholder,
                                          positions,
                                          kv_seq_lens,
                                          block_tables,
                                          /*block_size=*/4)
                   .has_value());
}

}  // namespace
}  // namespace xllm::kernel::npu
