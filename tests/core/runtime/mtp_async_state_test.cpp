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

#include "core/runtime/mtp_async_state.h"

namespace xllm::mtp_async {
namespace {

TEST(MtpAsyncStateTest, EnablesCombinedDecodeOnlyForDeclaredCapabilities) {
  EXPECT_TRUE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));

  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/false,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/false,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/false,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/true,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/true,
      /*draft_is_eager=*/true,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/false,
      /*dp_size=*/1));
  EXPECT_FALSE(can_use_combined_decode(
      /*enable_schedule_overlap=*/true,
      /*supports_device_target_context=*/true,
      /*supports_combined_draft_attention=*/true,
      /*has_recurrent_layers=*/false,
      /*enable_atb_spec_kernel=*/false,
      /*draft_is_eager=*/true,
      /*dp_size=*/2));
}

TEST(MtpAsyncStateTest, BuildsMixedAcceptanceStateWithoutHostRoundTrip) {
  const torch::Tensor accepted_tokens = torch::tensor(
      {{10, 11, 12, 13}, {20, 21, -1, -1}, {30, -1, -1, -1}},
      torch::kLong);
  const torch::Tensor accepted_embeddings =
      torch::arange(24, torch::kFloat).reshape({3, 4, 2});
  const torch::Tensor placeholder = torch::tensor({-100.0, -101.0});
  const torch::Tensor base_positions = torch::tensor({100, 200, 300});
  const torch::Tensor base_kv_seq_lens = torch::tensor({101, 201, 301});

  const AcceptedState state = build_accepted_state(accepted_tokens,
                                                   accepted_embeddings,
                                                   placeholder,
                                                   base_positions,
                                                   base_kv_seq_lens);

  EXPECT_TRUE(torch::equal(state.accepted_lengths,
                           torch::tensor({4, 2, 1}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.all_draft_accepted,
                           torch::tensor({true, false, false})));
  EXPECT_TRUE(
      torch::equal(state.last_tokens, torch::tensor({13, 21, 30})));
  EXPECT_TRUE(
      torch::equal(state.previous_tokens, torch::tensor({12, 20, 30})));
  EXPECT_TRUE(torch::equal(
      state.last_embeddings,
      torch::stack({accepted_embeddings[0][3],
                    accepted_embeddings[1][1],
                    accepted_embeddings[2][0]})));
  EXPECT_TRUE(torch::equal(
      state.previous_embeddings,
      torch::stack({accepted_embeddings[0][2],
                    accepted_embeddings[1][0],
                    placeholder})));
  EXPECT_TRUE(torch::equal(state.base_positions,
                           torch::tensor({104, 202, 301}, torch::kLong)));
  EXPECT_TRUE(torch::equal(state.base_kv_seq_lens,
                           torch::tensor({105, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, BuildsRowMetadataForChunkedAndDecodeLayouts) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.base_kv_seq_lens = torch::tensor({105, 203, 302}, torch::kLong);
  const torch::Tensor offsets = torch::tensor({-1, 0}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      make_row_positions(state, offsets),
      torch::tensor({{103, 104}, {201, 202}, {300, 301}}, torch::kLong)));
  EXPECT_TRUE(torch::equal(make_kv_seq_lens(
                               state, offsets, /*use_chunked_prefill=*/true),
                           state.base_kv_seq_lens));
  EXPECT_TRUE(torch::equal(
      make_kv_seq_lens(state, offsets, /*use_chunked_prefill=*/false),
      torch::tensor({104, 105, 202, 203, 301, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, RedirectsUnusedRepairRowsToScratchPositions) {
  AcceptedState state;
  state.base_positions = torch::tensor({104, 202, 301}, torch::kLong);
  state.all_draft_accepted = torch::tensor({true, false, false});

  EXPECT_TRUE(torch::equal(make_repair_cache_positions(state),
                           torch::tensor({103, 203, 302}, torch::kLong)));
}

TEST(MtpAsyncStateTest, MapsPositionsAcrossCacheBlockBoundaries) {
  const torch::Tensor block_tables =
      torch::tensor({{10, 11, 12}, {20, 21, 22}}, torch::kInt);
  const torch::Tensor positions =
      torch::tensor({{3, 4}, {7, 8}}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      map_positions_to_cache_slots(block_tables, positions, /*block_size=*/4),
      torch::tensor({43, 44, 87, 88}, torch::kInt)));
}

TEST(MtpAsyncStateTest, SelectsCurrentRowsFromCombinedDsaTopk) {
  const torch::Tensor combined_topk =
      torch::arange(24, torch::kInt).reshape({6, 2, 2});
  const torch::Tensor current_rows = torch::tensor({1, 3, 5}, torch::kLong);

  EXPECT_TRUE(torch::equal(
      select_rows(combined_topk, current_rows),
      torch::stack(
          {combined_topk[1], combined_topk[3], combined_topk[5]})));
}

}  // namespace
}  // namespace xllm::mtp_async
