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

#include <acl/acl.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/layers/common/linear.h"
#include "core/layers/npu_torch/fused_moe.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/platform/stream.h"
#include "tests/core/layers/npu_torch/tests_utils.h"

namespace xllm::layer {
namespace {

TEST(NpuSharedExpertOverlap, LazyNzConversionRetainsSourceStorage) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261031);
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(Platform::type_torch(), 0);
  std::unique_ptr<ProcessGroup> process_group;
  test::create_default_parallel_args(process_group);
  KernelConfig& config = KernelConfig::get_instance();
  const bool saved_weight_nz = config.enable_weight_nz();
  config.enable_weight_nz(true);
  const QuantArgs quant;
  const torch::Tensor input_cpu = torch::randn({4, 2048}).to(torch::kBFloat16);
  const torch::Tensor input = input_cpu.to(options);
  const torch::Tensor delay_a = torch::randn({2048, 2048}).to(options);
  const torch::Tensor delay_b = torch::randn({2048, 2048}).to(options);
  Stream producer(c10_npu::getCurrentNPUStream());
  Stream conversion(options.device());
  CHECK(c10_npu::npuSynchronizeDevice(/*check_error=*/true));
  for (int32_t iteration = 0; iteration < 6; ++iteration) {
    SCOPED_TRACE(iteration);
    const torch::Tensor weight_cpu =
        (torch::randn({512, 2048}) * 0.02).to(torch::kBFloat16);
    ColumnParallelLinear linear(2048,
                                512,
                                /*bias=*/false,
                                /*gather_output=*/false,
                                quant,
                                process_group.get(),
                                options);
    const StateDict state({{"weight", weight_cpu}}, "");
    linear->load_state_dict(state);
    CHECK(c10_npu::npuSynchronizeDevice(/*check_error=*/true));
    conversion.wait_stream(producer);
    torch::Tensor output;
    {
      const auto guard = conversion.set_stream_guard();
      // Delay the first conversion while the producer can reuse freed storage.
      for (int32_t delay = 0; delay < 40; ++delay) {
        const torch::Tensor pending = torch::matmul(delay_a, delay_b);
      }
      output = linear->forward(input);
    }
    std::vector<torch::Tensor> replacements;
    replacements.reserve(6);
    for (int32_t replacement = 0; replacement < 6; ++replacement) {
      replacements.emplace_back(torch::full({512, 2048}, 17.0, options));
    }
    CHECK(c10_npu::npuSynchronizeDevice(/*check_error=*/true));
    const torch::Tensor actual = output.cpu().to(torch::kFloat32);
    const torch::Tensor expected = torch::matmul(
        input_cpu.to(torch::kFloat32), weight_cpu.to(torch::kFloat32).t());
    const double relative_error =
        ((actual - expected).to(torch::kFloat64).norm() /
         expected.to(torch::kFloat64).norm())
            .item<double>();
    EXPECT_TRUE(torch::isfinite(actual).all().item<bool>());
    EXPECT_LT(relative_error, 0.006);
  }
  config.enable_weight_nz(saved_weight_nz);
}

TEST(NpuSharedExpertOverlap, MatchesSerialOnNonDefaultStream) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261027);
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(Platform::type_torch(), 0);
  std::unique_ptr<ProcessGroup> process_group;
  const ParallelArgs parallel =
      test::create_default_parallel_args(process_group);
  ModelArgs args;
  args.model_type() = "qwen3_5_moe";
  args.hidden_size() = 128;
  args.moe_intermediate_size() = 128;
  args.n_routed_experts() = 8;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 2;
  args.norm_topk_prob() = true;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  QuantArgs quant;
  KernelConfig& config = KernelConfig::get_instance();
  const bool saved_finalize = config.enable_moe_finalize_routing();
  config.enable_moe_finalize_routing(false);
  config.enable_shared_expert_overlap(false);
  FusedMoE serial(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  // An enabled option must also remain valid with EP = 1, including graphs.
  config.enable_moe_finalize_routing(true);
  config.enable_shared_expert_overlap(true);
  FusedMoE overlap(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_shared_expert_overlap(false);
  config.enable_moe_finalize_routing(saved_finalize);

  const auto weight = [](const std::vector<int64_t>& shape) {
    return (torch::randn(shape) * 0.02).to(torch::kBFloat16);
  };
  std::unordered_map<std::string, torch::Tensor> weights = {
      {"gate.weight", weight({8, 128})},
      {"shared_expert_gate.weight", weight({1, 128})},
      {"shared_expert.gate_proj.weight", weight({128, 128})},
      {"shared_expert.up_proj.weight", weight({128, 128})},
      {"shared_expert.down_proj.weight", weight({128, 128})},
      {"experts.gate_up_proj", weight({8, 256, 128})},
      {"experts.down_proj", weight({8, 128, 128})},
  };
  const StateDict state(std::move(weights), "");
  serial->load_state_dict(state);
  overlap->load_state_dict(state);

  Stream compute(options.device());
  Stream loading(c10_npu::getCurrentNPUStream());
  compute.wait_stream(loading);
  const auto guard = compute.set_stream_guard();
  ModelInputParams params;
  for (const int64_t rows : {1, 2, 4, 16, 127, 2048, 4, 1}) {
    SCOPED_TRACE(rows);
    const torch::Tensor input = torch::randn({rows, 128}).to(options);
    const torch::Tensor expected = serial->forward(input, params).cpu();
    for (int32_t repeat = 0; repeat < 3; ++repeat) {
      const torch::Tensor actual = overlap->forward(input, params).cpu();
      ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
      ASSERT_TRUE(torch::equal(actual, expected))
          << "max_abs_diff="
          << (actual.to(torch::kFloat32) - expected.to(torch::kFloat32))
                 .abs()
                 .max()
                 .item<float>();
    }
  }

  // Match serving: capture several buckets into one pool, then alternate
  // replay and eager execution. Events must belong to each captured forward.
  const auto pool = c10_npu::graph_pool_handle();
  std::vector<std::unique_ptr<c10_npu::NPUGraph>> graphs;
  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> outputs;
  std::vector<torch::Tensor> expected_outputs;
  graphs.reserve(3);
  inputs.reserve(3);
  outputs.reserve(3);
  expected_outputs.reserve(3);
  for (const int64_t rows : {16, 4, 1}) {
    const torch::Tensor input = torch::randn({rows, 128}).to(options);
    expected_outputs.emplace_back(serial->forward(input, params).cpu());
    overlap->forward(input, params).cpu();
    auto graph = std::make_unique<c10_npu::NPUGraph>();
    graph->capture_begin(
        pool, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    outputs.emplace_back(overlap->forward(input, params));
    graph->capture_end();
    graph->replay();
    ASSERT_TRUE(torch::equal(outputs.back().cpu(), expected_outputs.back()));
    inputs.emplace_back(input);
    graphs.emplace_back(std::move(graph));
  }
  for (int32_t repeat = 0; repeat < 3; ++repeat) {
    for (size_t index = 0; index < graphs.size(); ++index) {
      graphs[index]->replay();
      ASSERT_TRUE(torch::equal(outputs[index].cpu(), expected_outputs[index]));
      ASSERT_TRUE(torch::equal(overlap->forward(inputs[index], params).cpu(),
                               expected_outputs[index]));
    }
  }

  std::vector<FusedMoE> layers;
  layers.reserve(40);
  config.enable_shared_expert_overlap(true);
  for (int32_t layer = 0; layer < 40; ++layer) {
    layers.emplace_back(
        args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
    layers.back()->load_state_dict(state);
  }
  config.enable_shared_expert_overlap(false);
  Stream update(options.device());
  c10_npu::NPUEvent update_ready(ACL_EVENT_EXTERNAL);
  const torch::Tensor chain_input = torch::randn({4, 128}).to(options);
  torch::Tensor expected_chain = chain_input;
  for (int32_t layer = 0; layer < 40; ++layer) {
    expected_chain = expected_chain + serial->forward(expected_chain, params);
  }
  expected_chain = expected_chain.cpu();
  const auto run_chain = [&]() {
    torch::Tensor value = chain_input;
    for (FusedMoE& layer : layers) {
      value = value + layer->forward(value, params);
    }
    return value;
  };
  ASSERT_TRUE(torch::equal(run_chain().cpu(), expected_chain));
  c10_npu::NPUGraph chain_graph;
  chain_graph.capture_begin(
      {}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
  update_ready.block(*compute.get_stream());
  update_ready.reset(*compute.get_stream());
  const torch::Tensor chain_output = run_chain();
  chain_graph.capture_end();
  for (int32_t repeat = 0; repeat < 3; ++repeat) {
    update_ready.record(*update.get_stream());
    chain_graph.replay();
    ASSERT_TRUE(torch::equal(chain_output.cpu(), expected_chain));
  }
}

TEST(NpuSharedExpertOverlap, MatchesSerialAtModelDimensions) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261028);
  const auto options = torch::TensorOptions()
                           .dtype(torch::kBFloat16)
                           .device(Platform::type_torch(), 0);
  std::unique_ptr<ProcessGroup> process_group;
  const ParallelArgs parallel =
      test::create_default_parallel_args(process_group);
  ModelArgs args;
  args.model_type() = "qwen3_5_moe";
  args.hidden_size() = 2048;
  args.moe_intermediate_size() = 512;
  args.n_routed_experts() = 256;
  args.n_shared_experts() = 1;
  args.num_experts_per_tok() = 8;
  args.norm_topk_prob() = true;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  QuantArgs quant;
  KernelConfig& config = KernelConfig::get_instance();
  config.enable_shared_expert_overlap(false);
  FusedMoE serial(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_shared_expert_overlap(true);
  FusedMoE overlap(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_shared_expert_overlap(false);
  const auto weight = [](const std::vector<int64_t>& shape) {
    return (torch::randn(shape) * 0.02).to(torch::kBFloat16);
  };
  std::unordered_map<std::string, torch::Tensor> weights = {
      {"gate.weight", weight({256, 2048})},
      {"shared_expert_gate.weight", weight({1, 2048})},
      {"shared_expert.gate_proj.weight", weight({512, 2048})},
      {"shared_expert.up_proj.weight", weight({512, 2048})},
      {"shared_expert.down_proj.weight", weight({2048, 512})},
      {"experts.gate_up_proj", weight({256, 1024, 2048})},
      {"experts.down_proj", weight({256, 2048, 512})},
  };
  const StateDict state(std::move(weights), "");
  serial->load_state_dict(state);
  overlap->load_state_dict(state);
  Stream compute(options.device());
  Stream loading(c10_npu::getCurrentNPUStream());
  compute.wait_stream(loading);
  const auto guard = compute.set_stream_guard();
  ModelInputParams params;
  for (const int64_t rows : {1, 4, 16, 127, 2048}) {
    SCOPED_TRACE(rows);
    const torch::Tensor input = torch::randn({rows, 2048}).to(options);
    const torch::Tensor expected = serial->forward(input, params).cpu();
    for (int32_t repeat = 0; repeat < 3; ++repeat) {
      const torch::Tensor control = serial->forward(input, params).cpu();
      const torch::Tensor actual = overlap->forward(input, params).cpu();
      const float control_max_abs =
          (control.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      const float overlap_max_abs =
          (actual.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      LOG(INFO) << "rows=" << rows << ", repeat=" << repeat
                << ", serial_repeat_max_abs=" << control_max_abs
                << ", overlap_max_abs=" << overlap_max_abs;
      ASSERT_TRUE(torch::isfinite(actual).all().item<bool>());
      ASSERT_TRUE(torch::allclose(actual, expected, 2e-2, 2e-3));
    }
  }
}

TEST(NpuSharedExpertOverlap, PreservesModelDtypeAndSelectedExpertSemantics) {
  torch::NoGradGuard no_grad;
  torch::manual_seed(20261028);
  using TestCase = std::tuple<std::string, torch::ScalarType, int64_t, bool>;
  const std::vector<TestCase> cases = {
      {"qwen3_5_moe", torch::kHalf, 1, true},
      {"deepseek_v3", torch::kBFloat16, 2, true},
      {"deepseek_v4", torch::kBFloat16, 1, false},
      {"qwen3_moe", torch::kBFloat16, 0, false},
  };
  for (const auto& [model_type, dtype, shared_count, load_gate] : cases) {
    SCOPED_TRACE(model_type);
    const auto options =
        torch::TensorOptions().dtype(dtype).device(Platform::type_torch(), 0);
    std::unique_ptr<ProcessGroup> process_group;
    const ParallelArgs parallel =
        test::create_default_parallel_args(process_group);
    ModelArgs args;
    args.model_type() = model_type;
    args.hidden_size() = 128;
    args.moe_intermediate_size() = 128;
    args.n_routed_experts() = 8;
    args.n_shared_experts() = shared_count;
    args.num_experts_per_tok() = 2;
    args.norm_topk_prob() = true;
    args.n_group() = 1;
    args.topk_group() = 1;
    args.hidden_act() = "silu";
    args.scoring_func() = "softmax";
    QuantArgs quant;
    KernelConfig& config = KernelConfig::get_instance();
    const bool saved_fc1 = config.enable_flashcomm1();
    config.enable_flashcomm1(true);
    config.enable_shared_expert_overlap(false);
    FusedMoE serial(
        args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
    config.enable_shared_expert_overlap(true);
    FusedMoE overlap(
        args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
    config.enable_shared_expert_overlap(false);
    config.enable_flashcomm1(saved_fc1);
    const auto weight = [dtype](const std::vector<int64_t>& shape) {
      return (torch::randn(shape) * 0.02).to(dtype);
    };
    std::unordered_map<std::string, torch::Tensor> weights = {
        {"gate.weight", weight({8, 128})},
        {"experts.gate_up_proj", weight({8, 256, 128})},
        {"experts.down_proj", weight({8, 128, 128})},
    };
    if (shared_count > 0) {
      weights.emplace("shared_expert.gate_proj.weight",
                      weight({128 * shared_count, 128}));
      weights.emplace("shared_expert.up_proj.weight",
                      weight({128 * shared_count, 128}));
      weights.emplace("shared_expert.down_proj.weight",
                      weight({128, 128 * shared_count}));
    }
    if (load_gate) {
      weights.emplace("shared_expert_gate.weight", weight({1, 128}));
    }
    const StateDict state(std::move(weights), "");
    serial->load_state_dict(state);
    overlap->load_state_dict(state);
    Stream compute(options.device());
    Stream loading(c10_npu::getCurrentNPUStream());
    compute.wait_stream(loading);
    const auto guard = compute.set_stream_guard();
    ModelInputParams params;
    for (const int64_t rows : {1, 4, 16, 127}) {
      SCOPED_TRACE(rows);
      const torch::Tensor input = torch::randn({rows, 128}).to(options);
      ASSERT_TRUE(torch::equal(serial->forward(input, params).cpu(),
                               overlap->forward(input, params).cpu()));
      const torch::Tensor ids = torch::tensor({0, 1}, torch::kInt32)
                                    .expand({rows, 2})
                                    .contiguous()
                                    .to(options.device());
      const torch::Tensor probabilities = torch::full({rows, 2}, 0.5, options);
      ASSERT_TRUE(torch::equal(
          serial
              ->forward_with_selected_experts(input, probabilities, ids, params)
              .cpu(),
          overlap
              ->forward_with_selected_experts(input, probabilities, ids, params)
              .cpu()));
    }
  }
}

TEST(NpuSharedExpertOverlap, MatchesSerialWithRealDistributedGroups) {
  const char* mode_env = std::getenv("XLLM_OVERLAP_TEST_TOPOLOGY");
  if (mode_env == nullptr) {
    GTEST_SKIP() << "Run with four ranks and XLLM_OVERLAP_TEST_TOPOLOGY.";
  }
  torch::NoGradGuard no_grad;
  const std::string mode(mode_env);
  const bool use_dp = mode == "dp2_ep4";
  const int32_t rank = std::stoi(std::getenv("XLLM_OVERLAP_TEST_RANK"));
  const int32_t port = std::stoi(std::getenv("XLLM_OVERLAP_TEST_PORT"));
  const torch::Device device(Platform::type_torch(), rank);
  Device(device).set_device();
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto world = create_process_group(rank,
                                    /*world_size=*/4,
                                    /*rank_size=*/4,
                                    port,
                                    /*trans=*/false,
                                    "127.0.0.1",
                                    "overlap_world",
                                    device);
  auto single =
      std::make_unique<ProcessGroup>(/*rank=*/0, /*world_size=*/1, device);
  std::unique_ptr<ProcessGroup> dp_group;
  std::unique_ptr<ProcessGroup> tp_group;
  if (use_dp) {
    dp_group = create_process_group(rank,
                                    4,
                                    2,
                                    port + rank % 2 + 1,
                                    true,
                                    "127.0.0.1",
                                    "overlap_dp",
                                    device);
    tp_group = create_process_group(rank,
                                    4,
                                    2,
                                    port + rank / 2 + 10,
                                    false,
                                    "127.0.0.1",
                                    "overlap_tp",
                                    device);
  }
  ParallelArgs parallel(
      rank, 4, use_dp ? 2 : 1, 1, world.get(), use_dp ? 4 : 1);
  parallel.tp_size(use_dp ? 2 : 4);
  parallel.tp_group_ = use_dp ? tp_group.get() : world.get();
  parallel.moe_tp_group_ = use_dp ? single.get() : world.get();
  parallel.moe_ep_group_ = use_dp ? world.get() : single.get();
  parallel.dp_local_process_group_ = dp_group.get();
  parallel.single_rank_group_ = single.get();
  ModelArgs args;
  args.model_type() = "deepseek_v3";
  args.hidden_size() = 128;
  args.moe_intermediate_size() = 512;
  args.n_routed_experts() = 8;
  args.n_shared_experts() = 2;
  args.num_experts_per_tok() = 2;
  args.norm_topk_prob() = true;
  args.n_group() = 1;
  args.topk_group() = 1;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  QuantArgs quant;
  KernelConfig& config = KernelConfig::get_instance();
  const bool saved_finalize = config.enable_moe_finalize_routing();
  const bool saved_overlap = config.enable_shared_expert_overlap();
  config.enable_moe_finalize_routing(false);
  config.enable_shared_expert_overlap(false);
  FusedMoE serial(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_moe_finalize_routing(true);
  FusedMoE fused_serial(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_shared_expert_overlap(true);
  FusedMoE overlap(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_moe_finalize_routing(false);
  FusedMoE legacy_overlap(
      args, FusedMoEArgs{.is_gated = true}, quant, parallel, options);
  config.enable_shared_expert_overlap(saved_overlap);
  config.enable_moe_finalize_routing(saved_finalize);
  torch::manual_seed(20261028);
  const auto weight = [](const std::vector<int64_t>& shape) {
    return (torch::randn(shape) * 0.02).to(torch::kBFloat16);
  };
  std::unordered_map<std::string, torch::Tensor> weights = {
      {"gate.weight", weight({8, 128})},
      {"shared_expert_gate.weight", weight({1, 128})},
      {"shared_expert.gate_proj.weight", weight({1024, 128})},
      {"shared_expert.up_proj.weight", weight({1024, 128})},
      {"shared_expert.down_proj.weight", weight({128, 1024})},
      {"experts.gate_up_proj", weight({8, 1024, 128})},
      {"experts.down_proj", weight({8, 128, 512})},
  };
  const StateDict state(std::move(weights), "");
  serial->load_state_dict(state);
  fused_serial->load_state_dict(state);
  overlap->load_state_dict(state);
  legacy_overlap->load_state_dict(state);
  Stream compute(device);
  Stream loading(c10_npu::getCurrentNPUStream());
  compute.wait_stream(loading);
  const auto guard = compute.set_stream_guard();
  const std::vector<std::pair<int32_t, int32_t>> batches = {
      {1, 1}, {4, 2}, {1, 0}, {0, 1}, {16, 16}, {127, 4}};
  for (const auto& [left, right] : batches) {
    const int32_t local_rows = use_dp && rank >= 2 ? right : left;
    ModelInputParams params;
    if (use_dp) {
      params.parallel.dp_global_token_nums = {left, right};
    }
    params.meta.num_sequences = local_rows > 0 ? 1 : 0;
    const int32_t padded_rows = local_rows > 0 ? local_rows : 1;
    torch::manual_seed(20261028 + (use_dp ? rank / 2 : 0));
    const torch::Tensor input =
        local_rows > 0 ? torch::randn({padded_rows, 128}).to(options)
                       : torch::zeros({padded_rows, 128}, options);
    const torch::Tensor expected = serial->forward(input, params).cpu();
    const char* output_dir = std::getenv("XLLM_OVERLAP_TEST_OUTPUT_DIR");
    if (output_dir != nullptr) {
      const std::string output_path = std::string(output_dir) + "/" + mode +
                                      "_rank" + std::to_string(rank) +
                                      "_batch" + std::to_string(left) + "_" +
                                      std::to_string(right) + ".pt";
      torch::save(expected, output_path);
    }
    for (int32_t repeat = 0; repeat < 3; ++repeat) {
      const torch::Tensor control = serial->forward(input, params).cpu();
      const torch::Tensor fused = fused_serial->forward(input, params).cpu();
      const torch::Tensor actual = overlap->forward(input, params).cpu();
      const torch::Tensor legacy = legacy_overlap->forward(input, params).cpu();
      const float control_error =
          (control.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      const float overlap_error =
          (actual.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      const float fused_error =
          (fused.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      const float legacy_overlap_error =
          (legacy.to(torch::kFloat32) - expected.to(torch::kFloat32))
              .abs()
              .max()
              .item<float>();
      LOG(INFO) << "topology=" << mode << ", rank=" << rank
                << ", batch=" << left << "/" << right << ", repeat=" << repeat
                << ", serial_repeat_max_abs=" << control_error
                << ", overlap_max_abs=" << overlap_error
                << ", finalize_serial_max_abs=" << fused_error
                << ", legacy_overlap_max_abs=" << legacy_overlap_error;
      EXPECT_TRUE(torch::equal(actual, expected));
      EXPECT_TRUE(torch::equal(control, expected));
      EXPECT_TRUE(torch::equal(fused, expected));
      EXPECT_TRUE(torch::equal(legacy, expected));
    }
  }
}

}  // namespace
}  // namespace xllm::layer
