/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/platform/npu/super_kernel.h"

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "core/platform/npu/static_kernel_manager.h"
#include "tests/npu_test_environment.h"

namespace {

class SuperKernelTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    xllm::testing::init_npu_test_runtime();
    google::InitGoogleLogging("super_kernel_test");
    google::SetStderrLogging(google::INFO);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
    google::ShutdownGoogleLogging();
    xllm::testing::finalize_npu_test_runtime();
  }
};

::testing::Environment* const kTestEnvironment =
    ::testing::AddGlobalTestEnvironment(new SuperKernelTestEnvironment);

torch::Tensor norm_chain(torch::Tensor value, const torch::Tensor& weight) {
  for (int32_t index = 0; index < 8; ++index) {
    value = std::get<0>(at_npu::native::custom_ops::npu_rms_norm(
        value, weight, /*epsilon=*/1e-6));
  }
  return value;
}

// Real graph dependencies, including a separately launched auxiliary stream and
// a task whose tensor argument changes after optimization. No model-specific
// exclusions are used to make the test graph fusible.
class TestGraph final {
 public:
  explicit TestGraph(bool multi_stream,
                     bool updatable,
                     bool auto_op_parallel = false)
      : main_(c10_npu::getStreamFromPool(/*isHighPriority=*/false,
                                         /*device_index=*/0)),
        auxiliary_(c10_npu::getStreamFromPool(/*isHighPriority=*/false,
                                              /*device_index=*/0)),
        updater_(c10_npu::getStreamFromPool(/*isHighPriority=*/false,
                                            /*device_index=*/0)),
        multi_stream_(multi_stream),
        updatable_(updatable),
        auto_op_parallel_(auto_op_parallel),
        ready_(ACL_EVENT_EXTERNAL) {}

  torch::Tensor forward(const torch::Tensor& input,
                        const torch::Tensor& bias,
                        int64_t alpha,
                        bool capturing) {
    torch::Tensor value = norm_chain(input, bias);
    torch::Tensor shared;
    if (multi_stream_) {
      main_ready_.record(main_);
      {
        c10_npu::NPUStreamGuard guard(auxiliary_);
        main_ready_.block(auxiliary_);
        shared = norm_chain(input, bias);
        shared_done_.record(auxiliary_);
      }
    }
    {
      const char* configured_pause =
          std::getenv("XLLM_SUPER_KERNEL_TEST_PAUSE");
      const bool pause_middle =
          configured_pause != nullptr && std::string(configured_pause) == "1";
      xllm::npu::SuperKernelScopePause pause(pause_middle);
      xllm::npu::SuperKernelScopePause nested_pause(pause_middle);
      value = norm_chain(value, bias);
    }
    if (capturing && updatable_) {
      task_input_ = value;
      task_output_ = torch::empty_like(value);
      ready_.block(main_);
      ready_.reset(main_);
      c10_npu::graph_task_group_begin(main_);
      torch::add_out(task_output_, value, task_biases_.front());
      handle_ = c10_npu::graph_task_group_end(main_);
      value = task_output_;
    } else {
      value = value + torch::full_like(bias, alpha);
    }
    if (multi_stream_) {
      shared_done_.block(main_);
      c10_npu::NPUCachingAllocator::recordStream(shared.storage().data_ptr(),
                                                 main_);
      value = value + shared;
    }
    return value;
  }

  void capture(const torch::Tensor& input,
               const torch::Tensor& bias,
               bool optimize,
               const std::filesystem::path& dump_path) {
    c10_npu::NPUStreamGuard guard(main_);
    // Change a kernel tensor argument without changing the task topology.
    // Scalar out fallbacks can insert SDMA Copy, which task groups do not
    // accept.
    task_biases_.reserve(3);
    for (const int64_t alpha : {2, 3, 4}) {
      task_biases_.emplace_back(torch::full_like(bias, alpha));
    }
    forward(input, bias, 2, /*capturing=*/false);
    torch::npu::synchronize();
    graph_.capture_begin();
    if (optimize) {
      xllm::npu::super_kernel_scope_begin();
    }
    output_ = forward(input, bias, 2, /*capturing=*/true);
    if (optimize) {
      xllm::npu::super_kernel_scope_end();
    }
    graph_.capture_end();
    if (optimize) {
      xllm::npu::optimize_super_kernel(graph_, auto_op_parallel_);
    }
    graph_.debug_dump(dump_path.string());
  }

  torch::Tensor replay(int64_t alpha) {
    {
      c10_npu::NPUStreamGuard guard(main_);
      graph_.replay();
    }
    if (updatable_) {
      c10_npu::NPUStreamGuard guard(updater_);
      c10_npu::graph_task_update_begin(updater_, handle_);
      torch::add_out(task_output_, task_input_, task_biases_.at(alpha - 2));
      c10_npu::graph_task_update_end(updater_);
      ready_.record(updater_);
    }
    torch::npu::synchronize();
    return output_.cpu();
  }

  torch::Tensor eager(const torch::Tensor& input,
                      const torch::Tensor& bias,
                      int64_t alpha) {
    c10_npu::NPUStreamGuard guard(main_);
    const torch::Tensor output =
        forward(input, bias, alpha, /*capturing=*/false);
    torch::npu::synchronize();
    return output.cpu();
  }

 private:
  c10_npu::NPUStream main_;
  c10_npu::NPUStream auxiliary_;
  c10_npu::NPUStream updater_;
  bool multi_stream_;
  bool updatable_;
  bool auto_op_parallel_;
  c10_npu::NPUEvent main_ready_;
  c10_npu::NPUEvent shared_done_;
  c10_npu::NPUEvent ready_;
  c10_npu::NPUTaskGroupHandle handle_;
  torch::Tensor task_input_;
  torch::Tensor task_output_;
  std::vector<torch::Tensor> task_biases_;
  torch::Tensor output_;
  c10_npu::NPUGraph graph_;
};

TEST(SuperKernelTest, StaticCompilationMultistreamAndTaskUpdate) {
  torch::NoGradGuard no_grad;
  const char* configured = std::getenv("XLLM_SUPER_KERNEL_TEST_CACHE");
  const std::filesystem::path cache =
      configured != nullptr
          ? std::filesystem::path(configured)
          : std::filesystem::temp_directory_path() /
                ("xllm-super-kernel-test-" + std::to_string(getpid()));
  std::filesystem::create_directories(cache);
  auto& manager = xllm::npu::StaticKernelManager::get_instance();
  const torch::Device device("npu:0");
  const char* configured_auto =
      std::getenv("XLLM_SUPER_KERNEL_TEST_AUTO_OP_PARALLEL");
  const bool auto_op_parallel =
      configured_auto != nullptr && std::string(configured_auto) == "1";
  const auto control = [&](const std::string& action) {
    return nlohmann::json::parse(manager.control(
        nlohmann::json({{"action", action},
                        {"cache_dir", cache.string()},
                        {"super_kernel", true},
                        {"super_kernel_auto_op_parallel", auto_op_parallel}})
            .dump(),
        device));
  };
  const std::vector<int64_t> shapes = {1, 16, 32};
  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> biases;
  inputs.reserve(shapes.size());
  biases.reserve(shapes.size());
  for (const int64_t rows : shapes) {
    inputs.emplace_back(
        torch::linspace(-1, 1, rows * 4096)
            .reshape({rows, 4096})
            .to(torch::TensorOptions().device(device).dtype(torch::kBFloat16)));
    biases.emplace_back(torch::full({4096}, 0.25, inputs.back().options()));
  }
  control("begin");
  for (size_t index = 0; index < shapes.size(); ++index) {
    TestGraph calibration(/*multi_stream=*/true, /*updatable=*/false);
    calibration.eager(inputs[index], biases[index], 2);
  }
  control("end");
  const auto result = control("prepare");
  ASSERT_TRUE(result.at("result").at("super_kernel").get<bool>());
  ASSERT_EQ(result.at("result").at("super_kernel_auto_op_parallel").get<bool>(),
            auto_op_parallel);
  control("load");
  for (size_t index = 0; index < shapes.size(); ++index) {
    for (const bool multi_stream : {false, true}) {
      for (const bool updatable : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "rows=" << shapes[index]
                                          << ", multi_stream=" << multi_stream
                                          << ", updatable=" << updatable);
        {
          TestGraph reference(multi_stream, updatable);
          TestGraph plain(multi_stream, updatable);
          TestGraph fused(multi_stream, updatable, auto_op_parallel);
          const std::string name = std::to_string(shapes[index]) + "-" +
                                   std::to_string(multi_stream) + "-" +
                                   std::to_string(updatable);
          plain.capture(inputs[index],
                        biases[index],
                        /*optimize=*/false,
                        cache / (name + "-plain.json"));
          fused.capture(inputs[index],
                        biases[index],
                        /*optimize=*/true,
                        cache / (name + "-super.json"));
          nlohmann::json plain_tasks;
          nlohmann::json fused_tasks;
          std::ifstream(cache / (name + "-plain.json")) >> plain_tasks;
          std::ifstream(cache / (name + "-super.json")) >> fused_tasks;
          ASSERT_LT(fused_tasks.size(), plain_tasks.size())
              << "SuperKernel must combine multiple kernels, not only return "
                 "success";
          torch::Tensor previous;
          for (int64_t sample = 1; sample <= 3; ++sample) {
            inputs[index].fill_(static_cast<double>(sample) * 0.125);
            biases[index].fill_(0.25 + static_cast<double>(sample) * 0.125);
            torch::npu::synchronize();
            const int64_t alpha = updatable ? sample + 1 : 2;
            const torch::Tensor expected =
                reference.eager(inputs[index], biases[index], alpha);
            if (previous.defined()) {
              ASSERT_FALSE(torch::equal(previous, expected));
            }
            ASSERT_TRUE(torch::equal(plain.replay(alpha), expected));
            ASSERT_TRUE(torch::equal(fused.replay(alpha), expected));
            previous = expected;
          }
        }
        // Destroy the graphs before releasing cached expandable segments.
        // Independent graph pools must not exhaust the virtual address space.
        c10_npu::NPUCachingAllocator::emptyCache();
      }
    }
  }
}

}  // namespace
