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
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "core/common/property_json.h"
#include "core/framework/model/model_args.h"
#include "core/platform/npu/static_kernel_manager.h"
#include "tests/npu_test_environment.h"

namespace {

class StaticKernelTestEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override {
    xllm::testing::init_npu_test_runtime();
    google::InitGoogleLogging("static_kernel_test");
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
    ::testing::AddGlobalTestEnvironment(new StaticKernelTestEnvironment);

TEST(StaticKernelTest, SharedCollectionLoadAndUnseenShape) {
  torch::NoGradGuard no_grad;
  const char* configured = std::getenv("XLLM_STATIC_KERNEL_TEST_CACHE");
  const std::filesystem::path cache =
      configured != nullptr
          ? std::filesystem::path(configured)
          : std::filesystem::temp_directory_path() /
                ("xllm-static-kernel-test-" + std::to_string(getpid()));
  const char* configured_devices =
      std::getenv("XLLM_STATIC_KERNEL_TEST_DEVICES");
  const int32_t device_count =
      configured_devices != nullptr ? std::stoi(configured_devices) : 1;
  ASSERT_GE(device_count, 1);
  ASSERT_LE(device_count, 2);
  auto& manager = xllm::npu::StaticKernelManager::get_instance();
  const auto control = [&manager, &cache](const std::string& action,
                                          const torch::Device& device,
                                          int32_t worker_slot = 0,
                                          const nlohmann::json& extra =
                                              nlohmann::json::object()) {
    nlohmann::json command = {{"action", action},
                              {"cache_dir", cache.string()},
                              {"startup_context",
                               {{"test", "two-shape-collection"},
                                {"schema", 1},
                                {"worker_slot", worker_slot}}}};
    command.update(extra);
    return nlohmann::json::parse(manager.control(command.dump(), device));
  };
  std::array<torch::Tensor, 2> inputs_a;
  std::array<torch::Tensor, 2> inputs_b;
  std::array<torch::Tensor, 2> expected;
  for (size_t index = 0; index < inputs_a.size(); ++index) {
    const int64_t rows = static_cast<int64_t>(index + 1);
    const torch::Device device("npu:" + std::to_string(index % device_count));
    const torch::Tensor cpu_a =
        torch::linspace(-1.0, 1.0, rows * 4096, torch::dtype(torch::kFloat16))
            .reshape({rows, 4096});
    const torch::Tensor cpu_b = torch::full_like(cpu_a, 0.25);
    inputs_a[index] = cpu_a.to(device);
    inputs_b[index] = cpu_b.to(device);
    expected[index] = cpu_a + cpu_b;
  }
  bool startup_hit = true;
  nlohmann::json prepared;
  std::string group_key;
  std::string session;
  for (size_t index = 0; index < inputs_a.size(); ++index) {
    const auto probe =
        control("probe", inputs_a[index].device(), static_cast<int32_t>(index));
    if (index == 0) {
      group_key = probe.at("compilation_group").get<std::string>();
      session = probe.at("session").get<std::string>();
      EXPECT_EQ(session, std::filesystem::canonical(session).string());
    }
    EXPECT_EQ(probe.at("compilation_group"), group_key);
    EXPECT_EQ(probe.at("session"), session);
    startup_hit =
        startup_hit && probe.at("result").at("startup_cache_hit").get<bool>();
    if (probe.at("result").contains("result")) {
      prepared = probe.at("result").at("result");
    }
  }
  const char* expected_hit =
      std::getenv("XLLM_STATIC_KERNEL_TEST_EXPECT_STARTUP_HIT");
  if (expected_hit != nullptr) {
    EXPECT_EQ(startup_hit, std::string(expected_hit) == "1");
  }
  if (!startup_hit) {
    std::promise<void> start_forward;
    std::promise<void> finish_collection;
    const auto forward_ready = start_forward.get_future().share();
    const auto finish_ready = finish_collection.get_future().share();
    std::array<std::promise<void>, 2> workers_ready;
    std::array<std::promise<void>, 2> workers_computed;
    std::vector<std::thread> workers;
    workers.reserve(workers_ready.size());
    for (size_t index = 0; index < workers_ready.size(); ++index) {
      workers.emplace_back([&, index]() {
        torch::NoGradGuard worker_no_grad;
        const torch::Device device = inputs_a[index].device();
        control("begin", device);
        workers_ready[index].set_value();
        forward_ready.wait();
        EXPECT_TRUE(torch::equal((inputs_a[index] + inputs_b[index]).cpu(),
                                 expected[index]));
        const auto repeated =
            xllm::npu::repeat_interleave(inputs_a[index], 2, 0);
        EXPECT_TRUE(torch::equal(
            repeated.cpu(), inputs_a[index].cpu().repeat_interleave(2, 0)));
        workers_computed[index].set_value();
        finish_ready.wait();
        control("end", device);
      });
    }
    for (auto& ready : workers_ready) {
      ready.get_future().get();
    }
    EXPECT_TRUE(manager.is_collecting());
    start_forward.set_value();
    for (auto& computed : workers_computed) {
      computed.get_future().get();
    }
    EXPECT_TRUE(manager.is_collecting());
    finish_collection.set_value();
    for (auto& worker : workers) {
      worker.join();
    }
    EXPECT_FALSE(manager.is_collecting());
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(session)) {
      if (entry.is_regular_file() && entry.path().extension() == ".json" &&
          entry.path().parent_path().filename().string().find("_opcompile") !=
              std::string::npos) {
        std::ifstream input(entry.path());
        nlohmann::json description;
        input >> description;
        EXPECT_NE(description.value("op_type", ""), "RepeatInterleave");
      }
    }
    nlohmann::json group = {{"leader_session", session + "-another-process"},
                            {"sessions", {session}}};
    EXPECT_TRUE(control("prepare",
                        inputs_a[0].device(),
                        0,
                        {{"compilation_groups", {{group_key, group}}}})
                    .at("result")
                    .is_null());
    group["leader_session"] = session;
    prepared = control("prepare",
                       inputs_a[0].device(),
                       0,
                       {{"compilation_groups", {{group_key, group}}}})
                   .at("result");
  }
  EXPECT_EQ(prepared.at("operator_count").get<int64_t>(), 2);
  // Load once per device before either covered shape executes. A second load
  // after the first kernel would hide lazy task-queue initialization problems.
  for (int32_t index = 0; index < device_count; ++index) {
    nlohmann::json extra = nlohmann::json::object();
    if (!startup_hit) {
      extra["compiled_packages"] = {{group_key, prepared}};
    }
    control("load", torch::Device("npu:" + std::to_string(index)), 0, extra);
  }
  for (size_t index = 0; index < inputs_a.size(); ++index) {
    EXPECT_TRUE(torch::equal((inputs_a[index] + inputs_b[index]).cpu(),
                             expected[index]));
    const torch::Tensor unseen_a =
        inputs_a[index].slice(0, 0, 1).expand({3, 4096}).contiguous();
    const torch::Tensor unseen_b =
        inputs_b[index].slice(0, 0, 1).expand({3, 4096}).contiguous();
    EXPECT_TRUE(torch::equal((unseen_a + unseen_b).cpu(),
                             expected[index].slice(0, 0, 1).expand({3, 4096})));
  }
  if (!startup_hit) {
    control("publish", inputs_a[0].device());
  }
}

TEST(StaticKernelTest, ReflectedLaunchPropertiesKeepOptionalAndSetValues) {
  xllm::ModelArgs args;
  args.hidden_size(2048);
  args.n_kv_heads(std::nullopt);
  args.stop_token_ids({9, 3, 7});
  const auto value = xllm::properties_to_json(args);
  EXPECT_EQ(value.at("hidden_size"), 2048);
  EXPECT_TRUE(value.at("n_kv_heads").is_null());
  EXPECT_EQ(value.at("stop_token_ids"), nlohmann::json::array({3, 7, 9}));
}

}  // namespace
