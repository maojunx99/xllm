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

#include "core/platform/npu/static_kernel_manager.h"

#include <ATen/Context.h>
#include <acl/acl.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern char** environ;

namespace xllm::npu {
namespace {

constexpr uint32_t kDumpAicoreArguments = 1U;

std::shared_ptr<void> open_library(const char* name) {
  void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  CHECK(handle != nullptr) << "Static kernels require " << name << ": "
                           << dlerror();
  return {handle, [](void* library) { dlclose(library); }};
}

template <typename Function>
Function load_symbol(void* library, const char* name) {
  void* symbol = dlsym(library, name);
  CHECK(symbol != nullptr) << "CANN does not provide required interface: "
                           << name;
  return reinterpret_cast<Function>(symbol);
}

std::filesystem::path cache_root(const std::string& configured) {
  if (!configured.empty()) {
    return std::filesystem::absolute(configured);
  }
  const char* xdg_cache = std::getenv("XDG_CACHE_HOME");
  if (xdg_cache != nullptr && xdg_cache[0] != '\0') {
    return std::filesystem::path(xdg_cache) / "xllm/static_kernel";
  }
  const char* user_home = std::getenv("HOME");
  CHECK(user_home != nullptr) << "Set --static_kernel_cache_dir or HOME";
  return std::filesystem::path(user_home) / ".cache/xllm/static_kernel";
}

std::filesystem::path compiler_helper() {
  // Embedded/C API servers load libxllm.so from a different directory than
  // their executable. Prefer resources installed beside the containing image.
  Dl_info module_info{};
  if (dladdr(reinterpret_cast<const void*>(&compiler_helper), &module_info) !=
          0 &&
      module_info.dli_fname != nullptr) {
    const std::filesystem::path adjacent =
        std::filesystem::absolute(module_info.dli_fname).parent_path() /
        "static_kernel_compile.py";
    if (std::filesystem::is_regular_file(adjacent)) {
      return adjacent;
    }
  }
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe");
  const std::filesystem::path installed =
      executable.parent_path() / "static_kernel_compile.py";
  if (std::filesystem::is_regular_file(installed)) {
    return installed;
  }
  const std::filesystem::path source = XLLM_STATIC_KERNEL_HELPER_PATH;
  CHECK(std::filesystem::is_regular_file(source))
      << "Static kernel compiler helper not found: " << source;
  return source;
}

const std::unordered_set<std::string>& blacklisted_ops() {
  static const auto blacklist = []() {
    const auto path =
        compiler_helper().parent_path() / "static_kernel_blacklist.json";
    std::ifstream input(path);
    CHECK(input.is_open()) << "Static kernel blacklist not found: " << path;
    nlohmann::json value;
    input >> value;
    CHECK(value.is_array())
        << "Static kernel blacklist must be an OpType array";
    std::unordered_set<std::string> result;
    for (const auto& entry : value) {
      CHECK(entry.is_string() && !entry.get<std::string>().empty());
      result.insert(entry.get<std::string>());
    }
    LOG(INFO) << "Static kernel blacklist: " << value.dump();
    return result;
  }();
  return blacklist;
}

bool repeat_on_host(const torch::Tensor& input) {
  return StaticKernelManager::get_instance().is_collecting() &&
         input.device().type() == c10::DeviceType::PrivateUse1 &&
         blacklisted_ops().count("RepeatInterleave") != 0;
}

nlohmann::json run_helper(const std::filesystem::path& session,
                          const std::filesystem::path& cache,
                          const std::string& soc,
                          bool super_kernel,
                          bool auto_op_parallel,
                          const std::string& action,
                          const std::filesystem::path& result_file,
                          const std::vector<std::string>& extra = {}) {
  std::vector<std::string> arguments = {
      "python3",
      compiler_helper().string(),
      "--dump-dir",
      session.string(),
      "--cache-dir",
      cache.string(),
      "--soc",
      soc,
      "--deterministic",
      torch::globalContext().deterministicAlgorithms() ? "1" : "0",
      "--result",
      result_file.string(),
      "--action",
      action};
  arguments.insert(arguments.end(), extra.begin(), extra.end());
  if (super_kernel) {
    arguments.emplace_back("--super-kernel");
    arguments.emplace_back("--super-kernel-auto-op-parallel");
    arguments.emplace_back(auto_op_parallel ? "1" : "0");
  }
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (std::string& argument : arguments) {
    argv.emplace_back(argument.data());
  }
  argv.emplace_back(nullptr);
  pid_t child = 0;
  // Spawn immediately executes Python without running Python after fork in
  // the threaded model server.
  CHECK_EQ(
      posix_spawnp(&child, argv[0], nullptr, nullptr, argv.data(), environ), 0);
  int32_t status = 0;
  pid_t waited = 0;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited == -1 && errno == EINTR);
  CHECK_EQ(waited, child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << "Static kernel helper action " << action
      << " failed; logs: " << session;
  std::ifstream result_input(result_file);
  CHECK(result_input.is_open());
  nlohmann::json result;
  result_input >> result;
  return result;
}

void write_json(const std::filesystem::path& path,
                const nlohmann::json& value) {
  std::ofstream output(path);
  CHECK(output.is_open());
  output << value.dump();
  CHECK(output.good());
}

std::string library_path(const void* symbol) {
  Dl_info info{};
  CHECK_NE(dladdr(symbol, &info), 0);
  CHECK(info.dli_fname != nullptr);
  return std::filesystem::canonical(info.dli_fname).string();
}

void flush_device(const torch::Device& device) {
  c10_npu::set_device(device.index());
  // This exported interface drains host task queues before device sync.
  CHECK(c10_npu::npuSynchronizeDevice(/*check_error=*/true));
}

std::string compilation_group(const std::filesystem::path& cache,
                              const std::string& environment) {
  std::ifstream boot_input("/proc/sys/kernel/random/boot_id");
  std::string boot_id;
  CHECK(static_cast<bool>(boot_input >> boot_id));
  struct stat metadata{};
  CHECK_EQ(stat(cache.c_str(), &metadata), 0);
  // A path alone cannot identify a shared cache across nodes or containers.
  // Group only workers seeing the same directory on the same booted host.
  return nlohmann::json(
             {{"node", boot_id},
              {"cache", std::filesystem::canonical(cache).string()},
              {"cache_device", static_cast<uint64_t>(metadata.st_dev)},
              {"cache_inode", static_cast<uint64_t>(metadata.st_ino)},
              {"environment", environment}})
      .dump();
}

}  // namespace

class StaticKernelManager::Impl final {
 public:
  using StartDump = aclError (*)(uint32_t, const char*);
  using StopDump = aclError (*)(uint32_t);
  using Reselect = int32_t (*)(const char*);

  std::mutex mutex_;
  std::shared_ptr<void> dump_library_;
  std::shared_ptr<void> opapi_library_;
  StartDump start_dump_ = nullptr;
  StopDump stop_dump_ = nullptr;
  Reselect reselect_ = nullptr;
  int32_t participants_ = 0;
  bool collected_ = false;
  bool super_kernel_ = false;
  bool auto_op_parallel_ = false;
  std::filesystem::path cache_;
  std::filesystem::path session_;
  nlohmann::json result_;
  std::unordered_map<int32_t, nlohmann::json> probes_;
  std::unordered_set<std::string> startup_keys_;
  bool published_ = false;
  std::string compilation_group_;
};

torch::Tensor repeat_interleave(const torch::Tensor& input,
                                int64_t repeats,
                                std::optional<int64_t> dim,
                                std::optional<int64_t> output_size) {
  if (repeat_on_host(input)) {
    return input.cpu()
        .repeat_interleave(repeats, dim, output_size)
        .to(input.options(), /*non_blocking=*/false, /*copy=*/false);
  }
  return input.repeat_interleave(repeats, dim, output_size);
}

torch::Tensor repeat_interleave(const torch::Tensor& input,
                                const torch::Tensor& repeats,
                                std::optional<int64_t> dim,
                                std::optional<int64_t> output_size) {
  if (repeat_on_host(input)) {
    return input.cpu()
        .repeat_interleave(repeats.cpu(), dim, output_size)
        .to(input.options(), /*non_blocking=*/false, /*copy=*/false);
  }
  return input.repeat_interleave(repeats, dim, output_size);
}

StaticKernelManager::StaticKernelManager() : impl_(std::make_unique<Impl>()) {}
StaticKernelManager::~StaticKernelManager() = default;

StaticKernelManager& StaticKernelManager::get_instance() {
  static StaticKernelManager manager;
  return manager;
}

bool StaticKernelManager::is_collecting() const {
  return collecting_.load(std::memory_order_acquire);
}

std::string StaticKernelManager::control(const std::string& command,
                                         const torch::Device& device) {
  const auto request = nlohmann::json::parse(command);
  const std::string action = request.at("action").get<std::string>();
  CHECK(action == "probe" || action == "begin" || action == "end" ||
        action == "prepare" || action == "load" || action == "publish");
  flush_device(device);
  std::lock_guard<std::mutex> lock(impl_->mutex_);
  const int32_t device_index = static_cast<int32_t>(device.index());
  const char* soc = aclrtGetSocName();
  CHECK(soc != nullptr);
  if ((action == "probe" || action == "begin") && impl_->session_.empty()) {
    impl_->super_kernel_ = request.value("super_kernel", false);
    impl_->auto_op_parallel_ =
        impl_->super_kernel_ &&
        request.value("super_kernel_auto_op_parallel", false);
    impl_->cache_ = cache_root(request.at("cache_dir").get<std::string>());
    std::filesystem::create_directories(impl_->cache_ / "sessions");
    impl_->cache_ = std::filesystem::canonical(impl_->cache_);
    std::string pattern = (impl_->cache_ / "sessions/startup-XXXXXX").string();
    CHECK(mkdtemp(pattern.data()) != nullptr);
    impl_->session_ = pattern;
  }
  CHECK_EQ(impl_->super_kernel_, request.value("super_kernel", false))
      << "Workers in the same process must agree on SuperKernel compilation";
  CHECK_EQ(impl_->auto_op_parallel_,
           impl_->super_kernel_ &&
               request.value("super_kernel_auto_op_parallel", false))
      << "Workers in the same process must agree on SuperKernel scheduling";
  nlohmann::json report_result;
  if (action == "probe") {
    CHECK(!impl_->collected_ && impl_->participants_ == 0);
    auto context = request.at("startup_context");
    context["runtime_image"] =
        library_path(reinterpret_cast<const void*>(&compiler_helper));
    context["runtime_libraries"] = nlohmann::json::array(
        {library_path(
             reinterpret_cast<const void*>(&c10_npu::npuSynchronizeDevice)),
         library_path(reinterpret_cast<const void*>(&torch::globalContext))});
    const auto context_file =
        impl_->session_ / ("context-" + std::to_string(device_index) + ".json");
    write_json(context_file, context);
    report_result = run_helper(
        impl_->session_,
        impl_->cache_,
        soc,
        impl_->super_kernel_,
        impl_->auto_op_parallel_,
        "probe",
        impl_->session_ / ("probe-" + std::to_string(device_index) + ".json"),
        {"--startup-context", context_file.string()});
    impl_->probes_[device_index] = report_result;
    const std::string group = compilation_group(
        impl_->cache_,
        report_result.at("compilation_environment").get<std::string>());
    CHECK(impl_->compilation_group_.empty() ||
          impl_->compilation_group_ == group)
        << "Workers in one process must use the same compiler environment";
    impl_->compilation_group_ = group;
    impl_->startup_keys_.insert(
        report_result.at("startup_key").get<std::string>());
  } else if (action == "begin") {
    CHECK(!impl_->collected_) << "Static kernel collection is startup-only";
    if (impl_->participants_ == 0) {
      blacklisted_ops();
      impl_->dump_library_ = open_library("libascend_dump.so");
      impl_->opapi_library_ = open_library("libopapi.so");
      impl_->start_dump_ = load_symbol<Impl::StartDump>(
          impl_->dump_library_.get(), "aclopStartDumpArgs");
      impl_->stop_dump_ = load_symbol<Impl::StopDump>(
          impl_->dump_library_.get(), "aclopStopDumpArgs");
      impl_->reselect_ = load_symbol<Impl::Reselect>(
          impl_->opapi_library_.get(), "aclnnReselectStaticKernelWithPath");
      CHECK_EQ(
          impl_->start_dump_(kDumpAicoreArguments, impl_->session_.c_str()),
          ACL_SUCCESS);
      collecting_.store(true, std::memory_order_release);
      LOG(INFO) << "Static kernel argument collection: " << impl_->session_;
    }
    ++impl_->participants_;
  } else if (action == "end") {
    CHECK_GT(impl_->participants_, 0);
    --impl_->participants_;
    if (impl_->participants_ == 0) {
      CHECK_EQ(impl_->stop_dump_(kDumpAicoreArguments), ACL_SUCCESS);
      collecting_.store(false, std::memory_order_release);
      impl_->collected_ = true;
    }
  } else if (action == "prepare") {
    CHECK(impl_->collected_ && impl_->participants_ == 0);
    std::vector<std::string> extra;
    bool is_leader = true;
    if (request.contains("compilation_groups")) {
      const auto& group =
          request.at("compilation_groups").at(impl_->compilation_group_);
      is_leader = group.at("leader_session").get<std::string>() ==
                  impl_->session_.string();
      if (is_leader && impl_->result_.is_null()) {
        const auto directories_file = impl_->session_ / "dump_dirs.json";
        write_json(directories_file, group.at("sessions"));
        extra = {"--dump-dirs", directories_file.string()};
        LOG(INFO) << "Static kernel node compilation leader: " << getpid()
                  << ", collection_count=" << group.at("sessions").size();
      }
    }
    if (is_leader && impl_->result_.is_null()) {
      impl_->result_ = run_helper(impl_->session_,
                                  impl_->cache_,
                                  soc,
                                  impl_->super_kernel_,
                                  impl_->auto_op_parallel_,
                                  "compile",
                                  impl_->session_ / "result.json",
                                  extra);
    }
  } else if (action == "publish") {
    CHECK(impl_->result_.is_object());
    if (!impl_->published_ && !impl_->startup_keys_.empty()) {
      auto keys = nlohmann::json::array();
      for (const std::string& key : impl_->startup_keys_) {
        keys.emplace_back(key);
      }
      const auto keys_file = impl_->session_ / "startup_keys.json";
      write_json(keys_file, keys);
      run_helper(impl_->session_,
                 impl_->cache_,
                 soc,
                 impl_->super_kernel_,
                 impl_->auto_op_parallel_,
                 "publish",
                 impl_->session_ / "published.json",
                 {"--startup-keys",
                  keys_file.string(),
                  "--compiled-key",
                  impl_->result_.at("cache_key").get<std::string>()});
      impl_->published_ = true;
    }
  } else {
    if (request.contains("compiled_packages")) {
      const auto& package =
          request.at("compiled_packages").at(impl_->compilation_group_);
      CHECK(impl_->collected_ && impl_->participants_ == 0);
      if (!impl_->result_.is_null()) {
        CHECK_EQ(impl_->result_.at("cache_key"), package.at("cache_key"));
        CHECK_EQ(impl_->result_.at("install_path"), package.at("install_path"));
      }
      impl_->result_ = package;
    }
    report_result = impl_->result_;
    if (report_result.is_null()) {
      const auto& probe = impl_->probes_.at(device_index);
      CHECK(probe.at("startup_cache_hit").get<bool>());
      report_result = probe.at("result");
      report_result["startup_cache_hit"] = true;
    }
    CHECK(report_result.is_object());
    if (impl_->reselect_ == nullptr) {
      impl_->opapi_library_ = open_library("libopapi.so");
      impl_->reselect_ = load_symbol<Impl::Reselect>(
          impl_->opapi_library_.get(), "aclnnReselectStaticKernelWithPath");
    }
    const std::string install_path =
        report_result.at("install_path").get<std::string>();
    CHECK(std::filesystem::is_directory(install_path));
    const auto reselect = impl_->reselect_;
    auto select_kernel = [reselect, install_path]() -> int32_t {
      CHECK_EQ(reselect(install_path.c_str()), 0)
          << "Static kernel reselection failed: " << install_path;
      return 0;
    };
    // ACLNN workspace selection runs on the caller in queue mode 1 and on
    // the execution queue in mode 2. Configure both contexts before any model
    // operator runs. Mode 0 invokes the same idempotent selection
    // synchronously.
    select_kernel();
    at_npu::native::OpCommand::RunOpApiV2("xllm_static_kernel_load",
                                          select_kernel);
    CHECK(c10_npu::npuSynchronizeDevice(/*check_error=*/true));
    LOG(INFO) << "Static kernels selected on device "
              << static_cast<int32_t>(device.index()) << ": "
              << report_result.dump();
  }
  if (report_result.is_null()) {
    report_result = impl_->result_;
  }
  return nlohmann::json({{"pid", getpid()},
                         {"device", device.index()},
                         {"session", impl_->session_.string()},
                         {"compilation_group", impl_->compilation_group_},
                         {"action", action},
                         {"result", report_result}})
      .dump();
}

}  // namespace xllm::npu
