include(cc_library)

cc_library(
  NAME
    platform_npu
  HDRS
    npu_layer_synchronizer.h
    npu_batch_memcpy.h
    device_capture_lock.h
    acl_graph_task_update_context.h
    npu_profiler.h
    static_kernel_manager.h
    super_kernel.h
  SRCS
    npu_layer_synchronizer.cpp
    npu_batch_memcpy.cpp
    npu_profiler.cpp
    static_kernel_manager.cpp
    super_kernel.cpp
  DEPS
    torch_npu
    glog::glog
    nlohmann_json::nlohmann_json
    torch
    ascendcl
    pybind11::embed
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(platform_npu PRIVATE
  XLLM_STATIC_KERNEL_HELPER_PATH="${PROJECT_SOURCE_DIR}/tools/static_kernel_compile.py")
