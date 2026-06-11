/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <torch/torch.h>

#include "third_party/acl/inc/acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace xllm::npu {

class NPUExternalEvent {
 public:
  NPUExternalEvent();
  ~NPUExternalEvent();

  NPUExternalEvent(const NPUExternalEvent&) = delete;
  NPUExternalEvent& operator=(const NPUExternalEvent&) = delete;
  NPUExternalEvent(NPUExternalEvent&& other);
  NPUExternalEvent& operator=(NPUExternalEvent&& other);

  void record(const c10_npu::NPUStream& stream);
  void block(const c10_npu::NPUStream& stream);
  void reset(const c10_npu::NPUStream& stream);

  aclrtEvent event() const { return event_; }

 private:
  aclrtEvent event_ = nullptr;
  bool is_created_ = false;
  c10::DeviceIndex device_index_ = -1;

  void createEvent();
};

}  // namespace xllm::npu
