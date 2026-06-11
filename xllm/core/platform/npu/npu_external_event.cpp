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

#include "xllm/core/platform/npu/npu_external_event.h"

#include <glog/logging.h>

#include "third_party/acl/inc/acl/acl_rt.h"

namespace xllm::npu {

NPUExternalEvent::NPUExternalEvent() { createEvent(); }

NPUExternalEvent::~NPUExternalEvent() {
  if (is_created_ && event_ != nullptr) {
    aclError err = aclrtDestroyEvent(event_);
    if (err != ACL_SUCCESS) {
      LOG(WARNING) << "Failed to destroy external event: " << err;
    }
    event_ = nullptr;
    is_created_ = false;
  }
}

NPUExternalEvent::NPUExternalEvent(NPUExternalEvent&& other)
    : event_(other.event_),
      is_created_(other.is_created_),
      device_index_(other.device_index_) {
  other.event_ = nullptr;
  other.is_created_ = false;
  other.device_index_ = -1;
}

NPUExternalEvent& NPUExternalEvent::operator=(NPUExternalEvent&& other) {
  if (this != &other) {
    if (is_created_ && event_ != nullptr) {
      aclrtDestroyEvent(event_);
    }
    event_ = other.event_;
    is_created_ = other.is_created_;
    device_index_ = other.device_index_;
    other.event_ = nullptr;
    other.is_created_ = false;
    other.device_index_ = -1;
  }
  return *this;
}

void NPUExternalEvent::createEvent() {
  int32_t device_id = -1;
  aclError err = aclrtGetDevice(&device_id);
  CHECK_EQ(err, ACL_SUCCESS) << "Failed to get current device: " << err;
  device_index_ = static_cast<c10::DeviceIndex>(device_id);

  err = aclrtCreateEventWithFlag(&event_, ACL_EVENT_EXTERNAL);
  CHECK_EQ(err, ACL_SUCCESS)
      << "Failed to create external event with ACL_EVENT_EXTERNAL flag: "
      << err;
  is_created_ = true;
}

void NPUExternalEvent::record(const c10_npu::NPUStream& stream) {
  CHECK(is_created_) << "External event not created";
  aclError err = aclrtRecordEvent(event_, stream);
  CHECK_EQ(err, ACL_SUCCESS) << "Failed to record external event: " << err;
}

void NPUExternalEvent::block(const c10_npu::NPUStream& stream) {
  CHECK(is_created_) << "External event not created";
  aclError err = aclrtStreamWaitEvent(stream, event_);
  CHECK_EQ(err, ACL_SUCCESS) << "Failed to wait on external event: " << err;
}

void NPUExternalEvent::reset(const c10_npu::NPUStream& stream) {
  CHECK(is_created_) << "External event not created";
  aclError err = aclrtResetEvent(event_, stream);
  CHECK_EQ(err, ACL_SUCCESS) << "Failed to reset external event: " << err;
}

}  // namespace xllm::npu
