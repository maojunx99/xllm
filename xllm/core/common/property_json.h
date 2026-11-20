/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/xLLM-AI/xllm/blob/main/LICENSE
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <algorithm>
#include <nlohmann/json.hpp>
#include <utility>

#include "core/common/property_reflect.h"

namespace xllm {

class JsonPropertyVisitor final : public PropertyVisitor {
 public:
  void visit(const std::string& name, bool value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, int32_t value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, int64_t value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, float value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, double value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, const std::string& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::vector<int32_t>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::vector<int64_t>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::vector<float>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::vector<double>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name, const std::vector<bool>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::vector<std::string>& value) override {
    value_[name] = value;
  }
  void visit(const std::string& name,
             const std::unordered_set<int32_t>& value) override {
    std::vector<int32_t> ordered(value.begin(), value.end());
    std::sort(ordered.begin(), ordered.end());
    value_[name] = std::move(ordered);
  }
  void visit_absent(const std::string& name) override {
    value_[name] = nullptr;
  }
  nlohmann::json take_value() { return std::move(value_); }

 private:
  nlohmann::json value_ = nlohmann::json::object();
};

template <typename Owner>
nlohmann::json properties_to_json(const Owner& owner) {
  JsonPropertyVisitor visitor;
  visit_properties(owner, visitor);
  return visitor.take_value();
}

}  // namespace xllm
