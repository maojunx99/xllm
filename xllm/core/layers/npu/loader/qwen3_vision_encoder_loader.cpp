/* Copyright 2025-2026 The xLLM Authors.

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

#include <cmath>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include "qwen3_vision_encoder_loader.h"
#include "qwen_loader_constants.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

using namespace qwen3_vision_encoder_constants;

namespace {

constexpr char kWeightSuffix[] = "weight";
constexpr char kDeqScaleSuffix[] = "deq_scale";
constexpr char kInputScaleSuffix[] = "input_scale";

bool supports_w8a8_fallback(int index) {
  return index == IN_QKV_WEIGHT || index == IN_WATTENTION_OUT_WEIGHT ||
         index == IN_LINEAR_FC1_WEIGHT;
}

std::string sibling_tensor_name(const std::string& weight_name,
                                const char* suffix) {
  CHECK(absl::EndsWith(weight_name, kWeightSuffix));
  return weight_name.substr(0,
                            weight_name.size() - (sizeof(kWeightSuffix) - 1)) +
         suffix;
}

torch::Tensor dequantize_w8a8_weight(const StateDict& state_dict,
                                     const std::string& weight_name,
                                     torch::ScalarType target_dtype) {
  const auto deq_scale_name = sibling_tensor_name(weight_name, kDeqScaleSuffix);
  const auto input_scale_name =
      sibling_tensor_name(weight_name, kInputScaleSuffix);
  const auto quant_weight = state_dict.get_tensor(weight_name);
  const auto deq_scale = state_dict.get_tensor(deq_scale_name);
  const auto input_scale = state_dict.get_tensor(input_scale_name);

  CHECK(quant_weight.defined()) << "missing quantized weight " << weight_name;
  CHECK(deq_scale.defined()) << "missing dequant scale " << deq_scale_name;
  CHECK(input_scale.defined()) << "missing input scale " << input_scale_name;
  CHECK_EQ(quant_weight.scalar_type(), torch::kInt8)
      << "unexpected dtype for " << weight_name;
  CHECK_EQ(quant_weight.dim(), 2) << "unexpected rank for " << weight_name;
  CHECK_EQ(deq_scale.numel(), quant_weight.size(0))
      << "dequant scale does not match output channels for " << weight_name;
  CHECK_EQ(input_scale.numel(), 1)
      << "input scale must be scalar for " << weight_name;

  const float input_scale_value = input_scale.item<float>();
  CHECK(std::isfinite(input_scale_value) && input_scale_value != 0.0f)
      << "input scale must be finite and nonzero for " << weight_name;
  const auto channel_scale = deq_scale.to(torch::kFloat32) / input_scale_value;
  return (quant_weight.to(torch::kFloat32) * channel_scale.reshape({-1, 1}))
      .to(target_dtype)
      .contiguous();
}

}  // namespace

Qwen3VisionEncoderLoader::Qwen3VisionEncoderLoader(uint64_t weight_count,
                                                   const ModelContext& context,
                                                   LoadMode mode)
    : BaseLoader(weight_count, context, mode) {
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  working_tensors().resize(weight_count);
  if (load_to_host()) {
    auto host_options =
        torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}, host_options);
    }
  } else {
    at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
    for (int i = 0; i < weight_count; ++i) {
      working_tensors()[i] = torch::zeros({1}).to(options);
    }
  }
}

void Qwen3VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  const bool to_host = load_to_host();
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    const bool use_w8a8_fallback =
        supports_w8a8_fallback(index) &&
        state_dict.has(sibling_tensor_name(name, kDeqScaleSuffix));
    if (use_w8a8_fallback) {
      auto weight = dequantize_w8a8_weight(state_dict, name, dtype_);
      const auto shard_it = WEIGHT_SHARD.find(index);
      if (shard_it != WEIGHT_SHARD.end() && encode_param_world_size_ > 1) {
        weight = weight.chunk(encode_param_world_size_, shard_it->second)
                     .at(encode_param_rank_);
      }
      working_tensors()[index] = weight.to(target_device());
      continue;
    }

    const auto shard_it = WEIGHT_SHARD.find(index);
    if (shard_it != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, shard_it->second, to_host);
      continue;
    }
    set_weight(state_dict, name, index, to_host);
  }
}

void Qwen3VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(working_tensors()[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3VisionEncoderLoader::merge_host_at_weights() {
  get_weights_col_packed_qkv();
  if (encode_param_world_size_ > 1) {
    auto& w = working_tensors();
    w[IN_QKV_WEIGHT] = torch::cat(
        {w[IN_VISION_Q_WEIGHT], w[IN_VISION_K_WEIGHT], w[IN_VISION_V_WEIGHT]},
        0);
    w[IN_VISION_Q_WEIGHT] = torch::zeros({1}).to(target_device());
    w[IN_VISION_K_WEIGHT] = torch::zeros({1}).to(target_device());
    w[IN_VISION_V_WEIGHT] = torch::zeros({1}).to(target_device());

    w[IN_QKV_BIAS] = torch::cat(
        {w[IN_VISION_Q_BIAS], w[IN_VISION_K_BIAS], w[IN_VISION_V_BIAS]}, 0);
    w[IN_VISION_Q_BIAS] = torch::zeros({1}).to(target_device());
    w[IN_VISION_K_BIAS] = torch::zeros({1}).to(target_device());
    w[IN_VISION_V_BIAS] = torch::zeros({1}).to(target_device());
  }
}

void Qwen3VisionEncoderLoader::get_weights_col_packed_qkv() {
  auto& w = working_tensors();
  qkv_weight_ = torch::chunk(w[IN_QKV_WEIGHT], 3, 0);
  qkv_bias_ = torch::chunk(w[IN_QKV_BIAS], 3, 0);
  w[IN_VISION_Q_WEIGHT] =
      (qkv_weight_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_K_WEIGHT] =
      (qkv_weight_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_V_WEIGHT] =
      (qkv_weight_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_Q_BIAS] =
      (qkv_bias_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_K_BIAS] =
      (qkv_bias_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  w[IN_VISION_V_BIAS] =
      (qkv_bias_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];
}

}  // namespace layer
}  // namespace xllm
