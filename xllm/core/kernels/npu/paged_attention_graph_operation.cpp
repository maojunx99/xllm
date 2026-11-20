/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/kernels/npu/paged_attention_graph_operation.h"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <glog/logging.h>

#include <cstring>
#include <utility>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace xllm::kernel::npu {
namespace {

atb::Tensor make_atb_tensor(const torch::Tensor& tensor) {
  CHECK(tensor.is_contiguous());
  atb::Tensor result{};
  result.desc.format = ACL_FORMAT_ND;
  if (!tensor.device().is_cpu()) {
    const aclFormat format =
        static_cast<aclFormat>(at_npu::native::get_npu_format(tensor));
    if (format != ACL_FORMAT_NCHW && format != ACL_FORMAT_NHWC &&
        format != ACL_FORMAT_NCDHW) {
      result.desc.format = format;
    }
  }
  switch (tensor.scalar_type()) {
    case torch::kBFloat16:
      result.desc.dtype = ACL_BF16;
      break;
    case torch::kHalf:
      result.desc.dtype = ACL_FLOAT16;
      break;
    case torch::kInt32:
      result.desc.dtype = ACL_INT32;
      break;
    case torch::kInt64:
      result.desc.dtype = ACL_INT64;
      break;
    case torch::kUInt8:
      result.desc.dtype = ACL_UINT8;
      break;
    case torch::kInt8:
      result.desc.dtype = ACL_INT8;
      break;
    case torch::kFloat32:
      result.desc.dtype = ACL_FLOAT;
      break;
    default:
      LOG(FATAL) << "Unsupported PA graph tensor dtype: "
                 << tensor.scalar_type();
  }
  result.desc.shape.dimNum = tensor.dim();
  for (int64_t index = 0; index < tensor.dim(); ++index) {
    result.desc.shape.dims[index] = tensor.size(index);
  }
  if (tensor.device().is_cpu()) {
    result.hostData = tensor.data_ptr();
  } else {
    result.deviceData = tensor.data_ptr();
  }
  result.dataSize = atb::Utils::GetTensorSize(result);
  return result;
}

}  // namespace

class PagedAttentionGraphOperation::Impl final {
 public:
  Impl(torch::Tensor query,
       torch::Tensor key,
       torch::Tensor value,
       torch::Tensor block_table,
       torch::Tensor output,
       double scale)
      : query_(query.view({-1, query.size(-2), query.size(-1)})),
        key_(std::move(key)),
        value_(std::move(value)),
        block_table_(std::move(block_table)),
        output_(output.view(query_.sizes())) {
    lengths_ = torch::empty({block_table_.size(0)},
                            torch::TensorOptions().dtype(torch::kInt32));
    atb::infer::PagedAttentionParam param;
    param.headNum = query_.size(1);
    param.kvHeadNum = key_.size(-2);
    param.qkScale = scale;
    param.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
    param.batchRunStatusEnable = false;
    param.quantType = atb::infer::PagedAttentionParam::TYPE_QUANT_UNDEFINED;
    param.outDataType = ACL_DT_UNDEFINED;
    param.hasQuantOffset = false;
    param.compressType =
        atb::infer::PagedAttentionParam::COMPRESS_TYPE_UNDEFINED;
    param.calcType = atb::infer::PagedAttentionParam::CALC_TYPE_UNDEFINED;
    param.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_TOR;
    param.inputLayout = atb::infer::TYPE_BSND;
    param.mlaVHeadSize = 0;
    atb::Operation* operation = nullptr;
    CHECK_EQ(atb::CreateOperation(param, &operation), atb::NO_ERROR);
    operation_.reset(operation);
    atb::Context* context = nullptr;
    CHECK_EQ(atb::CreateContext(&context), atb::NO_ERROR);
    context_.reset(context);
    pack_.inTensors = {make_atb_tensor(query_),
                       make_atb_tensor(key_),
                       make_atb_tensor(value_),
                       make_atb_tensor(block_table_),
                       make_atb_tensor(lengths_)};
    pack_.outTensors = {make_atb_tensor(output_)};
  }

  void prepare(const std::vector<int32_t>& kv_seq_lens,
               torch::Tensor& shared_workspace) {
    CHECK_EQ(kv_seq_lens.size(), static_cast<size_t>(lengths_.numel()));
    std::memcpy(lengths_.data_ptr<int32_t>(),
                kv_seq_lens.data(),
                kv_seq_lens.size() * sizeof(int32_t));
    CHECK_EQ(
        context_->SetExecuteStream(c10_npu::getCurrentNPUStream().stream()),
        atb::NO_ERROR);
    CHECK_EQ(operation_->Setup(pack_, workspace_size_, context_.get()),
             atb::NO_ERROR);
    if (!shared_workspace.defined() ||
        static_cast<uint64_t>(shared_workspace.numel()) < workspace_size_) {
      shared_workspace = torch::empty({static_cast<int64_t>(workspace_size_)},
                                      query_.options().dtype(torch::kUInt8));
    }
    // Keep this allocation alive even if a later layer grows the shared buffer.
    workspace_ = shared_workspace;
  }

  void execute() {
    const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    auto launch = [this, stream]() -> int32_t {
      const atb::Status status = context_->SetExecuteStream(stream);
      if (status != atb::NO_ERROR) {
        return status;
      }
      return operation_->Execute(
          pack_,
          workspace_size_ == 0 ? nullptr : workspace_.data_ptr<uint8_t>(),
          workspace_size_,
          context_.get());
    };
    at_npu::native::OpCommand::RunOpApiV2("PagedAttentionGraph", launch);
  }

  uint64_t workspace_size() const { return workspace_size_; }

 private:
  torch::Tensor query_;
  torch::Tensor key_;
  torch::Tensor value_;
  torch::Tensor block_table_;
  torch::Tensor output_;
  torch::Tensor lengths_;
  torch::Tensor workspace_;
  atb::VariantPack pack_;
  std::unique_ptr<atb::Context, decltype(&atb::DestroyContext)> context_{
      nullptr,
      atb::DestroyContext};
  std::unique_ptr<atb::Operation, decltype(&atb::DestroyOperation)> operation_{
      nullptr,
      atb::DestroyOperation};
  uint64_t workspace_size_ = 0;
};

PagedAttentionGraphOperation::PagedAttentionGraphOperation(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor block_table,
    torch::Tensor output,
    double scale)
    : impl_(std::make_unique<Impl>(std::move(query),
                                   std::move(key),
                                   std::move(value),
                                   std::move(block_table),
                                   std::move(output),
                                   scale)) {}

PagedAttentionGraphOperation::~PagedAttentionGraphOperation() = default;

void PagedAttentionGraphOperation::prepare(
    const std::vector<int32_t>& kv_seq_lens,
    torch::Tensor& shared_workspace) {
  impl_->prepare(kv_seq_lens, shared_workspace);
}

void PagedAttentionGraphOperation::execute() { impl_->execute(); }

uint64_t PagedAttentionGraphOperation::workspace_size() const {
  return impl_->workspace_size();
}

}  // namespace xllm::kernel::npu
