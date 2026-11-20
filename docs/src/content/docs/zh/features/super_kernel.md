---
title: "CANN SuperKernel"
sidebar:
  order: 83
---

## 使用

```shell
--enable_graph=true \
--enable_static_kernel=true \
--enable_super_kernel=true \
--super_kernel_auto_op_parallel=true \
--static_kernel_cache_dir=/path/to/private/cache
```

`enable_super_kernel` 默认关闭。它要求 NPU ACL Graph、提供 SuperKernel
接口的 CANN/TorchNPU，并依赖静态核编译。CANN 编译器能接受
`--enable_super_kernel`、TorchNPU 能调用优化接口，均不能单独证明运行环境
支持融合。

运行时还需要 `aclmdlRITaskGetParams` 等图任务接口可用。该接口依赖软件 SQ
能力，包括 TS 固件支持和驱动 SQ 动态绑定能力。仅更新容器中的 CANN，
不会更新宿主机的驱动和固件。以实际接口调用与融合结果为准，不能仅按
容器名称或 CANN 版本号判断支持情况。

## 执行流程

启动校准沿用静态核的实际算子采集流程，编译时增加
`op_compiler --enable_super_kernel`。此配置参与缓存键计算，普通静态包和
SuperKernel 静态包使用不同的安装目录；原有普通静态缓存继续有效。

C++ 和 Python ACL decode graph runner 都在模型 forward 前后插入
`xllm_model` scope，在 capture 完成后调用 TorchNPU 的
`NPUGraph::super_kernel_optimize` / `NPUGraph.super_kernel_optimize`。
优化只发生在每个 graph 首次 capture 时，replay 不重复编译。

缓存维护策略与 vLLM-Ascend 对齐：`dcci_after_kernel_end=[".*"]`。
`super_kernel_auto_op_parallel` 默认关闭；设为 `true` 时，编译和运行时均传入
`auto_op_parallel=1`，让 CANN 在保持依赖关系的前提下调整 Cube/Vector 排布。
此配置不改变 scope，也不增加模型或并行策略限制；配置参与静态包和启动索引的
缓存标识，切换时不能误用另一种排布配置的缓存。其性能效果需要实测。
不启用激进依赖绕过。原有多流事件、通信和 graph task update 依赖交由
CANN 在融合时处理；不根据模型名称、TP 或 EP 大小添加开关限制。

## 覆盖与验证

SuperKernel 处理算子二进制融合，具体融合范围由 CANN 和图中的任务依赖决定。
它不会自动把 ATB 内部 kernel、直接 launch 的自定义 kernel 变成 ACLNN
静态核，也不能保证每个算子序列都能融合。

日志中的 `SuperKernel graph optimization completed` 只确认优化接口调用
成功。需要检查 graph debug dump 或 profiling 中生成的 SuperKernel 节点，
核对融合前后的任务数量、核数、延迟和正确性。

NPU 集成测试 target 为 `super_kernel_test`。测试覆盖多个 batch shape、
辅助流依赖、优化后的 graph task 参数更新，并在每次 replay 后立即检查
输出。设置 `XLLM_SUPER_KERNEL_TEST_CACHE` 可保留编译产物和 graph debug dump。

## 当前验证状态

框架接入及静态包编译已通过编译和 CPU 测试。74 环境上 CANN 9.1、9.2
的最小图均在融合阶段失败：`aclmdlRITaskGetParams` 返回不支持，底层日志
报告 `CheckCaptureModelSupportSoftwareSq: tsfw does not support`，且宿主机
驱动缺少 `FEATURE_TRSDRV_SQ_SUPPORT_DYNAMIC_BIND`。该环境的融合能力未通过
验证，不能仅升级容器后开启此功能。
98 环境使用相同 CANN 9.1 Runtime、较新驱动及固件后，已通过任务参数查询，
并在最小测试中把连续 12 个 RmsNorm 合成一个 SuperKernel。batch 为
1、16、32 时，每组 3 次改变预期输出的 replay 均逐位一致。

C++ 集成测试在 `TASK_QUEUE_ENABLE=0/1` 两种队列设置下分别通过。
每种设置覆盖 batch 1/16/32、单流/辅助流、启用/关闭 graph task update，
共 72 次改变输入的 replay；每次立即核对普通图和融合图的输出。
24 对图的节点数均下降：单流从 18 降至 3，有可更新任务时从 27 降至 12；
多流从 33 降至 7，有可更新任务时从 42 降至 16。

融合成功不保证更快。12 个连续 RmsNorm 的微基准在 batch 1 时为
42.03 → 40.59 us，batch 16 为 46.01 → 51.23 us，batch 32 为
52.16 → 73.36 us。这里测量整段 replay 延迟，不能外推整模型收益。

这不代表所有静态核都可融合。例如当前编译生成的 Add、Sigmoid、Mul 和
MatMulV2 二进制可能不带 SuperKernel bind map，融合器会保留这些任务。
在 RmsNorm 和不可融合 MatMul 交替的测试中，每个 RmsNorm 各自被改写为
单算子 SuperKernel，未减少任务数并导致性能下降。因此需要检查实际
多算子融合范围，不能根据 `sk_entry` 名字或接口成功日志判断收益。

此能力检查不引入模型或并行策略限制；开启后若优化失败，会保留错误并
停止执行，不静默退回普通 graph。

2026-11-02 在 98 环境、CANN 9.1、Qwen3.5-35B-A3B BF16、TP2/EP2、
physical die 0/2 上进行同一可执行文件的开关对照。32k 输入、单并发、
12 个请求、每请求强制输出 200 token，EvalScope 1.10.0 客户端结果为：

| 模式 | TPOT (ms) | TTFT (ms) | 输出吞吐 (token/s) |
| --- | ---: | ---: | ---: |
| Static Kernel | 8.5602 | 255.63 | 101.6553 |
| Static Kernel + SuperKernel | 16.5699 | 253.26 | 56.0786 |

两组实际 HTTP 请求、数据集哈希、平均输入/输出 token 和缓存命中率均一致，
HCCL 确定性环境变量保持默认未设置。此场景 TPOT 变慢 93.57%，没有收益。

两 rank 的 profiling 按完整 decode step 归一化后，每步 kernel 数从
1079 降至 879，产生 162 个 SuperKernel，其中 40 个融合了 MoE 路由、
GroupedMatmul 和最终还原。该 MoE 融合核为 `MIX_AIC`，申请 24 个 Cube
block；共享专家 GateUp 申请 22 个 Cube block。rank 0 的 GateUp 任务
区间中位数从 6.72 增至 205.30 us，rank 1 从 6.60 增至 259.35 us，
接近各自 MoE 融合核的完整区间。PagedAttention 中位数仍约 55 us。
这些现象指向融合核的整段占核影响多流重叠；任务区间可能包括资源等待，
尚未通过硬件 stall 计数器或消融实验确认精确机制，不能把该区间全部视为
有效计算时间，也不能将不同流的耗时相加解释 TPOT。

两组的 3 条 greedy 请求和 9 道 CEval 输出文本全部一致，CEval 均为 7/9；
`computer_network_0` 均生成到 16384 token 上限。原有 Static Kernel 的
整模型精度问题未解决，不能将此次输出一致视为整模型精度验收通过。

因此本特性保留默认关闭，当前版本未通过整模型性能与精度验收。

## auto_op_parallel 完整 scope 对照（2026-11-02）

新增 `--super_kernel_auto_op_parallel=true` 后，在 98、CANN 9.1、physical die 1/3、
TP2/EP2 上使用同一二进制对照，保留完整 `xllm_model` scope。32k、单并发、
12 个请求、每请求输出 200 token，HCCL 确定性环境变量未设置。

| 模式 | EvalScope TPOT (ms) | 输出吞吐 (token/s) |
| --- | ---: | ---: |
| Static Kernel | 8.6273 | 101.0396 |
| SuperKernel auto=0 | 8.7321 | 99.6951 |
| SuperKernel auto=1 | 8.6646 | 100.9468 |

三组实际 HTTP 请求、输入输出 token、数据集哈希和缓存命中率一致。
3 个生成用例和 9 个 CEval 用例逐项一致；CEval 仍为 7/9，保留了原有
一道错误答案及一道 16k 输出截断，不能据此声称原有静态核精度问题已修复。

两种排布均通过任务队列 0/1 的原生多 shape、多流、task update 回归，
共 288 次 replay 与参考输出逐位一致。自动排布配置贯通编译和运行时，并参与
静态包、启动索引及节点编译环境的缓存标识。

profiling 在性能测试完成后单独采集，按完整 decode step 统计。
各 rank 共享专家 GateUp 任务区间中位耗时（us）为：

| 模式 | rank 0 / die 1 | rank 1 / die 3 |
| --- | ---: | ---: |
| Static Kernel | 6.600 | 7.080 |
| SuperKernel auto=0 | 59.422 | 58.181 |
| SuperKernel auto=1 | 61.162 | 59.861 |

完整 trace、CSV、请求数据库及参数清单保存在实验目录。任务区间可能包括资源等待；
不能把融合核外层与共享流的时间条交叠当成内部 Cube/Vector 算子已并行的证据。
当前开关默认关闭，未增加模型或 TP/EP 策略限制，也未缩小 scope。
