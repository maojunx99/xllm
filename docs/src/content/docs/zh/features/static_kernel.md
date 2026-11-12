---
title: "CANN Static Kernel"
sidebar:
  order: 82
---

## 使用

NPU 启动参数：

```shell
--enable_static_kernel=true \
--static_kernel_cache_dir=/path/to/private/cache
```

默认关闭。缓存目录为空时使用 `$XDG_CACHE_HOME/xllm/static_kernel`，或
`~/.cache/xllm/static_kernel`。容器需要提供 `op_compiler`、
`aclopStartDumpArgs`、`aclopStopDumpArgs` 和
`aclnnReselectStaticKernelWithPath`；CANN 9.1 提供这些接口。
该实现直接接入 CANN，不依赖 TorchAir 或 FX。

## 启动流程

1. 所有 worker 在各自计算线程上查询启动索引。只有每个 worker 都命中且安装
   文件校验通过，才直接在其计算队列中加载静态包，跳过参数采集和采集校准。
   同一节点、共享缓存目录且编译环境一致的 worker 必须选择相同的静态包。
2. 任一 worker 未命中时，全部 worker 走完整采集流程：同一进程共用一次
   CANN dump 会话，执行原有合成请求和 decode batch 档位的 eager 校准。
   校准期间，C++ 和 Python executor 都跳过图捕获及重放。
3. 所有 worker 同步设备和任务队列后停止采集。通过现有 worker RPC 汇总各进程
   的采集目录；同一节点、共享缓存目录且编译环境一致的进程组成一个编译组。
   每组只由一个进程合并、去重所有描述，复用编译缓存，或调用一次
   `op_compiler -f true` 并安装到独立缓存目录；编译器仍可使用多个 CPU 工作进程。
   不写全局 OPP，不新增 Gloo 依赖。
   各组编译完成后，协调端将静态包结果发送给对应 worker，再分别加载。
4. 所有 worker 成功重新选核后，原子发布启动索引。队列模式 2 在任务队列线程
   上调用选核接口，并等待队列排空。
   同时在调用线程配置相同静态包，覆盖模式 1 在调用线程查询 workspace、
   模式 2 在队列线程查询 workspace 的两条路径。此步骤不执行模型校准。
5. 执行原有性能预热和图捕获，然后启动服务。启动索引不保存进程内设备地址，
   不替代本次进程的图捕获。

启动索引包含实际模型参数、并行配置、dtype、KV/状态池形状与步长、执行配置、
二进制和 TorchNPU 库内容、模型 JSON 元数据、SoC、确定性模式、CANN 与自定义算子
环境。只读取权重元数据，不扫描权重内容；同布局的权重数值不影响静态算子签名。
网络端口、日志路径和物理设备编号不参与索引。配置或代码变化、缓存缺失或损坏时，
在改变选核状态前选择完整采集流程；加载或编译失败直接报错。

编译组通过宿主机 boot ID、缓存目录实际设备号与 inode，以及编译环境指纹识别，
不会仅凭相同的路径字符串假定远端节点或不同容器共享文件。
相同算子描述按完整 JSON 内容规范化后去重；不同 shape、属性和运行时地址均保留，
不通过删除 `ffts_addr` 等属性强行合并。各 worker 的启动索引指向组内同一个编译结果。

原有编译缓存仍有效。升级后首次启动需要完成一次采集并建立启动索引，后续相同
启动配置可直接加载。新的算子签名仍会产生新的编译缓存键。编译与安装日志保留
在缓存的 `build-*` 目录，启动查询和采集记录保留在 `sessions/startup-*`。

普通 Linear 的 NZ 权重仍在首次实际 forward 时按需转换，保留原有语义。
`Linear` 也可能仅作为卷积等权重的保存容器，不能仅按模块类型遍历转换。

## 覆盖范围

此功能没有按模型名称、TP 或 EP 大小限制启用范围，使用每个 rank 的实际算子
描述生成静态包。校准档位决定静态核覆盖范围，不限制运行时输入；未命中的
shape 继续使用 CANN 的动态选核。CANN 不支持静态编译的算子、自定义直接 launch
的 kernel 和 ATB 内部算子，不会因为开启此选项自动变成静态核。

该功能可以与图模式同时使用，也可以用于 eager 执行。它不改变 attention
graph task update、tiling 更新、HCCL 确定性设置，也不启用 SuperKernel。
静态核的性能收益需要按实际模型和并行策略测量。

## 验证

编译缓存的 CPU 测试：

```shell
python3 -m unittest discover -s tests/scripts -p test_static_kernel_compile.py -v
```

NPU 集成测试 target 为 `static_kernel_test`。它验证同进程多 worker 采集收尾、
两种 shape 的去重、编译与私有路径加载、逐值正确性，以及未采集 shape 的动态执行。
同一缓存目录运行两次，可用 `XLLM_STATIC_KERNEL_TEST_EXPECT_STARTUP_HIT=0/1`
分别断言第一次采集、第二次跳过采集直接加载。
可分别使用 `TASK_QUEUE_ENABLE=0`、`TASK_QUEUE_ENABLE=1` 和 `TASK_QUEUE_ENABLE=2` 验证两种队列模式。
设置 `XLLM_STATIC_KERNEL_TEST_DEVICES=2` 可让两个 worker 分别使用两个可见设备。
该测试需要实际 NPU 和 CANN 编译器，首次执行会产生编译开销。

### 算子黑名单

`tools/static_kernel_blacklist.json` 按 CANN OpType 精确列出不参与静态编译的算子，随编译助手安装。源码修改后需重新构建或安装；已部署版本可直接修改编译助手旁的同名 JSON，然后重启服务。黑名单参与编译缓存和启动缓存的指纹，旧缓存不会被误用。

当前排除 `RepeatInterleave`：CANN 9.1 的参数采集会直接读取其设备端依赖值并崩溃。MTP、展开解码元数据、解码掩码和 DFlash 的调用通过公共封装，仅在启动采集期间将该算子放在 Host 求值并回传原设备。正常预热、图采集和在线推理仍使用原 NPU 动态算子，不更改模型、TP/EP 或输入形状的支持范围。编译助手也会过滤已存在的该类型 JSON。其他新增黑名单项只排除静态编译；若同样存在采集故障，需另行处理其采集路径。
