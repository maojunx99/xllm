# Qwen3-VL W8A8 Decode 算子优化总结

## 1. 优化目标和适用范围

本次优化面向 Qwen3-VL-32B W8A8、TP=8 的低并发 decode 场景，重点关注单次 decode 推理 `M=1–9` 个 token，并覆盖图模式 bucket `M=1–16`。

目标矩阵如下：

| 投影 | 每卡矩阵形状 |
| --- | --- |
| Gate/Up | `[M,5120] × [5120,6400]` |
| Down | `[M,3200] × [3200,5120]` |

输入激活为 INT8 ND，权重为 INT8 FRACTAL_NZ，per-channel dequant scale 为 FP32，bias 为 INT32。

自定义路径只在以下条件全部满足时启用：

- Qwen3 W8A8；
- hidden size 为 5120；
- TP 后每卡 intermediate size 为 3200；
- BF16 模型路径；
- ACLNN matmul backend；
- ACL Graph paged attention；
- Ascend A2；
- 没有 LoRA；
- 没有 FlashComm；
- 自定义算子已经正确注册；
- 输入 rank 为2或3；
- 实际 token 数在1到16之间。

不满足条件时继续使用原有实现，因此 prefill、较大 batch、其他模型和其他 shape 不受影响。

> 当前正式定制优化覆盖 Gate/Up 和量化 Down Projection。QKV 虽然已经是 W8A8，但仍走原来的 ACLNN QuantMatmul，没有进入本次定制核。

## 2. Gate/Up 融合算子

新增算子：`QuantMatmulNzSwigluDecode`。

原始路径大致为：

```text
INT8 QuantMatmul
    ↓
INT32 accumulator
    ↓
bias + dequant → BF16
    ↓
独立 SwiGLU kernel
```

新算子在一个物理核中完成：

```text
INT8 activation × INT8 NZ weight
    ↓
INT32 gate/up accumulator
    ↓
bias + per-channel FP32 dequant
    ↓
SiLU(gate) × up
    ↓
BF16 输出或直接量化为 INT8
```

这样消除了：

- 独立 SwiGLU kernel 的启动开销；
- 中间 BF16 tensor 的 GM 写回和再次读取；
- QuantMatmul 与 SwiGLU 之间的核间调度间隙；
- Down 量化时独立 requant kernel 和中间数据搬运。

### 2.1 Matmul tiling

Gate/Up 固定使用：

- 20个 mixed AIC/AIV core；
- BF16输出路径 L1 tile：`16 × 160 × 512`；
- INT8输出路径 L1 tile：`16 × 160 × 1024`；
- L0 tile：`16 × 128 × 256`；
- 双 workspace stage；
- Catlass async preload；
- 权重使用 NZ `zN` layout；
- 关闭权重 L2 cache，避免一次性大权重污染 L2；
- 非对称 L1 N=160 / L0 N=128。

INT8输出路径额外启用 K-direction shuffle。`K=5120` 被分成5个
L1 K tile 后，不同 core 从不同 K tile 起步，降低连续 Gate/Up、Down
大权重流量在同一内存分区集中爆发的程度。BF16输出路径保留原来的
K tile 和遍历顺序，因此本轮调度优化不会改变该路径。

为支持非对称 N tile，对 Catlass 增加了受编译宏控制的 L1/L0 shape 支持。该宏仅为新算子开启，不改变其他 Catlass kernel。

### 2.2 SwiGLU epilogue

Epilogue 针对小 M 做了以下优化：

- Gate 和 Up accumulator 分开读取；
- bias、scale 提前搬入 UB；
- 手工完成 FP32 SiLU，降低通用激活模板的额外开销；
- `M>=2` 使用二维 burst，一次搬运多行；
- `M=1` 使用轻量逐行路径，避免二维描述符的固定成本；
- AIC/AIV 使用 cross-core event 协作；
- 减少不必要的 workspace clear 和通用 KFC queue 初始化。

### 2.3 两种输出模式

算子通过编译期 tiling key 支持两种输出模式。

#### BF16输出

用于 Down 仍接收 BF16 输入的模型，保留原始数值语义：

```text
QuantMatmul BF16 rounding
→ BF16 SwiGLU
```

#### INT8输出

用于 Down Projection 也量化的模型：

```text
FP32 dequant
→ FP32 SwiGLU
→ FP32 quant
→ INT8
```

INT8 路径不经过中间 BF16 rounding，从而与官方 `npu_dequant_swiglu_quant` 保持完全一致。

## 3. Down Projection 定制算子

新增算子：`QuantMatmulNzDecode`。

算子完成：

```text
INT8 activation × INT8 NZ weight
→ INT32 accumulate
→ INT32 bias
→ FP32 per-channel dequant
→ BF16 output
```

### 3.1 按 M 分桶定制

Down shape 与 Gate/Up 差异较大，因此使用独立的 tiling 策略。

| M 范围 | L1 tile | L0 tile | 设计目的 |
| --- | --- | --- | --- |
| M=1 | `16×64×3200` | `16×64×512` | 尽量覆盖完整 K，减少 K 循环调度 |
| M=2–4 | `16×128×1792` | `16×128×256` | 平衡 N 并行和 K 搬运 |
| M=5–16 | `16×256×896` | `16×128×256` | 增大 N tile，提高多行搬运效率 |

共同优化包括：

- 20 core；
- 两级 workspace stage；
- K-direction shuffle，平衡不同 core 的权重搬运；
- 权重 L2 bypass；
- 多行二维 accumulator load；
- 多行二维 BF16 output store；
- 去掉未使用的 KFC workspace 清零。

Host tiling 只接受以下矩阵：

- Gate/Up：`K=5120,N=6400`；
- Down：`K=3200,N=5120`；
- `1 <= M <= 16`。

其他 shape 会被拒绝，避免误用定制 kernel。

## 4. 图模式接入

新增了两个正式 ACLNN wrapper：

- `QuantMatmulNzDecodeOperation`；
- `QuantMatmulNzSwigluDecodeOperation`。

它们作为 ATB graph 中的正式 operation node 存在，因此自定义 kernel 会被图捕获，不是图外 C++ wrapper 临时调用。

Qwen3 decoder 会建立两套 decode graph node：

```text
普通 decode graph
优化 decode graph（包含自定义 NZ kernel）
```

运行时按照模型配置、算子可用性和 token 数选择对应 graph。自定义算子不存在、shape 不匹配或功能条件不满足时，继续执行普通 graph。

该方案不需要增加用户专属开关，同时保证其他 shape 和功能组合不发生性能回退。

## 5. Down 量化链路

当 checkpoint 中检测到 Down Projection 是 W8A8 时：

- Gate/Up 自定义算子直接输出 INT8；
- Down 接收 INT8 activation；
- Down 使用 `QuantMatmulNzDecode`；
- Prefill 和 eager 参数保留官方兼容路径；
- 只有优化 decode graph 使用定制 Down kernel。

完整量化 MLP 链可以变为：

```text
AddRMSNormQuant
→ QuantMatmulNzSwigluDecode(INT8 output)
→ QuantMatmulNzDecode
→ residual/reduce
```

从而避免：

```text
Gate/Up INT32
→ 独立 DequantSwiGLUQuant
→ Down QuantMatmul
```

## 6. 精度验证

### 6.1 BF16输出

验证范围：

- M=1–16；
- 与原始官方路径逐 bit 一致；
- 8 streams × 100轮交替执行一致。

BF16 路径保留了两个关键 rounding 边界：

1. QuantMatmul dequant 后转 BF16；
2. SwiGLU 结果转 BF16。

因此没有为了性能改变当前 BF16 模型的数值语义。

### 6.2 INT8 Down输入

验证范围：

- M=1–16；
- quant offset 分别测试0和-3；
- 共435,200个 INT8 输出值；
- 与官方 `QuantMatmul INT32 + npu_dequant_swiglu_quant` 全部一致；
- 8 streams × 100轮交替执行一致。

初版曾在 FP32 dequant 后错误加入 BF16 rounding，导致约5.29%的 INT8 值相差1。通过 rounding probe 定位后，已改成保持全 FP32 直到最终 INT8 quant，目前逐值完全一致。

## 7. 性能收益

### 7.1 当前 BF16 Gate/Up

控制微基准中的最新融合算子延迟：

| M | 自定义融合核延迟 |
| ---: | ---: |
| 1 | 30.78 µs |
| 2 | 30.96 µs |
| 4 | 31.09 µs |
| 8 | 31.32 µs |
| 16 | 31.96 µs |

真实业务 profiling 的配对结果：

- 原 QuantMatmul + 激活链 p50：约39.80 µs；
- 新融合核 p50：约35.66 µs；
- 每层减少约4.14 µs；
- 降幅约10.44%；
- 64层累计每次 decode 约减少265 µs。

不同 profiling 期间设备负载会造成绝对延迟波动，因此性能判断采用同一环境中的配对差值，而不是跨 profiling 直接相减。

### 7.2 Gate/Up直接输出INT8

与官方 `QuantMatmul INT32 + DequantSwiGLUQuant` 相比：

| M | 节省时间 | 降幅 |
| ---: | ---: | ---: |
| 1 | 15.16 µs | 32.47% |
| 2 | 15.94 µs | 33.59% |
| 4 | 15.65 µs | 33.30% |
| 8 | 14.86 µs | 31.79% |
| 16 | 14.57 µs | 31.09% |

这里的31%–33%是“Gate/Up Matmul + dequant + SwiGLU + quant整条链”的融合收益，不是 Down Matmul 本身的收益。

### 7.3 Down Matmul

Down 单算子的提升小于 Gate/Up 融合：

- 不同 M 和设备状态下约为几个百分点到十余个百分点；
- 已记录示例中，M=8 约3.2%，M=16 约8.2%；
- 业务 shape 加权结果曾约12.2%。

Down 定制算子的主要价值不仅是单核加速，还在于让 Gate/Up 可以直接输出 INT8，消除中间量化链。

### 7.4 QKV回退分析和联动优化

真实量化权重 profiling 中，QKV 仍使用官方 `aclnnQuantMatmul`，但在
开启 Gate/Up 和 Down 定制核后，QKV 平均延迟由19.223 µs增至
20.408 µs，即增加1.185 µs（6.17%）。退化只集中在前面执行了
定制 MLP 链的32层，说明问题不是 QKV kernel 本身被替换，而是前序
大权重读取改变了后续 kernel 的带宽稳态。

使用真实算子顺序、32组独立 NZ 权重和相同 stream 做2×2归因：

| Gate/Up | Down | QKV变化 | 结论 |
| --- | --- | ---: | --- |
| 官方 | 定制 | +0.032 µs | Down单独影响很小 |
| 定制 | 官方 | +0.608 µs | Gate/Up是主要单项来源 |
| 定制 | 定制 | +1.445 µs | 连续定制链还有约0.8 µs交互项 |

最终只对 Gate/Up 的 INT8输出路径启用 `L1 K=1024 + K shuffle`。同一
profiling 内的配对差值变为：

| 指标 | 修改前定制链 | 最终定制链 | 改善 |
| --- | ---: | ---: | ---: |
| QKV附加延迟 | +1.445 µs/层 | +1.168 µs/层 | 下降19.2% |
| Gate/Up+Down+QKV净节省 | 12.125 µs/层 | 13.146 µs/层 | 增加1.021 µs/层 |

按受影响的32层估算，本轮调度调整可额外减少约32.7 µs/decode 的
MLP到QKV局部链时间，其中约8.9 µs来自直接缩小 QKV 尾部回退。
由于真机服务 profiling 的绝对负载不同，最终端到端数值仍应以相同
业务请求重新做开关配对为准。

Gate/Up 单算子复测结果如下：

| M | 最终延迟 |
| ---: | ---: |
| 1 | 30.80 µs |
| 2 | 30.94 µs |
| 4 | 30.99 µs |
| 8 | 31.20 µs |
| 16 | 31.82 µs |

相对修改前定制核，重复测试中各 bucket 约提升1.2%–2.5%，没有观察到
目标 M 范围内的回退。M=1/2/4/8/16 的最终 INT8 输出 SHA256 均与
修改前定制核一致，说明本轮变化只改变调度，不改变数值结果。

## 8. 带宽和理论上限

针对 NZ 权重搬运的独立测试：

- M=1：约1.568 TB/s；
- M=16：约1.519 TB/s；
- 板卡可用上限约1.6 TB/s。

对应约为理论带宽的98%和95%。

这说明小 M decode 下计算单元利用率低并不代表还有大量计算优化空间。主要瓶颈是每层必须读取的大量 INT8 权重，继续调整普通 Matmul tile 的剩余空间已经较小。后续更大的收益需要依靠跨算子融合，减少 kernel launch、中间 tensor 和核间调度间隙。

## 9. 配套 Loader 修复

以下修改不是计算 kernel 本身，但用于保证 W8A8 模型能够稳定进入新路径。

### 9.1 权重加载卡住修复

HF loader 从手工 `BlockingCounter` 改为异常安全的 `TaskGroup`。某个 safetensors 加载线程抛出异常时，不会因为 counter 未递减而永久等待。

### 9.2 mmap权重物化

只对 Qwen3 W8A8 loader 启用 CPU contiguous materialization，避免 mmap-backed tensor 在异步传输到 NPU 时卡住。其他模型默认关闭，不承担额外整权重复制。

### 9.3 Vision W8A8兼容

Vision encoder 暂不支持对应 INT8 linear 时，根据下式恢复 BF16 权重：

```text
FP weight = INT8 weight × deq_scale / input_scale
```

同时校验 scale 有限、非零，并且 channel 数量一致，然后继续使用原 Vision 路径。

### 9.4 O-proj和Down自动识别

Loader 根据 checkpoint 中 bias、scale 等参数是否真实存在，判断 Down 和 O-proj 是否量化。参数不存在时回退 BF16 配置，避免把占位 tensor 当作量化参数。

## 10. 已测试但未采用的实验

以下方案已经实际编译或测量，但因为无收益或产生回退而没有合入：

- L1 N=320；
- L0 N=160/K=192；
- K=640；
- L0C双缓冲；
- resident A；
- 参数提前搬运；
- 单 MTE2路径；
- 去掉最终 MTE3 wait；
- Catlass callback变体；
- constexpr scheduler；
- AscendC通用Silu API；
- 跳过更多 workspace 清理；
- 多种 N/K tile组合。

针对本轮 QKV 回退还验证并拒绝了：

- Down恢复普通 L2 cache：QKV仅改善约0.06 µs，但Down自身慢约0.45 µs；
- Down关闭 K shuffle：整链收益低于最终方案；
- Gate/Up `L1 K=512/768`：QKV附加延迟仍约1.27–1.29 µs；
- Gate/Up `L1 K=1280`：M=1–16均慢于 `K=1024`，M=16回退更明显。

### 10.1 AddRMSNormQuant物理融合

还验证了下面的单核物理融合原型：

```text
AddRMSNormQuant
+ Gate/Up QuantMatmul
+ SwiGLU
```

设备侧测量结果：

- 候选融合核：35.21 µs；
- 官方 AddRMSNormQuant：3.46 µs；
- 当前 Gate/Up 融合核：29.02 µs；
- 候选比两个 kernel 的内部时间总和慢2.82 µs；
- 即使扣除原图约1.17 µs的核间隙，仍预计回退约1.65 µs。

因此该原型没有合入正式代码。

## 11. 当前未覆盖的部分

当前尚未完成以下定制优化：

- QKV Projection：`K=5120,N=1280`；
- Attention O-proj 的专属小 M kernel；
- AddRMSNormQuant + QKV融合；
- AddRMSNormQuant + Gate/Up的有效物理融合；
- Attention与QKV的跨算子融合。

当前 `QuantMatmulNzDecode` host tiling 会拒绝 QKV shape，因此 QKV 仍使用官方 `aclnnQuantMatmul`。

## 12. 代码和提交位置

优化 worktree：

```text
/home/g00510989/xllm_xmj/xllm-qwen3vl-decode-opt
```

分支：

```text
perf/qwen3vl-decode-kernels
```

提交：

| 仓库 | Commit | 说明 |
| --- | --- | --- |
| xLLM根仓库 | `80c0cf1e` | Qwen3-VL loader、图节点和shape路由 |
| xllm_ops | `fe4ccc9` | 两个NZ decode定制算子 |
| Catlass | `48e2d23` | 非对称L1/L0 tiling支持 |
| xllm_atb_layers | `47bc8ce` | ACLNN wrapper和ATB图内接入 |

主要源码：

- `third_party/xllm_ops/xllm_ops/quant_matmul_nz_swiglu_decode/`
- `third_party/xllm_ops/xllm_ops/quant_matmul_nz_decode/`
- `third_party/xllm_atb_layers/operations/aclnn/ops/quant_matmul_nz_swiglu_decode_operation.*`
- `third_party/xllm_atb_layers/operations/aclnn/ops/quant_matmul_nz_decode_operation.*`
- `xllm/core/layers/npu/npu_qwen3_decoder_layer_impl.*`
- `xllm/core/layers/npu/loader/qwen3_decoder_loader.*`
