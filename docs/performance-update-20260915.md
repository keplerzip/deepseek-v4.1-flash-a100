# 20260915-perf1：A100 性能更新

本次更新继续使用已有 `deepseek-v4.1-flash-a100:20260913-r1` 镜像和同一份权重，以经过哈希校验的只读文件覆盖加载修正。目标机不下载资源、不安装宿主软件，只使用普通用户和 `sudo -n docker`。**构建机没有 GPU；新代码的 A100 数值、完整模型输出和速度需要目标机验收。README 原性能数据继续作为旧配置的历史记录。**

## 已接入

| 项目 | 当前默认与范围 |
|---|---|
| NCCL | 自动选择算法和协议，显式取消原来的 Ring/Simple 限制；保留其他通信设置和 custom all-reduce 配置 |
| mHC | 移植 vLLM #56633 的 delayed post/pre 融合、辅助状态复用及 mean 融合；主模型和 DSpark 草稿均接入 |
| A100 调优 | 保留旧 backport 的小 token 选择、16-token 截止及大批量 cuBLAS 路径；不套用 GB300 的阈值 |
| 图像预处理 | 移植 #56554，移除 V4.1 不应有的 compressor-alignment padding，修正图像 token 预算与 span 配置 |
| 模型契约 | 仍为 `DeepSeek-V4.1-Flash`、262144 上下文、k=5、Engram RAM、image=999，按实际 KV 计算至少 C32 |

新 mHC epilogue 单独放在 `vllm/model_executor/kernels/mhc/dsv41_sm80.py`，保留共享 V4 kernel 的旧 ABI。切换回旧 mHC 时同时恢复主模型和 DSpark 的 layer 返回值接口。大批量 verifier 不强行进入小形状融合，辅助状态处理则可继续省掉重复计算。

Engram 异步预取、HBM 搬迁、NUMA 策略此次没有默认启用。公开新补丁仍有平台和执行方式限制，本次继续使用已运行的同步 RAM 查表路径。不能从其他硬件上的算子倍率推算本机整模型提速。

## 升级已有隔离实例

在目标机解压小更新包后执行，路径指向已有 deploy：

```bash
bash install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

安装器先校验文件与已安装镜像，只备份并更新部署源码，保留权重、镜像、实例配置中的模型路径和 containerd 路径、运行证据。已有 selected/tuning 配置中的 k 和图片数会对齐为 5/999。备份记录在 `deploy/runtime/update-backups/`。

`--restart` 会停止本项目服务并启动更新版。新 mHC 第一次启动先在一张 SM80 上测试真实 kernel、辅助状态和 CUDA Graph，不加载模型；同一补丁、镜像和驱动通过后不重复该检查。随后模型加载和 KV 容量校准仍可能需要数分钟。若新的检查或启动失败，安装器恢复备份源码并尝试启动原服务，保留失败日志。

省略 `--restart` 只更新文件。此时现有服务不会自动加载更新，之后执行 `bash ./restart.sh`。若后续发现问题，可在 deploy 内执行 `bash ./rollback-update.sh`，恢复最近一次更新备份并重启。回退会拒绝覆盖更新后又被手工修改的源码文件，避免丢失修改。

## 验收与速度对照

```bash
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
```

后者分别测 off/high 的 C1 与 C32，记录每请求 decode、TTFT、输入与输出 token、缓存命中和 DSpark 草稿接受量。请暂停其他客户端流量。纯文本成绩不能代替多图历史成绩。

可选小图检查：

```bash
bash ./tests/performance-benchmark.sh --images 8
```

这些是包内的小 OCR 图片，不代表真实长截图历史。要复现慢请求，把脱敏的 **Chat Completions 请求体**保存为 `deploy/runtime/benchmark-chat.json`，图片使用已有 data URI；再运行：

```bash
bash ./tests/performance-benchmark.sh --request-file /state/benchmark-chat.json
```

测试保留该请求的消息和图片，固定采样、k=5、1024 输出 token，移除提前停止设置。它先预热两次，随后复用同一历史前缀；日志只记录请求哈希和统计，不写入图片正文。请求应要求持续文本生成；如果工具调用或其他停止条件导致输出不足 1024 token，会报告失败。原始私有 JSON 由操作者自行保管。

用于拆分收益的两项诊断开关位于 deployment.env：

```text
PERF_NCCL=auto
PERF_MHC=1
```

`legacy/0` 对照原通信与旧 mHC，`auto/0` 单独测通信，`auto/1` 为更新后的默认组合。每次改动后用 `restart.sh` 重启，按 A→B→A 的顺序测同一输入。图像修复在这三个组合中均启用，所以它们只用于拆分通信/mHC 收益；完整旧版对照使用备份回退。配置与容量会复制进每轮性能报告，不能把 C1/C32 或不同输入混比。

## 源码与验证边界

- [vLLM #56633](https://github.com/vllm-project/vllm/pull/56633)，固定合并提交 `5372e72a9884c141874465f0eb8054a4a81da6bb`；感谢 zyongye 与 vLLM 维护者。
- [vLLM #56554](https://github.com/vllm-project/vllm/pull/56554)，感谢 Isotr0py 与 vLLM 维护者。
- 继续使用 wtdcode/vllm-backport 的 SM80 算子、PyTorch、Triton、TileLang、NCCL；mHC epilogue 保留其 vLLM/SGLang 来源与 Apache-2.0 版权声明。

CPU 构建验证覆盖文件完整性、基底匹配、回退 ABI、实际解码层控制流与 CPU 参考算子对照、图像 token 处理、warmup key、fake tensor 接口及更新事务。它们不能代替 CUDA 数值测试与目标机完整验收，不填入预期 tok/s。
