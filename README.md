# DeepSeek-V4.1-Flash on 8×A100 — v1.0.0

## 实测性能：默认 DSpark k=5

**8×A100-SXM4-80GB + NVSwitch，256K 上下文，TP8，Engram 主存卸载。** 同机 k5/k7 对照结果如下，最终默认选择 **k=5**。

| 思考模式 | DSpark k | 单流 decode tok/s | 单流端到端 tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|---:|---:|
| off | **5** | 139.80 | 137.22 | **1379.01** | **24.56** |
| off | 7 | 146.04 | 143.50 | 1359.60 | 25.28 |
| high | **5** | **109.95** | **108.58** | **1039.91** | **34.93** |
| high | 7 | 95.69 | 94.44 | 960.82 | 37.96 |

high 模式下，k=7 的单流速度与 C32 总吞吐分别下降约 **12.97% / 7.61%**。off 模式下，k=7 单流约 +4.46%，但 C32 总吞吐约 −1.41%。因此 **v1.0.0 固化 k=5**；不会因 k7 功能测试通过就声称它更快。

**测试口径：** 操作者回传的隔离目标机实测；固定顺序 5→7，每组 2 次预热、5 个单流样本、200 个 C32 请求，每请求 1024 输出 token，off/high 分别测量。高思考模式的 TPS 包含思考 token；C32 为短输入负载，并非 32 个满 256K 请求同时驻留 KV。五个单流样本不足以证明小幅差异具有统计显著性。

另一次较早的 **k5 / high / C56** 验收结果单独记录：

| 指标 | 实测结果 |
|---|---:|
| 单流 decode 中位数 | **101.73 tok/s** |
| C56 总输出吞吐 | **1334.67 tok/s** |
| 请求数 / 失败数 | **200 / 0** |
| C56 请求延迟 P50 / P95 | 39.38 / 47.53 秒 |
| C56 TTFT P95 | 1.65 秒 |
| 热前缀命中 | **14208 / 14398 token，98.68%** |
| 最大单请求检索输入 | **257482 token** |

这组 C56 成绩与上面的 C32 对照条件不同，不能直接合并比较。文本/SSE、三协议流式与非流式工具调用、1/2/4 图 OCR、长输入检索和 C32/C56 短请求均报告通过。**C56 是调度并发，不代表 56 个满 256K 窗口同时驻留。**

[完整性能报告与方法](docs/performance.md) · [机器可读实测记录](reports/) · [测试代码](deploy/tests/dspark_benchmark.py)

**2026-09-15 性能更新已接入源码：** NCCL 自动选择、保留 SM80 调优的 V4.1 mHC 融合、图像 padding 修复。k=5、256K 和并发公式不变。上述表格仍是更新前实测；本次新增代码尚无目标 A100 速度成绩。[更新、回退与多图对照说明](docs/performance-update-20260915.md)

目标机原始逐请求 JSONL 留在隔离环境，尚未传到构建工作区；仓库记录明确标注“用户回传”。2h/24h 稳定性、满窗口并发驻留、Engram 位置对照和空闲 KV 保留问题均没有在本次发布中宣称已验证。

## 项目与默认方案

本项目提供 `DeepSeek-V4.1-Flash` 在八卡 A100 上的离线部署、验收和运维工具，是从实际隔离环境交付整理的 **源码发行版**。联网构建侧准备全部资源后，目标机仅用普通用户和 `sudo -n docker` 启动，无需公网下载或在宿主安装 Python 依赖。

| 项目 | 当前源码设置（基于 v1.0.0） |
|---|---|
| 唯一模型 / API 名称 | `DeepSeek-V4.1-Flash` |
| 上下文 | 262144 token，输入与输出合计 |
| 并行 | TP=8，PP=1 |
| 权重 | 官方混合 FP8/FP4，固定 ModelScope revision |
| DSpark | **固定 k=5**，CUDA Graph |
| 当前源码性能更新 | NCCL 自动选择、V4.1 mHC 融合；首次启用先做 SM80 kernel 检查 |
| Engram | 主存卸载 |
| KV | `fp8_ds_mla`，prefix caching 与命中 token 明细 |
| 并发公式 | `C=max(32, 2*floor(effective_KV_tokens/262144))` |
| 多模态 | 图片数量配置为 999（与固定版本 vLLM 默认值一致）；拒绝公网媒体抓取 |
| API | Chat Completions、Responses、Anthropic Messages |
| 访问 | `127.0.0.1:8005` 和本机 Docker bridge，客户端免密 |
| NewAPI | 接已有实例；不安装、不修改网关 |

参考机曾标定 N=28、C=56。每次配置启动按实际 KV 池重新计算 C，不把 TP8 容量重复乘八。权重、Docker 镜像、商业 CLI 二进制和目标运行数据均不进入 Git，需在构建侧准备。[构建与离线包](docs/building.md)

## 快速开始

完整离线资源目录：

```text
deepseek-v4.1-flash-a100/
├── DeepSeek-V4.1-Flash/       # 固定 revision 完整模型快照
└── deploy/
    ├── deployment.env
    ├── images/runtime-image.tar
    ├── configs/runtime.json   # 默认 k=5
    ├── manifests/deploy.sha256
    ├── deploy.sh
    └── runtime/               # 运行生成，不入 Git
```

在普通用户拥有的 `deploy` 目录执行：

```bash
umask 077
mkdir -p runtime
set -o pipefail
bash ./deploy.sh 2>&1 | tee runtime/first-deploy.log
```

首次导入、模型全量校验、GPU 小算子检查和模型加载均需要时间；KV 标定可能要求重启。启动后访问 `http://127.0.0.1:8005/v1/models`。容器内访问 `http://host.docker.internal:8005/v1`，Linux 容器需有 `host.docker.internal:host-gateway` 映射。[详细部署](docs/deployment.md)


## 文档

- [准备离线包](docs/building.md) · [部署](docs/deployment.md) · [运维与恢复](docs/operations.md)
- [性能报告](docs/performance.md) · [KV 缓存](docs/cache.md) · [故障定位](docs/troubleshooting.md)
- [20260915 性能更新与回退](docs/performance-update-20260915.md)
- [协议、NewAPI 与 CLI](docs/clients.md) · [固定实现](docs/architecture.md)
- [上游致谢](docs/acknowledgements.md) · [变更记录](CHANGELOG.md)

## 致谢与许可证

感谢 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 提供模型，[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) / lazymio 提供 SM80 适配与镜像，以及 [vLLM](https://github.com/vllm-project/vllm)、[PyTorch](https://github.com/pytorch/pytorch)、[Triton](https://github.com/triton-lang/triton) 和 [NVIDIA NCCL](https://github.com/NVIDIA/nccl) 的基础设施。本仓库负责固定版本、离线交付、接口修正、验收和结果记录，不将上游模型或 SM80 实现归为原创。

原创部署代码采用 Apache-2.0；第三方代码保留其版权与许可证，模型权重遵循模型自身许可。见 [LICENSE](LICENSE)、[NOTICE](NOTICE)、[完整致谢](docs/acknowledgements.md)。
