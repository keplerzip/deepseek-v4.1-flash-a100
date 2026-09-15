# 上游项目与致谢

| 项目 | 作用 |
|---|---|
| [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 模型、参考编码与部署资料。 |
| [ModelScope](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash) | 固定 revision 模型镜像和元数据。 |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) / [lazymio 镜像](https://hub.docker.com/r/lazymio/vllm-backport) | SM80、V4.1、Engram、DSpark 支持与预构建栈，是本项目核心推理基础。 |
| [vLLM](https://github.com/vllm-project/vllm) | 推理、API、KV、调度与投机解码。 |
| [PyTorch](https://github.com/pytorch/pytorch)、[Triton](https://github.com/triton-lang/triton)、[NVIDIA NCCL](https://github.com/NVIDIA/nccl) | GPU 计算与多卡通信。 |
| [NewAPI](https://github.com/QuantumNous/new-api) | 外部网关兼容目标，不随项目安装。 |
| [Codex](https://github.com/openai/codex)、[Claude Code](https://github.com/anthropics/claude-code)、[OpenCode](https://github.com/anomalyco/opencode) | 开发客户端兼容目标；二进制不随 Git 分发。 |
| [shi3z/deepseekv4.1-A100-custom](https://github.com/shi3z/deepseekv4.1-A100-custom) | 早期 A100 可运行性与算子调研参考，不是本交付引擎；其速度不计入本仓库实测。 |
| [SGLang](https://github.com/sgl-project/sglang) | V4.1 部署方案调研参考，未纳入本版引擎。 |
| [deepseek-v4-flash-a100](https://github.com/keplerzip/deepseek-v4-flash-a100) | 前代离线交付、验收与操作经验。 |

固定源码 wtdcode/vllm-backport@a350766628514d597670d2a5217777fdee2ba7f4，响应文本块修正参考上游 [4f4498f](https://github.com/wtdcode/vllm-backport/commit/4f4498fea2ec91848d1b131e82ae1401a09d4d3a)。

派生文件保留 SPDX 和版权头。[vLLM Apache-2.0 许可](../licenses/vllm-LICENSE)、[DeepSeek MIT 许可](../licenses/DeepSeek-LICENSE) 保留原文。原创部署脚本采用 Apache-2.0，没有重新许可模型、客户端或未随 Git 分发的镜像依赖。

## 20260915 性能更新

感谢 zyongye 与 vLLM 维护者提供 [#56633](https://github.com/vllm-project/vllm/pull/56633) 的 delayed mHC 融合与 DSpark 辅助状态优化，以及 Isotr0py 提供 [#56554](https://github.com/vllm-project/vllm/pull/56554) 的图像 padding 修复。本项目将其适配到固定的 SM80 backport，保留既有 A100 调优；新增 kernel 仍需目标 GPU 验收。mHC epilogue 的 SGLang 来源与原版权声明一并保留。
