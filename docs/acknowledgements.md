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

## R1.1 / 20260916-perf2

感谢 Danylo Storozhev / [Zanooda 的 CMP170HX 项目](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx/tree/2445d7db07a2023f1d0a4bbb5b1ee4154b299590) 的紧凑候选索引实现，以及 [Tokha233 的 A100 Turbo 项目](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/tree/fae324ae62ac5cef31b7d38f5d369618e1cae1fa) 的稳定 MoE 排序融合。前者移植到本项目固定基底的 TP query sharding 路径，后者保持原稳定排序语义；两者均保留 Apache-2.0 声明。引用其实现不等于将其硬件、整套参数或速度结果用于本项目背书。

## R1.2 / 20260918-perf3

继续感谢 Tokha233/deepseek-v4.1-flash-a100-turbo@fae324ae62ac5cef31b7d38f5d369618e1cae1fa 的 TP4×DP2＋EP8 部署经验、dense BF16 分派、custom AG/RS、DP 草稿 metadata 与 EPLB 隔离补丁。其公开整模数据来自 A800，不能作为本项目 A100 的实测。R1.2 保留图文能力，未启用该项目的 language-model-only。

感谢 wtdcode/vllm-backport 的 [a5bb095f](https://github.com/wtdcode/vllm-backport/commit/a5bb095f0b97d513ac0c657c2c481334e9302ad8)（8-warp decode）、[9925c45a](https://github.com/wtdcode/vllm-backport/commit/9925c45ab4c0740dfc8e77b4715f665e8b205447)（CUDA shared-expert overlap）与 [#89](https://github.com/wtdcode/vllm-backport/pull/89)（DSML 拼写兼容）。源码清单逐文件固定原始与派生哈希；CPU/编译检查不替代 GPU 验收。

## R1.3 / 20260920-perf4

感谢 Juntian777 的 [vLLM #57627](https://github.com/vllm-project/vllm/pull/57627)（`f0defc90f6645242ee61bccecd6146f385cff4b7`）与 Fangzhou-Ai 的 [vLLM #57434](https://github.com/vllm-project/vllm/pull/57434)（`bea2fcdfcf2ef385db7b93352d98c7ee35739126`）。本版分别适配 DSpark context KV 合并投影和跨层 ragged top-k 复用，针对固定原镜像保留原生 scale 加载、SM80 继承路径和更严格的输入缓冲区检查；不宣称上游其他 GPU 的测试结果适用于本机。修改仍使用 Apache-2.0，并保留原文件 SPDX。

## R1.4 / 20260922-perf5

感谢 Woosuk Kwon / WoosukKwon 的 [vLLM #57885](https://github.com/vllm-project/vllm/pull/57885)（`73ce1d9220507a4d06412983e6cb599ed163eb43`）及 nehaprakriya 的 [#57860](https://github.com/vllm-project/vllm/pull/57860)（`2031422bfd952cc65fc1a46a9b570f6f04dc318d`）。本版移植稀疏注意力冗余元数据清理和 ragged indptr/pack 融合，针对 A100 实际继承路径增加 1024 行上限、空输入处理及独立回退，继续保留 R1.3 索引复用。代码保留 Apache-2.0 SPDX；其他 GPU 的微基准不作为本机提速承诺。

R1.9还感谢vLLM #60091、#57435、#57632、#60874的作者和审阅者，分别提供同步消除、逆RoPE融合、draft context图捕获、工作区预算机制。本仓库适配到旧SM80 ABI、添加边界和离线验证；不将上游优化归为原创。固定提交及具体文件见deploy/overrides/performance/manifest.json。调研涉及LvLLM、Lsglang、XTU、shi3z、DGX社区方案；未采用其全部运行时，也不把其硬件成绩当成本机成绩。
