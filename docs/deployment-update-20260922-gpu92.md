# R1.5.1 / v1.5.1：1M 固定方案补齐 0.92 显存预算

本版将 R1.5 的默认 `gpu_memory_utilization=0.90` 修正并锁定为 **0.92**。配置校验拒绝旧 0.90 / 0.94；安装器备份并迁移已有调优配置，调优工具仅比较 batch token 预算，保持 0.92。复用初始 R1 镜像、原始模型和 R1.4 性能覆盖，可从 R1 至 R1.5 直接累计升级。

| 要求 | 固定实现 |
|---|---|
| 单一模型名 | `DeepSeek-V4.1-Flash`；模型列表只允许此 ID，请求别名被拒绝 |
| 上下文 | 1048576（1M）token，输入与输出合计，原权重原生支持 |
| 自动并发 | `C=max(32,2*(N0+N1))`，N0/N1 是各 DP 池实际可容纳的完整 1M 窗口数，每池 scheduler 上限 C/2 |
| 显存预算 | 固定 `gpu_memory_utilization=0.92`，实际 KV 在图捕获和工作区等开销后重新测量 |
| 图片 | `limit_mm_per_prompt={"image":999}`，三协议保留多模态 |
| 客户端 API key | 不需要；引擎强制空 `VLLM_API_KEY`，前端也不要求鉴权 |
| 宿主访问 | `http://127.0.0.1:8005/v1` |
| Docker 访问 | `http://host.docker.internal:8005/v1`；客户端容器需已有 `host.docker.internal:host-gateway` 映射 |
| 网络范围 | 引擎位于 internal bridge，前端只监听 loopback 与枚举确认的本机 Docker bridge 网关，并检查来源网段 |
| 并行 / DSpark | TP4×DP2＋EP8，k=5；主模型与草稿 decode CUDA Graph 保留 |

0.92 是 vLLM 显存预算比例，不要求 `nvidia-smi` 任意时刻恰好显示 92%。999 是单请求图片数量上限，包含重传的历史图片；实际请求还受总上下文、图片尺寸、视觉显存和 128 MiB 请求体限制。C32 是调度下限，不表示 32 个满 1M 窗口同时驻留。

**构建机没有 GPU，本版仍为待目标 A100 验收的预发布版。** 配置已满足清单，但不将静态配置和离线 CPU 检查等同于目标机已完成 1M / 999 图实测。旧 `runtime/api-key.txt` 仅为验收工具兼容保留，服务不使用它要求客户端鉴权。现有 NewAPI 和客户端容器不由本项目安装或修改；若后续新增 Docker bridge，可运行 `bash ./restart.sh` 刷新访问地址。

## 一行升级

传输本版 `.tar.gz` 和 `.sha256`，在两文件所在目录执行，会重启本项目模型：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.5.1-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.5.1-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.5.1-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

仍只需要普通用户及 `sudo -n docker`，无公网依赖，无新镜像或权重下载。升级失败时使用原有事务回退；手工恢复最近安装前的配置可运行 `bash ./rollback-update.sh`。

在目标 deploy 目录检查免密连接与功能：

```bash
bash ./tests/local-access.sh
bash ./tests/run-all.sh
```

前者实际从 host 与 bridge 两种容器网络发送无 Authorization 的模型列表和 Responses SSE 请求；后者在两个 DP 池分别检查多协议、缓存、图片和近 1M 检索，再测短请求吞吐。满窗口压力为 `bash ./tests/extended.sh full-window`。详见 [1M 容量与 CUDA Graph 范围](context-update-20260922.md)；其中旧 R1.5 安装命令由本页 R1.5.1 命令替代。

[构建检查](../reports/gpu92-update-20260922.build.json)记录本版固定参数、旧调优迁移和原镜像 CLI 的检查结果；最终累计包的安装/回退证据随 `r151-delivery.json` 独立发布。历史性能仍是 TP8 / 256K 实测，不改写为本版成绩。
