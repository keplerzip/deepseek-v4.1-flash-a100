# R1.8 / v1.8.0：本机前端修复、C32 与公开模型名

R1.7 在隔离目标机通过八个 rank 的 EP8 通信测试、整模启动、两个 DP pool 的文本、前缀缓存和 DSpark smoke。最后一步的本机 API 前端以固定 UID 65534 读取私有 `/frontend.py` 时失败，安装器按设计回滚。R1.8 让前端使用部署操作者的 UID/GID，并在加载八卡模型前检查同一个只读挂载是否可读；生成 `/access.json` 后也在启动前检查两个真实挂载。

唯一公开模型 ID 改成 **`DeepSeek-V4-Flash`**，`/v1/models`、Chat Completions、Responses、Anthropic Messages、离线 guard 与验收请求统一使用此名。底层权重、ModelScope ID、容器内模型路径继续是 DeepSeek-V4.1-Flash，**不用重传权重或镜像**。如果已有 NewAPI 渠道仍按旧名称转发，需在已有网关里把渠道上游模型名同步成新 ID；本包不会修改 NewAPI。

全局调度并发硬上限固定 **C32**，两个 DP pool 各 `max_num_seqs=16`。实际 KV 容量仍记录，每池至少能容纳一个满 1M 窗口才能启动，但容量不会再自动把 C32 提高到 C56。上下文仍为 1,048,576 token（输入输出合计），显存比例 0.92，DSpark k=5，prefix caching 和 `cached_tokens` 明细继续开启。C32 不是 32 个满窗口同时驻留的承诺。

本次按操作者要求**跳过启动前 EP8 通信重测**。如果目标机保留 R1.7 的通过收据，则直接复用；如果没有，仍运行继承的 SM80 kernel 检查，跳过 EP8 通信测试。整模启动时的 NCCL 通信、双池 smoke、缓存与 DSpark 检查照常运行。R1.8 的 GPU 完整验收和性能数字须以目标机结果为准；构建机无 GPU。

## 累计更新

传输以下两个文件到隔离目标机：`deepseek-v4.1-flash-a100-R1.8-update.tar.gz` 和同名 `.sha256`。从初始 R1 至 R1.7 可直接升级；不传模型或镜像。等待前一个安装/回滚进程结束后，在两个文件所在目录执行，命令在前台等待模型启动：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.8-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.8-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.8-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

安装器会备份并替换源码与配置，保留权重、镜像和 `runtime` 验收证据；新启动失败时恢复之前的源码并尝试重启旧服务。它只管理本项目容器，不触碰 NewAPI。

启动成功后，在 `deploy` 目录检查：

```bash
bash ./status.sh
curl --fail http://127.0.0.1:8005/v1/models
bash ./tests/local-access.sh
bash ./tests/run-all.sh
```

`/v1/models` 应只返回 `DeepSeek-V4-Flash`。`runtime/capacity.json` 应显示 `required_concurrency: 32` 和 `formula: fixed(32)`；`runtime/runtime.resolved.json` 中应是 `max_model_len: 1048576`、`max_num_seqs: 16`、`enable_prefix_caching: true`、`num_speculative_tokens: 5`。缓存冷/热验收会报告 `cached_tokens`，性能报告由 `tests/run-all.sh` 生成。旧 TP8/256K 性能表不代表本版速度。
