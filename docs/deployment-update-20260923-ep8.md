# R1.6 / v1.6.0：EP8 验收 dtype 修正

**历史版本：R1.6 在目标机又暴露了 reduce-scatter 输入默认 int64 的第二处测试错误。请使用 [R1.7 累计更新](deployment-update-20260923-ep8-r17.md)。**

R1.5.1 在隔离的八卡 A100 首次启动时，单卡 SM80 算子检查全部通过，随后 EP8 检查在 `performance_ep8.py:61` 失败并自动回滚。八个 rank 均报告 `torch.float32 != torch.int64`：变长、非 BF16 的 NCCL 回退分支输出是 float32，但测试使用 `torch.full` 的默认整数 dtype 构造期望值。这条日志证明验收断言错误，不证明通信结果的数值正确，也不证明 EP8 内核或整模失败。

R1.6 只把该期望值显式设为输入 `x.dtype`。数值仍用 `atol=0, rtol=0` 严格比较；保留等长 BF16 custom AG/RS、CUDA Graph 修改输入重放、变长 float32 回退和八个 rank 的全部断言。若数值有误，仍会失败并自动回滚。模型权重、原始 R1 镜像、R1.4 性能覆盖、TP4×DP2＋EP8、DSpark k=5、1M 上下文、0.92 显存预算、自动 C≥32、999 图和本机/Docker 免密访问均不变。

## 累计升级

只需传输 `deepseek-v4.1-flash-a100-R1.6-update.tar.gz` 及同名 `.sha256`。从初始 R1 到 R1.5.1 均可直接安装，无需先安装中间版本，不传镜像或权重。在两文件所在目录执行；命令在前台等待模型启动：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.6-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.6-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.6-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

安装器先校验累计包和既有原镜像，再备份当前部署文件；启动失败会恢复安装前文件并尝试重启旧服务。`ROLLED_BACK` 只表示文件恢复，旧服务健康还需等待启动结束。不会修改已有 NewAPI、宿主驱动或模型权重。

在目标 `deploy` 目录验收：

```bash
bash ./status.sh
bash ./tests/local-access.sh
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
```

首次启动会运行单卡 SM80 数值/图回放测试和八卡 EP8 测试。EP8 检查最多 300 秒；报告位于 `runtime/performance-ep8-*/probe.log` 和每个 rank 的 JSON。随后可能因 1M KV 容量标定而多次加载模型。构建机没有 GPU；R1.6 的 EP8 修正、整模、1M 容量和速度仍待目标机实际验收。历史 TP8/256K 性能不作为本版成绩。完整的 0.92 固定配置见 [R1.5.1 方案](deployment-update-20260922-gpu92.md)。
