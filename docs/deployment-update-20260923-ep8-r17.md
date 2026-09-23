# R1.7 / v1.7.0：EP8 全链路验收修正

R1.6 在隔离八卡 A100 上通过变长 all-gather 比较，随后在 `performance_ep8.py:67` 失败：reduce-scatter 的输入 `y` 由整数填充值创建，默认为 `torch.int64`；期望值由 float32 的 `x` 生成，因此八个 rank 报告 `torch.int64 != torch.float32`。这仍是验收脚本的 dtype 错误，日志没有证明数值正确，也没有证明整模能启动。安装器因此自动回滚；`ROLLED_BACK` 后仍须确认旧服务已恢复健康。

R1.7 把测试输入、期望和输出的构造统一为两套明确的 dtype：等长自定义通信为 BF16，变长 NCCL 回退为 float32。`y` 显式沿用 `x.dtype`；所有数值比较继续使用 `atol=0, rtol=0`。同时验证 BF16 自定义 AG/RS 确实被调用、float32 回退没有进入自定义路径、CUDA Graph 修改输入重放正确，并将 EP8 测试源码哈希纳入启动验收签名，避免更新测试后被旧通过标记跳过。新增 CPU 参考验收，逐 rank 检查四种等长形状及变长形状的 dtype、shape、聚合数值和修改输入后的期望值。

这些构建侧检查验证的是**测试自身的数学与 dtype 契约**，不能代替 A100 上的 NCCL、自定义通信或完整模型实测。模型、初始 R1 镜像、R1.4 算子覆盖、1M 上下文、0.92 显存预算、TP4×DP2＋EP8、DSpark k=5、自动 C≥32、视觉及本机/Docker 内部免密访问不变。

## 累计升级

等待上一次安装器结束并确认旧服务恢复后，只需传输 `deepseek-v4.1-flash-a100-R1.7-update.tar.gz` 和同名 `.sha256`。从初始 R1 至 R1.6 均可直接安装；不传镜像或权重。在两文件所在目录执行，命令在前台等待模型启动：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.7-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.7-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.7-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

安装器校验累计包与既有镜像，备份当前部署文件；启动失败会恢复安装前文件并尝试重启旧服务。不会修改 NewAPI、宿主驱动或模型权重。

在目标 `deploy` 目录验收：

```bash
bash ./status.sh
bash ./tests/local-access.sh
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
```

首次启动先跑 SM80 单卡数值/图回放和八卡 EP8 测试，然后才加载完整模型；可能因 1M KV 标定再次重启。EP8 日志在 `runtime/performance-ep8-*/probe.log`，逐 rank 结果在同目录。构建机没有 GPU；R1.7 的八卡 EP8、整模、1M 容量与速度仍待目标机验收。历史 TP8/256K 性能不作为本版成绩。
