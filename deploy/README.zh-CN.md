# DeepSeek-V4.1-Flash 离线服务 R1.5.1 / v1.5.1

固定显存预算 gpu_memory_utilization=0.92，DSpark k=5，1048576（1M）总上下文，TP4×DP2＋EP8，Engram 主存卸载。只管理模型服务，不安装 NewAPI。

20260922-context1m-gpu92 补齐并锁定 0.92，容量与验收沿用 R1.5。20260922-context1m 将容量与验收改为 1M，忽略旧 256K 容量种子。沿用 20260922-perf5 的 SWA 有效性直接输出、prefill 专用长度计算及小批 ragged indptr/pack 融合（最多 1024 行，较大批次保留原扫描）。累计保留 R1.3 的 DSpark context KV 合并投影与索引复用，以及 dense BF16、EP8 custom AG/RS、8-warp decode、共享专家条件重叠、主干 EPLB 和此前修正。两池分别计算满 1M 容量，C=max(32,2×两池满窗口数之和)，每池 max_num_seqs=C/2。复用最初 R1 镜像；首次检查新增及累计 GPU 算子、图回放和 EP8，两个池均做 smoke。新版 GPU 速度和完整模型验收待目标机回传。

已有实例可用小更新包中的 install.sh 升级并重启，自动保留源码备份；源码更新后也可运行 bash ./restart.sh。使用 bash ./rollback-update.sh 恢复最近备份。独立检查为 tests/performance-kernels.sh，C1/C32 与多图历史对照为 tests/performance-benchmark.sh。包顶层 README.zh-CN.md 包含安装说明，版本记录见 manifests/update.json；仓库详细说明见 docs/deployment-update-20260922-gpu92.md。

源码树需要在构建侧补齐 images/runtime-image.tar 并生成 manifests/deploy.sha256；模型位于相邻 DeepSeek-V4.1-Flash 目录。完整准备流程见 ../docs/building.md，目标部署见 ../docs/deployment.md。

完整离线包中执行 bash ./deploy.sh，后续运维用 start.sh / stop.sh / status.sh。k5 已直接写入配置并由参数校验固定。

图片数量上限固化为 999，与本次固定版本 vLLM 的默认值一致。实际可处理数量还受 1M 总上下文、图片尺寸、视觉编码显存、并发和 128 MiB 请求体约束。历史实机报告覆盖 1/2/4 图；新增超过 4 图的验证需在目标机执行，不能将配置上限当作 999 图实测能力。

本版固化 `PERF_DECODE_METADATA=1`，同时保留 `PERF_DSPARK_KV=1`、`PERF_TOPK_REUSE=1`，k=5 固定。单独将 `PERF_DECODE_METADATA` 设为 0 并重启，只恢复 R1.3 元数据实现。累计小包可从初始版本直接升级，失败时恢复安装前源码并尝试重启旧服务。开关不改变图片限制、TTL 或 Engram 位置。

主模型和 DSpark 的 decode CUDA Graph 保持启用（FULL_DECODE_ONLY）；视觉编码器图未移植。两池各需至少容纳一个满 1M 窗口；C32 为调度下限，不代表 32 个满 1M 同时驻留。tests/run-all.sh 在两池分别增加 512K 与近 1M 检索，输出预算计入总窗口。
