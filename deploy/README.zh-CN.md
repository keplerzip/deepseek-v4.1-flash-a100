# DeepSeek-V4.1-Flash 离线服务 R1.3 / v1.3.0

固定 DSpark k=5，262144 上下文，TP4×DP2＋EP8，Engram 主存卸载。只管理模型服务，不安装 NewAPI。

20260920-perf4 新增 DSpark context KV 合并投影及 ragged top-k 元数据复用，累计包含 dense BF16 分派、EP8 custom AG/RS、8-warp sparse decode、共享专家条件重叠及主干 EPLB，并修复 DP 草稿 metadata、隔离草稿 EPLB。两池分别计算满 256K 容量，C=max(32,2×两池满窗口数之和)，每池 max_num_seqs=C/2。保留 R1.1 的索引/mHC/MoE 与图片修复。复用最初 R1 镜像；首次检查原生 FP8 KV 投影、索引缓存图回放、其他新增算子与 EP8，两个池均做 smoke。新版 GPU 速度和完整模型验收待目标机回传。

已有实例可用小更新包中的 install.sh 升级并重启，自动保留源码备份；源码更新后也可运行 bash ./restart.sh。使用 bash ./rollback-update.sh 恢复最近备份。独立检查为 tests/performance-kernels.sh，C1/C32 与多图历史对照为 tests/performance-benchmark.sh。包顶层 README.zh-CN.md 包含安装说明，版本记录见 manifests/update.json；仓库详细说明见 docs/performance-update-20260916.md。

源码树需要在构建侧补齐 images/runtime-image.tar 并生成 manifests/deploy.sha256；模型位于相邻 DeepSeek-V4.1-Flash 目录。完整准备流程见 ../docs/building.md，目标部署见 ../docs/deployment.md。

完整离线包中执行 bash ./deploy.sh，后续运维用 start.sh / stop.sh / status.sh。k5 已直接写入配置并由参数校验固定。

图片数量上限固化为 999，与本次固定版本 vLLM 的默认值一致。实际可处理数量还受 256K 总上下文、图片尺寸、视觉编码显存、并发和 128 MiB 请求体约束。历史实机报告覆盖 1/2/4 图；新增超过 4 图的验证需在目标机执行，不能将配置上限当作 999 图实测能力。

本版固化 `PERF_DSPARK_KV=1`、`PERF_TOPK_REUSE=1`，k=5 保持固定。累计小包可从初始版本直接升级，失败时恢复安装前源码并尝试重启旧服务。两个开关设为 0 只用于隔离性能对照，需要重启；不改变图片限制、TTL 或 Engram 位置。
