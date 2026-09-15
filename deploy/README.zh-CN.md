# DeepSeek-V4.1-Flash 离线服务 v1.0.0

固定 DSpark k=5，262144 上下文，TP8，Engram 主存卸载。只管理模型服务，不安装 NewAPI。

20260915-perf1 默认接入 NCCL 自动选择、SM80 mHC 融合和 V4.1 图像 padding 修复。更新代码以 overrides/performance/ 的已校验文件只读挂载到原镜像；目标首次启用前自动做一次新 mHC kernel 回归，结果按补丁/镜像/驱动缓存。新性能仍需目标机实测，原报告不变。

已有实例可用小更新包中的 install.sh 升级并重启，自动保留源码备份；源码更新后也可运行 bash ./restart.sh。使用 bash ./rollback-update.sh 恢复最近备份。独立检查为 tests/performance-kernels.sh，C1/C32 与多图历史对照为 tests/performance-benchmark.sh。详细说明见 ../docs/performance-update-20260915.md。

源码树需要在构建侧补齐 images/runtime-image.tar 并生成 manifests/deploy.sha256；模型位于相邻 DeepSeek-V4.1-Flash 目录。完整准备流程见 ../docs/building.md，目标部署见 ../docs/deployment.md。

完整离线包中执行 bash ./deploy.sh，后续运维用 start.sh / stop.sh / status.sh。k5 已直接写入配置并由参数校验固定。

图片数量上限固化为 999，与本次固定版本 vLLM 的默认值一致。实际可处理数量还受 256K 总上下文、图片尺寸、视觉编码显存、并发和 128 MiB 请求体约束。历史实机报告覆盖 1/2/4 图；新增超过 4 图的验证需在目标机执行，不能将配置上限当作 999 图实测能力。
