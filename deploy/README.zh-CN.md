# DeepSeek-V4.1-Flash 离线服务 v1.0.0

固定 DSpark k=5，262144 上下文，TP8，Engram 主存卸载。只管理模型服务，不安装 NewAPI。

源码树需要在构建侧补齐 images/runtime-image.tar 并生成 manifests/deploy.sha256；模型位于相邻 DeepSeek-V4.1-Flash 目录。完整准备流程见 ../docs/building.md，目标部署见 ../docs/deployment.md。

完整离线包中执行 bash ./deploy.sh，后续运维用 start.sh / stop.sh / status.sh。k5 已直接写入配置并由参数校验固定。

图片数量上限固化为 999，与本次固定版本 vLLM 的默认值一致。实际可处理数量还受 256K 总上下文、图片尺寸、视觉编码显存、并发和 128 MiB 请求体约束。历史实机报告覆盖 1/2/4 图；新增超过 4 图的验证需在目标机执行，不能将配置上限当作 999 图实测能力。
