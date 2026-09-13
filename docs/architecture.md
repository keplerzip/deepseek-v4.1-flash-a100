# 实现与固定版本

项目版本 **1.0.0** 与参考运行镜像 `deepseek-v4.1-flash-a100:20260913-r1` 是不同维度；保留镜像名称以复用已交付字节。

[release.json](../release.json) 固定 SM80 基础镜像 `lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3`，核对源码提交 `a350766628514d597670d2a5217777fdee2ba7f4`。参考栈 PyTorch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7。

[ModelScope 镜像](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash) revision 为 `f877e3ed82d979be83e828b52762cdb86152bb9c`；[逐文件清单](../deploy/manifests/model-files.json) 固定路径、大小、SHA256。保持官方 FP8/FP4，不全量展开 BF16；Engram 使用分支提供的主存路径。

模型引擎在独占 internal Docker 网络，宿主网络前端只绑定 loopback 和检查过的 bridge 网关，检查来源网段再转发。普通操作者 UID/GID 挂载运行目录，模型只读；Python/编译器等依赖在镜像内，Triton/CUDA 派生缓存存 runtime。设置离线与遥测关闭，并提前拒绝公网输入抓取。

build/image/patched 保存固定基底的编码修正和 KV/Graph 观测覆盖文件，安装前核对原文件 SHA256。deploy/overrides 保存 schema guard。部署配置和运行参数校验固定 k5，与权重原生 dspark_block_size=5 一致。
