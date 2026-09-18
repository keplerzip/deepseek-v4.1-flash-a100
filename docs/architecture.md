# 实现与固定版本

交付版本 **R1.2** 对应项目 / Git 标签 **v1.2.0**，累计更新编号 **20260918-perf3**。参考运行镜像仍为 `deepseek-v4.1-flash-a100:20260913-r1`，保留其名称和字节以复用初始交付。模型 revision、镜像 digest、覆盖层逐文件哈希与安装收据分别记录；不能单看镜像 tag 判断正在运行哪个更新版本。

[release.json](../release.json) 固定 SM80 基础镜像 `lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3`，核对源码提交 `a350766628514d597670d2a5217777fdee2ba7f4`。参考栈 PyTorch 2.13.0+cu130、CUDA 13.0、NCCL 2.29.7。

[ModelScope 镜像](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash) revision 为 `f877e3ed82d979be83e828b52762cdb86152bb9c`；[逐文件清单](../deploy/manifests/model-files.json) 固定路径、大小、SHA256。保持官方 FP8/FP4，不全量展开 BF16；Engram 使用分支提供的主存路径。

模型引擎在独占 internal Docker 网络，宿主网络前端只绑定 loopback 和检查过的 bridge 网关，检查来源网段再转发。普通操作者 UID/GID 挂载运行目录，模型只读；Python/编译器等依赖在镜像内，Triton/CUDA 派生缓存存 runtime。设置离线与遥测关闭，并提前拒绝公网输入抓取。

build/image/patched 保存固定基底的编码修正和 KV/Graph 观测覆盖文件，安装前核对原文件 SHA256。deploy/overrides 保存 schema guard。部署配置和运行参数校验固定 k5，与权重原生 dspark_block_size=5 一致。

R1.2 默认 TP4×DP2＋EP8。专家权重由八个 EP rank 分片，attention/dense/草稿的其他状态按实现分片或复制。主干 EPLB 不增加冗余专家，草稿关闭 EPLB；FP4 专家保留 Marlin 路径。原镜像提供 custom AG/RS 的原生符号，无需重新编译 C++ 扩展。

全局并发为 `max(32,2*sum(floor(KV_i/262144)))`；每个 DP 的 scheduler 限额为 C/2。容量记录必须包含两池坐标，graph 与真实 replay 必须各覆盖两池的四个 worker。FULL_DECODE_ONLY、编译 mode=0 保留 decode CUDA Graph；prefill 走 eager 路径。旧 TP8 调优文件随事务备份，更新为新拓扑配置。

会话按 prompt_cache_key 或首条用户消息的稳定哈希固定 DP pool；显式 X-data-parallel-rank 可指定 0/1。无明确亲和 key 的 stateful Responses continuation 留给原生路由。响应 X-DSV41-DP-Rank 可用于诊断。相同会话集中在一池有利于缓存，多个独立会话才能充分利用 DP2。
