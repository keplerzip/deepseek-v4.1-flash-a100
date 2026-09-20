# R1.3 / v1.3.0：DSpark KV 合并投影与注意力索引复用

## 实测性能与本版状态

| 配置 | 单流 decode tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|
| 历史 R1，TP8，k5 / off，用户回传 | 139.80 | 1379.01 | 24.56 |
| 历史 R1，TP8，k5 / high，用户回传 | 109.95 | 1039.91 | 34.93 |
| R1.3，TP4×DP2＋EP8，k5 | **待目标 A100 实测** | **待实测** | **待实测** |

历史测试每组 5 个单流样本、200 个 C32 短文本请求，每请求 1024 输出 token；high 包含思考 token。原始 JSONL 尚未回传，不能将这些数字当作 R1.3 或多图长历史成绩。保留完整的 [性能口径](performance.md) 和 [机器可读历史报告](../reports/dspark-k5-k7.user-reported.json)。

本版是**待目标 GPU 验收的预发布版**。构建机没有 GPU；CPU 权重加载、控制流和 SM80 编译通过也不能代替完整模型推理、质量和速度验收。目标机尚未安装 R1.2，本包已累计包含其改动。

## 交付与一键升级

只需传两个文件：`deepseek-v4.1-flash-a100-R1.3-update.tar.gz` 和同名 `.sha256`。可从初始 R1、R1.1 或 R1.2 直接安装，复用已导入的 `deepseek-v4.1-flash-a100:20260913-r1` 镜像和完整模型，**不用先升级 R1.2，不传权重或镜像**。

在目标机拷包目录执行这一行：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.3-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.3-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.3-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

命令在前台运行，**会停止并重启本项目模型服务**。只使用普通用户与 `sudo -n docker`，全部依赖来自已有镜像；不联网、不安装宿主服务、不操作外部 NewAPI。安装前检查包哈希、镜像身份和源码兼容性，事务保存旧源码、权限与调优配置，保留原权重路径、端口、containerd 路径和运行证据。

第一次启动自动运行新增 GPU 数值/图回放检查以及 EP8 通信检查，通过后加载整模、重标定两个 KV 池、验证两池的模型响应和实际图回放。算子检查设有 30 分钟上限，EP8 检查 300 秒上限；这些是超时界限，不是预计耗时。KV 标定可能需要再加载一次模型。启动失败会恢复备份并尝试重新启动旧服务，错误报告保留在 `deploy/runtime/`。

成功安装后需要主动回退时执行：

```bash
cd /ai/services/deepseek-v4.1-flash-a100/deploy && bash ./rollback-update.sh
```

回退恢复本次安装前的版本（不一定是 R1.2）；如果升级后又手工编辑了待恢复文件，回退会拒绝覆盖。省略安装命令中的 `--restart` 仅更新磁盘文件，运行中的容器仍是原版本。

## 本版固定方案

唯一模型/API 名仍为 **DeepSeek-V4.1-Flash**，使用原 ModelScope revision `f877e3ed82d979be83e828b52762cdb86152bb9c`。

| 项目 | R1.3 |
|---|---|
| 并行 | TP4×DP2＋EP8、PP1；专家跨八卡分片 |
| DSpark | **固定 k=5**，合并三层 context KV 投影 |
| 注意力 | 复用同一步、同一索引来源且输入兼容的 ragged top-k 元数据 |
| 累计优化 | R1.2 dense BF16、EP8 custom AG/RS、8-warp decode、条件共享专家重叠、主干 EPLB；R1.1 mHC/索引/MoE 优化 |
| 上下文 / 并发 | 262144；`C=max(32,2*sum(floor(KV_DP_i/262144)))`，每池调度 C/2 |
| Engram | 主存卸载；保留 DP2 两份表预算，未改成 GPU 常驻 |
| 多模态 | 保留视觉，image=999、128 MiB 请求体限制；拒绝公网媒体抓取 |
| 协议 | Chat Completions、Responses、Messages，保留工具与缓存 usage |
| 访问 | loopback 和本机 Docker bridge 的 8005 端口，客户端免密 |

有关 TP4×DP2＋EP8、共享专家条件、两份 Engram 表和 KV 会话亲和的细节见 [R1.2 固定方案](performance-update-20260918.md)。DP2 不复制两份完整主干到四卡组。历史 TP8 标定出的 C56 不沿用，两个池必须各自按实际可用 KV 重新计算。

## 新增优化怎样执行

**DSpark context KV 投影。** 原实现逐层计算 `Wq_a | Wkv`，随后丢弃 context 不需要的 Q 部分。本版将三层 Wkv 堆叠成一次投影，再逐层进行原有 normalization、RoPE、量化和 KV 写入。三层权重互相独立，主干与草稿各层原有 attention 投影仍保留。它不改变 k、接受算法或模型权重文件。

加载时从原 checkpoint 的 `mtp.i.attn.wkv.weight/scale` 复制，通过镜像原生的 merged-linear loader 处理 MXFP8 32×32 scale；缺少任何一个 KV weight/scale 分片就中止。没有把已重排的 Marlin 内存简单拼接。新增常驻投影和 dense BF16 副本粗略合计约 **23 MiB/卡**，另有临时输出与工作区；准确峰值由目标机标定。此项针对重复投影，不等于全部 decode 计算量同比减少。

**Ragged top-k 元数据复用。** A100 的 attention 继承 backport 中 `amd/rocm.py` 的通用 Triton 路径，因此虽然上游 PR 标为 ROCm，此处可以在实际 SM80 调用路径上移植。索引生产层每次 eager 执行或图捕获先清空本次缓存；只有压缩率、块大小、批次、top-k、block table、请求映射和有效掩码的缓冲地址/布局匹配，消费层才复用结果。CUDA Graph 记录一次构建，回放时重新执行该构建，不能复用上一轮 token 的值。

本地额外保留了输入地址与布局检查，避免跨 KV group 错用物理槽位。这可能比上游只按压缩率复用的方案命中更少；实际减少多少次 pack、能快多少要实测。它只缓存注意力索引的临时整理结果，**不设置 prefix KV 的 TTL，也不承诺修复空闲十分钟后命中归零的问题**。

## 验证与诊断

[构建检查记录](../reports/performance-update-20260920.build.json) 单独标明没有 GPU 执行。构建侧在原始镜像、`network=none` 下检查：原生 merged-linear 的权重与非均匀 scale 顺序、模拟 TP rank3/4 的完整复制、缺分片拒绝、mHC 开关两种实现、独立 CPU KV 参考；以及索引来源逐步失效、DP 状态隔离、不同 block table/压缩率、batch 缩小与增大、关闭复用的原路径。SM80 AOT 编译不执行 GPU。

目标机新增检查使用实际模型维度 H=5120、Q-rank=1280、KV=512、三层草稿，比较原路径/合并路径与独立 BF16 参考，覆盖 Marlin/dense 分派和修改输入后的图回放。索引检查比较真实 Triton pack 与 CPU 物理槽位参考，并交替执行不同 batch 的 eager 和已捕获图。相关检查自动写入 `runtime/performance-kernels-*/`；通过记录绑定覆盖版本、镜像、驱动和开关。

模型启动后，在 deploy 目录依次运行：

```bash
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
bash ./tests/performance-benchmark.sh --images 8
```

更贴近慢请求的测试是保存脱敏 Chat 请求（图片用 inline data URI）为 `runtime/benchmark-chat.json`，执行 `bash ./tests/performance-benchmark.sh --request-file /state/benchmark-chat.json`。报告保留 off/high、C1/C32、TTFT、decode、端到端吞吐、接受率、实际参数和逐请求日志。不要将短文本与多图历史混算，也不要用端到端时间反推纯 decode。

默认在 `deployment.env` 固化 `PERF_DSPARK_KV=1`、`PERF_TOPK_REUSE=1`。诊断时可将其中一个设为 0，再执行 `bash ./restart.sh`，保持其他条件不变对照；两个都关闭保留 R1.2 的计算路径和当前 R1.3 运维工具。关闭投影会改变一点显存占用，应重新标定；测试结束恢复两个 1 并重启。完整退回安装前状态仍应使用 `rollback-update.sh`。

建议固定输入、图片、思考模式、输出长度和冷热缓存，暂停其他流量，采用 A→B→A。一次启动通过不能替代 2h/24h 稳定性、满窗口并发和真实 Agent 任务质量验收。

## 固定来源与致谢

- [vLLM #57627](https://github.com/vllm-project/vllm/pull/57627)，Juntian777，固定提交 `f0defc90f6645242ee61bccecd6146f385cff4b7`：DSpark context KV 堆叠投影。
- [vLLM #57434](https://github.com/vllm-project/vllm/pull/57434)，Fangzhou-Ai，固定提交 `bea2fcdfcf2ef385db7b93352d98c7ee35739126`：跨层 ragged top-k 复用。

两项在调研快照中均为未合并 PR。本版是针对固定 backport 命名空间、SM80 和原生权重加载的本地适配，不是声称上游正式版本已验证八卡 A100。上游 GB200 微基准与 ROCm 数据不作为本机提速承诺。感谢 DeepSeek、wtdcode/vllm-backport、vLLM、Tokha233、Zanooda、PyTorch、Triton、TileLang 和 NCCL；原 SPDX 与许可声明保留，见 [完整致谢](acknowledgements.md)。

视觉编码器 CUDA Graph、inverse RoPE 融合、Engram THP/跨 DP 共享等仍属于后续候选，本次没有加入。
