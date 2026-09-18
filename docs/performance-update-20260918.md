# R1.2 / v1.2.0：V4.1 的 TP4×DP2＋EP8 激进方案

## 实测性能与本版状态

| 配置 | 单流 decode tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|
| 历史 R1，TP8，k5 / off，用户回传 | 139.80 | 1379.01 | 24.56 |
| 历史 R1，TP8，k5 / high，用户回传 | 109.95 | 1039.91 | 34.93 |
| R1.2，TP4×DP2＋EP8，k5 | **待目标 A100 实测** | **待实测** | **待实测** |

历史数据为短文本、每组 5 个单流样本和 200 个 C32 请求，每请求 1024 输出 token；high 包含思考 token。不能用这些成绩承诺 R1.2 的提升，也不能当成多图长历史 decode 成绩。原始逐请求 JSONL 未回传；更新包保留用户回传的机器可读报告。

构建机没有 GPU。本版完成可在构建侧执行的检查后交付；首次目标机启动必须通过新增 GPU 数值与 EP8 通信检查。源码可安装、内核可编译不等于八卡完整模型或速度验收通过。

## 固定选择

唯一模型仍是 **DeepSeek-V4.1-Flash**，ModelScope revision `f877e3ed82d979be83e828b52762cdb86152bb9c`；不更换权重。唯一默认方案如下：

| 部分 | R1.2 |
|---|---|
| 并行 | TP4×DP2＋EP8，单机八卡，PP1 |
| DSpark | 固定 k=5、greedy draft、标准 rejection、local argmax |
| 主干专家 | 原 FP4 Marlin；EP8 分布；EPLB 无冗余专家、async，草稿明确关闭 EPLB |
| Dense | SM80 上预备部分 MXFP8 dense 的 BF16 副本，≥32 行用 BF16 GEMM，其余保留 Marlin |
| 通信 | EP8 等长 BF16 AG/RS 尝试镜像已有 custom 算子，不适用时走原 NCCL；NCCL 自动选择 |
| Attention | 8-warp split-K decode；累计紧凑候选索引 |
| Shared experts | CUDA aux stream 重叠；只在原兼容性条件允许时使用 |
| CUDA Graph | FULL_DECODE_ONLY、mode=0；decode 保留图，prefill eager；DP padding metadata 修正 |
| Batch / HBM | 8192 token / DP 池，gpu_memory_utilization=0.90；实际 KV 重新标定 |
| 上下文 / 图片 | 262144 输入输出合计；保留视觉，image=999 和 128 MiB 请求体限制 |
| Engram | 继续主存卸载，不更改宿主 NUMA/THP |
| API | Chat、Responses、Messages；唯一模型名；三协议工具与图片回归 |
| 访问 | 本机 loopback 与 Docker bridge 的 8005 端口，客户端免密；不操作 NewAPI |

累计保留 R1.1 的 mHC 融合、稳定 MoE 排序、候选索引与图像 padding 修正。稳定 MoE 排序仍开启。新增 DSML 兼容无空格/旧包装拼写，避免长对话中合法工具调用被当成可见文本。

## 552B 为什么可以讨论 DP2

这是 TP 与 EP 组合下的 DP，**不是完整复制两份 552B 主干到四卡组**。路由专家沿跨两个 DP 组的 EP8 分片，attention、共享专家、草稿的其他状态按照各自实现复制或 TP 分片。

本地原始 48 分片头的核算如下，单位是 GiB（2³⁰ bytes）：

| 权重分类 | 原始存储量 | R1.2 位置 |
|---|---:|---|
| 路由专家，含 DSpark 草稿专家和 scale | 275.67 GiB | EP8 分布；scale 转为 BF16 后合计每卡约 36.49 GiB |
| Engram embedding 表和 scale | 188.83 GiB | 主存 |
| 非路由专家权重，含视觉及 Engram 投影 | 10.74 GiB | 分片或复制到 GPU |
| 全部候选非专家 MXFP8 权重的额外 BF16 副本 | 13.79 GiB | 只是 dense fast path 的副本预算；不是全模型 BF16 化 |

Engram 按 TP 组分片，两个 DP 组各持有一份表，因而主存表预算约 **377.66 GiB**，还需加载临时空间和其他进程内存。这里按你的约 2 TiB 主存目标机设计，不能沿用 TP8 的单份主存预算。启动检查的可用主存下限相应提高到 512 GiB；这只是本方案的检查下限，不是实测加载峰值或官方最低要求。

极保守地把所有非专家数据与 dense BF16 副本都按每卡一份计算，得到约 **61.01 GiB/卡的权重场景**，相比 0.90×80 GiB=72 GiB 的配置预算有差额。这**不是运行峰值上界**：不含打包 padding、加载临时副本、KV、activation/workspace、CUDA Graph、通信和 EPLB staging。实际通常有 TP 分片节省，但能否启动和可保留多少 KV，必须看目标机日志。

核算脚本是 `build/audit_r12_weights.py`，结果附为 `reports/r12-weight-budget.json`。记录读取分片头，不再次扫描 510 GB 权重数据；已有整包 SHA256 验证记录仍保留。

## 并发与缓存池

每个 DP 池的有效容量独立计算：`N_i=floor(KV_i/262144)`，再取 **`C=max(32,2*(N_0+N_1))`**。vLLM 每个 DP scheduler 的 `max_num_seqs=C/2`；不会把 TP rank 当额外容量，也不会合并两个池中不足一个满窗口的碎片。原 TP8 的 C56 不能直接沿用。每池至少装下一个满 256K 窗口才允许启动；满窗口压力验收仍独立进行。

两池不共享 KV。默认用显式 `prompt_cache_key`，否则用首条用户消息的稳定哈希做会话亲和路由；追加历史保持原池。诊断可以传 `X-data-parallel-rank: 0` 或 `1`，响应头为 `X-DSV41-DP-Rank`。未指定亲和 key 的 stateful Responses continuation 留给原生路由。

亲和路由有取舍：同一会话或同一首条用户消息会集中到一池，充分利用 DP2 吞吐需要多个独立会话。跨协议编码不同也不保证同池。此改动不设置 TTL，不能据此认定过去的空闲缓存问题已经修复。

## 只传累计小包

传输 `deepseek-v4.1-flash-a100-R1.2-update.tar.gz` 及其 `.sha256` 文件即可。已导入的原始镜像 `deepseek-v4.1-flash-a100:20260913-r1` 继续复用；不需要 R1.1 中间包，也不携带模型或镜像。

在目标机拷包目录执行：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.2-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.2-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.2-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

这是前台命令，会停止本项目服务并重启。安装前核对镜像 ID、源码兼容哈希和包校验；全部工具来自原镜像。只使用普通用户和 `sudo -n docker`，不修改宿主服务，不联网。

安装器保存旧源码、权限和 TP8 调优配置。旧 `runtime/selected-config.json` / `tuning-active.json` 如存在，会改为 R1.2 配置并保留备份，避免它们盖过新拓扑。保留已有权重路径、API 端口、containerd 路径及运行日志；不删除旧编译缓存作为升级前置。

新增算子首次检查包含 dense、稀疏 decode、mHC、索引与 MoE 数值/图回放，最长 30 分钟；新增 EP8 八卡检查最长 300 秒，包含等长 custom 路径、变长 NCCL 回退和图回放。它不是重跑之前的全部基础探针。检查通过记录按覆盖版本、镜像、驱动、开关缓存。

模型启动后分别向 DP0/DP1 发 smoke 请求并要求两个池共八个 worker 的 target/draft 图实际回放。KV 重新标定可能再次加载模型。安装器遇到检查或启动失败会恢复旧源码并尝试启动旧服务，报告保留在 runtime。

已成功升级后，主动回退：

```bash
cd /ai/services/deepseek-v4.1-flash-a100/deploy && bash ./rollback-update.sh
```

回退拒绝覆盖升级后又手工修改的源文件。省略 `--restart` 只更新磁盘文件，不能把仍运行的旧容器算成 R1.2。

## 验收与速度对比

在 deploy 目录执行：

```bash
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
bash ./tests/performance-benchmark.sh --images 8
```

完整验收包括两池的三协议工具/图片/缓存检查与分别 256K 单请求检索，再做全局并发和性能测试。性能脚本记录 off/high、C1/C32、实际配置、KV 容量、DSpark 接受率和逐请求计时，同时报告 `load_requests_by_dp`，区分两池负载与同会话单池负载；R1 历史吞吐并非本版结果。

更有代表性的是相同截图历史：将脱敏 Chat 请求（inline data URI）存为 `runtime/benchmark-chat.json`，运行 `bash ./tests/performance-benchmark.sh --request-file /state/benchmark-chat.json`。重复会话会固定一个 DP 池，这时 C32 是同会话压力，不代表两池总吞吐；独立会话压测需提供不同 prompt_cache_key。图片正文不写入统计报告。

对照应暂停其他流量，固定输入、思考模式、输出长度、冷热缓存，采用 A→B→A。EPLB 按 3000 步调平衡，短测可能未触发重排；需要长测覆盖至少一次重排再检查工具和图片质量。2h/24h 稳定性、满窗口并发、多图长历史和真实 Agent 任务仍是签收条件。

## A100、A800 与来源

同为 80GB SXM 的公开规格中，两者 BF16 和 HBM 带宽相近；A100 NVLink 600 GB/s、A800 400 GB/s。A100 对 EP8 通信更有条件，但 50% 的链路带宽差不能换算成 50% 的模型提速。来源：[NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/)、[Lenovo A800 规格](https://lenovopress.lenovo.com/lp1813-thinksystem-nvidia-a800-pcie-gpu)。

感谢 [Tokha233 的 V4.1 Ampere Turbo](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/tree/fae324ae62ac5cef31b7d38f5d369618e1cae1fa) 提供 TP4 DP2 EP8、dense 分派、custom AG/RS 和 DP/EPLB 修正。其整模实测是 A800，且默认 `language-model-only`；本项目保留视觉，需独立验证。

感谢 [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)、vLLM、Zanooda、PyTorch、Triton、TileLang、NCCL。8-warp 与共享专家调优的上游微基准含旧 V4-Flash 案例；移植依据是 V4.1 调用相同 SM80 底层算子，**没有更换为 V4 模型，也不照搬其收益数字**。派生源码保留原声明和逐文件哈希。
