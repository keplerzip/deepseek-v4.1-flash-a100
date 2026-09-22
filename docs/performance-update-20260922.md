# R1.4 / v1.4.0：A100 decode 元数据融合累计更新

## 实测性能与本版状态

| 配置 | 单流 decode tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|
| 历史 R1，TP8，k5 / off，用户回传 | 139.80 | 1379.01 | 24.56 |
| 历史 R1，TP8，k5 / high，用户回传 | 109.95 | 1039.91 | 34.93 |
| R1.4，TP4×DP2＋EP8，k5 | **待目标 A100 实测** | **待实测** | **待实测** |

历史每组 5 个单流样本、200 个 C32 短文本请求，每请求 1024 输出 token；high 包含思考 token。原始 JSONL 尚未回传，表中历史数字不是 R1.4 或多图历史对话成绩。完整口径见 [性能报告](performance.md)。

本版选择两项实际经过 SM80 路径的整数元数据优化，累计包含 R1.3 及此前改动。**构建机没有 GPU，本版按待目标 GPU 验收的预发布版交付。** 不承诺截图 decode 从约 50 tok/s 恢复至短文本历史水平，也不将不同硬件的上游收益相加。

## 只传小包，直接升级

传输 `deepseek-v4.1-flash-a100-R1.4-update.tar.gz` 和同名 `.sha256` 即可。可从初始 R1、R1.1、R1.2、R1.3 直接安装，**不用先装 R1.3，不重传镜像或权重**。复用已导入的 `deepseek-v4.1-flash-a100:20260913-r1` 镜像。

在目标机拷包目录执行：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.4-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.4-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.4-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

命令前台运行，**停止并重启本项目模型服务**，只需普通用户及 `sudo -n docker`。全部目标依赖来自原镜像，不联网、不安装宿主服务、不操作 NewAPI。安装器核对哈希、镜像和底层源码；保留操作者的模型路径、端口、containerd 路径和运行日志，事务备份旧源码、权限及旧调优配置。

首次启动先运行启用的 GPU 数值/图回放检查和 EP8 通信检查，再加载整模、重算两个 DP 池 KV、检查两池模型响应与图回放。30 分钟算子检查及 300 秒 EP8 检查是超时上限，不是预计时间。KV 标定可能再次加载模型；不直接沿用 TP8 的历史 C56。

升级启动失败会恢复旧源码并尝试启动旧服务，日志保存在 `deploy/runtime/`。主动回退本次安装前状态：

```bash
cd /ai/services/deepseek-v4.1-flash-a100/deploy && bash ./rollback-update.sh
```

如果升级后手动修改了待恢复文件，回退会拒绝直接覆盖。省略安装命令的 `--restart` 只写磁盘文件；正在运行的服务不会因此换版。

## 固定方案与选择理由

| 项目 | R1.4 固定值 |
|---|---|
| 唯一模型 / API 名 | **DeepSeek-V4.1-Flash** |
| 原权重 | ModelScope revision `f877e3ed82d979be83e828b52762cdb86152bb9c`，FP8/FP4 |
| 并行 | TP4×DP2＋EP8、PP1；专家跨八卡分片 |
| DSpark | **k=5**，保留 R1.3 三层 context KV 合并投影 |
| 新增优化 | SWA 有效性直接写入持久缓冲；prefill 才计算压缩长度；小批 ragged indptr 与 pack 融合 |
| 累计优化 | R1.3 同一步内索引复用；R1.2 dense BF16、EP8 custom AG/RS、8-warp decode、条件共享专家重叠、主干 EPLB；R1.1 mHC/索引/MoE |
| 上下文 / 并发 | 262144；`C=max(32,2*sum(floor(KV_DP_i/262144)))`，每池上限 C/2 |
| Engram | 保留主存卸载，DP2 两份表预算约 377.7 GiB |
| 图文与协议 | 视觉开启，image=999；Chat、Responses、Messages；工具与缓存 usage 保留 |
| 本地访问 | loopback 和本机 Docker bridge 的 8005 端口，客户端免密 |

`image=999` 是配置上限，实际受总上下文、图像尺寸、视觉显存、并发和 128 MiB 请求体限制；不是 999 图已测承诺。DP2 不复制两份完整 552B 模型，权重预算见 [R1.2 核算](../reports/r12-weight-budget.json)。R1.4 没有增加权重副本，准确可用 KV 仍由当前 graph/工作区运行占用标定。

**为何先选元数据优化：** 它们命中本地 Ampere 继承的 `amd/rocm.py` 与通用 metadata builder，只改变整数索引准备；不改变浮点累加顺序、DSpark 接受规则、图片分辨率或上下文。这使其能作为独立、小范围、可回退的更新。上游尚无这些改动在本项目八卡 A100 上的整模型速度证明。[本轮候选审查](performance-review-20260922.md)

inverse RoPE 融合会改变浮点执行顺序，视觉编码器 CUDA Graph 需要适配旧协议与重新验证图显存预算；本版暂不合入。共享 Engram 主要减少 RAM，本机约 2 TiB 主存下尚无优先采用它提高 decode 的证据。

## 本次改动如何工作

1. **SWA 有效性原地输出。** 将比较后再复制改为 `torch.ge(..., out=persistent_buffer)`，仍保持供 CUDA Graph 使用的输出地址。
2. **纯 decode 跳过 prefill 长度除法。** 含 prefill 的批次继续按原压缩率计算，pure decode 不再准备未使用的压缩长度。源自已合并的 [vLLM #57885](https://github.com/vllm-project/vllm/pull/57885)。
3. **合并 ragged indptr 和 pack。** 先计算每行有效长度，再由打包内核直接计算该行前缀并写出物理槽位，省掉单独的 indptr 清零及 cumsum。源自仍为 draft 的 [#57860](https://github.com/vllm-project/vllm/pull/57860)，本地限制为 **1–1024 行且 top-k 宽度大于 0**；大批次保留线性扫描路径，空输入返回合法空结果。

第三项的“行”是注意力 token 行，含草稿验证与 padding，不能等同于请求并发 C。直接前缀求和的工作量随行数平方增长，因此设置形状上限；这不是 A100 最优阈值实测结论。空行/无效行也必须完整写出 indptr，且不能解引用无效请求。R1.3 的跨层复用仍在，所以不能把上游“每步多次 pack”的收益再重复叠加。

## 验证范围与目标机测试

[构建记录](../reports/performance-update-20260922.build.json) 记录原镜像、网络隔离、源码哈希和检查结果。新增检查包括：

- 10 组生产 builder 片段执行：压缩率 1/4/8，pure decode/混合 prefill，有效掩码空/短/较长行；检查整数结果、持久输出与 ATen 实际调用次数。
- 10 组生产 Triton 的 **CPU 解释执行**，每组修改输入后再执行；覆盖空输入、零宽、全 padding、非连续 top-k、带 stride 的 block table、1024/1025 行边界。与独立逻辑槽位参考精确比较。这不属于 GPU 执行。
- 新融合内核 7 个 SM80 编译形状，加上此前 37 个形状。编译出 cubin 不代表数值或性能已在 GPU 验证。
- 既有原生 MXFP8 分片/scale 加载、索引复用生命周期、DP/EPLB/DSML 控制流和三协议图像/schema 检查。
- 源码、哈希、安装事务、历史版本累计安装与逐字节/权限回退检查。最终归档的独立收据随 GitHub release 发布，避免把归档自身哈希写入归档造成循环。

目标机自动新增 10 组真实 CUDA ragged 对照与修改输入后的图回放，以及 3 组 SWA 有效缓冲图回放。算子结果精确对照独立 CPU 槽位参考和 R1.3 原路径；保留其他累计优化的 GPU 检查。结果在 `runtime/performance-kernels-*/`。一次通过不代表整模速度、长上下文和任务质量都已验收。

服务启动后在 deploy 目录分别运行：

```bash
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
bash ./tests/performance-benchmark.sh --images 8
```

最贴近慢请求的测试：将脱敏的 Chat 对话保存为 `runtime/benchmark-chat.json`，图片使用 inline data URI，运行 `bash ./tests/performance-benchmark.sh --request-file /state/benchmark-chat.json`。保留 off/high、C1/C32、TTFT、纯 decode、接受率、缓存命中与原始日志；固定输入、输出长度及流量，采用 A→B→A。

## 独立关闭新增项

默认 `deployment.env` 固化 `PERF_DECODE_METADATA=1`。诊断时仅改为 `0`，再执行 `bash ./restart.sh`：三个新增覆盖文件恢复 R1.3 的实现，R1.3 的 KV 合并和索引复用继续保留。恢复 `1` 并重启即可重新启用。开关及覆盖 manifest 哈希参与启动验收签名，旧检查结果不会冒充新配置通过。

该开关不调整 KV TTL、Engram 位置、鉴权、图片数量或 k。完整退回升级前版本请用 `rollback-update.sh`；切换开关只用于 R1.4 新增项的对照。

## 固定来源与致谢

- Woosuk Kwon / WoosukKwon，[vLLM #57885](https://github.com/vllm-project/vllm/pull/57885)，`73ce1d9220507a4d06412983e6cb599ed163eb43`：稀疏注意力冗余元数据清理。
- nehaprakriya，[vLLM #57860](https://github.com/vllm-project/vllm/pull/57860)，`2031422bfd952cc65fc1a46a9b570f6f04dc318d`：ragged indptr 与打包融合。本项目保留原 kernel 作为大批/诊断路径，并增加形状边界与空输入处理。

逐文件原始、R1.3 基线和 R1.4 覆盖哈希固定在 [manifest](../deploy/overrides/performance/manifest.json)。上游状态以本次调研快照为准；原 Apache-2.0 SPDX 保留。感谢 DeepSeek、wtdcode/vllm-backport、vLLM、Tokha233、Zanooda、PyTorch、Triton、TileLang 和 NCCL；见 [完整致谢](acknowledgements.md)。
