# DeepSeek-V4.1-Flash on 8×A100 — R1.9 / v1.9.0

## 实测性能：保留历史成绩，R1.9 / 500K / C64 待目标机验证

**8×A100-SXM4-80GB + NVSwitch，256K 上下文，TP8，Engram 主存卸载。** 同机 k5/k7 对照结果如下，最终默认选择 **k=5**。

| 思考模式 | DSpark k | 单流 decode tok/s | 单流端到端 tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|---:|---:|
| off | **5** | 139.80 | 137.22 | **1379.01** | **24.56** |
| off | 7 | 146.04 | 143.50 | 1359.60 | 25.28 |
| high | **5** | **109.95** | **108.58** | **1039.91** | **34.93** |
| high | 7 | 95.69 | 94.44 | 960.82 | 37.96 |

high 模式下，k=7 的单流速度与 C32 总吞吐分别下降约 **12.97% / 7.61%**。off 模式下，k=7 单流约 +4.46%，但 C32 总吞吐约 −1.41%。因此 **v1.0.0 固化 k=5**；不会因 k7 功能测试通过就声称它更快。

**测试口径：** 操作者回传的隔离目标机实测；固定顺序 5→7，每组 2 次预热、5 个单流样本、200 个 C32 请求，每请求 1024 输出 token，off/high 分别测量。高思考模式的 TPS 包含思考 token；C32 为短输入负载，并非 32 个满 256K 请求同时驻留 KV。五个单流样本不足以证明小幅差异具有统计显著性。

另一次较早的 **k5 / high / C56** 验收结果单独记录：

| 指标 | 实测结果 |
|---|---:|
| 单流 decode 中位数 | **101.73 tok/s** |
| C56 总输出吞吐 | **1334.67 tok/s** |
| 请求数 / 失败数 | **200 / 0** |
| C56 请求延迟 P50 / P95 | 39.38 / 47.53 秒 |
| C56 TTFT P95 | 1.65 秒 |
| 热前缀命中 | **14208 / 14398 token，98.68%** |
| 最大单请求检索输入 | **257482 token** |

这组 C56 成绩与上面的 C32 对照条件不同，不能直接合并比较。文本/SSE、三协议流式与非流式工具调用、1/2/4 图 OCR、长输入检索和 C32/C56 短请求均报告通过。**C56 是调度并发，不代表 56 个满 256K 窗口同时驻留。**

[完整性能报告与方法](docs/performance.md) · [机器可读实测记录](reports/) · [测试代码](deploy/tests/dspark_benchmark.py)

**R1.9 / v1.9.0：固定 500K＝1024,000 token、C64、调度预算 1024，唯一公开名称 `DeepSeek-Flash`。** 原 V4.1 权重不变，复用初始 SM80 镜像，累计保留 TP4×DP2＋EP8、k5、0.92、Engram 主存卸载、前缀缓存、图片上限 999 与三协议。新增 prefill 去同步、inverse RoPE 融合、DSpark context KV 图捕获、indexer 工作区上界优化。**尚无本版 A100 实测 TPS；上表是历史 TP8 / 256K 成绩，不能当作新版成绩。** [完整进版方案与传输说明](docs/deployment-update-20261015-r19.md) · [全网方案筛选](docs/performance-review-20261015.md)。

R1.8 用户回传还观察过：文本预热 176.25 / 192.10 tok/s；8 图 off 单流 39.49–177.29 tok/s，中位数 128.02。因后台请求造成全局计数不一致，两轮正式结果均 FAIL，无有效聚合成绩。这些观察值不能据此声称升级幅度。新版测速会保留污染数据并标记 CONTAMINATED，也支持只暂停本项目入口后隔离测速。

[权重容量核算](reports/r12-weight-budget.json)来自本地 V4.1 全部 48 个分片头：专家含草稿约 275.7 GiB，由 EP8 分摊；Engram 单份表约 188.8 GiB，DP2 的两份主存表预算约 377.7 GiB；其余权重约 10.7 GiB。DP2 并非复制两份完整模型，但非专家副本与额外 BF16 dense 缓冲会消耗显存，实际 KV 容量必须重测。

目标机原始逐请求 JSONL 留在隔离环境，尚未传到构建工作区；仓库记录明确标注“用户回传”。2h/24h 稳定性、满窗口并发驻留、Engram 位置对照和空闲 KV 保留问题均没有在本次发布中宣称已验证。

## 项目与默认方案

本项目提供 `DeepSeek-V4.1-Flash` 在八卡 A100 上的离线部署、验收和运维工具，是从实际隔离环境交付整理的 **源码发行版**。联网构建侧准备全部资源后，目标机仅用普通用户和 `sudo -n docker` 启动，无需公网下载或在宿主安装 Python 依赖。

| 项目 | R1.9 当前设置 |
|---|---|
| 唯一模型 / API 名称 | `DeepSeek-Flash`（底层仍是 V4.1 权重） |
| 上下文 | **512000 token（500 × 1024，约定 500K）**，输入与输出合计 |
| 并行 | TP=4 × DP=2，EP=8，PP=1；专家跨八卡分布 |
| 权重 | 官方混合 FP8/FP4，固定 ModelScope revision |
| DSpark | **固定 k=5**，主模型与草稿 decode CUDA Graph |
| 性能更新 | decode 元数据清理、小批 indptr/pack 融合；累计 DSpark context KV 合并、top-k 复用、dense BF16 ≥32 行、EP8 custom AG/RS、8-warp decode、条件共享专家重叠、主干 EPLB、R1.1 优化 |
| token 调度预算 | **每 DP pool 1024**，chunked prefill 开启；k5 每池最多 192 验证 token |
| 显存预算 | **固定 0.92**，启动后测量实际 KV 容量 |
| Engram | 主存卸载 |
| KV | `fp8_ds_mla`，prefix caching 与命中 token 明细 |
| 并发上限 | **固定全局 C64**；每 DP pool `max_num_seqs=32` |
| 多模态 | 图片数量上限999；**单图完整视觉token上限1024**，按固定调度预算缩放，大图细节可能减少；拒绝公网媒体抓取 |
| API | Chat Completions、Responses、Anthropic Messages |
| 访问 | `127.0.0.1:8005` 和本机 Docker bridge，客户端免密 |
| NewAPI | 接已有实例；不安装、不修改网关 |

C64 是两池总调度并发，不承诺 64 个满 512,000 窗口同时驻留。启动读取两池实测 KV，要求各池至少容纳一个完整窗口，并报告按池取整后的满窗口容量；不会自动改变用户指定 C64。1024 token 是每池每步调度额度，不是输出上限，长输入会分成更多块，可能增加冷 prefill 延迟。[构建与离线包](docs/building.md)

## 快速开始

本版提供**不含权重、含原运行镜像的完整包**，以及复用已装镜像的累计小包。已有目标机无需重新传权重。完整包解压后执行：

```bash
bash ./install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

已有部署的模型目录、端口和 runtime 保留，旧调优配置在事务中迁移；失败自动恢复旧源码并尝试重启旧服务。全新部署需先放好原 V4.1 权重；可先不带 `--restart` 安装，再检查目标 `deployment.env`。只需普通用户与 `sudo -n docker`，不安装系统服务。详细包名、校验、回退和首次启动见 [R1.9 说明](docs/deployment-update-20261015-r19.md)。

服务就绪后：

```bash
curl --fail http://127.0.0.1:8005/v1/models
cd /ai/services/deepseek-v4.1-flash-a100/deploy
bash ./tests/performance-benchmark.sh --isolate-clients
bash ./tests/performance-benchmark.sh --isolate-clients --images 8
```

隔离测速会暂停本项目本机/Docker入口，完成后恢复，期间已有客户端无法访问；不停止模型、不操作 NewAPI。宿主地址 `http://127.0.0.1:8005/v1`，容器地址 `http://host.docker.internal:8005/v1`；Linux 容器自行配置 `host.docker.internal:host-gateway` 映射。

## 文档

- [准备离线包](docs/building.md) · [部署](docs/deployment.md) · [运维与恢复](docs/operations.md)
- [性能报告](docs/performance.md) · [KV 缓存](docs/cache.md) · [故障定位](docs/troubleshooting.md)
- [R1.9：500K / C64 / 1024 与四项优化](docs/deployment-update-20261015-r19.md) · [R1.8：前端权限修复、C32 与 API 名称](docs/deployment-update-20260923-r18.md) · [R1.7 历史验收修正](docs/deployment-update-20260923-ep8-r17.md) · [R1.6 历史修正](docs/deployment-update-20260923-ep8.md) · [R1.5.1：0.92 固定方案](docs/deployment-update-20260922-gpu92.md) · [R1.5：1M 容量说明](docs/context-update-20260922.md) · [R1.4 历史优化](docs/performance-update-20260922.md) · [R1.3 历史方案](docs/performance-update-20260920.md) · [R1.2 历史方案](docs/performance-update-20260918.md) · [R1.1 历史更新](docs/performance-update-20260916.md) · [perf1 历史说明](docs/performance-update-20260915.md)
- [协议、NewAPI 与 CLI](docs/clients.md) · [固定实现](docs/architecture.md)
- [上游致谢](docs/acknowledgements.md) · [变更记录](CHANGELOG.md)

## 致谢与许可证

感谢 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 提供模型，[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) / lazymio 提供 SM80 适配与镜像，以及 [vLLM](https://github.com/vllm-project/vllm)、[PyTorch](https://github.com/pytorch/pytorch)、[Triton](https://github.com/triton-lang/triton) 和 [NVIDIA NCCL](https://github.com/NVIDIA/nccl) 的基础设施。本仓库负责固定版本、离线交付、接口修正、验收和结果记录，不将上游模型或 SM80 实现归为原创。

原创部署代码采用 Apache-2.0；第三方代码保留其版权与许可证，模型权重遵循模型自身许可。见 [LICENSE](LICENSE)、[NOTICE](NOTICE)、[完整致谢](docs/acknowledgements.md)。

R1.9单图上限1024的实际处理器预算为1020，另留DSpark k5的4个新增槽，总计1024；保留整张图片的双向注意力，不允许为缓存命中或混合负载而拆图片段。
