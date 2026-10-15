# 2026-10-14：继续追踪 A100 性能进展

本次对照 [10 月 7 日调研](performance-review-20261007.md)，检查近期提交、PR 合并状态、测试口径和 R1.8 实际调用链。**公开进展有增加，但本次仍未找到在原 V4.1 权重、8×A100、多模态、1M、C32 条件下，已证明超过 R1.8 的直接替换运行栈。**

最值得新增到后续移植评估的是：**prefill 去掉不必要的主机同步、缩减 indexer 工作区，以及按有效长度做 candidate selection**。持续 decode 的第一优先仍是上次选出的 DSpark context KV 图捕获。其他硬件的百分比不代表本机收益。

这是研究记录，未改动 serving 源码、默认值、权重、镜像或版本，也没有新增目标机 GPU 测速。合并时间以 UTC 计，状态截至本次查询。

## 基线与支持现状

以操作者最后确认的 R1.8 / v1.8.0 为比较基线：实际权重 DeepSeek-V4.1-Flash，对外别名 DeepSeek-V4-Flash，TP4×DP2＋EP8，k5，1,048,576 token 上下文，GPU 预算 0.92，总调度并发 32，前缀缓存和视觉开启，Engram 主存卸载。构建机没有 GPU，隔离目标机由操作者手动传包。

- `wtdcode/vllm-backport` 的 `master` 和 `master-v013` 自 10/7 起均没有新提交，标签最高仍为 v0.13.1。本次更新的 PR 主要是 GLM draft KV 与 MiMo，不是已合并的 V4.1 提速。[默认分支提交](https://github.com/wtdcode/vllm-backport/commits/master/)
- Tokha233 turbo、shi3z custom、concentrate1 fork 和 Zanooda CMP170HX 配方主干自 10/7 起也没有新提交。不能根据页面爬取时间重新计为新性能版本。
- 官方 vLLM 最近正式发布仍是 [v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)。本次读取的 main 中，V4.1 [FlashMLA backend](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/sparse_mla.py) 仍接受 SM90/SM100，[FlashInfer backend](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py) 仍接受 SM100/SM120；没有 SM80 分派。[SM8x 跟踪 issue](https://github.com/vllm-project/vllm/issues/57144) 仍开放，没有本周新评论。
- [SGLang V4.1 文档](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1) 增加了 agentic 长上下文部署内容和缓存/router 说明，但验证矩阵仍为 H200/B200/B300/GB300/MI355X，没有 A100；NVIDIA 配方仍使用 `dev-dsv41`。不是 A100 的现成升级包。

## 有明确本地落点的通用改动

### 1. Prefill 去掉 repeat_interleave 的输出长度同步

[#60091](https://github.com/vllm-project/vllm/pull/60091) 为两次 `torch.repeat_interleave` 加 `output_size=num_tokens`，消除读取 GPU repeats 来确定输出长度的同步。它 10/5 已创建、10/12 更新，仍未合并，是本轮新补充到候选清单的项，不能称为 10/7 之后新建。

公开测试：MI350X、TP4＋EP、8K 输入，C1 TTFT 340.7→321.1 ms（−5.7%）；混合服务 C64 聚合输出 1717→1763 tok/s（+2.7%），范围与基线不重叠，保留 GSM8K 和 metadata 输出一致性检查。这些是该硬件/工作负载的结果，不是 A100 的预期承诺。

本项目的实际调用链是 NVIDIA 模型选择 Ampere attention，而 Ampere 继承 `amd/rocm.py`。R1.8 该文件的 `combine_topk_swa_indices` 中，两次调用仍缺少 `output_size`，prefill 路径实际调用这个函数。因此，不能因 PR 写 ROCm 就排除它；这是有代码依据的 A100 适配候选。改动只针对 prefill，不直接缩短无 prefill 的纯 decode 步。

核对位置：

- `deploy/overrides/performance/patched/vllm/models/deepseek_v4_1/amd/rocm.py`：两次 repeat_interleave 和 `_forward_prefill` 调用。
- `source/vllm-backport-a350766628514d597670d2a5217777fdee2ba7f4/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py`：继承关系。

进一步的 [#60936：token 并行 Top-K/SWA 合并](https://github.com/vllm-project/vllm/pull/60936) 是 10/10 新建、未合并。B300 组件 49.152→20.480 µs，但整模型仅有 optimized-only 运行与历史基线约 +0.60% 的描述性差异；尚无最终 selector 的受控 A/B，六个图像集成测试受依赖阻碍。本项目需要将机制适配到继承的 Torch 路径，不能照搬 common ops 文件就假设已经启用。

### 2. 长上下文 indexer 工作区缩减

[#60874](https://github.com/vllm-project/vllm/pull/60874)，10/9 新建、开放。在 H20 TP8、1M、k5 上把每卡 KV 可用空间从 19.69 GiB 提高到 23.39 GiB，即释放约 3.70 GiB；长输入和多 session TTFT 基本不变。主要是容量改进，不是 decode TPS 提速。

R1.8 的 `DeepseekV4Indexer.max_total_seq_len` 仍按 `get_max_prefill_buffer_size` 分配；后者为 `40 * max_model_len`。FP8 K-gather 行的值与 scale 共 132 bytes。根据真实 chunk 的 logits 预算给 key rows 设上界，是可以评估的通用机制。

但 H20 的 3.70 GiB 同时包含 FlashMLA prefill gather 缩减。A100 走不同 attention 实现，并共享 workspace；只缩减 indexer 后，其他 reservation 可能成为最大值。**不能直接写成 A100 每卡释放 3.70 GiB，也不能直接换算新增满 1M 并发。** 应对照所有 reservation、warmup 与真实 chunk planner，修改后重新测两个 DP pool。

### 3. 按有效上下文长度选择 candidate blocks

[#61039](https://github.com/vllm-project/vllm/pull/61039)，10/10 新建、开放。新路径只对有效块评分，使用 DeepSelect per-row end，部分路径输出 BF16 logits。在 GB300、1M 服务上限、实际约 131K 上下文的受控 decode 测试中，ITL 26.52→25.65 ms（约 −3.28%）。性能测试使用 DSpark 合成接受长度；另有真实拒绝采样的 GSM8K 检查。

R1.8 已有 candidate-only consumer indexer，但 candidate-source 的 `candidate_write` 分支仍调用旧 `_select_candidate_blocks`，所以仍有可评估落点。旧镜像没有新 `indexer_topk` / DeepSelect 集成接口；BF16 logits 部分又绑定新的 DeepGEMM 路径。不能用一份 Python diff 在 SM80 上直接获得表中收益。优先评估保留现有 FP32/software-FP8 数学的有效长度选择，独立检查 ties、NaN、padding 与图内变长回放。

## 本周合并的进展与适用限制

| 项目 | 本次状态／公开证据 | 当前方案判断 |
|---|---|---|
| [#59923：packed FP8 Engram DP exchange](https://github.com/vllm-project/vllm/pull/59923) | 10/9 合并；GB300 查表/交换组件明显更快 | 最终快路径仍要求 TP1、相应直接 MXFP8 wkv；decode peer exchange 还需 SM100。我们的 TP4/SM80 保留 BF16 路径，不能算直接收益 |
| [#59039：ROCm decoder SWA bounded replay](https://github.com/vllm-project/vllm/pull/59039) | 10/13 合并；MI325X、TP4、16K、无 DSpark/前缀缓存，TTFT 1176.37→629.97 ms（−46.4%），TPOT 10.59→10.58 ms | 明确是 prefill 提速；最终代码仍禁用 sequence parallel，当前 TP4×DP2＋EP8 正好开启 SP。不是默认可开启的 A100 小改 |
| [#59894：最后一个 KV-source 也缩减 replay](https://github.com/vllm-project/vllm/pull/59894) | 10/9 合并，是 replay 依赖；[#59532](https://github.com/vllm-project/vllm/pull/59532) 的 replay 图已在 10/6 合并 | 说明 replay 方案成熟度增加，不解除本项目 SP 与 attention 适配条件 |
| [#60397：MI355X mono decode](https://github.com/vllm-project/vllm/pull/60397)、[#61082：覆盖全部 backbone](https://github.com/vllm-project/vllm/pull/61082) | 分别 10/10、10/13 合并；后者相对已启用 mono 的前者，AgentX P90 541.6→562.6 tok/s/user，使用合成接受长度 | 真正的新融合路线，但基于 AMD FlyDSL、MFMA、peer memory 和新权重布局，没有 A100 实现。不能设置 `--monokernel` 就迁移 |
| [#60731：gfx942 mono](https://github.com/vllm-project/vllm/pull/60731) | 开放；MI325X C1 AgentX P50 216.85→503.53 tok/s/user（约 2.32 倍），一轮每侧 3600 秒 | 是 AMD 交互测试，不是 A100 单流 decode。该版本每 rank 权重约 170.11 GiB，且只在 gfx942/TP4 小批启用；更不能原样部署到 80GB A100 |
| [#46963：SM8x NVFP4 KV](https://github.com/vllm-project/vllm/pull/46963) | 10/7 合并 | 是普通 MHA/FA2 K/V，不是 V4.1 的压缩 MLA，仍不能给本项目直接换 KV 格式 |
| [#57893：small Engram lookup](https://github.com/vllm-project/vllm/pull/57893) | 10/13 关闭，未合并 | 不标记为可用的新优化；先前 serving 结果仍没有显著收益 |

Mono kernel 的整层融合、减少 launch 与通信开销的思路值得长期研究，不能因 AMD 标签排除机制。但真正移植到 A100，需要 CUDA/SM80 实现、压缩权重布局与内存预算、同步和数值验证，超出原镜像少量源码替换的范围。

## 不纳入默认的其他新项

- [#61088：compression＋RoPE＋FP8 cache 写入](https://github.com/vllm-project/vllm/pull/61088)，draft。算子 1.18–1.53 倍，但 H20 的两组 TP4 crossover 主指标 cold 16K/C16 为 −0.05%、warm C16 为 −0.77%，没有稳定整服务增益；代码又明确 gated 到 SM90。不能宣传为模型提速 50%。
- [#60938：Engram 与 delayed mHC 融合](https://github.com/vllm-project/vllm/pull/60938)，draft。尚无 GPU correctness/performance/model 验证；仅 Blackwell、TP1、无 SP、无 overlap stream。不是本项目现成优化。
- [#61395：异构主干／draft 专家数的 EPLB](https://github.com/vllm-project/vllm/pull/61395)，draft。这项终于覆盖 V4.1 draft 专家拓扑差异，但 GPU、GSM8K 和速度验证尚缺，还依赖 redundant expert loader 修正。当前不据此开启草稿 EPLB。
- [#61530：MXFP4 Engram 表](https://github.com/vllm-project/vllm/pull/61530)，开放。减少主存占用，但需要换成 MXFP4-packed checkpoint，违背本次保持原权重的范围；公开表格也不是干净的速度对照。
- [#60909：精确大小 pinned 分配](https://github.com/vllm-project/vllm/pull/60909)，开放。主要减少 CPU KV tier／weight-offload 的内存浪费，不是已证明的 V4.1 Engram/A100 decode 提速。
- [#59342：nvfp4_ds_mla](https://github.com/vllm-project/vllm/pull/59342) 仍开放，新的 native decode 仍是 SM100/103，未解锁 SM80。
- [#59511：完整 context preparation 纳入 draft 图](https://github.com/vllm-project/vllm/pull/59511) 仍未合并，10/14 更新；既有 GB200 C1 约 −2% 至 −6% 的 ITL 证据没有变成 A100 结果。继续作为 #57632 之后的候选。

## 下一轮顺序

1. **Prefill 小改：#60091。** 改动点在本项目实际路径，保留数学与动态长度，只去除不必要的输出长度同步；先验证空请求、混合 batch、query_start 非零基址和输出等价。
2. **持续 decode：#57632，再独立测 #57435。** 延续上次选择，不把 #59511 的百分比叠加进去。两个 DP worker 的 slot/padding、超出图容量的 prefill 和动态回放仍需验收。
3. **1M 容量：#60874 的 indexer 边界机制。** 检查 shared workspace 的所有最大项，重新标定两个 KV 池；总调度 C32、k5 不因此自动更改。
4. **长截图：#56720 gather、#61039 有效长度选择，以及 #56625＋#58499 视觉图分别评估。** 保留软件 FP8、图像分辨率和原模型；区分 TTFT、持续 decode、接受长度和聚合输出。

这些方向不需要改变权重；#60091 很可能能复用原镜像，其他项还需检查旧 runner/Triton/算子接口，不能把研究记录当作已经完成的升级包。真正决定是否固化仍需目标机交错 A/B，并隔离后台客户端流量。不同硬件、不同工作负载和不同基线的百分比不能相加。

本次结构化状态与固定提交见 [JSON 记录](../reports/performance-review-20261014.json)。
