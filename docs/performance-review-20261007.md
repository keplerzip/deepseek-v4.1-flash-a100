# 2026-10-07：V4.1 在 A100 上的支持与提速进展

截至 2026-10-07，本次核对的公开源码和测试中，**尚未找到同时保留原 V4.1 权重、多模态、1M 上下文和 C32，且已证明快过本项目 R1.8 的新 A100 运行栈**。有值得继续移植的优化，优先是 DSpark 上下文 KV 图捕获，随后是 inverse RoPE 融合和长前缀 KV gather。官方新版本和其他硬件的成绩不能直接换算成本机收益。

本次是研究记录，没有改变部署默认值、镜像、权重或发行版本。构建机无 GPU，未新增目标机实测。合并时间以下以 UTC 计；GitHub PR 的状态是本次查询时的快照。

## 比较基线

本地基线为 R1.8 / v1.8.0，Git 提交 `89bd2ef`。实际权重仍是 **DeepSeek-V4.1-Flash**，ModelScope revision 为 `f877e3ed82d979be83e828b52762cdb86152bb9c`；按照操作者后续要求，对外 API 名称是 **DeepSeek-V4-Flash**，名称变化不代表权重变成 V4。

目标为 8×A100-SXM4-80GB、NVSwitch、约 2 TiB RAM。当前方案为 TP4×DP2＋EP8、DSpark k=5、1,048,576 token 上下文、GPU 利用率 0.92、总 C32、前缀缓存开启、Engram 主存卸载。镜像继续复用 `deepseek-v4.1-flash-a100:20260913-r1`；原 backport 源码提交为 `a350766628514d597670d2a5217777fdee2ba7f4`。

R1.8 已有 stable MoE 对齐、dense BF16 分派、EP8 通信、mHC 优化、candidate-only indexer、共享专家重叠、DSpark context KV 合并投影、ragged top-k 复用和 decode 元数据融合。它已使用主模型和草稿 CUDA Graph；下面讨论的**上下文 KV 预计算纳入草稿图**是另外一个尚未移植的步骤。

## 官方支持现状

- [vLLM v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) 于 10 月 5 日发布，包含多项 V4.1 融合、Engram 和视觉优化。不过该固定标签的 [sparse MLA 后端](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/deepseek_v41/sparse_mla.py#L124) 接受 SM90/SM100，[FlashInfer V4.1 后端](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py#L137) 接受 SM100/SM120，均不包含 SM80。该版本不能作为本项目 A100 镜像的直接替换。
- [官方 SM8x V4.1 跟踪 issue #57144](https://github.com/vllm-project/vllm/issues/57144) 仍开放。这里的结论同时依据固定标签源码，不只依据 issue 状态。
- [SGLang 最新 V4.1 部署文档](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1) 的验证矩阵为 H200、B200、B300、GB300 和 MI355X，没有 A100；NVIDIA 路线仍要求 `dev-dsv41` 预览镜像。社区其他 SGLang 分支的 SM80 支持需要单独判断。

## 新进展与移植顺序

| 优先级 | 改动、状态 | 公开证据 | 本项目判断 |
|---|---|---|---|
| 第一批 | [#57632：draft context KV 预计算纳入 FULL 图](https://github.com/vllm-project/vllm/pull/57632)，9/24 合并 | GB200、DSpark k5 的受控测试，C1 TPOT 1.27→1.08 ms，约 −15.2%；C8/C64 收益很小。性能测试使用合成接受长度 | R1.8 对应步骤明确在图外；机制不要求原生 FP8 Tensor Core，值得优先适配。A100 收益未测，不能承诺 15% |
| 第一批 | [#57435：inverse RoPE 融合到 sparse decode reduce](https://github.com/vllm-project/vllm/pull/57435)，9/22 合并 | MI355X 原独立旋转核每步约 43 次 launch、196 µs；作者补充实际融合路径和 GSM8K 验证 | Ampere 继承同一套 ROCm/Triton 实现，改动点可达。必须适配本项目 split、软件 FP8、DSpark noncausal 和图内动态输入 |
| 长输入专项 | [#56720：按 gather 长度选网格及 one-pass 解码](https://github.com/vllm-project/vllm/pull/56720)，10/6 合并 | MI355X、TP4、无 DSpark、C4、8 输出 token：131K 热缓存 TTFT −43.2%，冷输入 −10.6%；不是持续 decode 结果 | 按长度选网格可研究 SM80 移植；新 one-pass 内核仅默认分派到 ROCm。A100 需保留软件 FP8 并重测网格，不能照搬“22 倍”微基准 |
| 图片专项 | [#56625：vision encoder 图](https://github.com/vllm-project/vllm/pull/56625) ＋ [#58499：消除 replay metadata 主机同步](https://github.com/vllm-project/vllm/pull/58499)，后者 9/26 合并 | 修复后的 H200 视觉图测试通过，不再触发主机同步检查 | R1.8 尚未移植视觉编码器图；应完整移植并加入同步修复。主要改善图片编码、TTFT 及混合负载干扰，不等于单请求持续 decode 同比例变快 |
| 跟踪项 | [#59511：context combine/anchor 也纳入草稿图](https://github.com/vllm-project/vllm/pull/59511)，开放、非 draft | GB200 的 C1 ITL 约 −2% 至 −6%，基线自身有约 5% 漂移；C8/C64 无可测收益。性能测试使用合成接受长度 | 依赖 #57632，适合作为后续独立开关；尚未合并，不宜与前项百分比相加 |
| 次要 prefill 项 | [#60320：DSpark 只投影尾部窗口](https://github.com/vllm-project/vllm/pull/60320)，10/7 新建、开放 | GB200 的草稿 context GPU 时间显著缩短，但整请求 TTFT 仅改善约 0.3%–1.3%，区间包含无收益 | 有 prefix cache、KV connector、anchor 和 SWA 布局限制。本项目开启前缀缓存，需逐条件判断，不列为首选 |

`#57632` 本地对照点：`deploy/overrides/performance/patched/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` 的注释和调用仍明确说明 context KV 预计算在图外。移植时应处理 padding slot、超出图容量的 prefill、多个 KV group 和两个 DP worker 的形状一致性，不能只把旧 eager 调用挪进 capture。

`#57435` 的早期修订曾出现流畅输出但任务质量下降，因此使用已合并实现的设计，并验证批次相关行数不被冻结进编译/图状态。只测“能输出文字”不能确认这种融合正确。

长输入 gather 与视觉图各自占用额外工作区或图显存。任何实际移植后都应重新读取两套 DP KV 池，不能继续假设历史满 1M 容量保持不变。

## Engram：进展确实存在，A100 收益仍未建立

| 改动 | 核对后的限制 |
|---|---|
| [#59327：排序 host lookup、较大查询不走后台 stream](https://github.com/vllm-project/vllm/pull/59327)，10/1 合并 | GB200 大批查表微基准明显改善；排序阈值在 Blackwell 启用，其他架构默认不启用。我们的旧 backport 已在 main stream 准备 lookup，串行部分不是新增收益 |
| [#56926：串行 lookup＋可选 THP](https://github.com/vllm-project/vllm/pull/56926)，9/25 合并 | 主存大页方向可研究，但目标 Linux 5.15 不支持 6.1 才有的 `MADV_COLLAPSE`。不能把 THP 覆盖率当成现有条件，也无需先更换宿主驱动或内核 |
| [#59849：按精确大小注册主存表](https://github.com/vllm-project/vllm/pull/59849)，draft、未合并 | GB10 显示更少主存和 lookup 微基准收益；作者仍要求 GB200 与整服务验证。未提供 A100 整模型成绩 |
| [#58678：wkv 按 TP 分片](https://github.com/vllm-project/vllm/pull/58678)，9/26 合并 | 有 sequence parallel 时继续使用 replicated 投影。本项目 TP4＋DP2＋EP8 的 `_use_sequence_parallel` 为 true，默认走不到这项分片收益 |
| [#59923：packed FP8 DP exchange＋直接 MXFP8 wkv](https://github.com/vllm-project/vllm/pull/59923)，开放 | 快路径要求 TP1 和相应直接 MXFP8 核；公开性能来自 GB300。本项目 TP4/A100/Marlin 不符合该快路径条件 |
| [#57893：小批 lookup tile](https://github.com/vllm-project/vllm/pull/57893)，draft、开放 | GB300 组件变快，但整服务配对测试平均仅 +0.11%，区间跨 0，没有证明持续 decode 提速 |

因此，不能据这些结果认定“RAM 卸载让 A100 速度掉一半”，也不能承诺新 Engram 补丁能把它补回来。应测目标机上的 lookup、NUMA/PCIe、attention、dense/MoE、草稿和通信占时，再决定优化哪一段。跨 DP 共享主存表主要减少 RAM 副本；并不等额增加 GPU KV 空间。

## 社区 A100/SM80 方案

- **[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport/commits/master/)**：默认分支是 `master`。相对 9/22 核对点，新 HEAD 为 `29e66dad461b11caae6039f510eaa7913f9dfb19`，新增四个提交为 Qwen4、MiMo 和 PP KV 分配修正，没有新的 V4.1 decode 优化；`master-v013` 在这段时间无新提交。不能因 repo 的 `pushed_at` 更新就判断运行镜像变快。
- **[Tokha233 turbo](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo)**：HEAD 仍是 `fae324ae62ac5cef31b7d38f5d369618e1cae1fa`，未发现相对上次的新源码。其八卡 A800 短 coding 请求 C32 high 聚合约 1459 tok/s，C128/256 约 3040 tok/s；都是聚合吞吐，非单流。dense BF16 和 EP8 通信等核心改动已在 R1.8 中，不能重复算新增收益。[StellarVoyager 模型卡](https://huggingface.co/StellarVoyager/DeepSeek-V4.1-Flash-A100-A800-SM80) 指向同一配方，并非新权重或新的独立 A100 基准。
- **[shi3z custom](https://github.com/shi3z/deepseekv4.1-A100-custom)**：主干 HEAD 仍为 `580568edd6f84537ff042ded4ee842b892bf3ff4`。新看到的 [concentrate1 fork](https://github.com/concentrate1/deepseekv4.1-A100-custom) 增加模型 worker/网关分离、批处理、图文缓存和取消处理，最近推送为 9/27。这是服务支持进展，但未看到可与本项目八卡、C32、1M、多协议相对照的性能报告；文档入口仍需安装依赖，也不是现有离线包的直接替换。
- **[LvLLM](https://github.com/guqiong96/Lvllm) / [Lsglang](https://github.com/guqiong96/Lsglang)**：明确新增 V4.1 SM80＋DSpark 支持，使用 CPU/GPU 混合 MoE 和 NUMA 调度。公开 V4.1 样本主要为两张消费卡加 EPYC CPU，LvLLM 单流约 24–40 tok/s，Lsglang 另一硬件样本约 60 tok/s。它们说明更多硬件可以运行模型，未证明八卡 A100、计算权重已在显存时会更快。本项目不应为这些数字主动把路由专家计算迁到 CPU。
- **[5090 Accel](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel)**：主干最近推送仍在 9/16。“5.6 倍”是相对其 eager/CPU expert-offload 基线，SM120、PCIe 拓扑和驻留策略不同，不能拿来预测本项目增益。

## 通用底层优化也纳入，但按实际调用链判定

1. **[backport #110：小 q 的 3D split-KV verify](https://github.com/wtdcode/vllm-backport/pull/110)** 在 CMP170HX/SM80、MiMo-V2.6 67K 上有 16.5→33.7 tok/s 的整服务证据，单个注意力核约 7–20 倍。这是有效 SM80 证据，不能因为型号不是 A100 就忽略。但改的是 `triton_unified_attention_diffkv.py`，本项目走 `TRITON_MLA_SPARSE_DSV41`，不会直接执行它。可借鉴 verify 并行度，不能直接安装后声称 V4.1 翻倍。
2. **[backport #82：FA2 verify split-KV](https://github.com/wtdcode/vllm-backport/pull/82)** 在 Qwen3.8-27B、SM80、DFlash2 下约 +12%；它不是本项目稀疏 MLA 后端，且当前未合并。
3. **[#58132：decoder SWA bounded replay](https://github.com/vllm-project/vllm/pull/58132)** 已在 9/29 合并，GB200 的长输入首 token 延迟明显改善；但 sequence-parallel MoE 下禁用，并非本项目默认方案可直接开启的优化。[#59039 的 ROCm 适配](https://github.com/vllm-project/vllm/pull/59039) 仍需额外索引边界改动和后端适配。
4. **[#46963：SM80 NVFP4 KV](https://github.com/vllm-project/vllm/pull/46963)** 针对普通 MHA/FlashInfer FA2 的分页 K/V，不是 V4.1 的压缩 MLA 记录。[#59342：`nvfp4_ds_mla`](https://github.com/vllm-project/vllm/pull/59342) 的新路径仍面向 SM100/103。二者均不能证明我们可直接把 1M KV 容量扩大约三倍。
5. **[#59668：candidate mask 单核](https://github.com/vllm-project/vllm/pull/59668)** 是通用 Triton 思路，但 R1.8 的 candidate-only 路径已绕过部分全宽 mask；先测剩余调用，不能按原基线计算整层收益。

## 对下一版的建议

继续沿用原权重、k5 和 TP4×DP2＋EP8。**以 #57632 为第一优先，#57435 独立开关；长输入另测 #56720；多图另测 #56625＋#58499。** Engram 排序/注册作为 profiling 后的专项，不先改宿主内核或把专家计算搬到主存。

这些候选大多是 Python/Triton 源码层面的适配，可能复用原镜像做小包；目前只是代码依赖判断，尚未验证旧 PyTorch/Triton/模型 runner 接口，所以没有将其标为“已可安装”。完整上游 0.31.0 或替换运行时不属于同一种小包升级。

目标机验证应隔离其他客户端的请求，固定文本和同一组真实截图历史，分别测 off/high、C1/C32。保留 decode/E2E/TTFT、DSpark 接受长度、缓存命中、抢占和两池 KV。交错重复 baseline→candidate→baseline，图修改还需动态输入/slot 回放和多模态回归。各 PR 在不同硬件上的百分比不能相加，微基准也不能当成整模型 TPS。

已有 R1.8 的 8 图 off 单流中位数 128.02 tok/s 来自操作者回传；那次测试检测到全局 token 计数与本轮请求不一致，后续确认可能存在后台流量。它可作为有噪声的观察记录，不能作为新补丁的干净对照；最高样本 177.29 tok/s 也不等于稳定平均速度。

结构化状态、固定提交和适用性记录见 [本次 JSON](../reports/performance-review-20261007.json)。
