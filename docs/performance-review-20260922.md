# 2026-09-22：R1.3 之后的 A100 优化进展

有实质更新，但尚未找到在 **8×A100、原版 DeepSeek-V4.1-Flash、256K、DSpark k5、多模态** 这个组合下，用可比整模型实测证明优于本项目的新部署栈。下一轮值得优先适配的是视觉编码器 CUDA Graph、稀疏注意力元数据融合、inverse RoPE 融合。Engram 共享内存主要改善 RAM 容量，不能当成 decode 大幅提速的证据。

本文保留选择 R1.4 前的调研快照，以 [R1.3 / v1.3.0](performance-update-20260920.md) 源码为比较基线。**用户于本次调研中确认目标机尚未安装 R1.3。** 后续已选取 `#57885` / `#57860` 适配为 [R1.4 累计更新](performance-update-20260922.md)，其余候选仍按本文边界评估。现有速度体验与历史成绩不代表 R1.3 或 R1.4 实测。历史 TP8 短文本成绩仍是 off 单流 139.80 tok/s、high 单流 109.95 tok/s，不能替代多图长历史或新版实测。[历史性能与证据范围](performance.md)

核对时间为 2026-09-22 上午（北京时间）；PR 状态及完整 head SHA 固定在 [机器可读记录](../reports/performance-review-20260922.json)。以下外部性能数字均为对应作者报告，本项目未在目标 GPU 上复现。

## 相对 9 月 20 日的实质变化

| 更新 | 本次状态 | 对本项目的意义 |
|---|---|---|
| [#57434：跨层复用 ragged top-k](https://github.com/vllm-project/vllm/pull/57434) | 9 月 20 日已合并，head 与上次审查相同 | **R1.3 已移植**，不能再次计为新增提速 |
| [#56625：V4.1 视觉编码器 CUDA Graph](https://github.com/vllm-project/vllm/pull/56625) | 9 月 20 日已合并；head 更新为 `ca70b6d…` | 从待审 PR 变成已合并实现；本项目尚未移植，截图场景优先级上升 |
| [#57435：decode reduce 内融合 inverse RoPE](https://github.com/vllm-project/vllm/pull/57435) | 仍开放；9 月 21 日补充真实内核测试和完整 GSM8K，head `6a33016…` | 命中本项目 Ampere 继承的 Triton 路径；证据比上次完善，仍需 SM80 适配与质量验证 |
| [#57885：去掉多余 SWA/索引元数据操作](https://github.com/vllm-project/vllm/pull/57885) | 新增，9 月 21 日已合并 | 通用元数据代码，可作为小范围源码更新候选；作者未宣称显著整模型收益 |
| [#57860：融合 ragged indptr 前缀和](https://github.com/vllm-project/vllm/pull/57860) | 新增，仍为 draft | 虽标 ROCm，本项目 SM80 也调用这一函数；能进一步减少 R1.3 剩余的 pack 开销 |
| [#57651：默认跨 DP 共享 Engram 主存表](https://github.com/vllm-project/vllm/pull/57651) | 9 月 21 日已合并 | DP2 的表存储有机会由约 377.7 GiB 降至 188.8 GiB；这是主存变化，不是 GPU KV 增量 |
| [#57893：小批 Engram 查表优化](https://github.com/vllm-project/vllm/pull/57893) | 新增，仍为 draft | 有查表微基准收益，但配对服务测试未建立 decode 提速，暂不作为主线 |
| [#57421：Marlin/Humming 持久工作区共享](https://github.com/vllm-project/vllm/pull/57421) | 9 月 20 日已合并 | 通用内存管理方向值得关注；公开节省 9.836 GiB 的结果来自 Humming grouped MoE，不适用于直接估算我们的 Marlin 配置 |

R1.3 的另一项 [#57627：DSpark context KV 合并投影](https://github.com/vllm-project/vllm/pull/57627) 仍未合并，head 仍是 `f0defc90…`。本次没有发现其 head 发生变化，R1.3 对它的移植仍需本机 GPU 验收，不能因另一项已合并就视为整个 R1.3 已获上游验证。

## 截图场景：视觉编码器图值得做，但不要混淆计时范围

`#56625` 将 ViT 与 aligner 的多图编码打包进 CUDA Graph，减少每张图片的 Python 调度、内核启动及部分主机侧准备。作者在 **4×GB200、TP4、C32、256 个图像请求、每请求只输出 32 token** 的测试中报告：

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 请求吞吐 | 4.00 req/s | 10.37 req/s |
| 平均首 token 延迟 | 4467 ms | 1894 ms |
| 平均 TPOT | 114.2 ms | 38.3 ms |

这些结果说明它值得用于图片请求密集的服务；其中 TPOT 改善也包含图片编码减少对其他请求 decode 的干扰。**不能据此承诺单个截图历史对话的纯 decode 从 50 tok/s 提高到 130 tok/s。** 单请求的图片编码发生在生成之前；如果编码已复用，持续生成还受长上下文注意力、DSpark 接受长度等影响。[测试条件及实现](https://github.com/vllm-project/vllm/pull/56625)

本项目旧镜像已有 encoder graph 管理接口，但固定版 V4.1 类缺少本 PR 的协议和打包实现。移植需要同时处理共享 vision 代码、V4.1 mixin、视觉后端及图预算，不能仅增加 `cudagraph_mm_encoder=true`。该实现拒绝 TORCH_SDPA/FLASHINFER 视觉后端，提示使用 FLASH_ATTN；还限制动态 FP8 视觉量化。需要核对旧镜像实际视觉路径，并保留本项目图像 padding 修正。[合并代码](https://github.com/vllm-project/vllm/pull/56625/files)

图捕获占用显存，两套 DP worker 都必须参与预算。若移植，应重新读取两池可用 KV、分别取满 256K 窗口数后再求和，不能继续套用历史 C56。

同期 [#57898：最大合法图像特征 profiling](https://github.com/vllm-project/vllm/pull/57898) 修正用正方形图片估算最大 patch 数的假设。它仍未合并；宽图可产生比旧 dummy 更多的 ViT patches。属于与多图图预算相关的正确性检查，不是已证明的 OOM 修复或速度提升。应与视觉图方案一起审查，而不是降低图像分辨率来隐含换取速度。

## 持续 decode：实际可达的通用优化

本项目并非看到 ROCm 标签就排除。固定源码的调用链仍是：

```text
nvidia/model.py：SM8x 分派
  → ampere/ampere_sparse.py
  → amd/rocm.py：V4.1 注意力实现
  → rocm_aiter_mla_sparse.py：可在 SM80 执行的 Triton 算子
```

### 1. inverse RoPE 融合：证据补强，继续列为直接 decode 候选

`#57435` 把独立旋转放进 split-K reduce 的尾部，作者报告 MI355X 每步净省约 193 µs。9 月 21 日新增验证实际调用融合内核，并报告 1319 题、30-shot、C256 的 GSM8K：基线与候选 strict-match 都为 0.9735；文本并非逐字一致。这比只测试两份 Torch 参考实现更有价值，但仍不是 A100 整模型提速数据。[最新验证讨论](https://github.com/vllm-project/vllm/pull/57435)

移植必须覆盖 causal / DSpark noncausal、SWA-only、混合 prefill/decode、软件 FP8 和图回放；不得把批次相关的“已旋转行数”冻结进编译状态。该 PR 的早期修订曾在流畅输出、无运行错误的情况下显著降低质量，说明仅冒烟测试不足。本项目保留数值与任务质量验收，不能直接以页面最新 patch 替换已有大文件。

### 2. 元数据融合：适合小包，但收益应以 R1.3 为基线

`#57860` 合并 `_pack_global_topk_ragged_kernel` 与 indptr 前缀和。PR 说明里“CUDA 不承担这项开销”针对上游另一条 CUDA 实现；**我们的 Ampere 类继承了这里的实现**。本地 R1.3 的 `amd/rocm.py` 仍有 `torch.zeros` 加 `torch.cumsum`，所以确实存在可适配的工作。

作者在 MI355X 的组件测试中记录 4→2 graph nodes/call；其整服务吞吐变化约 −0.1%，没有建立吞吐提升。R1.3 已做跨层复用，不能继续用“每步 38 次构建”计算本项目潜在收益。应测剩余调用、空行、padding、不同 batch 和图内地址稳定性。[#57860](https://github.com/vllm-project/vllm/pull/57860)

已合并的 `#57885` 则直接将 SWA 有效性写入持久缓冲，并只在含 prefill 的批次计算压缩序列长度；本地固定源码仍保留对应冗余操作。它不需要新模型或新数学算法，适合与前项组成一个可独立开关、独立验收的小更新；作者的 GB200 测试未显示显著端到端收益。[#57885](https://github.com/vllm-project/vllm/pull/57885)

### 3. split-K 与长前缀 gather：保留机制，重做 SM80 分派

新 `#57880` 的 split 数选择与 single-split reduce 融合已有 MI355X 服务测试：C32 报告约 +3.44%。但代码明确以 `_ON_GFX950` 为门槛，改的是 gfx950 partial kernel；不会因文件被 Ampere 复用而自动启用。需要基于 A100 的 SM 数、寄存器和软件 FP8 解码重新选形状。[#57880](https://github.com/vllm-project/vllm/pull/57880/files)

`#56720` 的长前缀 gather 新证据也值得保留：MI355X、TP4、无 DSpark、C4、8 输出 token，131K 冷输入延迟约降 10.6%，热缓存约降 43.2%。这是长输入/缓存续接的结果。其新 one-pass 内核目前只分派到 ROCm，使用原生 float8 转换；本项目 SM80 需保留软件解码并移植网格选择，不能照搬，也不能称为持续 decode 加速。[#56720](https://github.com/vllm-project/vllm/pull/56720)

底层 split-K 思路也与 [AITER #5736](https://github.com/ROCm/aiter/pull/5736) 相关；该补丁的微基准在 gfx950 上，单纯限制一个 wave 曾造成回退，补上 KV tile 工作量下界才改善。因此下一轮应测试一组 A100 形状，而不是把 AMD 的固定 split 数写入部署默认值。

## Engram：RAM 容量与速度分别判断

跨 DP 共享主存表 `#57651` 已合并，上游通过 `dp_shared_memory` 在同节点保留一份表。对我们的 DP2，理论上表存储可由约 377.7 GiB 降到 188.8 GiB；还需承担进程、临时缓冲和加载峰值。**减少的是 RAM，并不会凭空给 GPU 增加 188.8 GiB KV。** 本机已有约 2 TiB RAM，共享表还会改变两个 NUMA 节点的访问局部性，所以不排在 decode 优化之前。移植还需要较早的共享存储基础实现，不能只加入这个参数。[#57651](https://github.com/vllm-project/vllm/pull/57651)

9 月 21 日新增 `#57893` 对小批查表采用两行 tile。DGX Station GB300 上 1/2/4-token 查表微基准分别约改善 6.1%/19.4%/28.1%，但 6 组平衡重启配对、1728 个服务请求的 decode 变化仅 **+0.11%，95% 区间 −1.55% 至 +1.80%**，没有证明整模型加速。它不能支持“因为放 RAM，所以速度掉一半”的归因，也不能排除 A100 的 PCIe/NUMA 开销；后者需要目标机 profiling。[#57893](https://github.com/vllm-project/vllm/pull/57893)

`#56926` 的串行查表流与 THP 仍开放，9 月 21 日继续修改内存分配代码。我们固定 backport 已在主 stream 上准备 Engram，起点不是上游两个后台 stream；目标内核 5.15 也不能使用 Linux 6.1 引入的 `MADV_COLLAPSE`。可另评估 best-effort `MADV_HUGEPAGE` 和 NUMA 放置，但不修改宿主设置、不把 THP 覆盖率当成既得条件。[#56926](https://github.com/vllm-project/vllm/pull/56926)、[madvise 定义](https://man7.org/linux/man-pages/man2/madvise.2.html)

## 看起来很快，但不能算成本项目新增收益的项

| 项目 | 源码/证据核对后的结论 |
|---|---|
| [#57770：DSpark split-vocabulary argmax](https://github.com/vllm-project/vllm/pull/57770) | 优化上游全词表 `torch.argmax`；我们的默认 `use_local_argmax_reduction=true`、`VLLM_DSPARK_FUSED_MARKOV=1` 已从融合分片路径提前返回，通常不执行这个改动点。其服务性能还使用 synthetic acceptance=6.0，不能拿来推算真实接受率下的收益 |
| [#56694：DSpark Markov W2 top-k](https://github.com/vllm-project/vllm/pull/56694) | 作者约 5–8% 的吞吐结果来自 Hopper 上 Qwen3-4B/8B 草稿，非 V4.1；本 backport 已有 top-k 分支，但默认使用融合分片链。需要比较两条采样路径及接受率，不能直接叠加 |
| [#57549：DSpark draft EPLB](https://github.com/vllm-project/vllm/pull/57549) | PR 明确不支持 V4.1 draft，因为主干与草稿 routed expert 数不同；保留本项目“主干 EPLB、草稿不启用”的判断 |
| [#57421：共享工作区](https://github.com/vllm-project/vllm/pull/57421) | 9.836 GiB 来自 B300/Qwen/Humming grouped MoE permutation scratch；我们的 Marlin 锁缓冲按 SM 数分配，且已有部分复用。需要单独统计，不能将该数字加入 KV 预算 |
| [#57603：mHC 小批重叠](https://github.com/vllm-project/vllm/pull/57603)、[#57643：all-reduce+mHC](https://github.com/vllm-project/vllm/pull/57643) | 已合并，但仍受 SM100/DeepGEMM 或 MNNVL 等条件限制；不是本项目 A100 已有 mHC Triton 融合的直接替代 |
| [#57977：DeepGEMM 之间保持 MXFP8 激活](https://github.com/vllm-project/vllm/pull/57977) | 方向是减少重复量化，当前实现依赖原生 MXFP8/DeepGEMM，不能绕过 A100 的硬件限制 |
| [#57872：encoder-only Mooncake P/D](https://github.com/vllm-project/vllm/pull/57872) | 新增了实际代码，但仍为 draft、GPU 验证待做，首版 text-only，P 仍加载全目标权重；不符合当前单节点八卡、完整多模态的小包替换要求 |
| [#57122](https://github.com/vllm-project/vllm/pull/57122)、[#57632](https://github.com/vllm-project/vllm/pull/57632) | 前者已关闭且未合并；后者仍为 draft。当前 Ampere builder 的 ragged 更新不全，不能因此直接开启图内 draft 元数据刷新 |
| [#57971：SM8x sparse MLA “64 倍冗余”](https://github.com/vllm-project/vllm/issues/57971) | 作者已撤回：测的是其自有 GLM 移植，不是上游路径，且原瓶颈归因有误。不能拿标题推导本项目 V4.1 的提速空间 |

## 其他运行时与可靠性进展

- `wtdcode/vllm-backport` HEAD 仍为 `4136231…`，Tokha233 仍为 `fae324a…`，shi3z custom 仍为 `580568e…`；相对上次固定点无新增 HEAD。backport 新 PR #95 是文档内容解析功能，不是 decode 加速。[backport 提交](https://github.com/wtdcode/vllm-backport/commits/main/)、[Tokha233 提交](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/commits/main/)、[custom 提交](https://github.com/shi3z/deepseekv4.1-A100-custom/commits/main/)
- Zanooda 新增 [0018 / `c704b45…`](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx/commit/c704b457c3285d3e87e3dd921c5e3aa23c0fe332)，移植 [#54442](https://github.com/vllm-project/vllm/pull/54442) 修正结构化输出的 permissive grammar 行不应接受草稿的问题。属于工具/schema 可靠性，值得针对本项目真实调度路径复现；本次没有据此认定用户已遇到该故障，也未改动协议服务。
- SGLang 官方 V4.1 页面仍使用预览交付、没有 A100 验证项；另有 [H20 SM90 MegaMoE #791](https://github.com/bytedance-iaas/sglang/pull/791)、[Humming #801](https://github.com/bytedance-iaas/sglang/pull/801)、[H100 InferenceX #3345](https://github.com/SemiAnalysisAI/InferenceX/pull/3345) 的新进展。这说明 Hopper 路径持续改进，不能当成 SM80 已可替换。SM90 ReduceScatter 实现还对 DSpark 等路径设有限制。[官方部署文档](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
- 搜索发现的 [Flash-Lite](https://github.com/atandra2000/deepseek-v41-flash-lite) 是约 1.16B 的缩小复现，改变模型；[single-A100 部署](https://github.com/ouroborosscr/deepseek-v41-flash-single-a100-deploy) 是 8K、CPU/存储分层、移除视觉与 DSpark 的 GGUF 路线，公开生成约 8–17 tok/s。两者都不满足本项目约束。

## 下一轮建议

1. **保留原模型与 k5，以 R1.3 作源码基线。** 本次后续已提供 R1.4 累计包，目标机可直接升级，不需要先安装 R1.3。取得固定截图历史样本的 TTFT、纯 decode、接受长度、缓存命中与并发基线。
2. **先移植元数据小改，再独立评估 inverse RoPE。** 优先 `#57885` / `#57860`，随后 `#57435`；保持独立开关和逐项 A/B，保留软件 FP8、图回放、长上下文和任务质量验收。
3. **视觉编码器图独立进版。** 按 `#56625` 的合并实现适配旧接口，连同最大图像 profiling 和图显存预算检查；测试 1/5/多图、缓存冷热、C1/C32 与混合文本/图片。单请求持续 decode 和并发期间的 decode 分开报告。
4. **Engram 和 split-K 作为第二批定向 profiling。** 先确认查表/NUMA、注意力、草稿和通信各自耗时；共享表优先回答主存成本，SM80 split-K/one-pass gather 需要新形状调优。

以上是本次候选全景与选择依据；已交付子集及验证范围以 [R1.4 更新说明](performance-update-20260922.md) 为准。目前没有依据承诺某项能把截图 decode 从约 50 tok/s 恢复到历史短文本的 110–140 tok/s，也不把不同 PR、不同硬件、不同基线的百分比相加。
