# 2026-10-15：R1.9 的路线选择与性能更新依据

本轮保留原始 **DeepSeek-V4.1-Flash** 权重与现有 SM80 运行栈，加入四项能落到当前调用链、可以单独验证的改进。没有找到在同样原权重、8×A100-SXM4-80GB、NVSwitch、多模态和本次并发要求下，已经证明更快的整套替代运行时。**这不等于其他项目没有进展，也不代表四项优化已有本机提速成绩。**

本版按用户最新要求固定：对外名称 **DeepSeek-Flash**，上下文 **512,000 token（500×1024，输入与输出合计）**，总调度并发 **64**，每个 DP pool 32，调度 token 预算 **1024**，TP4×DP2＋EP8，DSpark k=5，GPU 预算 0.92。原权重、Engram 主存卸载、前缀缓存、999 图配置上限、免客户端 API key、本机与 Docker bridge 访问方式保留。并发64是调度上限，不承诺64个请求同时驻留满500K；实际 KV 容量由两个 pool 的启动日志和目标机测试确定。

**用户最终要求调度与单图上限改为1024，覆盖先前512选择。** 处理器使用`hf_overrides={"vision_max_n_token":1020}`，为DSpark k5留4个新增槽，1020＋4＝1024；不保留512降采样设置。`disable_chunked_mm_input=true`及原模型`vision_prefix_lm=true`保留，整张双向图片段不拆分。与原1024图像预算非常接近，临界宽高比仍可能出现细微缩放差异。999是图片计数上限，原权重不变。

本记录接续 [10/14 调研](performance-review-20261014.md)。2026-10-15 重新读取各仓库默认分支 HEAD、固定提交的 README／相关源码，以及四项所选 PR。提交时间采用 UTC，短 SHA 仅用于阅读；完整 SHA 和来源见 [JSON 快照](../reports/performance-review-20261015.json)。网页的抓取日期不作为代码更新日期。

## 路线对照

| 路线及本次固定提交 | 已有支持／公开证据 | 本版判断 |
|---|---|---|
| [wtdcode/vllm-backport `29e66dad461b`](https://github.com/wtdcode/vllm-backport/commit/29e66dad461b11caae6039f510eaa7913f9dfb19)，9/26 | V4.1、SM80、原生 FP4／FP8 和 Engram；本项目原镜像基于较早固定 `a3507666` | **保留**已验证镜像及本项目累计 overlay；最新主干没有提供新的同口径 V4.1 A100 提速实测 |
| [Tokha233 turbo `fae324ae62ac`](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/commit/fae324ae62ac5cef31b7d38f5d369618e1cae1fa)，9/15 | A800 TP4×DP2＋EP8＋k5 的高并发配方；C128 聚合约3040 tok/s | 核心 dense／MoE／EP 优化已经纳入；不能把旧 A800 高并发数据写成本版新收益 |
| [Zanooda CMP170HX `28c5729c0c1e`](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx/commit/28c5729c0c1e00ff81b3245bb2f849c94f409aa2)，10/14 | 新增 PP8 decode burst spreading；同机暖 C8 聚合274→560 tok/s，另有 C16 约806 | 确有新进展；依赖 PP ring 多批次槽位，本项目 PP=1，**不直接移植该调度策略**。[测试口径](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx/blob/28c5729c0c1e00ff81b3245bb2f849c94f409aa2/RESULTS.md) |
| [shi3z custom `580568edd6f8`](https://github.com/shi3z/deepseekv4.1-A100-custom/commit/580568edd6f84537ff042ded4ee842b892bf3ff4)，9/20 | 已有排队与多槽 batched decode；四卡128K八槽约110–120聚合，1M cold prefill 约95.5分钟 | A100 原生实现有参考价值；须修正旧“完全串行”判断，仍无匹配本版规模的替代性能证据 |
| [concentrate1 fork `53c7595491a6`](https://github.com/concentrate1/deepseekv4.1-A100-custom/commit/53c7595491a643e21cc827def619ce5a571b9612)，9/27 | 改善批处理、多请求 DSpark、视觉、取消请求、图文前缀复用及协议解析 | **不能再描述为无视觉或无 batch**；换运行时仍需重新验收协议、缓存和八卡规模，未采用 |
| [LvLLM `43ffc42a54ef`](https://github.com/guqiong96/Lvllm/commit/43ffc42a54ef4aa736ee585acb525e6c6afbad0e)，9/22 | V4.1 SM80＋DSpark、CPU/GPU MoE 混合推理；有 SM8x tiled prefill 机制 | 通用机制纳入评估；公开主线主要解决小显存与混合架构，未证明胜过本项目 GPU 常驻计算权重 |
| [Lsglang `8350d80fb924`](https://github.com/guqiong96/Lsglang/commit/8350d80fb924136cecda471bf9cfb7351070020a)，10/15 | 当日仍在调3090 TP2专家池；SM80支持、NUMA混合推理、DSpark与可选NVMe Engram | 不能因CPU标签忽略；但CPU专家／NVMe路径解决的瓶颈与现有八卡不同，不替换 |
| [XTU `fdfdeef31a5f`](https://github.com/yeungtuzi/vllm-xtu-moe/commit/fdfdeef31a5ff50b85db6a707e0c1c11f91b78dc)，10/15 | v0.2.7 缩减CPU专家的GPU prefill DMA staging；TP2合计约省11.3GB；明确未改decode | 当前没有这类CPU专家整层staging，不能套用节省量或decode增益。[发行说明及勘误](https://github.com/yeungtuzi/vllm-xtu-moe/blob/fdfdeef31a5ff50b85db6a707e0c1c11f91b78dc/RELEASE_NOTES_v0.2.7.md) |
| [antirez/ds4 `fc80bd695da7`](https://github.com/antirez/ds4/commit/fc80bd695da76ee14bcb86820bf900a5cdfda806)，10/7 | GGUF、SSD Engram、Spark流式专家和多会话；V4.1 CUDA文档仍明确不支持视觉与DSpark | 更换权重格式与推理栈，不能保留本版原权重＋CUDA多模态＋k5契约；不采用。[固定文档](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/docs/DGX_SPARK.md) |
| [Entrpi/ds4-on-spark `99fb7b170528`](https://github.com/Entrpi/ds4-on-spark/commit/99fb7b1705281a9767e18176c889d0b7dbbf6a8b)，8/27 | GB10/SM121 的 V4 Flash kernel、continuous batching、持久化KV与DSpark | 借鉴图捕获／缓存机制；不能把旧V4、GGUF、Blackwell速度当成V4.1／A100支持 |
| [tonyd2wild V4.1 Spark `d45538f67366`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/commit/d45538f67366c62da668c35ec1afd4fcd0e4c8b4)，9/19 | 单Spark有限统一内存与Engram存储路线 | 关注通用IO与缓存机制；GB10内存体系及容量约束不等同于A100＋主存 |
| [catid 双GB300 Station `ff8a496e5e02`](https://github.com/catid/dgx_station_benchmarks/commit/ff8a496e5e027bbc462f81643361ef5516072a68)，9/11 | 原权重、PP2/TP2、DSpark、双400GbE；C1最高252.9、C64聚合3401.6 tok/s | 方法与区分单流／聚合的口径有价值；**DGX Station GB300不是DGX Spark，也不是A100**，性能不外推。[原始条件](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/README.md) |
| [上游vLLM `7020600b1db3`](https://github.com/vllm-project/vllm/commit/7020600b1db32f239611b037f07e0dc5268e5b4f)，10/15 | 正式版仍v0.31.0；V4.1优化持续增加 | 固定源码FlashMLA只列SM90/100，FlashInfer列SM100/120；**尚非SM80直接替换镜像**。[后端能力检查](https://github.com/vllm-project/vllm/blob/7020600b1db32f239611b037f07e0dc5268e5b4f/vllm/models/deepseek_v41/sparse_mla.py#L124)、[另一后端](https://github.com/vllm-project/vllm/blob/7020600b1db32f239611b037f07e0dc5268e5b4f/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py#L137) |
| [上游SGLang `f7ad4345b86d`](https://github.com/sgl-project/sglang/commit/f7ad4345b86d147a1c3095a3cebabca9a33ebaa1)，10/15 | 官方V4.1配方验证H200/B200/B300/GB300/MI355X，NVIDIA仍用预览镜像 | 官方矩阵没有A100；社区Lsglang的SM80适配与上游承诺分开判断。[固定部署文档](https://github.com/sgl-project/sglang/blob/f7ad4345b86d147a1c3095a3cebabca9a33ebaa1/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) |

## 对 shi3z 并发能力的更正

不能再根据 `serve.py`／`engine.py` 顶部历史单锁说明推断整个服务串行。固定提交中的 [`Engine.generate_text`](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/dsv41/engine.py#L4386) 在 `max_seqs>1` 时进入 batch queue；[`_batch_worker_loop`](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/dsv41/engine.py#L4411) 维护活动槽、接纳新请求，并将多个序列合并给 `rt.step`。这确实是可补入活动 decode batch 的调度实现。

限制也要按代码写：prefill仍在同一worker循环中串行运行；Chat SSE调用 [`generate_text`](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/dsv41/serve.py#L263) 完成后再发送生成内容，heartbeat不等于实时token流。[128K八槽脚本](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/run_128k_8slots.sh) 使用 `--mtp 0`；因此这组110–120聚合不能与k5吞吐混称。README的1M预填是四卡PCIe、1,000,000实际token、5729.20秒，并非1M服务上限下的一小段短输入，也不是10月新增提交。本版未复测其数值。

## 本版纳入的四项

1. **Prefill去同步，参考 [#60091](https://github.com/vllm-project/vllm/pull/60091)，头提交 `82c6eaf85e22`，仍开放。** 当前Ampere确实继承Torch `combine_topk_swa_indices`；两次repeat使用已知 `output_size=num_tokens`，保留索引数学与混合batch非零基址。上游MI350X的TTFT −5.7%、C64聚合＋2.7%仅作为选型证据；本版12个CPU精确参考用例通过，不把其百分比外推到1024预算的A100。
2. **DSpark context KV纳入draft CUDA Graph，参考 [#57632](https://github.com/vllm-project/vllm/pull/57632)，`f2b00d1f989e`，9/24合并。** 保留每请求k+1个真实verify上下文行、PAD slot、多个KV group以及超出图容量的eager尾段。上游GB200 C1 TPOT约−15.2%，使用合成接受长度；本版仅据此采用机制，不宣称A100同等收益。
3. **Inverse RoPE融合进decode reduce，参考 [#57435](https://github.com/vllm-project/vllm/pull/57435)，`b1465b142c4c`，9/22合并。** 沿现有SM80 Triton路径移植；由eager attention处理未融合后缀，避免compiled `_o_proj` 冻结批次拆分。非split fallback仍执行独立旋转，NoPE保持不变；保留单独关闭融合的诊断开关。SM80 cubin编译和CPU解释器已通过，真实GPU数值／图回放仍须目标验收。
4. **按真实chunk界限缩减indexer reservation，参考 [#60874](https://github.com/vllm-project/vllm/pull/60874)，`fe1a30812d34`，仍开放。** 使用本镜像真实chunk planner检验，不照搬新主干的压缩比假设。只移植indexer容量机制，不改变Ampere attention四请求gather分组或软件FP8。H20每卡省3.70GiB的结果包含另一路FlashMLA改动，不能算成本项目节省量。

本版512,000 token上限、512MiB logits预算下，indexer行数上界为 ratio1=8,289,721、ratio2=5,861,718；旧行数分别为20,480,000与10,240,000。共享workspace还有Ampere gather等reservation，实际可归还KV的字节不能简单相加。1,440组真实chunk planner控制流／2,108,452个chunk的整数边界检查通过，这不是GPU分配或满窗口容量实测。

## 继续观察、未默认纳入的项

- **长KV gather [#56720](https://github.com/vllm-project/vllm/pull/56720)**：新单遍dequant分派在ROCm侧；SM80必须保留软件FP8并重新调grid。AMD核速度及热缓存TTFT不能证明本机持续decode收益，本版未盲替换。
- **候选有效长度 [#61039](https://github.com/vllm-project/vllm/pull/61039)**、并行Top-K/SWA [#60936](https://github.com/vllm-project/vllm/pull/60936)：机制通用，但依赖新DeepSelect／DeepGEMM或不同公共helper；原镜像没有相应ABI。已有candidate-only消费端优化继续保留，未假定新selector已启用。
- **Decoder bounded replay [#59039](https://github.com/vllm-project/vllm/pull/59039)**：已合并不等于当前可用；其sequence-parallel限制与本项目TP4×DP2＋EP8冲突。较大的prefill百分比不用于解释decode。
- **Engram packed DP exchange [#59923](https://github.com/vllm-project/vllm/pull/59923)**、排序查表 [#59327](https://github.com/vllm-project/vllm/pull/59327)：快路径分别受TP1／直接MXFP8和Blackwell阈值限制；不据此改变本版TP4或原权重。主存Engram是否为本机主瓶颈需要profile，不能仅凭“在RAM”归因。
- **视觉encoder图 [#56625](https://github.com/vllm-project/vllm/pull/56625)／[#58499](https://github.com/vllm-project/vllm/pull/58499)**：需要旧runner、模型encoder元数据及padding接口的完整移植和多图GPU回归；本版保留已验收视觉路径。主干／draft图开启不代表视觉encoder图已移植。
- **AMD mono kernel、Blackwell融合、普通MHA的SM8x NVFP4 KV**：硬件指令、权重展开预算或KV格式不匹配；分别不能作为A100可开参数、80GB可装实现或V4.1压缩MLA的新KV后端。
- **进一步把context combine／anchor放入draft图 [#59511](https://github.com/vllm-project/vllm/pull/59511)**、MXFP4 Engram表 [#61530](https://github.com/vllm-project/vllm/pull/61530)：前者继续独立观察，后者需要不同checkpoint。它们不与本次四项百分比叠加。

这些排除是针对当前八卡、原权重和原镜像交付范围；不是把其他硬件上的底层思路排除出研究。后续每项移植都应独立增加正确性和性能证据。

## 验证范围与性能口径

构建侧使用原镜像 `deepseek-v4.1-flash-a100:20260913-r1`，只执行CPU参考、真实Triton CPU解释器、来源完整性和显式 `sm_80` 编译。新增inverse RoPE的34个编译变体、6组可变输入解释器、6组混合attention分派、2组split／fallback wrapper，以及6个draft graph控制流测试通过；benchmark计数／前端恢复10项通过。报告位于 `reports/r19-runtime-build/`。**没有GPU推理、GSM8K、500K满窗口、64并发或999图实测通过的声明。**

目标机门禁继续检验BF16结果、NoPE不变、strided原地fallback及CUDA Graph输入变更回放；完整服务需再测文本／多图、缓存命中、三种协议和上下文。本版测速增加隔离前端流量与计数污染标记，支持C1样本以及所选C32或C64负载；污染时保留观测样本但不作为签字成绩。SSE事件数不能当token数，单流decode、HTTP端到端速度、聚合输出和TTFT分别报告。

独立审阅还发现原observer的限制：`memory_estimation_probe` 依据当前manager的 `_capture_mem_samples`；profiling生命周期只给target manager设置该字段，draft的临时capture可能也被标为非probe。因此，**单独看到capture记录不足以验收**；必须结合同manager的实际FULL replay、成功请求及CUDA错误检查。本版保留完整capture／replay门禁，不在临发版时扩散修改profiling生命周期。构建侧也直接读取原镜像manager接口确认 `capture(create_forward_fn, progress_bar_desc)`／`run_fullgraph(desc)` 及现有observer hook，未把CPU控制流测试当作CUDA回放实测。

旧TP8／256K的C32 off聚合1379.01 tok/s、高思考1039.91 tok/s，以及R1.8八图单流中位128.02 tok/s仍是历史记录。它们不是本版500K／C64成绩；更改上下文、调度预算或并发后，需要新的同负载测试才能判断总收益。

最终调度审查进一步确认DSpark k5需预留4个新增槽，故实际HF处理器上限为1020，加4等于1024；用户确认的单图1024是上限。图片缓存命中、重复图与图起点预算不足也必须整段推迟，不能拆分双向图片段。
