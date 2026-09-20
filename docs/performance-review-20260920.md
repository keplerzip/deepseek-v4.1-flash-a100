# 2026-09-20：DeepSeek-V4.1-Flash / A100 方案复核

有新增的底层优化值得移植，但本次没有找到同时满足本项目全部约束、且已用可比实测证明优于现有候选方案的另一套 A100 部署栈。推荐继续使用本项目的 vLLM-backport 路线，先取得 R1.2 的目标机基线，再逐项评估 DSpark 投影合并、Ampere 实际调用的索引复用、视觉编码器 CUDA Graph。

本文保存进版前的调研快照。当时只更新研究文档与来源记录；后续已按用户指示将第一批两项改动纳入 [R1.3 累计更新](performance-update-20260920.md)。以下候选状态与推荐顺序按调研时点保留，不是要求安装 R1.3 前先安装 R1.2。

## 比较基线

用户于 2026-09-20 确认：**目标机尚未升级 R1.2**。不能把当前体验归因于 R1.2 的 TP4×DP2＋EP8，也不能把 R1.2 的构建检查当成实机提速证明。

| 已有记录 | 单流 decode tok/s | C32 聚合 tok/s | C32 P95 秒 | 证据范围 |
|---|---:|---:|---:|---|
| 历史 TP8，k5，off | 139.80 | 1379.01 | 24.56 | 用户回传；短文本，1024 输出 token |
| 历史 TP8，k5，high | 109.95 | 1039.91 | 34.93 | 用户回传；包含思考 token |
| 图片或历史截图对话 | 约 50 | 未提供 | 未提供 | 用户观察；未固定输入、缓存、输出长度与计时口径 |
| R1.2，TP4×DP2＋EP8，k5 | 待实测 | 待实测 | 待实测 | 已有累计更新包；目标机未安装 |

历史原始逐请求 JSONL 未回传；现有数字来自 [性能报告](performance.md) 和 `reports/*.user-reported.json`。R1.2 的内存核算、更新与回退方式见 [R1.2 文档](performance-update-20260918.md)。

固定约束继续是原版 **DeepSeek-V4.1-Flash**、原 ModelScope revision、8×A100-SXM4-80GB＋NVSwitch、约 2 TiB RAM、256K、k5、多模态和三协议。并发按实际各池满窗口数量计算，最低 32；TP4×DP2＋EP8 不能理解为四卡各放一份完整 552B 主干。目标机完全离线，只使用普通用户及 `sudo -n docker`，优先复用原镜像的小更新包。本次未查看或修改 NewAPI。

## 整套运行时的最新状态

| 路线 | 本次查到的状态 | 判断 |
|---|---|---|
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport/commits/main/) | HEAD `4136231c0cf2b3b532db2e0e2af45913ab6d3a1e`，9 月 18 日最新提交更新 Qwen 配置；其前面的 DSML、8-warp、共享专家改动已纳入 R1.2 | 没有发现晚于本项目已选补丁的新增 V4.1 性能提交；不能仅换最新 tag 就承诺提速 |
| [Tokha233 Ampere Turbo](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/tree/fae324ae62ac5cef31b7d38f5d369618e1cae1fa) | HEAD 仍是 9 月 15 日的 `fae324ae…`；实测硬件为 A800，默认纯语言模型 | R1.2 已改编其主要并行、dense、通信优化，并保留本项目视觉路径；没有新的独立加速方案 |
| [shi3z A100 custom](https://github.com/shi3z/deepseekv4.1-A100-custom/tree/580568edd6f84537ff042ded4ee842b892bf3ff4) | 新增批处理调度、前缀复用及视觉模块；README 与服务文件头仍留有过时的单请求描述 | 应更新早期“只能串行”的判断，但尚不适合直接替换本项目 |
| [SGLang V4.1](https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1) | 验证列表仍以 H200、B200、B300、GB300、MI350X 为主，使用 V4.1 预览栈 | 未发现符合本项目约束的 A100 八卡完整验收；不把其他硬件支持推成 A100 性能保证 |

custom 项目的最新公开结果来自 **4×A100 PCIe**：k5 单流报告 84.7–93.0 tok/s；并发表中 C32 是 16K 配置，159.8 tok/s 端到端聚合、376.5 tok/s 峰值引擎 decode；475.6 tok/s 则是 **C64、8K** 的峰值。它们不能直接与我们的八卡、256K、high 或 C32 历史成绩比较。Jev 的上万 tok/s 是固定字段候选打分后的“有效提取吞吐”，不是通用自回归生成速度。[原始报告](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/README.md)

此外，该项目当前启动脚本优先选择 Abliterated 权重，找不到才回退原模型；服务入口还把设备列表过滤到 0–3，模型列表暴露多个别名。虽可改源码显式使用原权重，但最新默认交付与本项目的八卡、唯一模型名及协议约束有明显差异。新增视觉模块也不等于三协议多图链路已经验收。[启动脚本](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/run_server_batched.sh)、[服务源码](https://github.com/shi3z/deepseekv4.1-A100-custom/blob/580568edd6f84537ff042ded4ee842b892bf3ff4/dsv41/serve.py)

## 值得移植的底层优化

以下优先级依据实际调用路径、适用工作负载和离线移植成本，不依据 PR 标题里的 GPU 品牌。公开性能数据均是作者报告，本项目未在目标 A100 上验证。

| 优先级 | 改动及来源 | 公开证据 | 本项目适用性 |
|---|---|---|---|
| 第一批 | [#57627：合并 V4.1 DSpark context KV 投影](https://github.com/vllm-project/vllm/pull/57627) | GB200，M=6 时投影 GPU 时间 25.1→9.2 µs，CPU 提交 608→208 µs；仅投影微基准 | 原代码确实每个草稿层各做一次 Q＋KV 投影，再丢弃 Q。可合并成一次 KV-only 投影，保留 k5；需适配 SM80 Marlin 权重打包、scale 与 R1.2 dense 分派 |
| 第一批 | [#57434：复用 ragged top-k 元数据](https://github.com/vllm-project/vllm/pull/57434) | MI355X，38 次重复构建→8 次，相关内核每步约省 250 µs | **命中我们的 Ampere 路径**；Python 层复用现有 Triton 结果，有希望沿用原镜像。R1.2 尚未包含 |
| 截图场景优先评估 | [#56625：视觉编码器 CUDA Graph](https://github.com/vllm-project/vllm/pull/56625) | 4×GB200、32K、C32、每请求 32 输出 token：请求吞吐 4.00→10.37 req/s；平均 TTFT 4467→1894 ms | 原镜像已有编码器图管理接口，但 V4.1 模型未实现对应协议。需要移植打包图像的 ViT/aligner 路径；FLASH_ATTN、图预算、多图与 DP 均须验证 |
| 第二批 | [#57435：inverse RoPE 融合到 sparse decode reduce](https://github.com/vllm-project/vllm/pull/57435) | MI355X，每步约省 193 µs、去掉 43 次独立旋转调用 | 同样命中 Ampere 复用的 Triton 路径，但改变舍入位置，且旧修订曾因图编译状态冻结产生静默质量退化；需要数值、图回放和模型质量验收 |
| 第二批 | [#57122：SWA 有效性融合](https://github.com/vllm-project/vllm/pull/57122)；[#57632：草稿图内刷新元数据](https://github.com/vllm-project/vllm/pull/57632) | GB200，减少逐组 Python 调度/内核提交；后者仍是 draft PR | 思路可移植，但 A100 还维护 ragged indices/indptr。必须把它们也纳入图内刷新，不能只套上游通用 SWA 实现 |

截至本次查询，上述 PR 均未合并。没有足够证据将这些微秒级节省相加为服务 TPS 提升，更不能把“投影快 2.7 倍”说成“模型快 2.7 倍”。

`#57434` 与 `#57435` 的作者还公布了合用后的整步对比：MI355X 上 C1 为 45.57→46.15 ms/step，C32 为 44.65→44.41 ms/step，样本很少，尚未分辨出可靠的端到端收益。因此它们属于减少冗余工作的候选，不是大幅提速的证据。[整步数据与限制](https://github.com/vllm-project/vllm/pull/57434)

视觉编码器图的 2.60 倍是上述短输出、多请求图像负载的结果。它可以减少编码过程对其他请求 decode 的干扰；若只有一条长历史对话，且图片编码已经复用，就不能据此期待纯 decode 同倍提速。该补丁显式拒绝 TORCH_SDPA 和 FLASHINFER 视觉后端，不能只给当前命令加一个开关。[实现条件](https://github.com/vllm-project/vllm/pull/56625/files)

### 为什么 ROCm 补丁也进入 A100 评估

本地固定源码的调用链是：

```text
nvidia/model.py::_select_dsv4_attn_cls，SM8x
  → ampere/ampere_sparse.py::DeepseekV41AmpereMLAAttention
  → amd/rocm.py::DeepseekV41ROCMAiterMLAAttention
  → rocm_aiter_mla_sparse.py 中可在 CUDA 执行的 Triton 算子
```

这条路径在 `amd/rocm.py::_forward_decode` 逐层调用 `compute_global_topk_ragged_indices_and_indptr`，正是 `#57434` 的改动点。其与 `#57659` 的目标相似，但 [#57659](https://github.com/vllm-project/vllm/pull/57659) 修改的是 NVIDIA FlashMLA/FlashInfer 的另一种索引格式；单独套它不会优化当前 SM80 路径。两者不能作为两份独立加速收益相加。

移植时还要验证同一索引来源的 block table、compress ratio、有效 token 映射相同；检查 batch 缩小/扩大、混合 prefill/decode、两个 DP 池和 target/draft 图回放时不复用旧步的数据。

`#57632` 的前提也有同样差异：本地 ROCm/Ampere metadata builder 会额外生成 ragged SWA，并明确关闭未覆盖这些缓冲区的 draft metadata 更新。后续适配必须覆盖完整缓冲区，不能继承一个只刷新 dense SWA 的闭包。

## Engram 放 RAM 是否是主要瓶颈

**仍不能根据现有证据把约 50 tok/s 归因于 RAM 卸载。** 既不能认定没有开销，也不能认为把表搬回显存就会恢复旧 V4 的速度。

[vLLM #56926](https://github.com/vllm-project/vllm/pull/56926) 于 9 月 20 日更新了 Engram 串行预取与大页内存结果。在 GB200 上，1 token 查表是 2.69→2.68 µs，32 token 约 1.02 倍；大输入查表才有明显收益。其 8K/C32 的整服务平均 TTFT 改善是约 2.84%，32K/C32 约 1.10%。这些是其他平台结果，尤其不能把 32K 查表内核的 10 倍收益用于预测 A100 单流 decode。

本项目原始源码和 R1.2 的 `nvidia/model.py` 都在进入解码层前调用 `Engram.prepare_embeddings`；`common/engram.py` 明确在 **main stream** 上准备数据。上游 PR 修的是“两条后台流互相干扰”的实现，和我们的起点不同。若移植，必须重新设计并验证预取事件、缓冲区寿命和占用，不能声称只是修复了一个已经确认存在的双流问题。

THP 路线还受宿主约束：目标内核为 5.15.0-94；`MADV_COLLAPSE` 从 Linux 6.1 起提供，不能假设目标可用。`MADV_HUGEPAGE` 仍可评估，但是否得到大页取决于现有策略与内存状态；本项目不要求修改宿主 sysctl 或升级内核。[Linux 手册](https://man7.org/linux/man-pages/man2/madvise.2.html)、[内核 THP 文档](https://docs.kernel.org/admin-guide/mm/transhuge.html)

[vLLM #57651](https://github.com/vllm-project/vllm/pull/57651) 默认启用同机 DP 共享 Engram，依赖先前的共享存储实现。在 R1.2 的 DP2 情况下，理论上可把主存表从约 377.66 GiB 降到一份 188.83 GiB，但这是容量收益，没有已证实的 A100 decode 提升。你的双 NUMA、约 2 TiB 主存下，一份共享表的远端访问与两份靠近各 GPU 组的表，应以实测比较；当前 R1.2 也没有承诺 NUMA 本地放置。

该共享路径与 `#56926` 的私有表 THP 路径不能直接叠加成一套已有实现。另一个 [#57195 SP token-owner 共享查表方案](https://github.com/vllm-project/vllm/pull/57195) 当前显式排除 speculative decoding 和 DP 存储模式，也不符合我们的 k5＋DP2 固定方案。[#57491](https://github.com/vllm-project/vllm/pull/57491) 则是让 ROCm 启用已有卸载能力；我们的 CUDA 部署早已卸载，并非新增收益。

## 长上下文、缓存和其他通用方法

9 月 18 日的 [V4.1 技术报告](https://arxiv.org/html/2609.19969v1) 解释了 CED 下的 encoder/decoder SWA bounded replay：复用全局 KV，并只重放有限窗口，可显著减少 prefill 或缓存恢复工作。但有限窗口重放包含近似，不是通用、无条件的逐 token 等价替换。报告中官方服务的 72 小时全局缓存策略也不是本地 vLLM 的 TTL 承诺，不能用来解释此前“空闲后 cached_tokens=0”。

新出现的 [#57738 encoder-only P/D RFC](https://github.com/vllm-project/vllm/issues/57738) 仍是设计讨论；需要完整 KV 与 DSpark 状态交接。对单机八卡离线服务，它还没有可直接替换的实现与同口径性能报告。本次把 bounded replay 和更持久的 prefix/KV 管理列为中长期 prefill/恢复方向，不作为已完成的 decode 优化。

其他近期方案也检查了适用条件：

| 方案 | 未选为当前更新的原因 |
|---|---|
| [#57643 TP all-reduce＋mHC 融合](https://github.com/vllm-project/vllm/pull/57643) | 使用 SM100/MNNVL 路径，当前还不覆盖 DSpark；需要新的适用内核，不能复用 headline 收益 |
| [#57711 Mega Attention split-KV](https://github.com/vllm-project/vllm/pull/57711) | B200、NVFP4 KV 与新 FlashMLA 扩展；本项目是 SM80 软件 FP8 稀疏路径。split-K 思路已经在本项目使用 |
| [#57673 few-head sparse prefill](https://github.com/vllm-project/vllm/pull/57673) | 针对 SM90 的 head padding 问题；不是 A100 当前算子的直接替代，也不优化 decode |
| [#57428 GEMM＋reduce-scatter](https://github.com/vllm-project/vllm/pull/57428) | SM100 CuTeDSL MXFP8 内核，需单独开发 SM80 对应实现 |
| [#57081 ROCm FP8 o-projection](https://github.com/vllm-project/vllm/pull/57081) | 依赖 MI355X/AITER 原生 FP8；A100 不具备该计算路径 |
| [#55184 pre-SM90 sparse MLA omnibus](https://github.com/vllm-project/vllm/pull/55184) | 主要为旧 V4 的可运行性补齐；V4.1 已有独立 Ampere 后端，需逐算子去重审查，不能当作 V4.1 新整栈升级 |
| [#57184 DBO 不均衡图回放](https://github.com/vllm-project/vllm/pull/57184) | 原型当前验证范围为 DeepSeek-V2-Lite、特定通信栈，并排除 speculation/EPLB；不符合本项目 |
| [#57628 输入准备融合](https://github.com/vllm-project/vllm/pull/57628) | 截至查询已关闭、未合并；作者注明只省约 30 µs/约 20 ms 的一轮，不是主要提速途径 |

## 推荐顺序与验收依据

1. **先对比当前目标服务与现有 R1.2 累计包。** R1.2 的 TP4×DP2＋EP8、dense BF16 分派、通信与 EPLB，是比上述单个微优化更大的一组变化，但是否更快仍需实机证据。保留原图像输入和 256K，不能用纯文本 Turbo 成绩代替。
2. **后续第一批只移植 #57627 与 Ampere 对应的 #57434。** 保留 k5、原权重；预计可通过源码覆盖复用原镜像，但需先验证加载器和图回放，尚未构建新包。
3. **针对截图负载单独评估 #56625。** 确认原镜像的 FLASH_ATTN 支持、捕获实际图像尺寸预算；记录增加的图内存并重新标定 KV。不能因开启新图而沿用旧 C56。
4. **再评估 RoPE 融合与草稿 metadata 调度。** 后者要包含 Ampere 特有 ragged 缓冲区；前者必须覆盖 mixed prefill/decode 和完整模型质量，不能只做 kernel 单测。
5. **Engram THP、共享存储、GPU 热行缓存与 NUMA 放置以测得的瓶颈排序。** 首先分离 CPU 提交间隙、Engram lookup/交换、attention、MoE 和 DSpark 验收率。GPU 热行缓存需要命中率和显存预算证据，本次未找到可直接交付且通过本项目约束验证的实现。

实际对照至少包含：纯文本/真实截图历史，off/high，C1/C32，相同输入和输出长度，冷/热 prefix cache。分别报告 TTFT、token-bearing SSE 的 decode 速度、端到端速度、总吞吐、P95、DSpark 接受率和两池负载分布。已有 `tests/performance-benchmark.sh` 支持图片和脱敏请求文件；A→B→A 对照和独立会话样本用于减少缓存状态、负载不均衡与固定测试顺序的影响。

研究来源的 commit、PR 状态及证据摘要保存在 [机器可读记录](../reports/performance-review-20260920.json)。来源快照下载到工作区忽略目录 `research/sources/speed-review-20260920/`，未将外部仓库的大量源码加入交付。感谢 DeepSeek、vLLM、wtdcode、Tokha233、shi3z 及相关 PR 作者提供可检查的实现和测试条件。
