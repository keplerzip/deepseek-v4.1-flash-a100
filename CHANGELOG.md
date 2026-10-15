# R1.9 / v1.9.0 — 2026-10-15

- 固定DeepSeek-Flash、512000上下文、每池1024调度token、全局C64/每池32，k5和GPU0.92。
- 移植prefill去同步、inverse RoPE融合、DSpark context KV图捕获、indexer工作区上界；复用原权重和SM80镜像。
- 保留图片双向注意力/整图段，单图视觉token上限1024（用户确认大图细节取舍）；原镜像CLI/config、CPU数学/图控制与SM80编译检查；不宣称目标机实测速率。
- 测速支持隔离本项目入口，污染流量保留结果并标记不可签收；完整无权重包与累计小包、事务回退。
- 完整选型及局限见docs/deployment-update-20261015-r19.md与docs/performance-review-20261015.md。

# Changelog

## 1.8.0 / R1.8 — 2026-09-23

- 隔离目标机的 R1.7 八卡 EP8、整模、双池缓存与 DSpark smoke 已由操作者报告通过；最后前端以 UID 65534 读取私有 `/frontend.py` 失败并自动回滚。前端改用部署用户 UID/GID，增加加载 GPU 前的挂载可读预检。
- 唯一公开 API 模型名改为 `DeepSeek-V4-Flash`；底层 DeepSeek-V4.1-Flash 权重、ModelScope revision、镜像、项目目录均保持原样。离线 guard、三协议请求、模型列表与文档同步。
- 全局调度并发硬上限固定 C32、每 DP pool 16；继续检查实际 KV 容量、1M 总上下文、0.92 显存比例、prefix caching、DSpark k=5。按操作者要求，本次跳过启动前 EP8 通信重测；完整模型通信与双池 smoke 仍执行。
- 累计小包复用初始权重与镜像，可从 R1 至 R1.7 直接升级。构建机无 GPU；R1.8 整机性能待隔离目标机实测。

## 1.7.0 / R1.7 — 2026-09-23

- 修正 R1.6 实机 EP8 验收中的第二处 dtype 错误：变长 reduce-scatter 输入原为默认 int64，期望值为 float32；现在两者明确为 float32，严格数值断言不放宽。
- 审核整个 EP8 测试：等长 BF16 自定义通信、四种形状、CUDA Graph 修改输入重放、变长 float32 NCCL 回退、每 rank 形状和实际通信路径。新增 CPU 参考测试验证八 rank 数学与 dtype 契约，并将 EP8 脚本哈希纳入启动门禁签名，修订后强制重验。
- 原始权重、镜像、TP4×DP2＋EP8、1M/0.92、DSpark k5 和 R1.4 算子覆盖不变。构建机无法执行 GPU 数值测试；完整目标机验收仍待实机。

## 1.6.0 / R1.6 — 2026-09-23

- 修正八卡 EP8 验收中变长 float32 回退分支的期望张量 dtype。R1.5.1 目标机回传显示八个 rank 因 `float32 != int64` 在同一测试断言处失败并自动回滚；尚无修正后的目标 GPU 通过结果。
- 保留原始权重和镜像、TP4×DP2＋EP8、1M/0.92、DSpark k5、视觉及三协议。提供从 R1 至 R1.5.1 可直接安装的累计小包；不宣称速度提升。

## 1.5.1 / R1.5.1 — 2026-09-22

- 将 R1.5 默认显存预算 0.90 修正并固定为 0.92，运行参数拒绝其他值；调优工具只调整 batch token 预算。
- 补齐单一模型名、1M、自动并发至少 C32、999 图、免密、loopback/Docker bridge 的固定配置清单；原有模型、镜像、CUDA Graph 与访问实现保留。
- 累计安装迁移旧调优配置，可从初始版至 R1.5 直接升级；目标 GPU 验收仍待执行。

## 1.5.0 / R1.5 — 2026-09-22

- 固定原权重原生 1M（1048576）输入输出总上下文，沿用 TP4×DP2＋EP8、DSpark k5 和 R1.4 完整 decode CUDA Graph / 性能覆盖。
- 两池分别取满 1M 窗口数，`C=max(32,2*sum(floor(KV_DP_i/1048576)))`；不复用旧 256K 容量种子，迁移并备份旧调优配置。
- 两池长输入验收增加 512K 和近 1M，核对 tokenizer 与实际 usage，性能报告标明新上下文；历史 256K 成绩保留。
- 累计小包可从 R1 至 R1.4 直接升级，仍复用初始镜像与权重；构建侧检查不替代目标机 1M 容量、速度或质量验收。

## R1.4 / v1.4.0 — 2026-09-22

- 适配 vLLM #57885：SWA 有效掩码直接写入持久缓冲，压缩序列长度只在含 prefill 的批次计算。
- 适配 #57860：A100 继承路径的 ragged indptr/pack 融合，限 1–1024 行，保留大批原扫描与空输入处理。
- 新增独立 `PERF_DECODE_METADATA` 开关，关闭只恢复 R1.3 元数据实现；累计保留 TP4×DP2＋EP8、DSpark k5、256K、多模态及此前优化。
- 新增 CPU 生产片段/解释执行、SM80 编译与目标 CUDA 图回放检查，累计安装及失败回退覆盖至 R1.3。复用初始镜像和原权重，提供累计小包。
- README 继续以历史实测性能开头；新增来源、适用边界及未合入项说明。本版为待目标 GPU 验收的预发布版，未宣称 A100 新速度。

## R1.3 / v1.3.0 — 2026-09-20

- DSpark k=5 固定；三层 context Wkv 合并投影，原生 MXFP8 weight/scale 分片加载与缺失校验。
- 在 A100 实际继承的 Triton attention 路径中复用当步 ragged top-k，检查缓冲区身份和布局，图回放仍更新索引。
- 累计包含 R1.2 TP4×DP2＋EP8 与此前修正，从初始镜像直接安装源码小包；保留视觉、256K、三协议和 Engram 主存卸载。
- 新增 CPU 加载/缓存生命周期、SM80 AOT 与目标 GPU 数值/图回放检查，升级事务和失败自动回退覆盖 R1/R1.1/R1.2。
- 构建机无 GPU，发布为待目标八卡验收的预发布版；保留历史性能报告，不声称新版加速已实测。

## R1.2 / v1.2.0 — 2026-09-18

- 同一 DeepSeek-V4.1-Flash 权重，TP4×DP2＋EP8、DSpark5、256K、视觉和三协议。
- dense BF16 ≥32 行、custom EP8 AG/RS、8-warp sparse decode、符合条件的共享专家重叠；主干 EPLB 与草稿隔离。
- DP attention padding 修正、逐池 KV 标定、每池四个 worker 的 graph/replay 验证、会话缓存亲和路由。
- 修复 DSML 拼写兼容；更新和回退覆盖旧 TP8 调优文件。
- 累计包复用初始镜像；历史报告保留。A100 GPU 数值、容量、速度与稳定性待目标机验证，不宣称提速百分比。


## 1.1.0 / R1.1 — 2026-09-16

- Deliver cumulative source update `20260916-perf2`, installable over the initial R1 offline runtime or intermediate patches, without transferring weights or a new container image.
- Adapt compact candidate-only indexer scoring from Zanooda@2445d7d to the pinned TP query-sharded SM80 path. Keep source-layer/full scoring and short-width fallbacks; exclude DCP/PCP and FP4 KV paths.
- Adapt stable MoE alignment from Tokha233@fae324a without disabling deterministic routing order. Include reference, CUDA graph and dispatch checks; target GPU correctness and speed remain pending.
- Add release/update metadata, independent diagnostic switches, timeout cleanup, and transactional backup/rollback. Preserve the original model, k=5, Engram RAM, context, API identity and local-only access.

- Add `20260915-perf1`: unset forced NCCL Ring/Simple, port V4.1 delayed mHC fusion and draft auxiliary-state reuse from vLLM #56633 while retaining SM80 launch tuning and cuBLAS fallback. New CUDA math requires target acceptance.
- Port the V4.1 image-sentinel padding correction from #56554. Keep k=5, context 262144, image limit 999 and the measured KV concurrency formula.
- Ship verified read-only source overlays for the existing offline image, transactional update/rollback, cached first-use SM80 kernel regression, and direct C1/C32 benchmarks with optional image/history input. No new target TPS is claimed.

- Correct the Responses image acceptance payload to include `detail: auto`; add same-image schema regression checks and live five-image tests for Responses and Messages. An early operator check omitted this field and failed before inference, after the model had already restarted successfully.
- Persist the image count limit at 999, matching the pinned vLLM default, in both runtime configuration and protected constraints; keep native DSpark k=5.
- Add 5/8-image OCR acceptance cases using the four existing fixtures. These new cases require target GPU execution; historical performance and 1/2/4-image results remain unchanged.
- Document historical-image counting, upgrade configuration precedence, and practical context/memory/request-size limits. The original v1.0.0 release asset retains its original configuration.

## 1.0.0 — 2026-09-14

- First public source release for DeepSeek-V4.1-Flash on 8×A100-SXM4-80GB/NVSwitch.
- Fix deployment to native DSpark **k=5** based on the reported high/off comparison; retain historical k5/k7 performance data.
- Preserve 262144 context and the actual KV concurrency formula, minimum 32.
- Include Engram CPU offload, SM80 runtime overlays, CUDA Graph observations and offline startup checks.
- Include local/Docker-only keyless access, Responses schema handling, tool/vision/cache acceptance and cache retention diagnostics.
- Preserve operator-reported performance, benchmark methodology, offline packaging guidance and upstream acknowledgements.
- Publish a complete fixed deployment tree with source packaging and direct startup.

This is a source release. No new target GPU benchmark, 24h soak, Engram placement comparison or long-idle KV retention result is claimed for publication.
