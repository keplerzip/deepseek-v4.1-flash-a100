# Changelog

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
