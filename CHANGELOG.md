# Changelog

## 1.0.0 — 2026-09-14

- First public source release for DeepSeek-V4.1-Flash on 8×A100-SXM4-80GB/NVSwitch.
- Fix deployment to native DSpark **k=5** based on the reported high/off comparison; retain historical k5/k7 performance data.
- Preserve 262144 context and the actual KV concurrency formula, minimum 32.
- Include Engram CPU offload, SM80 runtime overlays, CUDA Graph observations and offline startup checks.
- Include local/Docker-only keyless access, Responses schema handling, tool/vision/cache acceptance and cache retention diagnostics.
- Preserve operator-reported performance, benchmark methodology, offline packaging guidance and upstream acknowledgements.
- Publish a complete fixed deployment tree with source packaging and direct startup.

This is a source release. No new target GPU benchmark, 24h soak, Engram placement comparison or long-idle KV retention result is claimed for publication.
