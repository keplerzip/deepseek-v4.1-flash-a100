# 验收工具

所有命令在完整离线包的 deploy 目录执行。运行模型测试前先确保服务健康，不与其他压测并行。

| 入口 | 用途 |
|---|---|
| `bash tests/run-all.sh` | 文本、工具、图片、长输入、C32/标定并发和 high 负载 |
| `bash tests/local-access.sh` | 本机与 Docker bridge 的免密访问 |
| `bash tests/guard-schema.sh` | 客户端工具 JSON Schema 合同检查 |
| `bash tests/vision-schema.sh` | 同镜像验证三协议图片请求参数和 Responses detail 字段；无需 GPU、不重启 |
| `bash tests/cache-retention.sh seed` / `check` | 固定前缀的空闲保留诊断 |
| `bash tests/newapi-test.sh` | 已有外部网关的手动验收，需要其连接设置 |
| `bash tests/offline-rebuild.sh` | 维护窗口重建派生 GPU 缓存，会重启模型 |

结果位于 runtime/ 下相应运行目录。历史 k5/k7 报告包含 off/high、单流和 C32；当前固定 k5 基于已回传结果，不承诺任意请求都更快。GPU 小算子检查通过不是完整模型验收，短请求高并发不等于满 256K 驻留。

图片验收覆盖 Chat Completions 的 1/2/4/5/8 图，以及 Responses、Messages 的 5 图。Responses 的每个 input_image 显式提供 detail: auto，避免固定镜像请求 schema 在推理前拒绝。参数校验通过不替代目标机的多图 OCR 实测。

extended.sh / extended.py 提供 full-window、稳定性等扩展测试，调用参数见脚本。长时测试不是默认启动步骤，公开历史报告未声称已完成 24h 验收。
