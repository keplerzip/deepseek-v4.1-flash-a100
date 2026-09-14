# 协议、已有 NewAPI 与客户端

唯一模型名为 `DeepSeek-V4.1-Flash`。客户端展示用的 -high/-max 后缀若没有网关映射，不应作为上游模型名发送。

| 客户端 / 接口 | 模型服务路径 |
|---|---|
| Chat Completions / OpenCode 相应 provider | /v1/chat/completions |
| Codex Responses | /v1/responses |
| Claude Code / Anthropic Messages | /v1/messages |
| Messages token count | /v1/messages/count_tokens |
| 模型列表 | /v1/models |

宿主根地址 `http://127.0.0.1:8005`；Docker 内 `http://host.docker.internal:8005`。Linux 容器需有 host.docker.internal:host-gateway 映射，容器自己的 127.0.0.1 不指向宿主。

模型前端允许本机 loopback 与检查过的 Docker bridge，客户端免密。SDK 如强制要求 key 字段，可以使用其本地占位值；服务不依赖该值。NewAPI 自身用户 token 策略独立。

```bash
curl --fail http://127.0.0.1:8005/v1/responses -H 'Content-Type: application/json' -d '{"model":"DeepSeek-V4.1-Flash","input":"你好","max_output_tokens":128}'
```

已有 NewAPI 可将上游 API base 指向 `http://host.docker.internal:8005/v1`，是否附带 /v1 取决于渠道拼接方式，避免 /v1/v1。参考版本是操作者确认的 **v1.0.0-rc.37**。曾有直连成功、转发 404，操作者补齐 Responses/Messages 路由后恢复；版本号并不保证渠道已配好。

最终在你的 NewAPI 和客户端上分别检查流式工具调用、取消、最终 usage 和多图。本仓库不安装或修改网关与客户端。构建侧曾检查 Claude Code 2.1.218、Codex 0.154.0、OpenCode 1.18.30，源码发行不包含这些二进制。

多模态已开，默认图片数量配置为 999，与固定版本 vLLM 的默认值一致；历史实机验收覆盖 1/2/4 张本地图片。新增 5/8 图用例会循环使用这四张夹具，检查超过旧上限的请求处理和顺序，需在目标机执行。使用内联图片 data URL，外部图片 URL 由离线 guard 提前拒绝。

图片数量包含一次请求重新提交的全部历史图片。999 是请求计数上限，实际容量还受 256K 总上下文、图片尺寸、视觉编码显存、并发负载和 128 MiB 请求体限制，不代表 999 张图片或相应高并发已经通过实测。
