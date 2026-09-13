# 缓存命中与空闲保留

默认启用 prefix caching 和 usage 缓存明细。直接模型验收中约 14K 前缀冷请求命中 0、热请求命中 14208，比例约 98.68%；这不等于所有网关都正确显示。

| 协议 | 响应 usage 字段 |
|---|---|
| Chat Completions | prompt_tokens_details.cached_tokens |
| Responses | input_tokens_details.cached_tokens |
| Anthropic Messages | cache_read_input_tokens，以及存在时的 cache_creation_input_tokens |

流式响应应查看最终 usage。NewAPI 需对应渠道路由和 usage 转换，可通过 tests/newapi-test.sh 检查已有网关。真实命中必须来自上游响应，不能以首 token 变快代替。

## 空闲后命中为 0

操作者报告无重启、无其他请求，十分钟内再次请求命中为 0。当前交付没有设置十分钟 TTL；源码检查没有发现该部署的定时前缀过期配置。**原因尚未解决**，不能直接解释为请求压力淘汰，也没有提供未经验证的延长 TTL 开关。

```bash
bash ./tests/cache-retention.sh seed
# 保持空闲 10–15 分钟
bash ./tests/cache-retention.sh check
```

该诊断直连引擎，记录请求摘要、容器/进程/配置身份和真实命中。不会重置 prefix cache；每次 check 本身发出请求并刷新前缀，不能连续轮询后称为完全空闲测试。

模型服务重启时旧 KV 不保留；与上述无重启反馈应分开判断。
