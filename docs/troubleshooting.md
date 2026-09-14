# 故障定位

| 现象 | 处理 |
|---|---|
| No such image | 新机先 deploy.sh 导入归档；verify.sh 需要已存在镜像。 |
| Cannot determine active containerd storage | deployment.env 中填已核实的 CONTAINERD_ROOT_DIR，只影响空间检查。 |
| 校验后显存仅约 1 GiB | 可能仍是小算子探测，检查 gpu-probe 日志，不是权重加载。 |
| torchrun 解析容器 hostname 超时 | 修正探测使用 static rendezvous 和 127.0.0.1，避免 DNS 依赖。 |
| 卡在 nccl_destroy | 历史探测需在销毁通信组前释放 CUDA Graph；公开版本已包含修正。 |
| Waiting for eight-GPU engine | 观察模型日志，加载与 Graph 编译可为分钟级；显存上涨不是单独的健康证明。 |
| C32→C56 后再次加载 | KV 标定以及 network=none 转 internal 网络均可能触发明确重启。 |
| unhashable type: dict | 已保留 schema guard 修正，不将 JSON Schema 对象/联合 type 当作媒体 discriminator。 |
| 直连成功、网关 404 | 检查 Responses/Messages 渠道路由、base URL 拼接和模型映射。 |
| At most 4 image(s) may be provided | 旧交付的图片数量保护值为 4；当前源码已固定为 999。升级需同步默认配置、选中/调优配置和 offline_ops.py 保护值及文件校验清单，再停止并启动模型。历史图片也计入本次请求数量。 |
| 5 图 Chat 通过，Responses 验收返回 400 | 先读取错误正文；早期图片数量补丁的测试请求漏了 input_image.detail，固定镜像 schema 要求这个字段。补上 detail: auto 并重跑验收即可，已生效的图片数量配置无需重启。tests/vision-schema.sh 可离线复现参数校验。 |
| 模型重启后缓存为 0 | 服务重启，旧 KV 不保留；无重启空闲问题另见缓存文档。 |

```bash
sudo -n docker logs --tail 100 deepseek-v4.1-flash-engine
bash ./status.sh
```

对照 runtime/current-run.txt 指向的本次启动，避免混用旧失败日志。不要为每次重启重复全量权重哈希；不要在八卡被占用时启动第二份模型。
