# 固定 k=5 方案的运维

本版本直接使用固定的 DSpark k=5 配置，普通部署和后续重启均沿用该方案。运行参数校验拒绝 k=7 等其他草稿长度，README 中的 k5/k7 数据作为历史选择依据保留。

图片数量配置和保护校验均固定为 999。升级已有实例时，还应将 runtime/selected-config.json、runtime/tuning-active.json 中存在的旧图片数量更新为 999，并保留 DSpark k=5；这些运行配置优先于 configs/runtime.json。完成修改后停止并重新启动，按真实 KV 容量重新标定并发，不沿用未经确认的旧 C 值。超过 4 图的能力需要目标机验证，历史性能报告仍对应当时的 4 图配置。

在普通用户拥有的完整 deploy 目录执行：

```bash
bash ./status.sh
bash ./stop.sh
bash ./start.sh
```

只管理本项目所有权标签的模型与前端容器，不管理已有 NewAPI 等服务。start.sh 检测到健康实例会返回；修改部署文件后应先停止再启动，以加载新的固定配置。

## 版本更新

20260915-perf1 可复用原镜像，通过小源码包原位备份升级。首次启用新的 mHC 时先执行一次 SM80 kernel 回归；依据覆盖层、镜像和驱动缓存通过结果。使用方法、诊断开关及恢复最近备份见 [性能更新](performance-update-20260915.md)。

在构建侧组装完整离线 deploy 目录，核对组件和镜像清单，复制到目标机的独立目录。在维护窗口停止旧实例，再按部署说明启动新版本。保留旧目录和相邻权重以便恢复，避免直接覆盖正在运行的脚本与运行状态。运行目录包含实例身份、缓存和验收结果，不应作为新源码包的一部分复制。

## 验收

```bash
bash ./tests/local-access.sh
bash ./tests/run-all.sh
```

首条检查本机/Docker 访问，第二条执行功能、长输入和固定 k5 的负载测试。输出日志和 runtime/current-run.txt 可定位当前实例；模型重载可能为分钟级，KV 容量重新标定可能再次加载。

extended.sh 提供 full-window/reuse/stability 扩展测试；offline-rebuild.sh 会停止模型，将旧派生缓存移至历史目录，再验证无公网重建。它们属于显式维护操作。长期稳定性与满窗口驻留不能由短请求吞吐代替。

目标机只需普通用户与 sudo -n docker，不自动升级宿主、变更防火墙或删除旧模型数据。
