# 固定 k=5 方案的运维

本版本直接使用固定的 DSpark k=5 配置，普通部署和后续重启均沿用该方案。运行参数校验拒绝 k=7 等其他草稿长度，README 中的 k5/k7 数据作为历史选择依据保留。

图片数量配置固定为999，上下文512000、全局C64、调度预算1024、k5、GPU0.92均为保护参数。更新安装器把旧selected-config/tuning-active作为事务备份并迁移，不要求手工编辑这些文件。实际KV容量按两个DP池分别测量；每池至少容纳一个满500K窗口，不承诺64个满窗口同时驻留。详见 [R1.9完整方案](deployment-update-20261015-r19.md)。

在普通用户拥有的完整 deploy 目录执行：

```bash
bash ./status.sh
bash ./stop.sh
bash ./start.sh
```

只管理本项目所有权标签的模型与前端容器，不管理已有 NewAPI 等服务。start.sh 检测到健康实例会返回；修改部署文件后应先停止再启动，以加载新的固定配置。

## 版本更新

R1.9 / 20261015-r19-500k-b1024-c64 复用原R1镜像与原V4.1权重，新增四项源码优化。完整包或累计小包均以包内install.sh执行升级，支持--restart和启动失败自动回退。版本身份见deploy/manifests/update.json，收据见runtime/latest-update.json。[安装、验收与回退](deployment-update-20261015-r19.md)。

在构建侧组装完整离线 deploy 目录，核对组件和镜像清单，复制到目标机的独立目录。在维护窗口停止旧实例，再按部署说明启动新版本。保留旧目录和相邻权重以便恢复，避免直接覆盖正在运行的脚本与运行状态。运行目录包含实例身份、缓存和验收结果，不应作为新源码包的一部分复制。

## 验收

```bash
bash ./tests/local-access.sh
bash ./tests/run-all.sh
```

首条检查本机/Docker 访问，第二条执行功能、长输入和固定 k5 的负载测试。输出日志和 runtime/current-run.txt 可定位当前实例；模型重载可能为分钟级，KV 容量重新标定可能再次加载。

extended.sh 提供 full-window/reuse/stability 扩展测试；offline-rebuild.sh 会停止模型，将旧派生缓存移至历史目录，再验证无公网重建。它们属于显式维护操作。长期稳定性与满窗口驻留不能由短请求吞吐代替。

目标机只需普通用户与 sudo -n docker，不自动升级宿主、变更防火墙或删除旧模型数据。
