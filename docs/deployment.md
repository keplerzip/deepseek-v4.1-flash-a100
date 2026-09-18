# 隔离目标机部署

## 宿主参考

实测机：8×A100-SXM4-80GB、NVSwitch NV12、双路 EPYC 7742、约 2 TiB RAM；Ubuntu 22.04.4，内核 5.15.0-94，NVIDIA 580.159.04，Docker 29.5.1，Container Toolkit 1.19.0。这是参考环境，不代表任意版本组合都经过验证。

预检要求八卡、MIG 关闭、NV12 拓扑，以及至少 512 GiB 当前可用主存作为基础检查；R1.2 两份 Engram 表预算约 378 GiB，按本机约 2 TiB RAM 设计，加载峰值仍须实测。八卡需由操作者从旧推理服务释放。已有 Docker、驱动、Toolkit 与 Fabric Manager 由管理员维护；部署只需普通用户和 `sudo -n docker`，不安装宿主服务。

## 资源与目录

模型为 89 文件、48 个 safetensors 分片，共 **510313354155 bytes**。模型保持一份物理快照，与 deploy 相邻，不在运行时拉取。

模型复制完成后，服务目录文件系统另需 64 GiB 派生缓存/运行空间与 20 GiB 余量。Docker/containerd 文件系统单独核算镜像导入与 50 GiB 余量，共享文件系统会合并预算；以 `runtime/storage-preflight.txt` 为准。

将完整离线 deploy 和模型复制至普通用户可写位置，例如 `/ai/services/deepseek-v4.1-flash-a100/`。`deployment.env` 中 `MODEL_DIR` 留空即采用相邻目录。

若普通用户无法自动读取 containerd 配置，可将 `CONTAINERD_ROOT_DIR` 填成**已核实的实际目录**，参考机为 `/var/lib/containerd`。这只指定空间检查路径，不修改 Docker/containerd 配置。

## 启动

```bash
cd /ai/services/deepseek-v4.1-flash-a100/deploy
umask 077
mkdir -p runtime
set -o pipefail
bash ./deploy.sh 2>&1 | tee runtime/first-deploy.log
```

流程：组件 SHA256 → 空间/硬件预检 → 本地镜像导入 → 模型全量校验 → 八卡小算子检查 → network=none 首启 smoke → 实际 KV 标定 → internal 网络正式服务。C 改变或隔离首启转入服务网络可能再次加载模型，日志会解释阶段。

固定 k5；运行参数校验拒绝其他草稿长度。配置优先级：`configs/runtime.json` < `runtime/selected-config.json` < 显式调优的 `runtime/tuning-active.json`。改配置须停止再启动；重复调用 start.sh 不会自动改动健康实例。

```bash
bash ./status.sh
curl --fail http://127.0.0.1:8005/v1/models
bash ./tests/local-access.sh
bash ./tests/run-all.sh 2>&1 | tee runtime/acceptance.log
```

run-all 包含功能、长输入与负载测试，避免同时运行其他压测。成功 smoke 不是 24h 稳定性或满窗口并发验收。

R1.2 的更新安装、目标机检查与回退见 [累计更新说明](performance-update-20260918.md)。首次多卡检查覆盖新增 EP8 集合通信，不重复此前的完整基础探针。
