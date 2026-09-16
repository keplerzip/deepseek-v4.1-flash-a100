# R1.1 / v1.1.0：复用初始镜像的累计更新

交付版本 **R1.1**，Git 标签 **v1.1.0**，更新编号 **20260916-perf2**。本包从最初的 R1 离线交付直接升级，也适用于已打过中间补丁的实例；**不要求先安装 perf1，不包含镜像或模型权重**。

继续复用已导入的 `deepseek-v4.1-flash-a100:20260913-r1` 镜像，镜像名称仍含 r1 是正常的。安装器核对本包的 SHA256 和已安装镜像的实际 ID，并核对被覆盖的底层源码。更新文件以只读挂载加载；新 Triton/TileLang kernel 在现有镜像内离线编译，工具链已包含在原镜像中，不下载依赖。

## 性能：历史基线与 R1.1 验收状态

| 版本 / 配置 | 单流 decode tok/s | C32 总 tok/s | C32 P95 秒 |
|---|---:|---:|---:|
| 原方案 k5 / off，用户回传 | 139.80 | 1379.01 | 24.56 |
| 原方案 k5 / high，用户回传 | 109.95 | 1039.91 | 34.93 |
| R1.1 k5 / off | 待 A100 实测 | 待实测 | 待实测 |
| R1.1 k5 / high | 待 A100 实测 | 待实测 | 待实测 |

历史数据是 8×A100-SXM4-80GB 上的文本负载，每组 5 个单流样本、200 个 C32 请求、1024 输出 token；high 计入思考 token。这不是多图历史成绩，不能用于承诺 R1.1 的提速。完整历史 JSON 记录附在更新包 reports/ 中，原始逐请求 JSONL 仍在隔离目标机。

## 本包包含什么

| 部分 | 更新内容 |
|---|---|
| R1.1 新增 | SM80 候选索引紧凑计算：后续索引层只计算已有候选块，避免完整列计算后再屏蔽；保留候选源层与短上下文的原路径 |
| R1.1 新增 | 融合稳定 MoE token 排序：保留 expert / 原 token 次序，不关闭确定性排序 |
| 累计 perf1 | NCCL 自动选择、保留 A100 调优的 mHC post/pre 融合与 DSpark 辅助状态复用、图像 padding 修复 |
| 累计接口修复 | 本机与 Docker bridge 免密访问、Responses 工具 schema 处理、三协议图片参数测试、image=999 配置 |
| 固定方案 | 同一份官方权重、唯一模型名 DeepSeek-V4.1-Flash、TP8、256K、DSpark k=5、Engram RAM、原 KV 并发公式 |

紧凑索引只进入 SM80 / FP8 / TP 路径，保留 query sharding 的切片和汇总；DCP/PCP、候选源层和不划算的短列宽沿用原实现。`2048×8` 是压缩索引坐标中的候选宽度，不能直接当成原始提示 token 数。数学上保留候选范围，但浮点舍入或并列分值可以改变 top-k 中的等分选择；需要实际模型质量验收。

这版不更换量化权重、不把 Engram 搬到 GPU，不调整宿主 NUMA/THP，不部署或修改 NewAPI。所有服务文件和检查输出保留在项目目录，只需普通用户与 `sudo -n docker`。

## 只传小包并升级

拷贝 `deepseek-v4.1-flash-a100-R1.1-update.tar.gz` 和同名 `.sha256` 文件到目标机普通用户拥有的目录，执行：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.1-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.1-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.1-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

不需要重跑 `deploy.sh`，不需要重新导入镜像或读取全部权重计算 SHA256。启动仍核对既有模型验证记录和文件状态；记录缺失或权重发生变化时会停止升级启动，不能绕过校验。

这是**前台命令**，会停止本项目服务再启动。新增算子首次启用先做单卡数值与 CUDA Graph 检查，日志逐组打印；检查最长 30 分钟，超时会退出并清理检查容器。此检查不重复原先八卡 NCCL 探针。通过结果按覆盖文件、镜像、驱动和开关组合记录，随后同组合不重复检查。模型加载、KV 重新标定仍可能触发重启。

安装器备份被改动的文件，保留已有 `MODEL_DIR`、端口、containerd 路径、权重、镜像和运行日志，并将 k / 图片数固定为 5 / 999。备份和版本收据位于 `deploy/runtime/update-backups/` 与 `deploy/runtime/latest-update.json`；发布身份位于 `deploy/manifests/update.json`。不删除旧日志和缓存作为升级前置。

若新增算子检查或服务启动失败，安装器自动恢复刚才备份的源码并尝试启动旧服务，失败日志仍保留。升级成功后需要主动回退时，在 deploy 目录执行：

```bash
bash ./rollback-update.sh
```

省略安装命令中的 `--restart` 只替换文件，**运行中的容器仍使用旧代码，不能据此判断更新效果**；之后执行 `bash ./restart.sh`。回退会拒绝覆盖升级后又手工改动的源文件。重复安装也产生独立备份，默认回退仅针对最近一次安装。

## 验收与同输入速度对照

构建侧没有 GPU。编译检查和 CPU 控制流测试不能代替 A100 数值检查、完整模型验收或速度实测。README 第一部分保留的 139.80 / 109.95 tok/s 等数字均为历史 k5 数据，**不是 R1.1 的速度**。

升级前可先在原服务运行 `bash ./tests/performance-benchmark.sh` 保存基线（原目录缺少该工具时，应先解包取用测试工具，或使用已有同口径测试）。升级后在 deploy 目录执行：

```bash
bash ./tests/run-all.sh
bash ./tests/performance-benchmark.sh
```

第二个命令测 off/high 下 C1 与 C32，并保留每请求计时、缓存命中和 DSpark 接受率。新增实现优先面对长历史，短文本结果不能替代截图历史。可对包内八张小 OCR 图运行 `bash ./tests/performance-benchmark.sh --images 8`；实际截图对照应把相同脱敏 Chat Completions 请求体存为 `runtime/benchmark-chat.json`，使用已有 data URI 后运行：

```bash
bash ./tests/performance-benchmark.sh --request-file /state/benchmark-chat.json
```

对照需暂停其他客户端流量，固定同一历史、思考模式、输出长度和缓存热度，按 A→B→A 重复。图片正文不写入统计报告。空闲 KV 保留、Engram 开销、多图 decode 是否恢复及长期稳定性仍需目标机证据。

需要拆分收益时，在 deployment.env 中设置四个开关后重启：

```text
PERF_NCCL=auto
PERF_MHC=1
PERF_INDEXER=1
PERF_MOE_ALIGN=1
```

`PERF_INDEXER=0`、`PERF_MOE_ALIGN=0` 分别恢复原索引与稳定排序，`PERF_MHC=0` 恢复原 mHC，`PERF_NCCL=legacy` 恢复 Ring/Simple。图像修复始终启用；完整旧版对照应使用升级备份回退。每轮报告记录实际开关与容量，不能把关闭一个开关的结果冒充初始 R1。

## 固定来源与致谢

感谢 Danylo Storozhev / [Zanooda 的 CMP170HX 项目](https://github.com/Zanooda/deepseek-v4.1-flash-cmp170hx/tree/2445d7db07a2023f1d0a4bbb5b1ee4154b299590) 提供候选索引计算实现；感谢 [Tokha233 的 A100 Turbo 项目](https://github.com/Tokha233/deepseek-v4.1-flash-a100-turbo/tree/fae324ae62ac5cef31b7d38f5d369618e1cae1fa) 提供稳定 MoE 排序融合。此次只移植上述明确组件，不套用其整套启动参数或第三方性能数字。

累计更新继续感谢 wtdcode/vllm-backport、vLLM #56633 / #56554、PyTorch、Triton、TileLang 和 NVIDIA NCCL。所有派生文件保留 Apache-2.0 声明。模型来自 DeepSeek，固定快照来自 ModelScope，均未重新分发到此更新包。
