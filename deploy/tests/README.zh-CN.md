# 验收工具

所有命令在完整离线包的 deploy 目录执行。运行模型测试前先确保服务健康，不与其他压测并行。

| 入口 | 用途 |
|---|---|
| `bash tests/run-all.sh` | 文本、工具、图片、长输入、全局C64/每池32和 high 负载 |
| `bash tests/local-access.sh` | 本机与 Docker bridge 的免密访问 |
| `bash tests/guard-schema.sh` | 客户端工具 JSON Schema 合同检查 |
| `bash tests/vision-schema.sh` | 同镜像验证三协议图片请求参数和 Responses detail 字段；无需 GPU、不重启 |
| `bash tests/performance-kernels.sh` | R1.4 单卡 SM80 dense/attention/共享专家/索引/MoE/mHC 数值与图重放检查；启用的组首次启动自动执行 |
| `bash tests/performance-benchmark.sh` | 固定 k5 的 off/high、C1/C64，--concurrency 32可对照历史，支持 `--images 8` 与 `--request-file /state/benchmark-chat.json` |
| `bash tests/cache-retention.sh seed` / `check` | 固定前缀的空闲保留诊断 |
| `bash tests/newapi-test.sh` | 已有外部网关的手动验收，需要其连接设置 |
| `bash tests/offline-rebuild.sh` | 维护窗口重建派生 GPU 缓存，会重启模型 |

结果位于 runtime/ 下相应运行目录。历史 k5/k7 报告包含 off/high、单流和 C32；当前固定 k5 基于已回传结果，不承诺任意请求都更快。GPU 小算子检查通过不是完整模型验收，短请求高并发不等于满500K 驻留。

图片验收覆盖 Chat Completions 的 1/2/4/5/8 图，以及 Responses、Messages 的 5 图。Responses 的每个 input_image 显式提供 detail: auto，避免固定镜像请求 schema 在推理前拒绝。参数校验通过不替代目标机的多图 OCR 实测。

extended.sh / extended.py 提供 full-window、稳定性等扩展测试，调用参数见脚本。长时测试不是默认启动步骤，公开历史报告未声称已完成 24h 验收。

R1.4 算子检查最长 30 分钟，超时退出并删除其检查容器。紧凑索引检查包括打包请求偏移、候选 -1、因果边界、TP query 切片、分页与图重放；MoE 与旧稳定排序逐项精确比较。CPU 调度测试与 SM80 离线编译分别记录，均不能替代这项 GPU 数值检查。通过算子检查后仍需 run-all.sh 与真实多图历史验收。

R1.2 新增 `bash tests/performance-ep8.sh`：8 卡 custom AG/RS、变长 NCCL 回退和图回放，300 秒超时，启动只在版本/驱动组合变化时执行。释放 CUDA Graph 后再关闭 communicator，避免先前探针的析构等待。

run-all.sh 分别固定 DP0 和 DP1 验证 full/long，三协议均检查 JSON/SSE 缓存命中；随后不固定池，验证全局C64 的独立短请求。报告区分每池与全局 C。R1.2 的 C 不直接复用历史 TP8 C56。

R1.3 新增实际维度的原生 MXFP8 合并 KV 投影/scale 加载、原路径与独立 BF16 参考、修改输入后的图回放；并检查 ragged top-k 在批次缩小/增大、block table 和 mask 改变、旧图与 eager 交替时的物理槽位结果。两个新优化默认开启，CPU 加载与 SM80 编译结果不充当目标 GPU 数值结果。

R1.4 累计保留上述检查，新增 10 组 ragged 融合/大批回退的真实 GPU 对照（空输入、零宽、全 padding、非连续输入、1024/1025 行边界），每组修改输入后图回放 2 次；另有 3 组 SWA 持久有效性缓冲图回放。`PERF_DECODE_METADATA` 控制本版新增组并参与验收签名，关闭后回到 R1.3 元数据实现。构建侧另执行生产 builder 片段与 Triton CPU 解释器对照、SM80 AOT 编译，报告明确不算 GPU 执行。

R1.5 继续使用相同 R1.4 算子覆盖；配置窗口改为 1048576。long 在 DP0/DP1 分别运行 32K、64K、128K、约 256K、512K、1044480 输入预算，另预留 4096 输出 token，逐项核对实际 usage 与 tokenizer。extended full-window 改为 1048064 输入目标加 512 输出预算；完整接纳和完成不等于全部同时驻留，C32 下限仍可能造成长请求排队或抢占。旧 256K 容量不复用，历史报告不改写为 1M 成绩。

## R1.9固定参数和测速

上下文512000，decoder每步预算1024，长输入测试最高507904预算并留4096输出；DP0/DP1分别检索，全局短负载C64。旧R1.5长输入范围描述仅为历史，当前预算见acceptance.py。

performance-benchmark.sh默认off/high、C64、200请求、1024输出，支持--concurrency 32、--images 8、--request-file。加--isolate-clients仅暂停本项目前端，EXIT/失败/TERM均恢复，不重启模型。全局计数受后台流量干扰时继续记录各阶段与聚合结果，但标记CONTAMINATED，不作正式性能验收。测试报告中P50/P95是请求完成延迟，聚合TPS是全部输出token/负载总墙钟时间。

新增inverse RoPE GPU数值与变输入图回放；启动必须收齐八rank DSpark context KV捕获/回放标志。CPU测试覆盖真实生产dispatch、slot padding/eager尾部、工作区chunk上界、原镜像CLI和512预算整图段与单图1024处理器缩放。SM80离线编译34变体，不声称执行GPU。tune.sh已改为固定参数测速入口，不再扫描或修改4096/8192等调度预算。

R1.9单图上限1024的实际处理器预算为1020，另留DSpark k5的4个新增槽，总计1024；保留整张图片的双向注意力，不允许为缓存命中或混合负载而拆图片段。
