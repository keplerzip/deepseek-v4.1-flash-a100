# R1.5 / v1.5.0：原权重 1M 上下文累计更新

R1.5 将服务总上下文固定为 **1,048,576 token，输入与输出合计**。继续使用 DeepSeek-V4.1-Flash 原始 ModelScope 快照、初始 R1 镜像、TP4×DP2＋EP8、DSpark k=5 和 R1.4 全部性能覆盖。升级包不包含模型或镜像，R1 至 R1.4 均可直接升级，无需逐版安装。

**本版是待目标 A100 验收的预发布版。** 构建机没有 GPU；历史 TP8 / 256K 吞吐与检索结果保留原口径，不作为 1M 实测。权重配置的 `text_config.max_position_embeddings=1048576` 已在固定快照中核对，不修改权重、YaRN 参数或模型名称。[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) · [固定文件清单](../deploy/manifests/model-files.json)

## 容量与并发

每次启动按实际初始化的 KV 分组、块大小与容量计算：

```text
L = 1048576
N_0 = floor(effective_kv_tokens_DP0 / L)
N_1 = floor(effective_kv_tokens_DP1 / L)
C = max(32, 2 * (N_0 + N_1))
每个 DP scheduler 的 max_num_seqs = C / 2
```

两个池分别取整，不能拼接不足一个窗口的余量，也不按 TP rank 乘四。两池都至少能容纳一个完整 1M 窗口才允许启动；不满足就报错并保留证据，安装器按已有事务机制恢复旧源码并尝试重启旧配置，不会静默降到 256K。

例如两池各能容纳 7 个满窗口时 N=14、C=32；两池各能容纳 9 个时 N=18、C=36。这是公式示例，不是本机容量预测。**C32 是调度上限，不表示 32 个满 1M 请求能同时驻留。** 长请求可能排队或发生抢占，必须另做满窗口负载验收。

升级会备份并重置旧 `selected-config.json` / `tuning-active.json` 到本版固定配置。旧 TP8 或 256K 容量报告仅作为历史证据保留，不再用作新启动的并发种子；第一次以 C32 重新测量，必要时重启以收敛公式。扩大总窗口不会把 8192 的 prefill 批次预算改成一次处理 1M。

## CUDA Graph 已开启的范围

| 路径 | R1.5 设置 |
|---|---|
| 主模型 decode / DSpark 草稿 | `FULL_DECODE_ONLY`，保留完整 decode CUDA Graph；启动要求八个 worker 的 capture 与实际 replay 记录 |
| 文本 prefill | 分块处理，eager；不把完整 1M 输入一次捕获为图 |
| 视觉编码器 ViT / aligner | 本版仍为 eager，尚未移植 encoder CUDA Graph |

R1.4 暂缓的是 [vLLM #56625 的视觉编码器图](https://github.com/vllm-project/vllm/pull/56625)，不是关闭整个 CUDA Graph。该实现已于 9 月 20 日合并，但上游报告使用 4×GB200、32K、FLASH_ATTN；本项目固定旧镜像仍需适配视觉协议、保留 padding 修正，并在 DP2 下实测额外图显存对 1M KV 容量的影响。直接加 `cudagraph_mm_encoder=true` 不能代替这些移植与验收。

1M 是服务允许的长度；实际请求越长，prefill、注意力及缓存占用通常越大。本版不承诺满 1M 请求维持历史短输入 decode TPS。视觉编码图主要减少图片编码与并发干扰，其上游短输出多图吞吐倍数也不能直接作为长历史纯 decode 的倍数。

## 升级

传输 `deepseek-v4.1-flash-a100-R1.5-update.tar.gz` 及同名 `.sha256` 文件。在两文件所在目录执行：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.5-update.tar.gz.sha256 && tar -xzf deepseek-v4.1-flash-a100-R1.5-update.tar.gz && bash deepseek-v4.1-flash-a100-R1.5-update/install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

会停止并重新加载本项目模型服务。复用已安装的 `deepseek-v4.1-flash-a100:20260913-r1`，全程不联网；宿主只需普通用户及 `sudo -n docker`。保持 `127.0.0.1:8005` / Docker bridge 免密访问。已部署的 NewAPI 与客户端不由安装器修改；客户端自身的上下文声明仍为 256K 时，应在客户端另外设置为 1048576。

性能覆盖仍精确复用 **20260922-perf5 / 1.4.0**，部署身份为 **20260922-context1m / 1.5.0**，两者在清单中分别记录。未改变的 GPU 算子无需因版本名字不同重复检查；实际模型启动仍重新测容量并验证图捕获/回放。构建机已删除的权重分片无需为制作这个小包重新下载；目标机的完整快照必须保留。

## 验收与回退

升级后，在目标 `deploy` 目录执行：

```bash
set -o pipefail
bash ./tests/run-all.sh 2>&1 | tee runtime/acceptance-r15-1m.log
```

每个 DP 池分别执行三协议、工具、图片、缓存及长输入检索。长输入阶梯为 32K、64K、128K、约 256K、512K、**1,044,480 输入预算＋4096 输出预算**。使用服务 tokenizer 计数，允许为模板预留最多 2048 token；最终 usage 必须与 tokenizer 一致，不能将截断后的短输入算作 1M 通过。分别记录 DP0/DP1 的实际输入 token、输出预算、检索结果和耗时。扩展的边界压力测试使用 1,048,064 输入目标＋512 输出预算。

启动 smoke 不自动跑长输入压测。长输入、满窗口压力和稳定性检查会占用服务，耗时以目标机为准。单独执行：

```bash
bash ./tests/performance-benchmark.sh
bash ./tests/extended.sh full-window
```

第一条仍是短输入 off/high、C1/C32 基准，不是满 1M 速度。第二条提交近满 1M 独立请求，报告完成数、延迟与抢占；它证明接纳/完成，不单独证明所有请求同时驻留。

恢复最近一次安装前的源码与固定配置：

```bash
bash ./rollback-update.sh
```

原有运行证据保留。回退后的启动会重新计算容量；若升级后手工改过受管理文件，回退工具会先指出冲突，避免覆盖这些修改。

## 构建侧证据

见 [R1.5 构建检查](../reports/context-update-20260922.build.json)。配置、容量分池取整、最少 C32、旧容量隔离、真实离线镜像 CLI，以及历史版本累计安装/回退分别验证；沿用的 R1.4 算子与编译证据见 [原报告](../reports/performance-update-20260922.build.json)。这些不是目标机的 1M GPU 验收。
