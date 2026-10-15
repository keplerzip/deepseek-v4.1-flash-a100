# R1.9 / v1.9.0：500K、C64、1024 调度预算的完整离线版本

本版使用原 DeepSeek-V4.1-Flash 权重，唯一公开 API 名称 `DeepSeek-Flash`。交付为不含权重的完整包，同时提供累计源码小包。构建机无 GPU；CPU、源码兼容性及 SM80 离线编译通过，不等于八卡整模型性能验收。历史性能保留在 [README 第一节](../README.md) 与 [性能报告](performance.md)。

## 固定方案

| 项目 | 固定值及边界 |
|---|---|
| 模型 | ModelScope deepseek-ai/DeepSeek-V4.1-Flash，revision `f877e3ed82d979be83e828b52762cdb86152bb9c`；不重下、不量化、不改变权重 |
| 唯一公开名称 | `DeepSeek-Flash`；旧别名不再接受，客户端与已有网关自行同步 |
| 总上下文 | **512000 token**；500K 按 500×1024，输入、历史、图像 token 与输出合计 |
| 并行 | TP4 × DP2 + EP8，PP1，八卡 A100 80GB NVSwitch |
| 并发 | **全局64，每池32**；不是64个满窗口同时常驻保证 |
| 调度预算 | **每池每步1024 token**；chunked prefill开；**单张图片段不拆分**，单图视觉token上限1024 |
| DSpark | **固定k5**；原生草稿权重与验收规则；贪心草稿、standard拒绝采样 |
| Graph | 主模型 FULL_DECODE_ONLY；每池k5查询图最高160行、k+1验证图最高192行；prefill和视觉编码器仍eager |
| 内存 | GPU0.92，FP8 DS MLA KV，Engram主存卸载；实际KV按池实测 |
| 缓存/视觉 | 前缀缓存与usage明细开启；图片计数上限999，**单图视觉token上限1024**；实际仍受总上下文、尺寸、请求体、显存约束 |
| 接口/访问 | Chat Completions、Responses、Messages；127.0.0.1:8005与已核对Docker bridge，客户端免key |
| 宿主/权限 | 普通用户可写服务目录；唯一提权是sudo -n docker；不修改驱动、Docker配置、系统服务或NewAPI |

C64×(k+1)不在同一池计算：每池32×6＝192，低于1024。调度预算固定1024，不自动上调到4096/8192。长输入的chunk数仍比历史8192预算更多，可能增加冷prefill延迟，不能承诺所有请求更快。

图片保留V4.1语言模型内的双向注意力，不能跨调度步拆分。用户最终选择**调度1024、单图上限1024**，覆盖先前的512选择；本版不保留512降采样配置。为给DSpark k5预留4个新增槽，实际`hf_overrides={"vision_max_n_token":1020}`，**1020完整图片token＋4草稿槽＝1024**。1020包含图片起止/换行；接近原1024预算，在临界宽高比仍可能出现小幅缩放差异。图片计数上限仍999。

## 本版纳入的更新

1. **Prefill去同步 #60091**：实际Ampere继承路径两处repeat_interleave指定已知output_size；数学不变，只避免为确定长度触发GPU到CPU同步。
2. **inverse RoPE融合 #57435**：把逆旋转并入分块稀疏attention归并，减少独立kernel；保留非split、非连续输入及不适用路径。行数在每次eager调用计算，不把第一次batch形状冻结进compiled函数。
3. **DSpark context KV图捕获 #57632**：三层context投影进入草稿CUDA Graph；严格分开k查询与k+1验证行数，按KV组维护slot mapping，padding不写真实KV，超图容量prefill尾部仍eager。
4. **Indexer工作区缩减 #60874机制**：按logits/chunk预算限制gather reservation，并按原镜像实际chunk builder证明上界；不照搬新上游接口。共享workspace最终由最大消费者决定，不把H20的3.70GiB宣传成A100收益。

另修复旧scheduler整图调度边界：guard先于encoder cache命中/重复图捷径执行，预算不足时推迟整段图片，覆盖图起点、部分prefix命中和混合decode负载；不改变KV或草稿预留量。退化极窄图缩放修复只针对原PIL会产生零像素的错误分支，正常缩放路径保持一致。

固定提交与逐文件base/patched SHA256见 [覆盖清单](../deploy/overrides/performance/manifest.json)。累计包含R1.1至R1.8的适用修正：mHC、compact indexer、stable MoE、BF16 dense阈值32、EP8 AG/RS、8-warp sparse decode、条件shared expert overlap、主干EPLB、stacked KV、top-k复用和decode metadata。草稿EPLB与新增冗余专家保持关闭。

容量估算：512000窗口下，indexer的cr1/cr2预留行上限约829万/586万；共享attention消费者可能使净预留减少只有约0.56GiB/卡。此为源码预算估计，非实测KV增量。两个池仍须分别读取容量；按池floor(KV_tokens/512000)后相加，不能先合并token再取整。

## 交付与传输

- `deepseek-v4.1-flash-a100-R1.9-full.tar.gz`：完整deploy、原运行镜像、源码覆盖、验收、文档、报告、许可证；**无模型权重**。
- `deepseek-v4.1-flash-a100-R1.9-update.tar.gz`：同一版本累计源码/文档/验收小包；需目标机已导入原镜像，可从R1至R1.8累计升级。
- 两者均有外层`.sha256`，内部`SHA256SUMS`及部署组件清单。权重目录、runtime、密钥、私有请求、旧base64补丁不进入包。

完整包复制到普通用户可写临时目录，保留解压目录不要与目标deploy重叠：

```bash
sha256sum -c deepseek-v4.1-flash-a100-R1.9-full.tar.gz.sha256
tar -xzf deepseek-v4.1-flash-a100-R1.9-full.tar.gz
cd deepseek-v4.1-flash-a100-R1.9-full
bash ./install.sh /ai/services/deepseek-v4.1-flash-a100/deploy --restart
```

已有部署：保留deployment.env、MODEL_DIR、端口、runtime编译缓存、模型校验指纹及历史报告；选中的旧runtime调优配置备份后迁移到固定512000/1024/C64/k5。升级不删除权重、不清空KV保留诊断记录；服务重启时内存KV自然丢失。镜像已存在且身份匹配则复用；缺失才从完整包导入。全新部署需先将原权重放在deploy相邻的DeepSeek-V4.1-Flash目录，或设置MODEL_DIR。默认只读检查现有宿主前置，不安装宿主包。

小包使用相同步骤，把full替换成update；install.sh必须传现有deploy路径。已有服务启停和重载约需数分钟，以健康检查为准，不应仅按显存涨到某数值判断成功。首次新增SM80数值/图回放门禁可能编译，最长30分钟超时；沿用操作者先前要求跳过的独立EP8通信重测，八rank整模graph/replay与双池smoke仍不能跳过。

## 验收与测速

```bash
cd /ai/services/deepseek-v4.1-flash-a100/deploy
bash ./status.sh
curl --fail http://127.0.0.1:8005/v1/models
bash ./tests/local-access.sh
set -o pipefail
bash ./tests/run-all.sh 2>&1 | tee runtime/r19-acceptance.log
bash ./tests/performance-benchmark.sh --isolate-clients
bash ./tests/performance-benchmark.sh --isolate-clients --images 8
```

run-all分别检查DP0/DP1文本、三协议工具/缓存/图片、长输入检索，再测全局C64短请求和high负载。最高输入预算507904，另留4096输出。测速off/high各2次预热、5个C1样本、200个C64请求，每请求1024输出；支持`--concurrency 32`作历史口径对照。`--request-file /state/benchmark-chat.json`可测本机私有历史，文件不发布。

`--isolate-clients`持操作锁，仅暂停且随后恢复本项目受管前端；不重启模型、不修改NewAPI；暂停期间客户端请求会失败。已有引擎直连流量仍需操作者停止。未隔离或计数不一致时，保存全部可得的逐请求与聚合数据，标记CONTAMINATED、不可正式签收；不在预热阶段丢掉后续测速结果。SSE事件数不是token数，思考token计入completion，报告同时列decode估计、端到端与聚合吞吐。

本版目标验收仍待执行：八卡启动、新inverse RoPE数值与变输入图回放、两池DSpark context图capture/replay、512k单请求检索、C64短负载、多图真实历史、缓存命中、2h/24h稳定性。CPU参考、SM80编译和安装回滚测试分别记录，不能代替上述实测。

## 回退

启动失败时安装器自动恢复备份源码并尝试启动旧服务；收据位于runtime/update-backups。需要手动回退最近一次更新时：

```bash
bash ./rollback-update.sh
```

该命令及前一次部署的模型身份/上下文应以目标目录脚本提示为准。备份会保留旧selected-config与tuning-active，回退恢复原文件与权限；不删除当前权重/缓存。完整安装与源码更新都只操作自己的目录与受管容器。详细操作见 [运维](operations.md)。
