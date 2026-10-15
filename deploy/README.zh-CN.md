# DeepSeek-V4.1-Flash A100 — R1.9

唯一API模型名DeepSeek-Flash；固定512000上下文、每池1024调度token、全局C64/每DP32、TP4×DP2＋EP8、DSpark k5、GPU0.92。原权重与运行镜像复用，Engram主存卸载、前缀缓存、999图片计数与单图1024视觉token上限（大图细节可能减少）、三协议、免key本机/Docker8005访问保留。

普通用户在本目录执行bash deploy.sh；首次启动需已有原V4.1完整权重与包内runtime-image.tar。CONTAINERD_ROOT_DIR使用用户已回传的/var/lib/containerd；若宿主布局改变需改为核实值。已有部署请用完整包/累计包的install.sh事务升级。

功能验收bash tests/run-all.sh；隔离测速bash tests/performance-benchmark.sh --isolate-clients，8图加--images 8；结果在runtime。隔离测速只暂停本项目前端并恢复，不重启引擎、不修改NewAPI。完整说明见包根目录docs/deployment-update-20261015-r19.md。构建机无GPU，新版本吞吐/容量待目标机验收。

R1.9单图上限1024的实际处理器预算为1020，另留DSpark k5的4个新增槽，总计1024；保留整张图片的双向注意力，不允许为缓存命中或混合负载而拆图片段。
