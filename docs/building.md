# 构建与完整离线包

GitHub 发布版本化源码，不托管约 510 GB 权重、运行镜像或宿主恢复二进制。R1.2 / v1.2.0 另提供相对初始 R1 的小累计更新包，复用原镜像和权重。以下准备发生在**联网构建机**；完成后目标机运行不依赖公网。使用 Python 3.10+、已有 Docker 和足够磁盘空间。

## 1. 准备固定模型快照

```bash
python3 build/download_model.py --dest /data/transfer/deepseek-v4.1-flash-a100/DeepSeek-V4.1-Flash
```

下载器使用清单中的固定 ModelScope revision、大小和 SHA256，支持 Range 续传，不产生第二份模型缓存。进度在 build-state/。已有正确快照可以复用；校验需读取所有文件。镜像地址和提交见 release.json，不使用随时间变化的 main 代替固定 revision。

## 2. 准备运行镜像

可复用已交付镜像归档和对应 image inspect 记录。否则在专用构建机运行：

```bash
bash build/build-runtime.sh /data/dsv41-runtime-build
```

该脚本拉取固定 SM80 基础镜像，检查基底文件 SHA256 后覆盖修正，以 network=none 构建覆盖层，导出并核对每个 blob、配置和 layer diff ID。它保留参考镜像 tag，会更新构建机上的该 tag；不要在生产目标机执行。image-inspect.json 是本地构建记录，不需要公开上传。

不同 Docker 构建得到的容器配置 ID 或 archive SHA256 可能不同。工具会固定**本次构建**的实际身份，而不是声称字节必然复现历史归档。重新构建的镜像仍需目标 GPU 验收；本仓库的历史性能只适用于报告中的栈。

## 3. 组装新离线目录

```bash
python3 build/assemble.py \
  --archive /data/dsv41-runtime-build/runtime-image.tar \
  --inspect /data/dsv41-runtime-build/image-inspect.json \
  --output /data/transfer/deepseek-v4.1-flash-a100/deploy
```

输出目录必须不存在。脚本验证归档、复制源码与镜像，生成 allowed-image-ids、image-integrity、baseline 和完整 deploy.sha256，不覆盖已有服务。模型与 deploy 相邻；脚本不复制权重、不下载目标运行依赖、不安装宿主包。

在同一文件系统内整理现有交付，可加 `--link-archive` 保留一份镜像数据；组装校验完成后可删除旧目录中的链接。此模式下两个路径指向同一归档，应保持只读。

当前性能更新放在 `deploy/overrides/performance/`，启动时验证基底与覆盖文件 SHA256，再只读挂载进原镜像。无需重新制作大镜像即可获得此更新。向已有隔离实例交付累计小包时运行 `python3 build/package_update.py --output /data/deepseek-v4.1-flash-a100-R1.2-update.tar.gz`；目标用包内 install.sh 应用，不需要先安装 perf1 或 R1.1。详细边界与回退见 [R1.2 更新](performance-update-20260918.md)。

将整个 transfer 下的项目目录拷到隔离机，由普通操作者持有文件。若只压缩 deploy，则模型另行复制一份，避免在归档里再占约 510 GB。最后按 [部署说明](deployment.md) 启动。

## 源码检查与发行

```bash
python3 build/check_source.py
python3 deploy/tests/dspark_cpu.py
python3 deploy/tests/performance_cpu.py
python3 deploy/tests/r12_contracts.py
python3 build/test_archive.py
python3 build/test_update.py
python3 build/test_runtime.py --output validation/r12/runtime
```

最后一项使用已安装的固定镜像，验证实际 CPU 控制流和无需 GPU 的 SM80 cubin 编译；它不会执行 CUDA 数值检查或生成 GPU TPS。其余检查仅需 Python 标准库。公开仓库的 deploy/manifests/deploy.sha256 是源码参考清单，包含历史镜像归档期望值，未补齐镜像时完整部署校验会失败。累计小包安装器则复用并核对已导入的镜像，不要求镜像归档仍在磁盘。assemble.py 会针对新完整输出重新生成清单。

R1.2 仍使用初始镜像。`build/test_runtime.py` 在原镜像 network=none 下检查 CPU 控制流及 SM80 AOT 编译；`build/prepare_r12_overlays.py` 需要构建侧归档的固定版本上游来源，只用于重建源码覆盖。已随源码提交的覆盖文件及 manifest 可直接用于 `build/package_update.py`，目标机不下载这些源码归档。
