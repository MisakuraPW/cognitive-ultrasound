# 本地全量转换，再上传 AutoDL

当前选择：本地 CPU 全量转换 10030 个 AVI，保持官方 CASL 极坐标/cubic 算法；AutoDL 关机，之后手动上传并运行评估与训练。无需 GPU、无需重新安装本机已验证的 `.venv`。不会下载新的 EchoNet 数据或修改原始 AVI。

## 内存紧张时的提速（2026-09-09）

30 帧真实图像对照中，缓存坐标与 Delaunay 三角网格后的插值部分用时 0.272 秒，原实现 3.659 秒（约 13.4 倍），输出逐像素一致。每帧的像素值、梯度估计和 Clough–Tocher 三次插值仍重新计算；不是缓存图像内容，也不是从 B-mode 恢复原始 RF 数据。整段视频还要解码与压缩，不能把这个倍数直接当作整体提速保证。每个新 HDF5 和转换清单记录计算实现，已完成的旧文件保留。

本次采样可用内存约 1.1–1.9 GiB，因此使用缓存实现配合 **1 个工作进程**，优先避免换页，而不是增加并行度。单进程启动门槛 1.5 GiB；原 2 进程入口仍要求至少 3 GiB。转换进度 JSON 每完成或记录一个失败视频都会更新可用内存和内存使用率。需要更换进程数时，先停止这套任务，再用同一输出目录续跑：

```powershell
Set-Location 'G:\科研项目\毕设\cognitive-ultrasound'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/stop_local_conversion.py
.\.venv\Scripts\python.exe -X utf8 scripts/local_conversion.py --raw 'G:\SRTP\dataset\EchoNet-Dynamic' --output 'G:\SRTP\dataset\CASL-EchoNet-polar' --workers 1 --start
```

停止脚本先核对进程命令、工作目录、原始/输出路径，随后只终止这套转换进程树，不停止其他应用或科研项目。已发布 HDF5 不改写。

发现一个独立数据问题：`0X1DFB35BA6E0A7B20.avi` 属于固定官方 train 清单，但上游首帧形状检查返回拒绝（右侧指标约 2.976，阈值为 5），导致原任务停在 1110 个文件。现在个别视频失败会写入 `conversion_failures.json` 并继续其他视频；不会放宽筛选、改划分或将失败病例伪装为成功。只要仍有失败病例，最终状态仍为 failed，数据不得视为完整复现输入。磁盘写满、内存分配失败或进程池损坏则立即停止。

- 原始目录：`G:\SRTP\dataset\EchoNet-Dynamic`
- 派生输出：`G:\SRTP\dataset\CASL-EchoNet-polar`
- 本机约 16GB 内存、20 个逻辑处理器，默认使用 2 个进程，每个进程计算线程数为 1。
- 已核对 10030 个 AVI 约 7.35 GiB，G 盘初始可用约 192 GiB。两段真实短视频转换的进程树峰值 RSS 实测约 1.1 GiB；这不是全数据内存峰值或总耗时保证。启动检查至少 3 GiB 可用内存、首次至少 180 GiB 可用磁盘。

## 本机 PowerShell 启动

```powershell
Set-Location 'G:\科研项目\毕设\cognitive-ultrasound'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -X utf8 scripts/local_conversion.py --raw 'G:\SRTP\dataset\EchoNet-Dynamic' --output 'G:\SRTP\dataset\CASL-EchoNet-polar' --workers 2 --start
```

后台运行，不依赖终端窗口保持打开；电脑必须保持开机且不进入睡眠。任务只转换和审计，不会启动训练或自动关机。不要在已有转换运行时重复启动；转换目录有进程锁。若电脑重启或中途失败，修复原因后执行同一条命令续转。

```powershell
Get-Content 'G:\科研项目\毕设\cognitive-ultrasound\results\local_conversion\run.log' -Tail 30 -Wait
```

Ctrl+C 仅退出查看。当前状态在 `results/local_conversion/status.json`；转换计数、剩余估时在派生目录的 `conversion_progress.json`。全量转换完成后自动完整审计，只有审计成功，status 才写入 `completed`。

完整 HDF5 会校验后复用。旧串行输出最新文件与损坏文件隔离到 `.conversion-quarantine` 再重做。并行转换先写 `.conversion-tmp`，完成后原子发布；中断最多需要重做当时在处理的 2 个文件。原始数据保持只读。

## 手动上传

上传到 `/root/autodl-tmp/datasets/CASL-EchoNet-polar`，保留 `train/val/test/rejected` 四个目录、`split.yaml` 和 `conversion_manifest.json`。不要上传 `.conversion-tmp`、`.conversion-quarantine`、`.conversion.lock` 等工作文件。约 7985 个正式病例应为 train=6985、val=500、test=500，其他原始病例在 rejected。

云端已有少量旧转换时，先将旧目录改名保留，再上传到空目标目录，避免混用。通过 SFTP 客户端传四个目录即可；大部分 HDF5 本身已压缩，不必为了打包再占用一份完整副本的本地空间。上传后云端完整审计会检查划分、数量、形状、有限值和像素范围；需要严格传输校验时，还应比较本地与远端文件 SHA256。

上传完成、重新以 GPU 模式开机后运行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate casl &&
source /etc/network_turbo &&
cd /root/autodl-tmp/cognitive-ultrasound &&
git pull --ff-only &&
python scripts/autodl_overnight.py --start --with-training --prepared-data
```

`--prepared-data` 跳过转换，完整审计上传数据后自动接官方权重评估、训练 pilot、正式训练和自训练权重评估。必须带该参数：转换续跑标识包含本地源路径和 AVI 修改时间，不能将 Windows 生成记录误当作云端原始数据转换的续跑记录。
