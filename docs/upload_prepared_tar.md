# 上传已经校验的 CASL tar

本地收尾顺序：本轮全部尝试 → 修复有哈希约束的已诊断冲突 → 完整审计 → 生成 tar → 逐成员校验内容及包内清单 → 只删除派生目录。任何失败都停止后续删除；原始 `G:\SRTP\dataset\EchoNet-Dynamic` 保留。

本地位置：`G:\SRTP\dataset\CASL-EchoNet-polar.tar`、同名 `.tar.sha256`、`.files.json` 与 `.verification.json`。`.partial` 也写在 G 盘相同目录。HDF5 本身已压缩，24 MiB 抽样额外 gzip 压缩仅减少约 1%，因此使用普通 tar，不承诺显著缩小体积。打包不降低图像精度，也不丢帧。

只有 `results/local_conversion/finish_status.json` 为 `completed`、`package_status.json` 为 `verified` 且 `source_deleted=true` 时，才表示这套收尾全部成功。日志在 `results/local_conversion/finish.log`。收尾期间保持电脑开机；协调器会请求 Windows 暂不自动休眠，退出时释放该请求，手动关机/休眠仍会中断。

## AutoDL 无卡模式：上传、校验、解压

通过 SFTP 上传 `.tar` 和 `.tar.sha256` 到 `/root/autodl-tmp/`。另两个 JSON 可一并上传备查。云端需要同时容纳 tar 与解压后的数据，预留至少约 150 GiB 可用空间，另给训练输出留余量。

```bash
cd /root/autodl-tmp
sha256sum -c CASL-EchoNet-polar.tar.sha256
mkdir -p datasets
```

目标 `/root/autodl-tmp/datasets/CASL-EchoNet-polar` 应不存在。若有之前的零散转换，先改名保留，避免直接混写。随后：

```bash
tar -xf /root/autodl-tmp/CASL-EchoNet-polar.tar -C /root/autodl-tmp/datasets
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
cd /root/autodl-tmp/cognitive-ultrasound
git pull --ff-only
python -m cognitive_ultrasound audit-data --data-root /root/autodl-tmp/datasets/CASL-EchoNet-polar --output reports/uploaded_dataset_statistics.md
```

包内包含 `train/val/test/rejected`、划分、转换/兼容清单、审计报告及 `transfer_manifest.json`。不包含工作锁、临时目录、隔离文件或原始 AVI。

## GPU 模式：启动评估与训练

在已通过 JAX/TensorFlow GPU 检查的原实例环境中：

```bash
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate casl &&
source /etc/network_turbo &&
cd /root/autodl-tmp/cognitive-ultrasound &&
git pull --ff-only &&
python scripts/autodl_overnight.py --start --with-training --prepared-data
```

保留 `--prepared-data`，避免再转换或把 Windows 的转换身份误当成云端续跑。后台顺序为 GPU/数据/权重检查、演示、评估试跑与估时、官方权重正式评估、训练试跑与估时、正式训练、自训练权重评估、结果导出。任何一步失败都会停止，故不能将本地数据完成当成云端全流程已验证成功。

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/overnight/run.log
```

结果包及训练恢复包在 `/root/autodl-tmp/casl_exports/`。当前正式训练配置仍为 500×10000 次更新，尚未核实为论文总步数；试跑估时不会自动缩短预算。数据兼容处理保持当前 PIL 灰度和固定官方划分，并不证明与历史发布数据逐像素相同，详见 [诊断说明](conversion_failure_diagnosis.md)。
