# AutoDL：复用已有 EchoNet 的准备方案

**2026-09-09 更新：当前系统 CUDA 安装方式、完整运行与下载命令以 [完整操作指南](autodl_runbook.md) 为准。**本页保留初始数据交接背景；第 3–5 节旧的分环境安装命令已由根目录 `requirement.txt` 的方案取代，不能混用。

目前已完成本地准备和个人 GitHub 仓库首次推送，未连接服务器、租用算力、创建云端目录或启动实验。下列命令供你确定实例后使用。

依据：用户提供的 `G:/SRTP/MAE/docs/autodl_echonet_shared_data_handoff.md`（2026-09-08），SHA256 为 `959ee3ba1df3dbe4c4ad5ff3a058f8a5770bbab0373b5bb287a1eaac61a49943`。交接文档描述历史事实和路径约定，不是本次服务器检查结果；其中 EchoRVM 的仓库、环境和训练命令不作为本项目操作指令。

## 1. 路径约定

| 用途 | 路径 | 使用方式 |
|---|---|---|
| 已有原始数据 | `/root/autodl-fs/datasets/EchoNet-Dynamic` | 只读；先确认共享挂载、CSV、Videos |
| 历史灰度缓存 | `/root/autodl-tmp/datasets/EchoNet-Dynamic` | 只读盘点，不用于 CASL 转换 |
| 约定的 RGB 缓存 | `/root/autodl-tmp/datasets/EchoNet-Dynamic-rgb` | 不假定存在，不修改 |
| CASL 项目代码 | `/root/autodl-tmp/cognitive-ultrasound` | 克隆自己的仓库 |
| CASL 极坐标数据 | `/root/autodl-tmp/datasets/CASL-EchoNet-polar` | 从已有 AVI 独立生成 |
| CASL 实验输出 | `/root/autodl-tmp/outputs_casl` | 日志、模型、评估、轨迹 |
| CASL 备份约定 | `/root/autodl-fs/projects/cognitive-ultrasound` | 核实挂载后备份重要产物 |

共享原始数据无需重新下载；高频读取的派生数据放在实例本地盘，重要产物另备份到文件存储，符合 [AutoDL 文件存储说明](https://www.autodl.com/docs/fs/)。新实例不一定有旧实例的本地缓存。共享目录暂时为空时先检查挂载和地区，不据此判断数据丢失。

`configs/autodl/paths.yaml` 用于只读检查；`evaluation.yaml`、`training.yaml` 分别覆盖基线配置中的运行路径。**路径文件不会自动改写评估或训练配置**：如果实际路径不同，要同步修改这三个文件；`demo.yaml` 会继承评估路径。

## 2. 数据复用边界

- 使用原始 `Videos/<ID>.avi`，保留字符串 ID 和原始帧序。现有灰度 NPY 由 OpenCV 转换，CASL 官方 AVI 加载使用 imageio/PIL；未证明逐像素一致，也未做极坐标转换，因此不能直接替代。
- CASL 划分以仓库中固定的 `configs/splits/split.yaml` 为准：6985/500/500。原始 FileList.csv 的 TRAIN/VAL/TEST 仅计数和交叉核对，不替换 CASL 划分。ID 互斥检查不等于独立的患者身份审计。
- 不继承 EchoRVM/MAE 的模型、16 帧、400 轮、归一化、增强或环境。CASL 继续 W=3、Np=2、FP32、官方极坐标处理和原版采样循环。
- `VolumeTracings.csv` 是否存在会记录，但现有 CASL 下游指标是重建图与完整图的**模型分割一致性**，不称为人工标签 Dice。人工轮廓仅覆盖有限帧，缺标签不能当背景；以后若增加人工标签评估，要另建协议并保留原视频帧索引。
- 原 EchoRVM 两个历史项目目录及 outputs、outputs_downstream、outputs_representation_full、outputs_temporal 都列为保护路径。

## 3. 上云后的首次只读检查

个人远程仓库已就绪，见 [Git 工作方式](git_workflow.md)。之后在 Linux 上克隆 `https://github.com/MisakuraPW/cognitive-ultrasound.git` 到上述 CASL 代码目录。以下命令假定目录已就绪，从项目根目录执行：

```bash
cd /root/autodl-tmp/cognitive-ultrasound
conda create -n casl-infer python=3.10 -y
conda activate casl-infer
python -m pip install PyYAML
python scripts/autodl_preflight.py
```

只读检查不依赖 Keras/JAX/TensorFlow；NumPy 存在时额外抽查至多三个旧 NPY 文件头，缺少它只记录无法检查缓存。脚本不解码 AVI、不创建缓存、不安装环境，只将报告写入 `/root/autodl-tmp/outputs_casl/preparation/preflight.json`；拒绝报告或 CASL 目标目录与保护路径重叠。

报告包含原始 CSV 计数、重复 ID、CASL 所需 AVI/CSV 缺失清单、空视频、划分交叉计数、文件哈希，以及当前 GPU、内存、磁盘容量/inode、挂载信息。返回码 2 表示输入清单未通过，先核实路径/挂载/缺失文件。Windows 入口会拒绝执行，避免将 Linux 路径误当本机目录。

`ready_for_conversion_inventory=true` **只表示输入文件清单通过**，不保证 AVI 可解码、GPU 可用、磁盘容量足够或论文复现成功。正式转换及 audit 才检查派生数据。空间需求应根据真实数据和首次小实验的轨迹体积估算；全量后验轨迹远大于稀疏观测，不套用旧实例容量。

## 4. 安装推理环境、转换、跑一个案例

确认检查报告后，在独立的 `casl-infer` 环境执行：

```bash
python scripts/bootstrap.py
python -m pip install -r requirements/inference.txt
python -m pip check
casl-repro doctor --output /root/autodl-tmp/outputs_casl/preparation/environment-infer.md
python -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'JAX GPU unavailable'"
python -m pip freeze > /root/autodl-tmp/outputs_casl/preparation/inference-resolved.txt

casl-repro prepare-data --raw /root/autodl-fs/datasets/EchoNet-Dynamic --output /root/autodl-tmp/datasets/CASL-EchoNet-polar
casl-repro audit-data --data-root /root/autodl-tmp/datasets/CASL-EchoNet-polar --output /root/autodl-tmp/outputs_casl/preparation/dataset_statistics.md
casl-repro fetch-assets --with-evaluation
casl-repro evaluate --config configs/autodl/demo.yaml
```

环境文件是待 AutoDL 验证的起点，不是已验证的云端 lock。医生报告记录硬件与依赖；上面的 JAX 断言单独确认计算后端。需要下载固定上游与模型，联网失败时保留报错，不替换成未固定版本。

转换调用官方筛选、扇形处理和 cubic 极坐标插值；输出应为 `(T,112,112)`、范围 `[-60,0]`。转换目标非空时会拒绝混写；当前转换器没有断点恢复入口。若转换中断，先检查原因和产物，不删除共享数据或旧缓存；如决定用新的派生目录重试，要同步更新配置。`prepare-data` 还会在本项目 reports 中写统计；显式 audit 将云端统计保存到独立输出目录。

demo 是 val 集 1 个病例、5 帧、7 条线，仍使用完整 500/450 步。检查 manifest、图像、实际选线和耗时后，再开始正式实验。

## 5. 正式评估与独立训练环境

```bash
casl-repro evaluate --config configs/autodl/evaluation.yaml
# 中断后，配置和权重不变时使用：
casl-repro evaluate --config configs/autodl/evaluation.yaml --resume
```

三方法和全部预算继承基线。恢复跳过完整病例，中断病例从头重跑。不要让其他训练作业与性能评测争用同一 GPU。

正式训练另外建立 `casl-train` 环境，不复用 `echocardmae` 或推理环境：

```bash
conda create -n casl-train python=3.10 -y
conda activate casl-train
python -m pip install -r requirements/training.txt
python -m pip check
casl-repro doctor --output /root/autodl-tmp/outputs_casl/preparation/environment-train.md
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU')); assert tf.config.list_physical_devices('GPU'), 'TensorFlow GPU unavailable'"
python -m pip freeze > /root/autodl-tmp/outputs_casl/preparation/training-resolved.txt
casl-repro train --config configs/autodl/training.yaml
# 中断后：
casl-repro train --config configs/autodl/training.yaml --resume
```

训练参数来自现有复现基线；`steps_per_epoch=10000` 是上游注释中的完整运行设置，尚不是从论文验证的值。云端正式训练前需结合训练协议核实，不能把此设置称为已复现论文训练预算。epoch 边界恢复包含模型/EMA/优化器，不保证恢复数据迭代器及随机流。详见 [训练协议](diffusion_training.md)。

训练后切回推理环境，单独输出结果：

```bash
conda activate casl-infer
casl-repro evaluate --config configs/autodl/evaluation.yaml --checkpoint /root/autodl-tmp/outputs_casl/training_fp32/hub --output /root/autodl-tmp/outputs_casl/trained_baseline
```

需要下游模型分割一致性时，在推理环境安装 `requirements/segmentation.txt`，再次 `pip check` 并记录依赖，再用独立 `--output` 加 `--segmentation`。不要混写到原先未启用分割的实验目录。

## 6. 备份与当前验收状态

备份前确认 `/root/autodl-fs` 实际挂载。保存 Git commit、完整配置、manifest、环境报告及 resolved 依赖、评估 CSV/图像/报告；训练保留整个训练输出目录，只有 `hub` 不能恢复优化器训练状态。完整轨迹体积大，可按实验打包后备份至项目专用备份目录，校验哈希再考虑后续磁盘管理。数据、模型、凭据、完整实验产物不进 Git。

本次已在 Windows 用合成文件清单验证路径保护、CSV/AVI 核对、旧缓存不回退替代、只读行为及配置继承；这不表示检查过真实云端数据。个人远程仓库已关联并推送；实例连接信息和真实 preflight 报告留待上云阶段补齐。
