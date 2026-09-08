# Cognitive Ultrasound · CASL 复现基座

本阶段只复现 **Patient-Adaptive Echocardiography Using Cognitive Ultrasound**。保留官方算法，建立可审计实验接口。真实 EchoNet 数据、完整训练和论文数值验收需在你后续安排的云算力阶段完成。合成测试结果不能当作论文复现结果。

## 先读这些文件

- `docs/CASL_architecture.md`：论文、上游代码、执行计划的对应关系与必要纠正。
- `docs/diffusion_training.md`：训练目标、参数、AMP、断点语义。
- `docs/git_workflow.md`：本地 Git 配置位置、创建空远程仓库和关联步骤。
- `docs/autodl_preparation.md`：复用已有云端 EchoNet、只读检查、独立路径与后续运行顺序。
- **`docs/autodl_runbook.md`：当前推荐入口，从开实例、安装 requirement.txt 到完整评估、可选训练和下载结果。**
- `docs/reproduction_protocol.md`：数据划分、指标、预算、计时及验收标准。
- `reports/CASL_reproduction_report.md`：目前完成程度与尚未运行的项目。
- `reports/environment.md`：实际本地环境；不是 RTX 4090 报告。

## 目录

```text
cognitive-ultrasound/
  vendor/casl/                 固定提交的官方 CASL（Git 子模块）
    zea/                      固定提交的官方 zea
  src/cognitive_ultrasound/
    acquisition/              Sampler.select_action(observation, state)
    models/                   官方 diffusion 适配器、BeliefModel 协议
    evaluation/               指标、分割一致性、报告
    visualization/            每帧可视化
    data.py                   官方数据转换与严格校验
    experiments.py            多策略/预算评估与轨迹保存
    training.py               训练、checkpoint、TensorBoard
  configs/                    可运行配置和官方患者划分
  scripts/                    获取上游、集成验证
  tests/                      数据、算法接口、评估测试
  data/ checkpoints/ logs/ results/  本地大文件，Git 忽略
```

相邻的 `../CASL/casl` 是你原先下载的仓库，保持原样。你自己的 Git 仓库是本目录；子模块固定 CASL `5f57aba...` 和 zea `192c0bb...`。个人远程 `origin` 已关联 [MisakuraPW/cognitive-ultrasound](https://github.com/MisakuraPW/cognitive-ultrasound)，`main` 已推送并跟踪 `origin/main`。数据、模型与本地环境不随 Git 上传。

## 本地检查

Windows PowerShell（当前 `.venv` 已安装本地测试依赖）：

```powershell
$env:PYTHONUTF8 = '1'
.venv/Scripts/python -X utf8 -m cognitive_ultrasound --help
.venv/Scripts/python -X utf8 -m pytest -q
.venv/Scripts/python -X utf8 -m cognitive_ultrasound doctor
```

本项目位于中文路径，Python 3.10 下请使用 UTF-8 模式。本地验证环境通过 `--system-site-packages` 复用原有基础包，只在项目 `.venv` 中增加缺失依赖；不把它当作可迁移的 Linux 环境。

新机器先取得上游：

```bash
python scripts/bootstrap.py
```

脚本使用 HTTPS，避免官方 zea 子模块默认 SSH URL 需要额外凭据。官方 CASL/zea 的代码不在本项目内重写。

## 环境文件

针对当前 PyTorch / Ubuntu 22.04 / CUDA 12.8 基础镜像，优先按 `docs/autodl_runbook.md` 创建 Python 3.10 环境，先运行 `python scripts/bootstrap.py`，再从仓库根目录执行 `python -m pip install -r requirement.txt`。该入口复用系统 CUDA/cuDNN，不下载 NVIDIA 运行库；`requirements.txt` 为同一入口的别名。安装后必须通过 `scripts/check_gpu.py` 的 JAX/TF 实际 GPU 计算检查。

正式计算目标为 **Linux + Python 3.10/3.11 + NVIDIA GPU**。官方为 Keras：JAX 推理，TensorFlow 训练；不是 PyTorch 重实现。

`requirements/inference.txt` 与 `requirements/training.txt` 是此前通过 pip 安装 CUDA 运行库的可选方案；当前用户选择复用系统 CUDA，**不要与根目录 requirement.txt 混用**。根目录入口已包含分割桥，不需再安装 segmentation.txt。这些文件固定主要计算库，但不是已在 AutoDL 验证的完整 lock；记录实际解析版本与 GPU 检查结果。

## 数据与权重准备入口

你已有 AutoDL EchoNet 数据可继续使用，先核实共享目录中的 `Videos/*.avi`。已按交接文档准备 `configs/autodl/` 与 `scripts/autodl_preflight.py`；它们尚未在服务器执行。旧 MAE NPY 缓存不直接作为 CASL 极坐标输入。本项目不提供医学视频。

```bash
casl-repro fetch-assets --with-evaluation
casl-repro prepare-data --raw /absolute/path/EchoNet-Dynamic
casl-repro audit-data
```

官方 train/val/test 患者清单已经放在 `configs/splits/split.yaml`，不是 EchoNet 原始 FileList.csv 的划分。转换复用官方筛选、分割扇形和 cubic 极坐标插值；不能直接把原 AVI 的笛卡尔列当作 112 条扫描线。

## 运行顺序（留待真实数据/云端）

以下是通用接口示例。当前 AutoDL 全流程请使用 `docs/autodl_runbook.md` 和 `configs/autodl/paper.yaml`，避免默认全轨迹输出占满磁盘；该配置保留论文五个主要预算及全部病例/指标，另有命令补充 56/112 预算。

先用验证集的一个案例检查官方权重推理：

```bash
casl-repro evaluate --split val --limit-cases 1 --frames 5 --methods casl --budgets 7 --output results/pretrained_demo
```

三种方法、全部预算、正式测试集：

```bash
casl-repro evaluate --config configs/baseline.yaml
casl-repro evaluate --config configs/baseline.yaml --resume
```

下一步训练自己的先验，训练后将 `hub` 用作推理 checkpoint：

```bash
casl-repro train --config configs/training.yaml
casl-repro train --config configs/training.yaml --resume
casl-repro evaluate --checkpoint checkpoints/training/hub --output results/trained_baseline
```

可选 AMP、下游分割和独立剖析：

```bash
casl-repro train --precision mixed_float16 --output checkpoints/training_amp
casl-repro evaluate --segmentation --output results/baseline_segmentation
casl-repro evaluate --profile --limit-cases 1 --output results/profile
```

分割需要额外权重及 `tf2jax`。正式推理速度不应与剖析模式混用。默认预算 2/4/7/14/28/56/112；后两个预算是执行计划要求的扩展检查，官方论文脚本默认扫前五个。

## 输出和续跑

每次评估保存 `manifest.json`（完整配置、Git 状态、各模型及划分哈希、实际环境），每位患者保存 `frames.csv` 和完成标记；全局生成 `patients.csv`、`summary.csv`、`budget_quality.png`、`runtime_analysis.md` 和 `CASL_reproduction_report.md`。

轨迹在 `方法/lines_预算/病例/trajectory/frame_编号/state.npz`。里面有当前观测/掩码、完整时序后验粒子、当前粒子、重建、测量缓存、下一步掩码缓存、种子、熵和动作。`acquired_action` 属于当前帧；`selected_action` 属于下一帧。PNG 每 20 帧导出一次，可改 `visualize_every`。

`--resume` 只跳过有完成标记的患者；中断的患者从头重跑。配置、输入划分和模型权重变化时拒绝混写结果。训练恢复保存到上一个完整 epoch，恢复参数/EMA/优化器；随机流和数据迭代位置不保证与不中断训练逐位相同。

## 合成检查的边界

```bash
casl-repro smoke
python scripts/check_pretrained.py
casl-repro train --smoke
```

`smoke` 只测指标和产物序列化，不运行 CASL。`check_pretrained.py` 真正运行官方权重但只用合成输入、2 个扩散步骤；`train --smoke` 只做合成训练检查。所有这些产物均有明确标记，不进入论文结果表。

Windows 的 TensorFlow 对中文路径存在文件系统错误；若在本机检查训练，请使用 ASCII 临时输出，例如 `--output "$env:TEMP/casl-training-smoke"`。本次通过的训练产物已复制到 `checkpoints/synthetic_fp32` 和 `checkpoints/synthetic_amp`，均非正式模型。

下游桥接使用 `tf2jax==0.3.7`：官方 Docker 的 0.3.6 会引用 JAX 0.6 已删除的 `jax.core.ClosedJaxpr`。这里只调整依赖兼容性，不修改 CASL 的选线或重建算法。
