# AutoDL 完整操作指南：CASL 的 EchoNet 复现

更新：2026-09-09。用户提供的实例日志已显示 JAX 和 TensorFlow GPU 卷积检查通过；全量数据转换、正式训练与评估尚待云端执行。路径沿用已有 EchoNet 交接约定，不修改 EchoRVM、旧缓存或共享原始数据。

## 已安装好环境：一次启动全流程

已完成安装并通过 GPU 检查时，直接在实例终端执行下面这段。不必重复创建环境、安装包或启动 tmux：

```bash
source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate casl &&
cd /root/autodl-tmp/cognitive-ultrasound &&
source /etc/network_turbo &&
git pull --ff-only &&
python scripts/autodl_overnight.py --start --with-training
```

看到 `Started PID ...` 表示后台进程已启动，可以断开 SSH。脚本依次执行 GPU/输入检查、获取权重、全量转换、完整数据审计、演示、评估 pilot 与耗时估算、官方权重正式评估、训练 pilot 与耗时估算、正式训练、自训练权重正式评估、打包。每一步成功才继续下一步，任何一步报错就记录失败并停止；无需手工接下一阶段。固定 `OMP_NUM_THREADS=8`，处理当前日志中的无效线程数警告。

`source /etc/network_turbo` 启用 AutoDL 内置学术资源加速，后台进程及其子进程会继承当前终端导出的代理环境。它用于访问官方 Hugging Face 权重与划分，仍固定原有仓库及提交版本，不重新下载 EchoNet AVI。[AutoDL 官方说明](https://www.autodl.com/docs/network_turbo/)。该服务不保证始终可用；本机检查无法证明实例上的连通性。

若旧任务在 `assets` 阶段出现 `[Errno 101] Network is unreachable`，且状态已经是 `failed`，在实例终端执行上面的完整启动命令即可重试。无需重装环境、清理数据或杀进程。对已经运行的后台任务，在另一个终端执行 `source` 不会改变它的环境。日志采用追加方式，重启后以最新 `STAGE` 时间和 `status.json` 为准，旧错误仍会保留。若启用加速后仍然下载失败，保留新的报错再排查；可再考虑官方文档列出的镜像或上传本地已下载并校验的资源。

**默认不自动关机。**如希望全流程完成或报错后自动关闭实例，在启动命令末尾加 `--shutdown-on-exit`；这会调用 AutoDL 的 `/usr/bin/shutdown`，也可能在早期检查失败时关机。它不释放实例。未加此参数时，流程结束后实例仍在运行，需要自行在控制台关机。强制杀进程、系统崩溃等不能保证执行自动关机。

全新转换要求派生数据所在盘至少 **180 GiB 可用空间**，否则在转换前停止；推荐数据盘总容量 300GB。这是保守启动门槛，不是压缩后大小保证。非空 polar 目录会直接完整审计，绝不自动覆盖或删除；如果上次只转换了一部分，会停止等待处理，不能当作已完成数据继续训练。

正式训练沿用 `500 epochs × 10000 steps`（500 万更新）的当前配置，pilot 不会替代正式训练，也不会根据耗时自动缩短预算。该预算仍不是已核实的论文总训练步数。全流程可能运行多天，估时文件保存在 `outputs_casl/preparation/evaluation-eta.json` 和 `training-eta.json`，生成后可查看。

启动后或次日登录实例时，查看状态与最近日志：

```bash
cat /root/autodl-tmp/outputs_casl/overnight/status.json
tail -n 60 /root/autodl-tmp/outputs_casl/overnight/run.log
```

`status=running` 时 `stage` 是当前阶段，`completed` 表示全部成功，`failed` 表示停止；将状态和日志末尾发回来即可定位。脚本有重复启动锁，已有流程运行时拒绝再开一份。修复失败原因后可重跑同一启动命令：评估按已完成病例恢复；训练有 manifest 时请求恢复完整 epoch 检查点，若尚无检查点则明确报错，不会静默重训。

全部成功后，结果包和单独的训练恢复包位于 `/root/autodl-tmp/casl_exports/`，同时生成 SHA256 和清单。`status.json` 记录实际包名。结果包默认不含模型和全量轨迹，训练包含 hub 与 resume；原始数据和转换后的 HDF5 不打包。后台日志和最终状态应以 `overnight/` 中的原文件为准，包内快照采集于打包时。按第 10 节下载，但将远端路径改为这里的 `casl_exports`。下载、校验或另行备份前不要释放实例。

以下分段指令仍可用于首次配置和人工排查；不要与已启动的后台流程同时运行。

## 0. 这次“完整复现”做到什么程度

建议先完成 **A：EchoNet 核心实验复现**，作为毕业设计后续创新的基线：固定官方代码/权重/划分，完整 500 例测试集，Random、Uniform、CASL 三方法，论文五个主要预算 2/4/7/14/28，每例最多 100 帧，PSNR/SSIM/LPIPS、下游分割一致性 Dice、实际时间、失败记录及报告。SSIM 是本项目增加的指标。正式配置为 `configs/autodl/paper.yaml`。

**B：从头训练先验后，重新完成同一评估**，是训练流程复现，需要额外算力。建议先完成 A，再执行本指南第 8 节。当前上游训练预算并未完全明确：本项目 500 epochs × 10000 steps 来自上游配置及注释，不是已验证的论文训练总步数。不能仅因训练跑满就宣称论文完全复现。

上述范围不包括论文的私有原始通道数据、在体采集硬件、3D Philips 数据，以及所有超参数消融。缺少这些数据时，准确称呼是“CASL 在 EchoNet 上的核心实验复现”。官方权重实验成功，并不代表已经从头复现了训练；合成 smoke 和 pilot 也不属于正式结果。

## 1. 开启实例和容量选择

- GPU：单张 RTX 4090 24GB；地区选择能挂载你已有 EchoNet 文件存储的地区。
- 基础镜像：你截图中的 PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8 可以作为起点。项目另建 Python 3.10 环境，预装 PyTorch 不参与 CASL 算法。
- 数据盘：**推荐 300GB 总容量**，不是额外加 300GB。预算紧时可先 200GB，但要保留后续扩容能力。此前 50GB 且最多再扩 20GB 的主机不适合本指南全量转换流程。
- 只配环境和少数病例试跑，50GB 可能足够；但本指南会转换全量原始视频，不能用这个小试跑容量直接运行第 5 节。

容量依据：发布清单约 139.5 万帧，官方转换同时保存 FP32 的 polar 与 image_sc，仅入选视频的两份数组未压缩约 140GB；还有 rejected 图像、HDF5 元信息、环境、模型、日志和余量。HDF5 默认 gzip，实际大小须 `du` 实测。原始 AVI 留在共享盘，无需复制或重新下载。

正式 paper 配置关闭全量 `state.npz` 和图片导出，保留所有逐帧指标和病例完成记录；demo 单独保留图像与完整轨迹。**不改变采样、重建或指标计算**。原 baseline.yaml 仍保留执行计划要求的 7 个预算和全轨迹能力。三方法 × 五预算 × 500 例 × 100 帧上限为 75 万帧次；只算 Np=2、W=3 的后验数组，未压缩也约 226GB，其余状态会继续增加占用。完整轨迹不是得出正式指标的必要条件。

## 2. 连接实例、下载代码

在 AutoDL 控制台复制 SSH 登录指令，在本机 PowerShell 执行；端口和域名以控制台为准，例如：

```powershell
# 替换 PORT 和 HOST；密码在 SSH 提示处输入，不写进脚本或仓库。
ssh -p PORT root@HOST
```

登录后的 Linux 终端执行：

```bash
cd /root/autodl-tmp
git clone https://github.com/MisakuraPW/cognitive-ultrasound.git
cd /root/autodl-tmp/cognitive-ultrasound
git log -1 --oneline
python scripts/bootstrap.py
```

`bootstrap.py` 获取固定提交的 CASL 和 zea。必须先执行它，再安装含本地 editable 包的依赖。若项目目录已存在，先查看 `git status`，干净时再 `git pull --ff-only`；不要覆盖目录或强制 reset。私有仓库使用你自己的 Git 凭据。

## 3. 创建环境、安装 requirement.txt

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n casl python=3.10 -y
conda activate casl
cd /root/autodl-tmp/cognitive-ultrasound
python -m pip install --upgrade pip --index-url https://pypi.org/simple
python -m pip install -r requirement.txt
python -m pip check
```

根目录的 `requirements.txt` 是同一入口的别名，任选其一，不必安装两遍。训练和推理共用这一环境的 Python 包，**每个命令是独立进程**，代码在导入 Keras 前选择后端；不要在同一 Notebook 内来回切换后端。GPU 训练和评估也不要同时运行。

依赖文件现在显式使用官方 `https://pypi.org/simple`，同时用于临时构建依赖，不修改机器全局 pip 配置。首个实例日志显示默认阿里云 HTTP 源找不到 zea 所需的 `poetry-core`，而官方 PyPI 可下载。手动把 poetry-core 装进 casl 环境不能解决原错误：pip 默认在独立的临时环境中再次安装构建依赖。[pip 构建隔离说明](https://pip.pypa.io/en/stable/reference/build-system/#build-isolation)

如果此前安装失败，保留已有环境，在项目目录 `git pull --ff-only` 后重新安装即可。不必重新克隆、创建环境或关闭构建隔离。**安装命令失败后出现 `No broken requirements found` 只说明已经安装的包没有声明冲突，不代表 requirement.txt 已安装成功。**重新安装成功后再执行 GPU 检查。

此入口固定主要数值库，使用 `jax[cuda12-local]==0.6.2`、普通 `tensorflow==2.20.0` 和 `tf2jax==0.3.7`，不请求 NVIDIA CUDA/cuDNN pip wheels。它安装 Python 包及 JAX GPU 插件，**不会安装 NVIDIA 驱动或替你补齐系统 CUDA/cuDNN**。这不是已在新镜像验证的完整依赖 lock；完成安装后保存实际 freeze。

系统依赖要点：JAX 0.6.2 支持本机 CUDA ≥12.1、cuDNN ≥9.1 且 <10，需要可见的 ptxas/nvlink/libdevice；TF 2.20 的官方测试组合为 CUDA 12.5/cuDNN 9.3。CUDA 12.8 的镜像仍需核对具体 cuDNN 和动态库路径，不能只看菜单就保证成功。[JAX 0.6.2 文档](https://github.com/jax-ml/jax/blob/jax-v0.6.2/docs/installation.md)、[TensorFlow 构建版本表](https://www.tensorflow.org/install/source#gpu)。

现在用同一套已安装的库做真实 GPU 卷积检查：

```bash
mkdir -p /root/autodl-tmp/outputs_casl/preparation /root/autodl-tmp/outputs_casl/console
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
python scripts/check_gpu.py
casl-repro doctor --output /root/autodl-tmp/outputs_casl/preparation/environment.md
python -m pip freeze > /root/autodl-tmp/outputs_casl/preparation/requirements-resolved.txt
git rev-parse HEAD > /root/autodl-tmp/outputs_casl/preparation/code-commit.txt
cp -r configs /root/autodl-tmp/outputs_casl/preparation/
```

**check_gpu 必须返回 0 且 JSON 中 passed=true。**它分别启动 JAX GPU 卷积、TensorFlow GPU 卷积及梯度计算，拒绝以 CPU 回退冒充通过。若失败，检查 `preparation/gpu.json`：镜像可能将 cuDNN 放在 base 环境的 site-packages，而新环境不可见；需要按真实位置修正库路径或通过系统包补齐兼容库。不要盲目拼多个 CUDA 目录，不使用跳过版本检查的变量，也不改回下载 CUDA wheels 的旧 requirements 文件。保留完整报错再处理。

## 4. 只读核对共享数据和磁盘

```bash
python scripts/autodl_preflight.py
df -h /root/autodl-tmp /root/autodl-fs
df -i /root/autodl-tmp
```

确认 `preflight.json` 的 `ready_for_conversion_inventory=true`，CASL 清单覆盖完整，确认实际挂载和剩余空间。它只检查 AVI 存在及大小，不证明视频可解码。默认原始目录为 `/root/autodl-fs/datasets/EchoNet-Dynamic`；旧灰度/RGB NPY 不会被用作 CASL 输入。

若原始目录不同，修改 `configs/autodl/paths.yaml` 的 raw_root，并同步修改下一节命令的 `--raw`。若代码/派生数据/输出目录变化，还须同步 evaluation.yaml、training.yaml 和所有显式命令；paths.yaml 不会自动重写运行配置。

## 5. 后台会话、转换全量数据与权重

长任务用 tmux，避免 SSH 断开导致退出。若镜像没有它，安装一次：

```bash
command -v tmux || (apt-get update && apt-get install -y tmux)
tmux new -s casl
```

进入 tmux 后执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
cd /root/autodl-tmp/cognitive-ultrasound
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true
set -o pipefail

time casl-repro prepare-data --raw /root/autodl-fs/datasets/EchoNet-Dynamic --output /root/autodl-tmp/datasets/CASL-EchoNet-polar 2>&1 | tee /root/autodl-tmp/outputs_casl/console/convert.log
```

转换成功后再执行：

```bash
casl-repro audit-data --data-root /root/autodl-tmp/datasets/CASL-EchoNet-polar --output /root/autodl-tmp/outputs_casl/preparation/dataset_statistics.md
du -sh /root/autodl-tmp/datasets/CASL-EchoNet-polar
casl-repro fetch-assets --with-evaluation
```

这一阶段 CPU 解码与 cubic 插值占主要时间，目前官方转换入口单进程，不能用 GPU 推理速度推算。全量转换可能耗时较长，日志中的实际处理进度才是依据。**已有完整 CASL 极坐标数据时只运行 audit，不重复转换。**输出目录非空会拒绝转换；中途失败没有转换续跑入口，先查清失败原因和剩余空间，再确定使用新的空目录，不能直接删除旧项目数据。

tmux 离开但保留任务：按 `Ctrl+B`，松开后按 `D`。重新 SSH 登录后用 `tmux attach -t casl` 回到任务。tmux 只能保护断线，实例关机会停止计算。

## 6. Demo 和真实数据计时 pilot

```bash
casl-repro evaluate --config configs/autodl/demo.yaml 2>&1 | tee /root/autodl-tmp/outputs_casl/console/demo.log
```

demo 是 val 集 1 例、5 帧、7 条线，完整 500/450 步，输出到 `pretrained_demo`，含图像及轨迹。它不启用分割，不能取代下面包含全部指标的 pilot。检查图像、采样掩码、报告及有限指标后，再运行：

```bash
time casl-repro evaluate --config configs/autodl/pilot.yaml 2>&1 | tee /root/autodl-tmp/outputs_casl/console/pilot.log
python scripts/run_tools.py estimate-evaluation --pilot /root/autodl-tmp/outputs_casl/pilot --config configs/autodl/paper.yaml --output /root/autodl-tmp/outputs_casl/preparation/evaluation-eta.json
```

pilot 使用 val 集 2 例、3 方法、5 预算、每例最多 100 帧，带 LPIPS 和分割一致性，总量最多 3000 帧次。新保存的 `case_wall_s` 包含实际指标计算和输出；算法自身的 `total_s` 不包含这些开销，不能拿它直接当租卡总时间。

`evaluation-eta.json` 给出按当前机器实测外推的全量小时数；不包含首次全数据哈希/模型加载、排队和故障恢复，不是保证。少量病例也可能不代表全量速度。更换 GPU、环境、模型或精度后应重新测 pilot。

## 7. A 阶段：官方权重完整测试集

```bash
time casl-repro evaluate --config configs/autodl/paper.yaml 2>&1 | tee /root/autodl-tmp/outputs_casl/console/paper-pretrained.log
```

出现断线后先看 tmux，不要重复启动进程。进程确实中断时，以相同配置恢复：

```bash
casl-repro evaluate --config configs/autodl/paper.yaml --resume 2>&1 | tee -a /root/autodl-tmp/outputs_casl/console/paper-pretrained.log
```

恢复跳过已经完整写出的病例组合，中断病例从头开始。配置、模型、划分或数据变化会拒绝混写结果。改变设置要另选输出目录，不能删除 manifest 来绕过检查。

完整输出在 `/root/autodl-tmp/outputs_casl/paper_pretrained`：

- `manifest.json`：状态、代码与模型/数据来源；最终应为 completed。
- `CASL_reproduction_report.md`：应显示 7500/7500 个病例/方法/预算组合完成。
- `frames.csv`（各病例子目录）、`patients.csv`、`summary.csv`：逐帧、患者平均及汇总指标。
- `budget_quality.png`、`runtime_analysis.md`：质量曲线与同步计时。
- 各病例 `complete.json`：实际预算不足、短视频、分割失败排除等记录。

分割 Dice 是完整图与重建图经同一模型分割后的**一致性**，不称为人工标注精度。报告不自动宣布达到论文水平；还需比对例如 7 条线的论文 PSNR 23.2dB，分析差异。可以在任务完成前用 `casl-repro report <输出目录>` 汇总已完成病例，但部分结果不能标记为完整复现。

如需逐模块耗时，另做一个小实验，不能把 profile 计时与正式 JIT 总耗时混在一起：

```bash
casl-repro evaluate --config configs/autodl/demo.yaml --profile --output /root/autodl-tmp/outputs_casl/profile
```

原执行计划额外要求的 56/112 条线可单独运行，保持同样的节省存储配置：

```bash
casl-repro evaluate --config configs/autodl/paper.yaml --budgets 56 112 --output /root/autodl-tmp/outputs_casl/extra_budgets
```

它额外增加约 40% 帧次，不包含在五预算 ETA 中。

## 8. B 阶段：训练先验和重新评估

这一阶段无需更换镜像或另装依赖；使用同一 `casl` 环境，独立进程自动切为 TensorFlow。先用真实数据做 2 epochs × 100 steps 的训练 pilot：

```bash
casl-repro train --config configs/autodl/training_pilot.yaml 2>&1 | tee /root/autodl-tmp/outputs_casl/console/training-pilot.log
python scripts/run_tools.py estimate-training --pilot /root/autodl-tmp/outputs_casl/training_pilot --config configs/autodl/training.yaml --output /root/autodl-tmp/outputs_casl/preparation/training-eta.json
```

pilot 权重不能用作正式训练复现结果，正式训练从头开始。计时器优先去掉首 epoch，再去掉每个测量 epoch 最初 5 步，计入验证和保存开销。正式配置现在为 **500 万次参数更新，batch size 32**；此训练预算来源有不确定性，先阅读 `docs/diffusion_training.md`，结合 ETA 决定是否执行这个预算或记录一个独立的预算实验。不要无意识地把它当成 500 次全数据遍历。

选择执行当前配置时：

```bash
casl-repro train --config configs/autodl/training.yaml 2>&1 | tee /root/autodl-tmp/outputs_casl/console/training.log
```

确实中断后：

```bash
casl-repro train --config configs/autodl/training.yaml --resume 2>&1 | tee -a /root/autodl-tmp/outputs_casl/console/training.log
```

每个完整 epoch 保存最新 hub、优化器恢复点（保留 3 个），每 50 epochs 及最后一个 epoch 额外归档权重。不同于原本每 epoch 都归档权重，这只改变存储频率，不改变训练更新。未完成一个 epoch 就中断时可能还没有恢复点。恢复参数/EMA/优化器，不保证随机流和数据迭代器逐位一致。

训练完成后，用训练所得 `hub` 重复完整测试：

```bash
casl-repro evaluate --config configs/autodl/paper.yaml --checkpoint /root/autodl-tmp/outputs_casl/training_fp32/hub --output /root/autodl-tmp/outputs_casl/paper_trained 2>&1 | tee /root/autodl-tmp/outputs_casl/console/paper-trained.log
```

如需续跑，在这条命令中添加 `--resume`。分别保留官方权重与自训练权重的结果表，不用后者覆盖前者。

## 9. 预计耗时：先预算，再用 pilot 替换

目前没有你的 4090 实测，不能承诺“几小时跑完”。以下是算术场景，不是性能测量：

| 一次五预算完整评估的平均每帧端到端时间 | 75 万帧次对应耗时 |
|---|---:|
| 0.05 秒 | 10.4 小时 |
| 0.2 秒 | 41.7 小时 |
| 1 秒 | 8.7 天 |

这还没加入转换、安装和首次检查。论文的极速吞吐涉及其他步数/精度/硬件设置，不能直接套给当前 500/450、FP32 的质量基线。

500 万训练步若实测 0.05 秒/步，纯更新约 2.9 天；若 0.2 秒/步，则约 11.6 天，再加验证/保存和自训练模型评估。A+B 总时间约为 **准备/转换 + 官方权重评估 + 正式训练 + 自训练权重评估**。按截图 2.18 元/小时仅作预算例子，连续运行一天约 52.32 元，另计实际存储等费用，价格以控制台为准。建议按需计费先测 pilot，再决定长跑安排。

## 10. 打包、备份和下载结果

先等对应实验完成或停止后再打包，避免得到写到一半的文件。确认共享文件存储已正确挂载：

```bash
findmnt -T /root/autodl-fs
df -h /root/autodl-fs
```

轻量结果包包含报告、指标、日志、配置快照、示例图片；默认排除模型/恢复点和 `state.npz`，不包含原始 AVI/派生 HDF5：

```bash
casl_export_tag=$(date -u +%Y%m%dT%H%M%SZ)
python scripts/run_tools.py export --source /root/autodl-tmp/outputs_casl --archive "/root/autodl-fs/projects/cognitive-ultrasound/casl-results-${casl_export_tag}.tar.gz"
```

记录脚本打印的实际 archive 路径。同时会生成 `.sha256` 和 `.json` 清单；同名包存在时拒绝覆盖。若需要完整备份训练恢复能力，另存一份：

```bash
python scripts/run_tools.py export --source /root/autodl-tmp/outputs_casl/training_fp32 --archive "/root/autodl-fs/projects/cognitive-ultrasound/casl-training-${casl_export_tag}.tar.gz" --include-checkpoints
```

这份训练包包括 hub 和 resume；只有 hub 能推理但不能恢复优化器训练。需要完整演示轨迹时，对 `pretrained_demo` 单独导出并加 `--include-trajectories`。

在**本机另开的 PowerShell**下载，替换 PORT、HOST 和文件名中的实际时间戳：

```powershell
New-Item -ItemType Directory -Force 'G:\科研项目\毕设\downloads'
scp -P PORT root@HOST:/root/autodl-fs/projects/cognitive-ultrasound/casl-results-TIMESTAMP.tar.gz 'G:\科研项目\毕设\downloads\'
scp -P PORT root@HOST:/root/autodl-fs/projects/cognitive-ultrasound/casl-results-TIMESTAMP.tar.gz.sha256 'G:\科研项目\毕设\downloads\'
```

`scp` 的端口参数是大写 `-P`，与 `ssh -p` 不同。下载完成后核验：

```powershell
$CaslArchive = 'G:\科研项目\毕设\downloads\casl-results-TIMESTAMP.tar.gz'
$CaslExpectedHash = ((Get-Content -LiteralPath "$CaslArchive.sha256").Trim() -split '\s+')[0]
(Get-FileHash -LiteralPath $CaslArchive -Algorithm SHA256).Hash -eq $CaslExpectedHash
```

返回 `True` 后再解压到新的本地目录查看报告。训练包用相同方式下载和校验。确认代码已推送、结果与必要模型备份并核验后，再在 AutoDL 控制台正常关机。退出 SSH 或 tmux 不会替你停止 GPU 计费；不要把释放实例当成关机，也不要因已下载结果就删除共享数据。

例如确认下面解压目录尚未存在后：

```powershell
$CaslExtract = 'G:\科研项目\毕设\downloads\casl-results-TIMESTAMP'
New-Item -ItemType Directory -Path $CaslExtract -ErrorAction Stop
tar -xzf $CaslArchive -C $CaslExtract
```
