# CASL 原生 PyTorch 与官方 JAX 对照（2026-09-22）

新增这项实验是因为出现了具体的工程线索：学长报告 4090 可超过 32 FPS。目前未收到对应代码、精度、迭代次数或计时边界，因此把它作为待验证主张。此前沿用官方框架是迁移成本决策，不是 PyTorch 无法提速的结论。

本次直接使用官方 EMA 权重，没有训练新模型，没有改动官方子模块，也没有切换现有实验默认后端。

## 已实现什么

- 从固定 Keras EMA 网络导出层拓扑、BN 统计与权重；转入原生 `torch.nn`，运行时没有 Keras/JAX 调用。不导出可执行 Lambda 字节码，不把整个模型当作 pickle 加载。
- NHWC→NCHW 权重/输入转换、channels-last 内存布局、卷积、BN、双线性上采样、正弦时间嵌入。保留原模型结构。
- `torch.func.grad_and_value` 的逐粒子 DPS：每个粒子独立求 L2 范数，再求和；不是把两粒子拼成一个范数，也不是 MSE。omega=10。
- 原 DDIM、初始 500 步、warm-start 的 `initial_step-1` 约定、3 帧观测缓存、2 粒子、熵与邻域抑制选线、choose_first 和官方“非零观测”硬投影。
- 三种 Torch 执行：eager、整段 DPS 单步与选线 `torch.compile(fullgraph=True)`、编译算子加完整 warm 帧 CUDA Graph。固定输入预热，图输出复制后归调用方所有，防止下一次 replay 覆盖前一次结果。

JAX 基线已有整段 recover JIT，不拿未编译 Python 循环作低效参照。双方 FP32，禁用 TF32；本批先隔离算子/图优化收益，没有用 FP16、少粒子或少 DPS 冒充等计算量加速。

50 步与 25 步是两个独立组，每一步都有 DPS。25 步组对应既有 `official25`，不是 `official25_fp16`；跨组变化属于采样算法设置变化，不能计作 Torch 框架收益。旧 FP16 路线的实验结论保留。

## 对照与判据

固定抽取官方 val 的 2 个病例，每例前 6 帧，seed=20260922，预算 14。抽样在看输出前完成；不按效果换样本，不使用 test。病例表保存于 `cases.json`，后续研究应把这些病例视为已用于工程调试，不当成全新确认样本。

首先在服务器比较真实 EMA 去噪、DPS 梯度、熵、选线。之后从原版实际轨迹保存每帧历史、掩膜、初始噪声和后验，再对 Torch 做同输入重放。噪声按原版 JAX 随机种子拆分重建；额外核对重放与原版后验、动作一致。不会只因为两边 seed 整数一样就认为输入相同。

浮点检查预设 atol=rtol=2e-4，选线精确一致。误差超限仍可以留下速度诊断，但不作为等价加速或 32 FPS 的通过证据，不自动放宽容差。

再运行 Torch 自己的连续闭环：下一帧使用自己的后验与动作，不能一直注入原版状态掩盖累计漂移。要报告“本批达到 32 FPS”，冷/热帧的数值与动作检查必须完整通过，连续动作一致、平均 MAE 不超过参照的 1.01 倍加 1e-4，并且含传输的稳定 FPS≥32。只有 2 个短视频，不能推出普遍实时性和医学质量结论。

## 三种速度，不能混用

1. **原仓库适配器速度**：实际原版 recover（含 RNG）与额外指标导出 wall time，跳过每段头两帧。
2. **同输入核心速度**：双方在 GPU 上接收同样的预生成输入，包含整个扩散、DPS、熵选线和投影；显式 GPU 同步，3 次重复。两边均不计 RNG、磁盘、传输。
3. **连续闭环含传输速度**：包含预加载数组到 GPU、观测历史更新、恢复、选线、投影、结果回传及同步；不含磁盘/NPZ 解码、显示、设备采集与 CPU 噪声生成。

FPS=帧数/同步总秒数，不能平均各帧的倒数。冷帧 500 步和编译/图捕获单独保存。首个 compiled 冷帧耗时包含单步编译，不能把它当成纯冷帧计算耗时。达到核心 32 FPS 而含传输未达到，不判作本批完整帧通过。

## 一次运行

沿用服务器已工作的 `casl` 环境和转换数据/权重。旧 GPU 实验结束后，在已有仓库执行；更新前查看 `git status`，有修改先保留处理，不强制覆盖：

```bash
cd /root/autodl-tmp/cognitive-ultrasound
git status --short
git pull --ff-only origin main
bash scripts/run_torch_casl.sh start
```

协调器先检查可用的 GPU PyTorch。依次尝试 `TORCH_PYTHON`、当前 casl Python、镜像 `/root/miniconda3/bin/python`，并记录实际版本。JAX 与 Torch 使用独立进程，可用不同 Python；不会为此在 casl 中安装或更换 CUDA。

如果镜像自带的 PyTorch 在 base，必要时明确指定：

```bash
TORCH_PYTHON=/root/miniconda3/bin/python bash scripts/run_torch_casl.sh start
```

Torch 解释器还需 numpy、h5py、PyYAML、psutil。缺这些轻量包时可用对应 Python 安装 `requirements/torch_benchmark_tools.txt`；这个文件没有 torch 或 NVIDIA 库。不自动替你装另一个 PyTorch，也不静默回退 CPU 测速。Linux 上 Inductor/Triton 的实际可用性在 compile 作业中检查，不支持时单独报错留档。

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/torch_casl_v1.console.log
# 各阶段详细输出：reference.log / eager.log / compile.log / graph.log
tail -n 40 -F /root/autodl-tmp/outputs_casl/torch_casl_v1/graph.log
```

固定四个阶段：原版参考 → eager → compile → CUDA Graph。累计子进程上限 90 分钟，每阶段最多 30 分钟；这是费用保护上限，不是预计时长，编译/预热计入上限。失败不追加更多候选或病例；原版失败则阻止后续比较。初始化、报告及打包另计，实例不自动关机。

```bash
bash scripts/run_torch_casl.sh status
bash scripts/run_torch_casl.sh stop
# 旧进程停止后，恢复原提交/配置/硬件；移除本批 STOP，再续跑：
rm -f /root/autodl-tmp/outputs_casl/torch_casl_v1/STOP
bash scripts/run_torch_casl.sh resume
```

完成的阶段跳过；未完成阶段整体重做（不是逐帧续跑）。已消耗时间仍计入上限，换卡/代码/依赖/数据需新输出目录。运行中不要 pull 或修改源码。不要与其它 GPU 作业并行抢卡测速。

## 看什么与下载什么

`REPORT.md` 有配对速度、原版实际速度、连续质量与 32 FPS 判据；`comparison.json` 保留机器可读结果，逐帧原始时间和一致性误差在各模式 JSON。固定第 3 帧生成图像对照。`identity.json` 记录 Git、源码、输入数据、权重、环境与 GPU 身份；EMA 导出另有校验。

```bash
cat /root/autodl-tmp/outputs_casl/torch_casl_v1/REPORT.md
```

结束时自动生成供下载的**结果包**（不是代码包）：

```text
/root/autodl-tmp/outputs_casl/torch_casl_v1.results.tar.gz
/root/autodl-tmp/outputs_casl/torch_casl_v1.results.tar.gz.sha256
```

可用 `bash scripts/run_torch_casl.sh report` 单独重建报告。未通过一致性、后端失败或超时的阶段会明确显示；不把 `completed` 当作科学假设成立。

## 来源与解释范围

算法对照依据仓库固定的 CASL `5f57aba...` 与 zea `192c0bb...`，实现来源是 `agent.py`、`diffusion.py`、`layers.py`、`selection.py`。这里的 native port 是项目新增实现，不声称已拿到学长代码。

PyTorch 官方提供 [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile) 和 [CUDA Graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) 机制，可以减少 Python 调度、算子启动开销并优化图；它们支持“值得实测”这个判断，并不保证任何特定模型达到 32 FPS。
