下面这份是给 **CODEX 执行用** 的，不是给人看的调研文档。目标是让 CODEX 明确：

* 先不要创新；
* 先建立 CASL 复现基线；
* 复现不仅是跑结果，还要整理成后续科研框架；
* 保留后续 WM / Agent / planning 的扩展接口。

我按 CODEX 比较容易执行的形式写。

---

# CASL 复现计划（Codex Execution Plan）

## 项目目标

完成 **Patient-Adaptive Echocardiography Using Cognitive Ultrasound (CASL)** 的本地复现，并建立可扩展实验框架，为后续：

* belief state 建模；
* World Model；
* planning；
* task-driven acquisition；
* dynamic budget sampling

提供代码基础。

本阶段目标：

> 在 EchoNet-Dynamic 数据集上复现 CASL 的核心 perception-action loop，包括 temporal diffusion reconstruction、posterior uncertainty estimation、scan-line active selection，并达到论文级 baseline。

不进行创新。

---

# Phase 0：环境与代码审计

## 目标

理解官方实现结构，确认运行环境。

## Tasks

### 1. 获取官方代码

检查：

* repository structure
* requirements
* pretrained checkpoints
* dataset preprocessing scripts

记录：

```
CASL/
├── diffusion
├── dataset
├── sampling
├── evaluation
├── utils
└── configs
```

（根据实际仓库调整）

---

### 2. 创建实验环境

目标环境：

```
Python >=3.10
PyTorch
CUDA
CUDA compatible with RTX4090
```

确认：

```bash
python
torch.cuda.is_available()
nvidia-smi
```

记录：

* GPU型号
* CUDA版本
* PyTorch版本
* 显存占用

---

### 3. 输出环境报告

生成：

```
reports/environment.md
```

内容：

* hardware
* software
* dependency versions

---

# Phase 1：理解 CASL 方法流程

## 目标

不要直接运行。

先建立代码-论文对应关系。

输出：

```
docs/CASL_architecture.md
```

内容：

---

## Overall pipeline

整理：

```
EchoNet video

        |
        v

Sparse scan-line observation

        |
        v

Measurement buffer

        |
        v

Temporal diffusion posterior sampling

        |
        v

Posterior particles

        |
        v

Entropy estimation

        |
        v

K-Greedy entropy minimization

        |
        v

Next scan lines
```

对应论文：

Perception:

$$
p(x_t|h_t)
$$

Action:

$$
a_t=\arg\max H(y|a,h)
$$

---

# Phase 2：数据准备

## 目标

复现 EchoNet 数据流程。

## Tasks

确认：

* train split
* validation split
* test split
* preprocessing

论文设置：

* image size:

$$
112\times112
$$

* temporal window:

$$
W=3
$$

---

## 输出

生成：

```
data/
├── train
├── val
└── test
```

并记录：

```
dataset_statistics.md
```

包含：

* number of videos
* frames
* resolution
* preprocessing

---

# Phase 3：运行官方 pretrained inference

## 目标

先不训练。

使用官方 checkpoint 跑 inference。

验证：

* reconstruction image
* selected scan lines
* entropy map

输出：

```
results/pretrained_demo/
```

每个case保存：

```
frame_x/

├── ground_truth.png
├── sparse_observation.png
├── reconstruction.png
├── entropy.png
└── selected_lines.png
```

---

# Phase 4：复现 diffusion training

## 目标

训练自己的 CASL diffusion model。

## Model

确认：

* temporal diffusion
* U-Net backbone
* noise schedule
* loss

记录：

```
docs/diffusion_training.md
```

包含：

* input shape

例如：

$$
(B,W,H,W)
$$

* training objective:

$$
L=
||\epsilon-\epsilon_\theta(x_t,t)||^2
$$

---

## Training setup

默认：

RTX4090

要求：

* AMP mixed precision
* checkpoint saving
* tensorboard/wandb

输出：

```
checkpoints/
logs/
```

---

# Phase 5：实现 CASL closed-loop evaluation

## 核心任务

完整复现：

```
for each frame:

    acquire initial lines

    while budget not reached:

        reconstruct posterior

        generate particles

        estimate entropy

        select next lines

    output reconstruction
```

---

## 必须保存中间状态

未来研究需要。

保存：

```
trajectory/

frame_t/

├── observation_mask
├── observation
├── reconstruction
├── posterior_samples
├── entropy_map
└── selected_action
```

---

# Phase 6：Baseline实现

必须加入：

## 1. Random sampling

随机选择scan lines。

## 2. Uniform sampling

固定间隔。

## 3. CASL

官方方法。

统一接口：

```python
Sampler.select_action(
    observation,
    state
)
```

方便未来：

```
WM sampler
Planning sampler
Task sampler
```

直接替换。

---

# Phase 7：Evaluation Framework

不要只复现PSNR。

建立统一evaluation。

---

# Reconstruction metrics

实现：

## PSNR

## SSIM

## LPIPS

---

# Acquisition efficiency

绘制：

```
number of lines

        vs

quality
```

预算：

```
2
4
7
14
28
56
112
```

---

# Downstream task evaluation

加入：

LV segmentation。

流程：

```
reconstruction

        |

EchoNet segmentation model

        |

Dice
```

比较：

```
GT image

CASL reconstruction

Random reconstruction

Uniform reconstruction
```

---

# Runtime evaluation

拆解：

记录：

```
total time

=
diffusion sampling

+
entropy estimation

+
action selection

+
projection
```

输出：

```
runtime_analysis.md
```

---

# Phase 8：实验报告生成

最终生成：

```
CASL_reproduction_report.md
```

结构：

---

# 1. Environment

硬件：

RTX4090

---

# 2. Dataset

EchoNet

---

# 3. Method

CASL pipeline

---

# 4. Reconstruction results

表格：

| Method  | PSNR | SSIM | LPIPS |
| ------- | ---- | ---- | ----- |
| Random  |      |      |       |
| Uniform |      |      |       |
| CASL    |      |      |       |

---

# 5. Sampling efficiency

曲线：

budget vs performance

---

# 6. Runtime

表格：

| Module    | Time |
| --------- | ---- |
| Diffusion |      |
| Entropy   |      |
| Action    |      |
| Total     |      |

---

# 7. Reproduction gap analysis

分析：

* 与论文差距；
* 速度差异；
* 参数差异；
* 数据处理差异。

---

# Phase 9：代码结构重构（为后续创新准备）

最终代码不要写死 CASL。

目标结构：

```
cognitive_ultrasound/

├── models/
│   ├── diffusion.py
│   ├── belief.py
│   └── wm.py

├── acquisition/
│   ├── random.py
│   ├── uniform.py
│   ├── casl.py
│   └── planner.py

├── evaluation/
│
├── configs/
│
├── experiments/
│
└── visualization/
```

未来创新只需要替换：

```
belief module

or

action module
```

---

# 完成标准（Definition of Done）

认为 CASL 复现完成，需要满足：

## 必须：

✅ 官方代码运行成功
✅ EchoNet 数据流程跑通
✅ diffusion inference成功
✅ reconstruction可视化
✅ entropy map生成
✅ active scan-line selection运行
✅ random/uniform/CASL比较
✅ PSNR/SSIM/LPIPS计算
✅ runtime profiling

## 推荐：

✅ LV segmentation downstream evaluation
✅ 保存完整trajectory
✅ 建立统一Sampler接口

---

# 注意事项

1. 不要修改核心算法。
2. 不要提前加入WM。
3. 不要优化速度。
4. 不要改变measurement model。
5. 所有实验必须可复现。

当前阶段唯一目标：

$$
\boxed{
\textbf{完全理解并复现 CASL perception-action loop}
}
$$

---

这份执行完成以后，下一阶段才开始：

```
CASL limitation analysis

↓

belief state analysis

↓

World Model / Planning proposal

↓

paper idea
```

这样后面的创新会建立在一个可靠基线上，而不是概念推演。
