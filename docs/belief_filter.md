# 学长论文方法的初版实现

实现入口：`python -m cognitive_ultrasound.belief_filter`。
这是依据《基于自适应采样及图像生成的感知型超声方法研究》第四章写的
`thesis_inspired_bf_v1`，不是学长代码的恢复版本，也不是已经达到其指标的模型。
目前没有学长的源代码、codec 权重、先验预测器权重或完整超参数。
这条实验路线独立于原 CASL 的 diffusion 训练和 paper 全量评估；不会启动它们。

## 哪些来自论文，哪些是补充实现

| 论文描述 | 本实现 | 补充假设/限制 |
| --- | --- | --- |
| 4.4：由历史潜变量预测当前先验 | `Models.predict` | 小型残差卷积网络；时间输入采用 `t/(t+100)`，并非已知原结构 |
| codec 编码/解码 | `Models.encoder/decoder` | 输入 112×112×1，潜变量 28×28×8，卷积宽度 16；论文没有给出这些尺寸 |
| 4.5–4.7：观测投影、重编码及残差特征 | `Models.update` | 特征下采样用 4×4 平均池化；history 用当前帧已采集掩膜 |
| 4.8：门控潜变量修正 | 残差 U-Net 的 delta/gate 两个头 | 网络宽度及有界残差系数为工程选择 |
| 4.9：不确定性 | 第三个 log-variance 头，softplus 后上采样 | 这是学习的误差代理，不是经验证的后验熵 |
| 4.11：硬数据一致性 | 已测列逐像素覆盖重建 | 每帧重新清空观测和硬掩膜；跨帧只保留软状态 |
| 4.12–4.14：分组选线与空间去冗余 | 每帧 10+4 条唯一列，两次观测反馈 | 分数归一化、权重、空间核和确定性平局规则为工程选择 |
| 观测残差参与后续选线 | 只计算已测像素误差，再向邻列平滑传播 | 未测位置的真实误差不可用；传播方式是补充假设，不能声称论文原实现如此 |
| 4.6 节冻结 codec 和先验，只训练更新模块 | 第三阶段只更新 filter 参数 | 因为缺少预训练权重，先用第一、二阶段准备紧凑模型，不会把随机冻结网络称为预训练模型 |
| 重建、观测、潜变量、不确定性、投影损失 | `filter_loss` | 默认权重见 YAML；每组选线后监督，并在最终硬覆盖之前计算重建/观测损失，避免观测损失恒为零 |

训练时以连续三帧为一段。每段从空状态开始，帧间潜状态停止梯度；帧内两次更新可反传。
这是控制内存的初版选择，长时序能力还需要单独验证。真值只用于采集模拟器和监督损失，
策略不接收完整真值；不把未观测的当前帧或真值潜变量塞进历史状态。

第一阶段以整帧训练 codec；第二阶段用真实连续帧的潜变量监督先验预测器；
第三阶段冻结二者，在模型自己的闭环状态上训练 filter。第一、二阶段的监督输入
不能与第三阶段推理时的可用信息混淆。

## 先做有界试跑

依赖使用已有 `casl` 环境的 TensorFlow 2.20/Keras 和基础科学计算包，不增加新的 CUDA 安装步骤。
直接读取此前生成的 polar HDF5 `data/image`，范围必须为 [-60, 0] dB；不重新转换全量视频。

默认 `configs/belief_filter/pilot.yaml`：固定种子选择 32 个 train 病例、4 个 val 病例；
codec 100 步、prior 100 步、filter 200 步。每步读一个连续三帧片段；这不是全数据一个 epoch。
随后可在 4 个 val 病例各前 20 帧检查图像和指标。**这些步数仅用于排错和计时，不是收敛预算。**

下面的命令需在新增代码已经同步到云端仓库后执行。它们不负责同步本地未提交的代码。
输出目录需为空；已有结果会被保护。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
cd /root/autodl-tmp/cognitive-ultrasound
export OMP_NUM_THREADS=4
export KERAS_BACKEND=tensorflow

python -u -m cognitive_ultrasound.belief_filter train \
  --stage codec --output /root/autodl-tmp/outputs_casl/bf_pilot/codec &&
python -u -m cognitive_ultrasound.belief_filter train \
  --stage prior \
  --initialize /root/autodl-tmp/outputs_casl/bf_pilot/codec/checkpoint.npz \
  --output /root/autodl-tmp/outputs_casl/bf_pilot/prior &&
python -u -m cognitive_ultrasound.belief_filter train \
  --stage filter \
  --initialize /root/autodl-tmp/outputs_casl/bf_pilot/prior/checkpoint.npz \
  --output /root/autodl-tmp/outputs_casl/bf_pilot/filter &&
python -u -m cognitive_ultrasound.belief_filter evaluate \
  --checkpoint /root/autodl-tmp/outputs_casl/bf_pilot/filter/checkpoint.npz \
  --output /root/autodl-tmp/outputs_casl/bf_pilot/eval_val
```

需要断开终端时，先把训练命令块保存为脚本，再用 `nohup bash 脚本路径 > 日志路径 2>&1 &`。
每步打印 JSON；日志可用 `tail -n 40 -F 日志路径` 查看。
本地 CPU 功能检查给每条命令添加 `--cpu`，并用单独 YAML 指向本地数据。

## 暂停、恢复与训练预算

`--stop-after 25` 表示本次最多再运行 25 步，保存为 paused；父阶段未完成时不会开始下一阶段。
每 25 步保存一个 `checkpoint.npz`，含全部网络、当前优化器变量和步号，采用临时文件替换。
中断后同阶段、相同配置和数据可续跑，例如：

```bash
python -u -m cognitive_ultrasound.belief_filter train --stage filter \
  --output /root/autodl-tmp/outputs_casl/bf_pilot/filter --resume
```

意外中断最多重做上次 checkpoint 之后的步；暂不支持评估中途续跑，默认评估只有 80 帧。
恢复要求配置、选中数据的路径/大小/mtime 和划分一致。每个实验单独用输出目录；
不要通过修改已完成实验的 YAML 然后 `--resume` 来伪造延长训练。
正式加大预算时新建配置、新建输出目录；可以用同架构已完成 codec 初始化 prior，
用已完成 prior 初始化 filter。codec 的跨实验继续预训练暂未实现。

日志 `train_step_s` 包含片段读取和一次训练更新；`wall_s` 另包括该步触发的验证；
运行十步后打印基于最近最多 50 步的 `projected_remaining_minutes`。
该估计不含模型启动、checkpoint 写盘和中断，应结合实际日志判断。
评估 FPS 包括同步后的模型、策略和模拟观测耗时，排除指标计算与文件保存。

不能由参数少或单步快推出几小时一定收敛。应分别检查 codec 的未见病例重建、
prior 的预测、filter 的闭环重建，再决定增加病例/步数或修改结构。
没有这些结果时，不报告“达到学长论文质量”所需时间。

## 输出与比较边界

- 每阶段：`training.jsonl`、`status.json`、`checkpoint.npz`，记录配置、父权重 SHA256、划分 SHA256、病例列表和参数量。
- 评估：`manifest.json` 和各病例 `frames.csv` / `complete.json`；PSNR、SSIM 在 polar uint8 域计算，病例均值再汇总。
- 每病例前三帧：真值、稀疏观测、重建、已采集列、绝对误差、不确定性 PNG；
  `actions.json` 保存分组选线，`state.npz` 保存原始数组、每组重建和分数。
- 绝对误差图固定 0–2 映射到灰度 0–255；不确定性图固定 0–4.1 映射，原数组不截断。
  两种图不能互相当作同一量。

相同 checkpoint、相同病例和帧数下，可用 `--policy uniform`、新输出目录做均匀选线对照。
默认均匀策略在 14 个等距位置按从左到右先取 10 条、再取 4 条，也执行两次更新。
这只能比较该协议下的策略；用 greedy 轨迹训练的 filter 可能偏向其训练策略。

本实现同一帧内获得第一组反馈再选第二组，原 CASL 的时序协议不能直接假定相同。
未经统一观测时序、病例、帧、预算和指标，不把两者的数字或 FPS 放在一起声称优劣。
当前未实现下游分割/EF 指标、完整消融或学长的全部实验。

## 本地验证

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_belief_filter.py -q
```

测试包含未观测真值扰动不改变当前选线/重建、唯一选线预算、帧间清空硬观测、
冻结权重、梯度、三阶段训练、优化器续跑一致性以及评估输出。
合成样本测试和真实样本几步试跑仅验证程序行为，不能支持重建质量结论。
