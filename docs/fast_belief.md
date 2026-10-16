# 独立前馈信念滤波的训练与运行

本轮依据学长论文第四章实现最终的 dynamic per-frame belief filter，固定每帧10＋4条线。它自行训练 codec、prior 和更新器，不调用在线 DPS，也不调用 EF 选线梯度。目标是建立可训练、可续跑、能检查长序列退化的前馈基线。没有学长的源码或权重，因此这是独立重实现，达到论文质量和帧率需要服务器实验验证。

入口是 `scripts/run_fast_belief.sh`，配置是 `configs/fast_belief.yaml`。旧 `belief_filter`、CASL 和动态预算实验保持独立，不读取其模型状态或重启历史任务。

## 方法和补充选择

| 项目 | 采用的机制 | 本实现补充选择 |
| --- | --- | --- |
| 潜空间 | 编码器与解码器 | 确定性卷积自编码器；112×112×1 ↔ 28×28×16，基础宽度32，GroupNorm。没有复制旧 latent diffusion 模型 |
| 跨帧先验 | 从上帧潜状态预测当前状态 | 残差卷积网络，时间输入 t/(t+100)，有界增量；不是已知的学长原网络尺寸 |
| 当前帧状态 | 每帧重新清空硬观测，继承软记忆 | 继承潜状态、上帧重建、不确定性代理和线价值EMA；不把上帧采集列当本帧真值 |
| 每组更新 | 投影观测、重新编码、门控潜更新、解码 | U-Net式更新器；输入先验、投影潜变量、差值、池化掩膜、正负残差、绝对残差、历史图。历史图定义为本帧之前分组的观测掩膜 |
| 输出 | 已测列 hard projection | 逐像素覆盖，验证误差必须为零；损失用覆盖前输出，防止观测损失恒为零 |
| 主动选线 | 不确定性、变化、价值、残差与空间覆盖 | 各分数按病例归一化；组内贪心去冗余，组间排除重复列。第一组只用历史；第二组加入第一组观测残差；不看完整当前帧 |
| 不确定性 | softplus后上采样 | 用异方差误差目标监督其原始log尺度，是选线代理，不是经校准的后验熵 |
| 损失 | 重建、观测、潜空间、不确定性与投影 | 具体权重和归一化见YAML；重建为L1＋SSIM；不确定性项允许负值，不能按曲线绝对大小直接比较不同阶段 |
| 确定性 | 固定FP32，关闭TF32，启用确定性算法 | 双线性上采样写成固定的两个线性映射，避免CUDA双线性反向的非确定性报错；CPU已核对值和梯度。见[PyTorch官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html) |

选线使用GPU批量张量计算，不逐个病例调用 `.item()` 决定动作。离散选线不求梯度；更新器通过实际选线后的自身历史训练。冻结codec不意味着把解码器包进 `no_grad`：filter训练仍需经过冻结解码器回传到潜状态。

## 数据与训练预算

需要已有转换结果和原始 `FileList.csv`，不重新转换视频。病例必须同时属于CASL对应划分和EchoNet原始对应划分，避免原始TEST病例进入训练。每个阶段共用启动时锁定的名单，不根据指标重新选病例。预选视频不足24帧、数据范围错误或损坏时明确失败，不悄悄换病例。

默认 **1024 TRAIN / 32 development VAL / 32 TEST**。这是比历史小试验更正式的一轮训练，**不是整个EchoNet训练集**。本轮不自动从1024扩大至全量。需要全量时，运行前复制配置、把 `cohorts.train` 改为 `null` 并使用新输出目录；不要在已经开始的批次上修改配置。

| 阶段 | 默认上限 | 每次更新 | 训练内容与选择标准 |
| --- | --- | --- | --- |
| codec | 30 epoch，至多3840次更新 | 8视频×16帧 | 学表示和重建；development浮点MSE选best |
| prior | 20 epoch，至多2560次更新 | 8视频×5连续帧 | 冻结codec，从真实首帧潜状态递归预测后4帧；验证同时保留一步预测、复制前帧和长递归诊断 |
| filter | 40 epoch，至多5120次更新 | 8视频×24连续帧 | 冻结codec/prior，先8帧自身闭环burn-in，再监督16帧，每8帧截断梯度；完整development视频选best |

这里一个epoch是每个入选视频抽一个连续片段，**不是遍历全部帧**。随机片段起点随epoch变化且可重放；每个batch做一次Adam更新，TBPTT多个窗口累积到这一次更新。三个阶段按顺序训练，后一个阶段继承前一个的best；不完整重训CASL。首次及每2个epoch验证一次，连续6次验证未改善则提前停止。上限和早停都不等于保证收敛。

本轮长序列检查不会每几帧清空状态。最终在完整development和TEST视频上比较：贪心10＋4、同一更新器的均匀10＋4、固定14条线插值。另对学习策略的实际观测位置计算同观测插值MSE。均匀掩膜不是该更新器的训练分布，因此这组是诊断对照，不把全部差异直接归因于选线价值。

主指标是连续[-1,1]极坐标PSNR、Gaussian SSIM、MSE、MAE和未观测区域MAE，另记录早/晚段误差、硬投影误差、实际线数、每帧耗时。先按帧求病例平均，再病例等权聚合。它们不直接等于学长表格或旧uint8图像指标。EF默认关闭；当前目标先验证成像基线，未评估EF不会写成“EF保持不变”。

## 服务器一次启动

推荐单卡24GB及以上，16 vCPU、64GB及以上内存；可先用4090，也可用5090。选择已有CUDA版Torch的镜像，例如PyTorch2.8 / Python3.12 / CUDA12.8。脚本继承镜像Torch建立独立虚拟环境，不重装CUDA，不激活原来的 `casl` 环境。

已有仓库时执行：

```bash
cd /root/autodl-tmp/cognitive-ultrasound
git pull --ff-only origin main
source /etc/network_turbo
bash scripts/run_fast_belief.sh start
tail -n 40 -F /root/autodl-tmp/outputs_casl/fast_belief_v2.console.log
```

新实例没有仓库时，先执行以下两行，再执行上面的启动段：

```bash
cd /root/autodl-tmp
git clone https://github.com/MisakuraPW/cognitive-ultrasound.git
```

默认数据路径为 `/root/autodl-tmp/datasets/CASL-EchoNet-polar`，CSV路径为 `/root/autodl-fs/datasets/EchoNet-Dynamic/FileList.csv`。如果CSV实际位于 `/autodl-fs/data/datasets/...`，在启动前复制一份配置修改 `file_list`，并设置：

```bash
cp configs/fast_belief.yaml configs/fast_belief.server.yaml
# 编辑上述副本中的 data_root / file_list；不修改已经运行的配置
export FAST_BELIEF_CONFIG=configs/fast_belief.server.yaml
bash scripts/run_fast_belief.sh start
```

镜像Python不在默认位置时设置 `FAST_BELIEF_BASE_PYTHON=/实际路径/python`。先用它运行 `-c 'import torch; print(torch.cuda.is_available())'` 确认继承的是CUDA版Torch。依赖安装失败会退出；不会偷偷转CPU跑正式实验。

首次启动自动完成环境建立、数据索引、独立短测、三个阶段训练、最终评价、报告与校验打包。`nohup`后台运行，断开网页终端不影响实验；不开自动关机。

## 短测与速度解释

短测使用新建模型，与正式检查点隔离。先对codec/prior各做一次真实更新，再对实际贪心filter工作负载比较2/4/8个Torch CPU线程，每项5次真实更新，以后3次中位时间选配置。候选必须通过损失、训练后参数和短闭环状态2e-4容差，以及离散选线完全相同的检查。用真实HDF5比较0与4个读取进程，核对输入内容、顺序和起点完全一致。没有“显存越满越好”的规则，不自动修改batch或训练预算；OOM明确失败，需要新配置重开。

这是有限候选的运行参数选择，不保证全局最快或GPU/CPU始终100%。`probe.json`报告短测显存和投影时间。codec/prior只测首更新，filter只测短段，因此投影只是当前训练上限的粗估，**不含反复验证、最终评价、初始化和是否收敛的不确定性**。推理计时分别保留同步暖帧、冷帧、读取传输和完整任务墙钟时间；不拿训练速度或包含质量计算的时间冒充帧率。

## 进度和续跑

```bash
bash scripts/run_fast_belief.sh status
bash scripts/run_fast_belief.sh stop
# 等 status=stopped、alive=false 后才关机
bash scripts/run_fast_belief.sh resume
```

日志依次出现 `INDEX`、`PROBE_STAGE`、`PROBE_CANDIDATE`、`PROBE_COMPLETED`、`TRAIN`、`VALIDATE`、`VALIDATION_END`、`STAGE_COMPLETED`、`EVALUATE`、`COMPLETED`。每个TRAIN包含阶段、epoch、更新数、当轮视频进度、损失分量、梯度范数、时间、RAM和显存。最终只有 `status=completed, phase=export_verified` 才表示结果包也已校验完，可结束本轮。

每25次更新及epoch边界原子保存last；best另存。正常stop会保存最近完成的更新。断电最多重算尚未写入检查点的更新；日志会回滚其未提交尾部。恢复包含模型、Adam、CPU/CUDA RNG、epoch、视频游标、固定片段起点、验证早停状态。CPU合成测试已验证三阶段连续训练与中途恢复的参数、Adam及RNG完全一致；GPU跨环境逐位重放不作保证。数据、代码、配置、GPU/驱动或运行环境变化时拒绝混写，使用新目录。

最终评价按病例提交，可跳过已完成且产物SHA256通过的病例；中断时最多重跑当前未提交病例，不缓存完整轨迹以逐帧续跑。

## 存储和结果下载

输出目录 `/root/autodl-tmp/outputs_casl/fast_belief_v2` 只保留：

- codec/prior/filter各自的 `last.pt`、`best.pt`，共6个模型/优化器检查点；不逐更新复制。
- 配置、环境、固定名单、真实训练JSONL、验证记录、逐病例/帧指标。
- 每组前3病例各4帧的小型PNG/NPZ样例，含真值、重建、误差、掩膜、不确定性；不保存整个训练视频或状态轨迹。
- `REPORT_中文.html`、训练曲线PNG/PDF、病例CSV和汇总JSON。

初始默认至少20GiB空闲；低于2GiB或输出超过10GiB时停止并保留证据，不擅自删除旧结果。压缩包在输出目录旁边，额外占用少量空间：

```text
/root/autodl-tmp/outputs_casl/fast_belief_v2.analysis.tar.gz
/root/autodl-tmp/outputs_casl/fast_belief_v2.models.tar.gz
```

前者用于报告和网页GPT讨论，后者用于模型恢复，分别校验所有成员及整包SHA256。`fast_belief_v2/export.json`记录路径、大小和哈希。不包含CASL、EF预训练权重，也不包含70GB数据集。用户手动下载；不自动删除原始结果。

## 可选EF与本地验证范围

如需最终EF评价，必须在启动前显式开启 `evaluation.ef_enabled`，提供已有冻结EF权重、TRAIN归一化统计和对应极坐标变换坐标；配置继承现有task-budget EF读取协议。只做全起点32帧/间隔2的视频读出，不训练EF、不做任务梯度选线、不自动下载资产。身份记录会锁定资产SHA256。默认不启用这部分，避免成像训练被旧实验资产阻塞。

本地测试使用原生Torch CPU和真实HDF5合成样本，覆盖因果观测、两组去重、每帧重置、硬投影、所有阶段真实梯度、冻结模块、三阶段中断恢复、HDF5多进程内容/顺序、数值拒绝、端到端评价/打包、完成后免重复运行、归档校验和修改身份拒绝。双线性固定映射另核对前向与梯度。CPU报告明确标注功能测试；真实GPU兼容性、显存、耗时、长序列质量和可选EF仍由启动短测及正式实验给出。
