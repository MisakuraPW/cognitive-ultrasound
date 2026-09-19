# 准备性实验：一键运行说明（2026-09-19）

这套代码回答“哪些机制值得继续做、能否以更低成本验证想法”，不启动全测试集 paper 评估，也不从头训练 CASL diffusion。学长路线是已有论文启发实现的有界试验，不是已恢复学长的原代码。不会改动 `vendor/casl`。

## 你现在需要上传什么

本地 `G:\科研项目\毕设\outputs\preparation_20260919` 中的两个文件：

- `cognitive-ultrasound-preparation-20260919-v1.tar.gz`
- `cognitive-ultrasound-preparation-20260919-v1.tar.gz.sha256`

上传到服务器 `/root/autodl-tmp/`。这个包只有源码、配置、测试和说明，包含尚未提交 Git 的学长式模型代码，不包含原始视频、70 GB 转换数据或大权重。`.sha256` 使用 Linux LF 换行，避免此前 Windows 回车导致找不到文件的问题。

现有服务器应保留：

```text
/root/miniconda3/envs/casl
/root/autodl-tmp/cognitive-ultrasound/vendor/casl
/root/autodl-tmp/cognitive-ultrasound/checkpoints/official
/root/autodl-tmp/datasets/CASL-EchoNet-polar/{train,val,test}
```

开 **有 GPU 的实例**，并确认旧 paper/训练任务已经结束或由你停止。新任务应独占这张卡，才能比较速度。不要为这个批次重新安装 PyTorch、TensorFlow、JAX 或 CUDA。已验证的 `casl` 环境就是本批次的起点。

## 一次粘贴启动

上传完后，在服务器终端粘贴整个代码块：

```bash
(
set -e
cd /root/autodl-tmp
sha256sum -c cognitive-ultrasound-preparation-20260919-v1.tar.gz.sha256
test ! -e cognitive-ultrasound-preparation-20260919-v1
tar -xzf cognitive-ultrasound-preparation-20260919-v1.tar.gz
cd cognitive-ultrasound-preparation-20260919-v1
bash scripts/run_preparation.sh start
)
```

它在独立目录解包，链接旧仓库的 vendor 和 checkpoints；Python 使用新目录的 `src`，不依赖 `git pull` 或重新安装 editable 包，不覆盖旧实验。若目录已存在，第一条启动块会停止，直接用后面的“续跑”命令即可。

启动器先在独立进程中检查 JAX 和 TensorFlow GPU 卷积，再进入流水线；失败则停下，日志写明原因。`nohup` 后台运行，关闭 SSH 窗口不影响任务。不会自动关闭实例，任务结束后 AutoDL 仍计费，需要你检查结果后自行关机。

## 看进度、停止和续跑

总日志显示当前任务、剩余上限和任务结束状态：

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_v1.console.log
```

看到例如 `"job": "debug_reference"`，可在另一个窗口看该任务每帧输出：

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_v1/jobs/debug_reference/console.log
```

第一帧含模型加载/冷启动/JIT，可能较久；有总任务时限看守，不会无限等待。`Ctrl+C` 退出 tail，不会停止后台实验。

```bash
cd /root/autodl-tmp/cognitive-ultrasound-preparation-20260919-v1
bash scripts/run_preparation.sh status
```

需要暂停实验时：

```bash
cd /root/autodl-tmp/cognitive-ultrasound-preparation-20260919-v1
bash scripts/run_preparation.sh stop
```

协调器发现 STOP 后终止当前 worker，保存清单、生成报告和结果包。CASL 已完成帧不重算；BF 恢复模型和优化器到最近 checkpoint（默认最多回退 50 个更新）；分支按单次分支、BF 评估和预算闭环按整病例恢复。TBIG 按整病例恢复。

普通断线无需重启；确实中断后，确认旧进程已结束，再运行：

```bash
cd /root/autodl-tmp/cognitive-ultrasound-preparation-20260919-v1
rm -f /root/autodl-tmp/outputs_casl/preparation_v1/STOP
bash scripts/run_preparation.sh resume
```

默认不重复已经明确失败的任务。若只是修复了外部环境/临时资源问题，且代码、配置、数据和权重未改变，可以显式重试失败项：

```bash
PREPARATION_RETRY_FAILED=1 bash scripts/run_preparation.sh resume
```

同目录有运行锁；若协调器被强杀但旧 worker 仍活着，程序会报告 PID 并拒绝重复启动。不要反复运行启动命令来“催进度”。

时间账本在续跑时继续累计。到达上限的任务不会因为 resume 就重新得到预算。代码、配置、权重或数据变更也不能混入旧结果；需要有意识地安排新配置、新输出目录，不能把调参后重复实验伪装成同一次运行。

## 默认实际执行什么

首轮把此前计划的大规模上限缩为更实际的小样本：验证集 debug 2 病例×8 帧、development 4 病例×16 帧、confirmation 4 病例×24 帧；训练集 8 病例。CASL 风险轨迹每个训练病例前 16 帧，BF 从这些训练病例完整视频中随机取连续 3 帧片段。三个验证子集互斥，且均不接触 test。

病例先按固定种子 20260919 选择，不能看结果后换病例。首轮轨迹用种子 42；分支及条件闭环用 42、31415。若预选病例缺失、格式错误或帧数不足，入口会报出病例名，不静默换成更容易的病例。启动只检查所选文件头及少量帧，逐帧读取时再验证数值；不再全量扫描/转换 10030 个文件。

| 终端任务名 | 做什么 | 为后续什么决定准备 |
|---|---|---|
| `debug_reference` | 原 CASL 权重、500 步冷启动/50 步 SeqDiff、FP32、W=3、2 粒子、14 条线 | 小规模正确性、冷/热态成本、原始可解释轨迹 |
| `debug_wrapper`、`parity` | 原生 recovery 与既有 wrapper 的逐帧数值及下一动作对比 | 查封装是否改变了算法，而非复核整篇论文 |
| `debug_profile` | 拆出 posterior sampling 与选线时间 | 判断真正的耗时位置；拆分计时可能改变 JIT，不能当原版总速度 |
| `fixed_*` | 从同一个原版历史、观测、随机状态出发比较采样器 | 区分采样器误差与闭环历史累积误差 |
| `debug_official25` | SeqDiff 改用后 25 步，FP32 | 优先试上游支持的短时推理配置；其起始噪声也改变 |
| `debug_official25_fp16` | 25 步加混合精度 | 验证速度收益和质量/不确定性损失 |
| `debug_sparse10` | 保留 50 次反向步，仅 10 个均匀分布位置计算 DPS 梯度（含首尾） | 直接验证“少做 DPS 能否更快且保持效果” |
| `confirm_reference`、`confirm_候选` | 只确认开发筛选出的一个候选，独立病例配对 | 决定是否允许日常小实验使用该加速版 |
| `bf_codec`、`bf_codec_qualify` | 紧凑编码器/解码器最多 1000 更新，检查重建误差及 resize 基线 | 避免把随机或尚不能重建的 codec 冻结下来 |
| `bf_prior`、`bf_prior_qualify` | 最多 500 更新；和复制上一帧/复制 codec 结果比较 | 判断时间先验是否至少不明显劣于简单持久性基线 |
| `bf_filter`、`bf_filter_qualify`、`bf_filter_confirm` | 冻结前两阶段，更新模块最多 1000 更新；统一固定观测，与独立 prior+projection 和空间插值比较 | 判断“新增小模块”是否值得深入；不冒充学长原结果 |
| `tbig` | 若独立官方安装、数据、权重与配置齐备，做 1–2 病例任务输出诊断；否则记录缺项 | 对接任务信念与任务输出，不拿不兼容 EchoNet 权重假装 TBIG |
| `branches` | 2 病例×2 状态，7/14/28 条线单帧预算分支；至多 4 组去重后的候选动作、2 种子、2 帧短分支 | 提前看预算边际收益和局部短视的可观测机会 |
| `risk_train/development/confirmation`、`risk_probe` | 训练集拟合轻量岭回归，预测下一帧未测区误差；与常数/当前不确定性/残差/变化量基线比较 | 看可用在线信号是否含有未来风险信息，确认预测主线的实验可行性 |
| `closed_loop` | 只有分支完整且风险探针过门槛才跑小闭环；同一空间采样规则、不同预算策略 | 检查动态预算接口与实际采集量；这是探索性 MVP，不是创新点定稿 |

独立确认轨迹会复用于风险分析，不为同一批结果重复推理。未通过固定历史检查的候选不能获得“可替换原版”的资格。未通过 BF 前置质量检查则跳过后续训练；失败不说明整个研究方向不可行，可能只是有限预算、结构假设或训练不足。

额外已实现但不默认遍历的 `sparse5`（50 反向步/5 DPS）及 `coarse10`（保持原 warm 起点与时间区间、10 个较大反向步/10 DPS）在 `configs/preparation_expanded.yaml` 中。所有自定义加速都保留原来的冷启动求解器，不把冷启动 500 步偷偷减掉。

## 如何限制成本与解释结果

实验子进程累计上限 12 小时：A（CASL/加速）6 小时，B（学长路线）3 小时，C（分支/风险/条件 TBIG）3 小时；单任务默认最多 1.5 小时，TBIG 最多 1 小时。每 2 秒检查预算与磁盘。进程结束等待、环境检查、清单、分析和打包不计入这 12 小时，所以它不是精确关机倒计时，也不是保证所有计划任务 12 小时完成。

若前一项耗尽阶段预算，剩余项记录 capped/blocked，仍执行其他独立阶段并打包。磁盘至少留 8 GiB，实验输出目标上限 8 GiB（单帧/检查点落盘及检查间隔可造成少量超出），结果包另需空间。结果和旧数据均不自动删除。

加速资格预先固定：热态同步推理至少 2 倍、含适配诊断的热态总耗时更低、病例平均 PSNR 降幅≤0.3 dB、SSIM 降幅≤0.01、未测区 MAE 增幅≤5%、最差病例平均 PSNR 降幅≤1 dB、不确定性与误差的病例内相关系数下降≤0.1。原版和候选都必须完整覆盖同一病例/帧/种子，固定历史诊断也须通过。门槛是开发筛选规则，不是统计非劣保证或临床风险标准；样本过少/关联不可计算时保守地不通过。

图像指标在原版极坐标 uint8 域计算 PSNR/SSIM，MAE 在 [-1,1] 浮点域；完整目标只做离线评分。官方基线保留它的非零观测硬覆盖语义，没有悄悄修补已知的“合法零值观测”边界。BF 使用显式 mask 硬投影，因此不能把两个实现声称为数值等价。

BF：codec MAE≤0.20 才进入下一阶段；prior 不比 codec 持久性误差高 5% 以上；filter 开发误差须低于较好简单基线的 98% 才做确认。模型结构等补充假设见 `docs/belief_filter.md`。prior 的真实上一帧输入/初始化 rollout 只用于“可预测性诊断”，不是合法采集策略成绩；filter 与闭环不使用真值历史状态。

风险探针的均值、标准差、系数和预算阈值均只从 train 拟合；开发集选择最强简单基线，确认集检验是否至少降低 5% 预测 MAE。当前误差只能作为离线标签，不能充当策略输入。条件闭环复用确认病例，空间位置统一等间隔，探针此前来自 CASL 策略，存在策略分布变化；超出训练特征范围时回退固定预算。随机对照使用自适应预算序列的事后打乱，匹配总预算但不是可部署的先验对照。因此该闭环只能检查接口和信号，不能当独立有效性证明。

## TBIG 的条件入口

默认会明确跳过，不会联网下载数据、权重或自动在现有 CASL 环境里升级 zea。未来准备好独立 TBIG 官方仓库与环境后，在新运行配置补全：

```yaml
tbig:
  python: /absolute/path/to/tbig-env/bin/python
  repo: /absolute/path/to/TBIG
  config: /absolute/path/to/TBIG/configs/your-task.yaml
  expected_commit: d878652811093d8a234ff023228d46306db65948
  sequences: [/absolute/path/to/compatible-task-sequence.hdf5]
  asset_files: [/absolute/path/to/diffusion.weights.h5, /absolute/path/to/downstream.weights.h5]
  frames: 12
  max_seconds: 3600
```

仓库需已有匹配的 `users.yaml`、下游任务权重与生成先验。桥接调用该版本官方 `active_sampling_single_file`，不移植 TBIG 模型，不替换它的数据物理模型。运行结果保存图像、信念、采集掩膜及任务输出；全图任务网络输出标记为 model reference，不标成临床真值。当前只有官方接口对照与条件检查，缺少该独立安装，尚未完成真实 TBIG 环境执行验证。

## 跑完下载什么

结束时自动生成：

```text
/root/autodl-tmp/outputs_casl/preparation_v1.results.tar.gz
/root/autodl-tmp/outputs_casl/preparation_v1.results.tar.gz.sha256
```

直接下载这两个文件。包内有 `REPORT.md`、`status.json`、实际配置/代码与权重指纹、固定病例清单、逐任务日志、逐帧 CSV/NPZ、定性图、加速门槛判定、各阶段权重与结果。NPZ 包括恢复所需状态；不是把整个70 GB数据再打一次包。

服务器端可快速阅读：

```bash
cat /root/autodl-tmp/outputs_casl/preparation_v1/REPORT.md
cat /root/autodl-tmp/outputs_casl/preparation_v1/selection.json
```

`finished_with_gaps` 表示调度走到末尾，但有缺条件、超时或失败项。以 `status.json` 和各任务 `result.json` 为准，不把“流水线结束”误认为“所有假设都验证成功”。若打包因空间不足跳过，保留的结果目录不受影响；腾出空间后可运行 `bash scripts/run_preparation.sh report`。

## 本地验证边界

已提供自动测试：稀疏 DPS 全指导时与上游求解器等价、JIT/vmap 下实际梯度调用计数、状态克隆和恢复、逐帧中断恢复、计时上限与重复运行保护、病例隔离、防标签泄漏、BF 三阶段 CPU 小样本训练及资格检查。测试使用合成数据，不能证明加速后医学图像质量，也不能替代 AutoDL 上的 CUDA/JIT 实测。

代码入口：`src/cognitive_ultrasound/preparation/`。首轮运行后，先看实际成本、失败类型及各门槛，再决定下一批具体创新实验；这次不会自动进行网格搜索、全量训练或全量论文复现。
