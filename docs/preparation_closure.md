# 最后一批准备工作：范围、运行与交接

2026-09-20。此批是进入正式创新探索前的固定收尾包，名称为 `preparation_closure_v4`。

**这次不再以“结果好，所以再做一点；结果不好，所以再补一轮”延长准备阶段。** 所有已确定的小实验一次编排。结果可以是正、负、不确定、缺资产或超限；运行结束后统一交接，不自动开展参数搜索或新的实验批次。

## 一、准备阶段的边界

已经完成且不重做：EchoNet 全量极坐标转换、CASL 环境与权重接入、基础封装核对、原版计时、25 步 FP16 筛选与确认、BF 原型三阶段小训练、风险预测小样本扩展及递归退化诊断。前两批原始结果保留。

本批补齐的内容：

| 编号 | 工作 | 固定设置与产物 |
|---|---|---|
| C00 | 输入与历史资产核对 | 排除前两批用过的 validation 病例；固定新名单；记录数据、源码、配置、设备和权重指纹 |
| C01 | 新卡及多预算加速确认 | 7/14/28 条各做原版与 25 步 FP16 对照；2 个调试病例、各 12 帧；不新增候选网格 |
| C02 | 多预算轨迹与候选预算标签 | 32 train、4 development、8 confirmation 病例，2 种子；每步从同历史比较 7/14/28 条，然后按预定混合预算推进真实历史 |
| C03 | 固定的轻量预测参照 | 训练只用 train；不确定性＋候选预算参照、多特征＋候选预算探针；开发与确认结果分别记录 |
| C04 | 公平闭环 | 固定、在线随机、当前不确定性、简单预测、多特征预测，共 5 策略；每段所有策略总线数严格等于 14×帧数 |
| C05 | BF 一次受控历史训练 | 从同一已有 filter 开始；两组各 200 更新×12 帧；一组每 3 帧重置，另一组连续 12 帧；只训练 filter |
| C06 | BF 小型独立评估 | 同时比较原冻结 filter、两组最终 checkpoint、空间插值；不按确认结果挑 checkpoint 或增加训练 |
| C07 | TBIG 条件检查 | 只有已提供匹配的独立环境、仓库、权重和兼容数据才运行小任务；否则明确登记缺条件 |
| C08 | 总结与交接 | REPORT.md、诊断图、逐帧指标、状态清单、Excel 事实导入表、结果 tar.gz 与 SHA-256 |

协调器共 18 个固定计算任务：6 个加速锚点、3 个预算数据集、1 个拟合、1 个确认预测、2 个闭环、2 个 BF 训练、2 个 BF 评估、1 个条件 TBIG。C00/C08 是输入与收尾操作。

此批不实现 Flow、world model、临床风险校准、完整规划、跨模态迁移、完整 CASL 重训或论文全表复现。这些需要针对具体创新点设定假设，不属于“无需额外研究决定的准备”。

## 二、一次运行的实验协议

### 病例与信息边界

新批从官方 train/val 划分抽样，读取前两批 manifest，把它们所有 debug/development/confirmation 病例从新 val 候选池排除；不接触 test。抽样在计算指标前完成，遇到选定病例过短或损坏直接报错，不按效果换病例。

train/development 各取 24 帧，confirmation 取 32 帧，2 个固定种子。不是完整视频或论文级样本规模。BF 使用相同新 train 病例，以及 development 24 帧、confirmation 32 帧；这些组彼此分开，但同一个确认组会用于本批预先固定的多个比较，不能当成多次独立确认。

预算数据的行为轨迹预先混合 7/14/28 条；每个未来帧的候选预算在同一历史和随机状态下比较。选择行为预算不查看候选误差。完整图只产生离线标签和评分，策略函数只接收上一帧可观测特征。

两个 ridge 模型均保持固定规则。简单模型用当前不确定性及候选预算；多特征模型用既有六种可观测特征及候选预算，包含预先固定的预算交互项。缩放、系数和预算提议阈值全部在 train 拟合。确认数据不能修改它们。

闭环风险策略把“假如下一帧使用 14 条，预计误差多大”映射到 7/14/28 条提议；这只是透明的轻量参照，不宣称已学会最优行动价值。当前不确定性策略采用自己的 train 分位数阈值。两者均保留特征越界回退并记录具体原因。

### 总预算公平性

每段初始帧均为 14 条，总额度为 14×帧数。每一步只允许选择使剩余额度仍可恰好兑现的动作；末帧结束时每个策略恰好花完同一额度。随机策略从当时可行的预算中抽样，不使用未来图像、自适应策略的未来序列或事后总额度。

记录全图 MAE、各自未测 MAE、所有策略共同未测区域的 MAE、预定阈值下低质量帧比例、回退次数、总额度约束次数、每帧实际线数，以及冷/热计时。质量阈值为归一化 [-1,1] 下全图 MAE>0.15，仅是本批工程诊断阈值，不是临床标准。换评价区域的补充结果不会回写加速准入门槛。

加速在不同预算上的锚点全部通过才默认使用 `official25_fp16`；否则在原版检查完整的情况下回退原版。原版锚点不完整时阻止预算实验，不能把未验证配置当作底座。首帧仍保留 500 步，后续才是 50→25 步。

### BF 控制变量

两组从同一个已完成的 filter checkpoint 初始化，重置 Adam；相同数据、采样种子、12 帧片段、200 次更新、学习率、损失和帧间 stop-gradient。区别仅为训练时状态每 3 帧清空，或完整 12 帧内连续保留。两组都冻结 encoder、decoder、prior，并逐数组检查它们没有变化。

每组开始前重新做 eager/graph 和线程短测；图模式必须通过损失、参数、优化器与显存检查。最终评估均连续运行，不在评估时给短历史组额外重置优势。只评价固定最终 checkpoint，不做最佳 epoch 搜索。这是定位已有实现的受控小实验，不是学长论文复现或模型上限检验。

## 三、费用与停止条件

- 所有计算子任务累计最多 **3 小时**：加速及条件 TBIG 30 分钟、预算实验 75 分钟、BF 75 分钟。普通单任务最多 30 分钟；TBIG 最多 10 分钟。
- 这些是上限，不是预计运行时长或完成保证。GPU/CPU 型号、编译、读取和实际回退会影响耗时。初始化、报告、打包及下载另计，实例不会自动关机。
- 有正常输出也可能得到负结果。不会因为负结果增加病例、训练步数、候选方法或新批次。
- 超限/失败会记录并保留已经完成的数据；其他独立任务继续，依赖未满足的任务明确 blocked。到此生成交接报告，不自动解锁无限准备工作。
- 至少保留 10 GiB 空闲空间，输出上限 6 GiB，打包还需要额外空间。仅保存少量固定位置图像及全部数值记录，避免再次逐帧保存重复恢复缓冲区形成很大的结果包。

## 四、Git 更新与一键启动

从 2026-09-20 起，代码统一通过 Git 获取，不再上传或解压源码包。此前的代码 tar 留作历史产物，不用于本批部署。数据、权重、实验结果仍单独保存。

已有实例应保留：

```text
/root/miniconda3/envs/casl
/root/autodl-tmp/cognitive-ultrasound/vendor
/root/autodl-tmp/cognitive-ultrasound/checkpoints
/root/autodl-tmp/datasets/CASL-EchoNet-polar
/root/autodl-tmp/outputs_casl/preparation_v1
/root/autodl-tmp/outputs_casl/preparation_followup_v3
```

最后两个是前批原始输出目录，包含 manifest、身份记录，以及 v1 的 BF 权重；只上传 v3 的结果包不能代替 v1 的继承权重。如果换实例，需要一起带上这些已有资产。不要重新转换 EchoNet 或从头训练 CASL。

开有 GPU 的实例，确认旧实验进程已经结束后，在服务器粘贴以下命令。它只允许干净工作区上的快进更新；若发现服务器有代码修改会停下显示文件，不覆盖或自动清除：

```bash
(
set -euo pipefail
cd /root/autodl-tmp/cognitive-ultrasound
if [[ -n "$(git status --porcelain --untracked-files=normal --ignore-submodules=untracked)" ]]; then
    git status --short --ignore-submodules=untracked
    echo "工作区有未提交文件，请先保留并处理这些改动，再更新。"
    exit 2
fi
test "$(git branch --show-current)" = main
git pull --ff-only origin main
git log -1 --format='%h %s'
bash scripts/run_preparation_closure.sh start
)
```

沿用原仓库中的 vendor/checkpoints 和已工作的 casl 环境，不重装 CUDA。新实验使用独立输出目录。启动后断开 SSH 不影响后台任务；运行期间不要在这个源码目录执行 pull、checkout 或修改代码。需要同时开发其他版本时用单独 clone。新实例先按 [Git 工作方式](git_workflow.md) clone 并准备已有资产。

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_closure_v4.console.log
```

总日志显示 `STAGE` / `STAGE_END`。详细帧和训练进度在对应任务日志，例如：

```bash
tail -n 30 -F /root/autodl-tmp/outputs_casl/preparation_closure_v4/jobs/budget_data_train/console.log
tail -n 30 -F /root/autodl-tmp/outputs_casl/preparation_closure_v4/jobs/bf_history_continuous12/console.log
```

状态/停止/续跑：

```bash
cd /root/autodl-tmp/cognitive-ultrasound
bash scripts/run_preparation_closure.sh status
# 需要暂停时执行：
bash scripts/run_preparation_closure.sh stop
# 只在已暂停、旧进程结束且代码/数据/硬件未变时续跑：
rm -f /root/autodl-tmp/outputs_casl/preparation_closure_v4/STOP
bash scripts/run_preparation_closure.sh resume
```

混合预算与闭环按病例/种子保存，断电最多重做当前小片段；BF 每 25 更新存权重及优化器。累计上限不会因续跑重置。修复环境后若明确要重试失败项可用 `PREPARATION_RETRY_FAILED=1 bash scripts/run_preparation_closure.sh resume`，也仍受原累计上限约束。换卡/代码/配置需新输出目录与新预检，不能绕过身份检查混写旧目录。

## 五、跑完看什么、下载什么

```bash
cat /root/autodl-tmp/outputs_casl/preparation_closure_v4/REPORT.md
```

下载：

```text
/root/autodl-tmp/outputs_casl/preparation_closure_v4.results.tar.gz
/root/autodl-tmp/outputs_casl/preparation_closure_v4.results.tar.gz.sha256
```

包内有 REPORT、数值结果、控制对照图、BF 时序图、固定位置重建示例、训练权重、配置与数据指纹、日志、`excel_facts.csv`。`completed` 是全部任务完成；没有 TBIG 资产时预期会是 `finished_with_gaps`，要看缺口原因。`preparation_closed: true` 表示固定批次已收尾，绝不等于每个方法有效或每项工程检查都通过。

报告生成需要重试时，可执行 `bash scripts/run_preparation_closure.sh report`，只重建报告和包，不训练。

## 六、怎样接入你的 Excel 和正式创新研究

沿用本地 `G:\科研项目\毕设\outputs\research_20260918\科研研究操作系统_认知超声_追加记录_20260918.xlsx`。本批不修改已有内容，也不填“kaiming 风”的人工判断。

运行后将 `excel_facts.csv` 作为单独的准备阶段留档表导入：任务 ID、状态、耗时、事实摘要、结果位置会填好；Prediction、Prediction Lock、Surprise、Belief Update、Next Decision 留空。不要用导入覆盖原来的核心实验表。

接下来你可以从具体创新点开始：先详细调研和写出自己的假设、预期与对照，再锁定实验；开发时沿用本批已确认的底座、病例隔离、额度控制、图像与指标导出、硬件预检和独立输出目录。没有预定你的创新方案，也不要求你先把所有旁支都做出正结果。

以下不作为进入主线的额外前置条件：TBIG 缺外部资产、BF 小试验仍不胜出、尚未完成大模型训练、尚未证实跨模态优势。这些结论明确留档，待对应创新点真正需要时再决定。
