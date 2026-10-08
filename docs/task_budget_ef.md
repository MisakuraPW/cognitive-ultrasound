# EF 驱动的两阶段动态预算：E0 / E1 / E2

2026-10-08。当前交付是代码与本地功能验证；未启动本轮服务器训练，尚无EF效果或GPU耗时结论。使用自己的CASL项目，保留`vendor/casl`和相邻TBIG官方仓库原样。TBIG提供任务选线思想，不切换到其LVH实验或权重。

## 本轮实现与固定边界

|实验|采集数量|训练|共享组件|
|---|---|---|---|
|E0|固定两批，主参照10+4；另有有限固定预算工作点|无需新增训练|官方CASL扩散权重、EF模型、EF任务选线、两阶段更新|
|E1|两个轻量MLP分别决定K1/K2|Straight-through Gumbel-Softmax，局部帧VJP|同上|
|E2|两个轻量MLP分别采样整数K1/K2|REINFORCE＋过去episode奖励的EMA baseline|同上|

仅优化预算控制器，CASL和EF权重冻结。训练目标为视频/片段EF绝对误差（百分点）＋`lambda × 平均线数/112`。无Oracle预算标签、无预算组合的感知穷举、无额外STOP网络、无多任务训练；分割评价也尚未接入本批。

新固定两阶段E0属于新实验协议，不声称与官方“一次感知后为下一帧选线”的入口逐位一致。共有首帧使用10+4初始化，第一批冷启动500步，第二批用热启动；之后每批热启动25步、FP32、两粒子、窗口3、逐步DPS、TF32关闭。25步是本批**明确使用的近似设置**，各方法在相同底座重建baseline，未改变历史官方参照。

预算候选第一批`[4,7,10,14]`，第二批`[0,2,4,7,14]`，总数不超过28；补采排除第一批位置。候选集、权重列表及病例在运行前锁定，失败不自动扩搜索。

## 在线闭环与任务选线

1. 第一控制器读取过去后验粒子、历史重建、过去线数与因果EF特征。将上一帧粒子末通道作为历史先验代理，不额外训练预测器。
2. 以过去重建构成因果EF窗口，把当前粒子假设放在窗口末尾，得到输入梯度。按TBIG官方代码约定形成评分：`图像粒子方差 × 粒子平均EF输入梯度的平方`，再沿深度汇总为线分数。
3. 用既定贪心和Gaussian邻域抑制选K1条，获取第一批真实观测，执行当前帧第一次CASL后验更新。
4. 第二控制器读取第一批掩膜、残差、中间粒子与更新后的因果任务信息，决定K2；按同一评分规则选择未采位置。
5. K2>0时以两批联合观测执行第二次后验更新；K2=0复用中间结果，**前向不执行第二次DPS**。
6. 观测历史只在帧开始推进一次；第二批覆盖当前通道，当前帧最终状态才传到下一帧。

两个控制器各用12项可观测统计量和32单元隐藏层。输入做固定`sign(x)log1p(abs(x))`变换，不用确认集拟合归一化。当前完整帧、EF真值及未来帧仅在模拟环境、训练损失或离线评价中出现。

全零任务评分采用确定性的空间分散平局规则，并明确排除已采位置。因此应称TBIG风格EF扩展；没有把此扩展等同于原论文所有任务和实现。评分采用官方代码“平均后平方”，并非悄悄改成论文公式的“平方后平均”。

所有方法的扩散随机键由episode seed、frame、stage确定，与预算策略抽样次数无关。动态方法不复用固定方法的重建轨迹。

## EF模型与图像协议

独立Torch进程加载EchoNet官方`r2plus1d_18_32_2_pretrained.pt`，构建原生R(2+1)D-18及标量输出头，严格加载权重，eval模式并冻结参数；在线评分仍保留**输入**梯度。使用32帧、间隔2的输入。

任务进程与JAX感知进程分离，复用现有Python环境，不安装/替换Torch、JAX或CUDA。EF进程通过有超时的本地文件/标准输入协议复用模型。退出或停止时回收对应进程，不建立服务器SSH服务或持久后台守护。

使用官方Zea坐标、轴交换及中央crop进行极坐标→Cartesian转换，插值采用显式双线性gather，支持输入梯度且避免CUDA `grid_sample`反向的非确定性限制。输出按0–255 RGB尺度和锁定mean/std归一化。该转换是本批共享的任务预处理，不声称还原了最初AVI像素。

首次资产准备从原始EchoNet TRAIN取固定128视频、16帧/间隔2片段计算RGB mean/std，并存文件哈希、抽样位置及EF权重哈希。EchoNet原训练统计也来自抽样，但这里不声称重现作者当年的同一次随机抽样。权重下载后记录URL与SHA256，已有文件需有`.source.json`来源收据；不静默接受不同模型文件。

评价同时保留：原始Cartesian完整视频EF、完整极坐标转换后视频EF、实际采样重建视频EF与EF真值。原始/极坐标帧索引不一致则失败。完整视频使用固定步长16的32帧窗口并补末窗口，窗口预测平均为视频预测。短视频重复末帧补齐；在线短历史重复最早已知帧左填充，**不会借用未来帧**。

## E1到底怎样获得梯度

GS硬前向执行整数预算，实际采集严格二值。每个候选预算对应同一贪心排序的前缀掩膜；soft类别权重提供反向，`hard + (soft-stop_gradient(soft))`保持前向二值。

官方inpainting对mask使用布尔`where`，没有mask导数。本实验实例增加custom JVP：二值前向仍用原`where`，反向采用分数掩膜的线性观测松弛。没有修改官方源码。对最终投影也保留可微掩膜路径。

先执行整段真实硬闭环，由冻结EF模型取得视频预测对所有重建帧的输入梯度；再在相同历史、线排序、观测和随机键下重放每个暖帧，沿两次CASL感知计算任务梯度与预算成本梯度。采用rematerialization降低反向内存，额外计算计入训练耗时。

**明确近似范围：**跨帧历史、控制器统计输入以及离散位置排序detach，属于`local_frame_vjp`，并非完整视频BPTT。K2=0的局部反向通过直通补采掩膜的直接观测投影近似补采收益，不虚构第二次DPS。温度从1.0降到0.3，部署使用hard argmax。

GPU入口强制检查：硬重放与真实轨迹差≤2e-4、非有限值拒绝、仅任务项对两个预算头都有非零有限梯度。失败时记录E1无法运行，不转成仅成本训练，也不把E1悄悄替换成RL。E2仍可在共享感知/任务检查通过后运行。

## E2与恢复

从两个合法预算分布采样，奖励为`-(EF误差 + lambda×归一化采集成本)`，将视频级奖励用于两批动作的log probability之和。EMA baseline仅包含之前episode，避免把当前奖励减掉自身。没有自动引入PPO等额外算法。

每次真实优化器更新保存控制器、Adam一二阶状态、global update和奖励baseline；按global update生成病例、连续片段位置和动作seed。中断最多重新执行当前更新；评价按完整病例提交收据，中断重算当前病例，完成病例跳过。CPU合成工作负载已核对中断恢复与连续更新逐位相同；GPU恢复的严格数值复现仍需服务器实测，不能扩张为任意训练入口的保证。

## 病例与有限批次

原始EchoNet划分与CASL划分不同。新策略TRAIN来自两者TRAIN的交集；development来自EchoNet VAL且属于CASL val/test；confirmation来自EchoNet TEST且属于CASL val/test。默认32/8/16个视频，seeded名单在结果前确定，文件SHA256锁定。确认集不参与训练、超参数调整或工作点挑选。

此隔离依赖官方EF预训练使用原始TRAIN的来源假设；若换自己的EF权重，必须核实其训练病例并调整协议，不能仅换文件后沿用旧结果。

默认`configs/task_budget_ef.yaml`：3种子、3个lambda（0/2/8）、E1/E2各50次64帧episode更新，合计18个训练工作点、900个episode。E0固定工作点6个。每个工作点记录开发和确认完整视频；所有工作点预先固定并完整展示，不靠确认结果挑“最好”的单点。**这是有限MVP研究批次，不是收敛保证，也没有承诺几分钟跑完。**

64帧是训练片段上限；锁定病例不足64帧时使用其全部实际帧，仅EF输入按已定义规则补齐。采集资源按实际帧数统计，不虚构额外采集、不因短视频换病例。

`configs/task_budget_ef_smoke.yaml`提供独立的全流程冒烟：4/2/2视频、1种子、1个lambda、每种方法2次更新、3个固定工作点。它只验证可运行性，不充当正式效果证据，也不混入正式目录。

## 主比较与成本记录

除固定预算曲线外，每个动态评价视频增加**按实际总线数匹配的周期固定对照**：从事先固定的两个相邻工作点按周期交替，首帧仍10+4。只读取动态策略已经花费的总资源，不读取其图像或EF误差。这是离线资源匹配对照，不能冒充可部署的预算预测策略。平均线数误差≤0.1条才标为资源匹配合格，否则保留不合格标志。

报告EF MAE/RMSE（百分点）、平均/总线数、预测保留误差、PSNR/SSIM/浮点MAE、感知调用数、各批DPS/任务评分耗时、视频计算耗时、预算轨迹及3帧快照。计算时间统一为预加载视频上的采集闭环＋最终EF推理；数据读取和完整输入参考的额外评价另记。减少扫描线数不直接写成实际采集时间或GPU加速。

统计先合并同一视频的种子，再等权合并视频，5000次视频级bootstrap；配对资源对照报告动态减固定的MAE差与95%区间。结果为描述性证据，不自动证明临床非劣效或宣布创新成功。

图只画观测工作点，不在缺失结果之间插出“更好”的Pareto曲线；失败任务留在ledger，部分完成工作点用空心标识。

## AutoDL运行

代码走Git，数据/权重/结果不走源码包。更新前检查服务器工作区，保持旧实验已经结束。

```bash
cd /root/autodl-tmp/cognitive-ultrasound
source /etc/network_turbo
git status --short
git pull --ff-only origin main
bash scripts/setup_task_budget_ef.sh
# 独立冒烟批次；其配置/结果不能复用为正式科研结果
TASK_BUDGET_CONFIG=configs/task_budget_ef_smoke.yaml \
  bash scripts/run_task_budget_ef.sh start /root/autodl-tmp/outputs_casl/task_budget_ef_smoke_v1
tail -n 60 -F /root/autodl-tmp/outputs_casl/task_budget_ef_smoke_v1.console.log
```

实际准备阶段会下载EF权重并计算一次原始训练集归一化统计；先确认YAML中的`file_list`与`raw_videos`指向共享数据。原始视频只读，极坐标数据继续沿用。

正式批次的一次启动：

```bash
bash scripts/run_task_budget_ef.sh start
tail -n 60 -F /root/autodl-tmp/outputs_casl/task_budget_ef_v1.console.log
```

自动流程：资产/数据锁定 → 真实GPU/GS梯度probe及估时 → E0固定对照 → E1/E2训练和统一评价 → 报告 → 小型分析包。顶层日志打印阶段，当前任务逐帧/更新在`jobs/<job>/console.log`；`status`查看任务PID和存活状态。首次编译包含在probe记录，估时为外推且不保证。

```bash
bash scripts/run_task_budget_ef.sh status
bash scripts/run_task_budget_ef.sh stop
bash scripts/run_task_budget_ef.sh resume
bash scripts/run_task_budget_ef.sh report
bash scripts/run_task_budget_ef.sh bundle
```

`resume`在身份相同条件下复用完成任务，保留失败收据后显式重试失败任务一次；不自动无限重试。换卡、框架版本、CUDA路径、科学配置或代码改变必须用新目录，不混写旧结果。STOP可在资产准备或任务执行时中止；Linux工作器及其EF子进程一起回收。

最终小包：`/root/autodl-tmp/outputs_casl/task_budget_ef_v1.analysis.tar.gz`与`.sha256`。小包包含报告、汇总、图、配置、身份和名单。策略checkpoint、逐帧轨迹、原始日志留在服务器完整结果目录，按需另行取回；不把EF/CASL权重或反复快照塞进默认分析包。下载遵循用户偏好，默认手动。

## 实现与验证索引

- `task_budget/perception.py`：固定CASL官方感知、观测松弛JVP。
- `task_budget/ef_worker.py`、`task.py`：冻结原生EF、可微转换、隔离任务进程。
- `task_budget/protocol.py`、`episode.py`：因果状态、任务贪心、两阶段闭环和GS重放。
- `task_budget/policy.py`、`experiment.py`：两个预算头、GS/RL训练、优化器恢复与配对评价。
- `task_budget/data.py`、`suite.py`：划分交集、资产/身份、任务清单、超时、停止与恢复。
- `task_budget/report.py`：视频级统计、事实CSV、图与校验分析包。
- `tests/test_task_budget_ef.py`：因果性、唯一线数、零补采、GS任务梯度、RL动作、数据隔离、恢复和任务输入梯度等。

本地验证明细见[本轮验收记录](task_budget_ef_validation.md)。

方法来源：[TBIG](https://arxiv.org/html/2601.20711v1)、[EchoNet官方EF实现](https://github.com/echonet/dynamic)、固定`vendor/casl`。图与证据呈现参考：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065)（核对当前v2；一般图规范，不是期刊合规认证）。
