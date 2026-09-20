# 准备实验第二轮：一次有依据的诊断扩展

> 历史运行留档：从 2026-09-20 起停止使用源码包部署。当前任务请按 [最后一批准备工作](preparation_closure.md) 通过 Git 更新和启动；下文旧批次不需要重跑。

承接 `preparation_v1`，2026-09-19 启动。输出单独写入 `preparation_followup_v3`；不覆盖先前研究、硬件校准或权重，不重训 CASL，也不重新筛选全部加速方案。

## 为什么继续

第一轮 linear-all 未来误差预测，相对在 development 选出的 current-uncertainty 参照，confirmation MAE 改善约 0.195%，没有达到预定 5% 门槛。训练只有 8 病例/120 个相邻帧对，因此允许一次固定模型、固定阈值的数据量扩展；不据此下结论称预测式研究已失败。

BF codec 的开发未测 MAE 约 0.0928，prior 约 0.1018，filter 却约 0.3395，而同预算空间插值约 0.1170。因此先固定已有权重定位问题，不盲目增加训练。

## 本轮锁定设置

- 固定病例抽样种子 20260920；32 个 train 病例，4 个 development，8 个新 confirmation。新 confirmation 与上一轮全部验证用途病例不重叠。不使用 test。
- 原计划 confirmation 64 帧；仅检查数据长度时发现固定名单中一例只有 56 帧，因此保留名单、统一改为 56 帧。在本轮任何质量计算前确定，没有按效果换病例。
- 原 CASL 与 25 步 FP16：8 病例 × 56 帧 × 两个固定种子；仅复核此前胜出配置，不再展开候选网格。
- 加速质量门槛不变。通过后才产生 32 train × 32 帧 × 两种子、4 development × 32 帧 × 两种子的加速轨迹；confirmation 复用刚完成的加速轨迹。
- 风险模型仍为同一 ridge 线性探针及同一组简单参照，5% 门槛不变。新的 confirmation 未用于调参。只有通过才接已有的少量分支和探索性闭环。该闭环复用 confirmation，仅属开发结果。
- BF 使用上一轮 4 个开发病例各 32 帧，固定 filter checkpoint、14 条均匀扫描线和 10+4 同帧反馈；比较正常递归、每 3 帧清空状态、仅用于诊断的上一帧真值 latent、直接插值、插值再经 codec、完整输入 codec。后两类真值相关条件明确标记为离线诊断，不能冒充在线策略。
- BF 的 update 先用 3 帧对比 eager/graph 输出与时间，正确且有收益才图编译；其余方法和采集策略不变，不进行训练更新。

BF 结果能区分短片段训练与长递归差异、状态初始化/传递问题及 codec 对部分观测填充图像的失真，但仍不是独立确认的改进结果。诊断后再决定是否修正模型；此轮不自动变更架构或搜索损失权重。

总上限 90 分钟，A/B/C 分别最多 36/18/36 分钟，单任务最多 24 分钟。程序达到限额、门槛失败或依赖不满足时记录原因并跳过后续，不绕过门槛。结果包自动生成，实例不会自动关机。

## 进度与恢复

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_followup_v3.console.log
```

每帧信息在对应任务日志，例如：

```bash
tail -n 30 -F /root/autodl-tmp/outputs_casl/preparation_followup_v3/jobs/confirm_official25_fp16/console.log
```

启动后断开终端不影响运行。确实中断、且确认没有旧 worker 活着时，使用同一份代码与配置恢复：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate casl
cd /root/autodl-tmp/cognitive-ultrasound-preparation-followup-v3
export PYTHONPATH="$PWD/src"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
python -m cognitive_ultrasound.preparation.followup \
  --output /root/autodl-tmp/outputs_casl/preparation_followup_v3 --resume
```

`source_run.json` 记录复用来源与 SHA-256，`identity.json` 记录本轮源码、硬件及配置。已完成帧/病例复用；暂停不会获得新的累计时间预算。修改代码、配置、数据或硬件后不能混写该目录。
