# 默认加速与运行前检查

从 auto-v2 起，新准备性实验默认使用 `configs/preparation_auto.yaml`。目标是在已测试、通过正确性门槛的候选中降低总耗时；不以 GPU/CPU 占用率达到 100% 作为成功条件，也不承诺找到全局最优参数。

## 两类加速分别验证

**CASL 推理**：保留小样本原始版本作参照，依次检查共同历史采样器、闭环质量与速度、独立验证病例。通过后，后续分支和风险轨迹使用自动选中的加速版本。目前候选是 25 步、25 步 FP16、稀疏 DPS；失败则回退原版本。不会为了提速缩减比较双方的扫描线预算、粒子数或改变测试划分。50→25 步属于近似算法变更，不能标成“原版 CASL”。正式结论仍需要更充分的配对验证。

**学长论文启发模型的训练**：每个 codec/prior/filter 阶段开始前，独立进程比较 eager 与 TensorFlow graph、1/4/8 线程（受容器 CPU 配额约束）。固定 FP32、模型、数据、学习率、更新次数和优化器，在相同初始状态上比较两次损失、全部模型参数及 Adam 状态；通过后用五个训练片段测速，读取结果同步 GPU，并计入数据读取。保留至少 1 GiB 或总显存 10% 的可用余量。用首次执行成本与剩余更新数估计总时间；graph 优势不足 10% 时保留 eager。正式验证损失也使用选中的执行方式。

graph filter 目前只支持准备实验中明确指定的 uniform 采集策略；greedy 仍用 eager，绝不通过更换策略来伪造提速。BF 的逐帧资格评估、通用 greedy 推理和未配置的 TBIG 还没有自动图编译适配器。未来新模型需新增相应的正确性与性能检查，不能直接假定已有参数最优。

BF GPU 运行现在明确关闭 TF32，使用完整 FP32。4090 首轮实测中，TF32 默认开启时 graph 虽然更快，但与 eager 的参数/优化器更新不满足严格容差，选择器拒绝了它。关闭 TF32 后重新验证，保留原容差，不通过放宽标准来接受候选。新的 BF 运行不能声称与旧默认 TF32 执行逐位一致，精度设置与源码一起留档。

## 换卡、断点与审计

记录 GPU 名称/UUID/显存/驱动、Python/框架版本、CPU 容器配额、内存限额、配置、源码和校准权重哈希。不假定始终使用 4090。每次新阶段重新短测，续跑训练也重新测，不盲用旧测速结果。每个候选最多 180 秒、最多三档线程；正常耗时通常远低于该上限，校准耗时单独记录且计入流水线时间预算。

流水线的硬件身份发生变化时拒绝向旧结果目录混写，需新输出目录重新做 CASL 质量/速度检查；旧结果保留。单独 BF `execution: auto` 训练可在相同数据/科学配置下从自己的完整断点恢复，并记录新的执行方式与校准指纹。旧 eager 实验不会被静默改写成 auto 结果。

校准用可丢弃的模型与优化器，不更新正式 checkpoint。所有候选失败、GPU 不可用或显存余量不足时停止并留下日志；没有“默认相信最快版”。显式 `--cpu` 仅用于功能测试，不产生 GPU 性能结论。

## 新实验的启动入口

在部署好的新目录中启动；首次使用新输出目录。现在已完成的 v1 不需要重跑。

```bash
cd /root/autodl-tmp/cognitive-ultrasound-preparation-auto-v2
bash scripts/run_preparation.sh start
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_auto_v2.console.log
```

总日志仍显示 STAGE；BF 子日志额外显示 `PREFLIGHT`、`PREFLIGHT_SELECTED`，随后是正式 TRAIN。断点仍每 50 次更新保存。单独查训练：

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/preparation_auto_v2/jobs/bf_filter/console.log
```

每阶段 `jobs/bf_<stage>/training.preflight/selection.json` 记录所选配置、原始测时、数值检查、显存信息及校准开销；`training/checkpoint.npz` 的 metadata 记录实际 execution 与 preflight_fingerprint。CASL 结果在 `selection.json`。原始对照只在小规模校准和需要正式对照时运行，后续候选实验走通过门槛的加速配置。

若只想在新卡测已训练的三个阶段，且不改变权重：

```bash
export PYTHONPATH="$PWD/src"
python scripts/calibrate_preparation.py \
  --from-run /root/autodl-tmp/outputs_casl/preparation_v1 \
  --output /root/autodl-tmp/outputs_casl/hardware_calibration_new_card
```

请在显卡没有其他实验争用时测速。脚本只跑有界短测，不重新训练三阶段；结果写独立目录。改卡后先确认 `casl` 环境的 GPU 检查通过，再运行。
