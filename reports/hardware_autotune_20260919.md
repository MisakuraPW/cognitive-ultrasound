# 加速默认入口与 4090 验证（2026-09-19）

用户要求后续优先加速，并在每次实验前根据实际硬件确认候选配置。已新增独立 GPU 预检、容器 CPU/内存检测、数值与显存门槛、短程计时、首次编译成本估计、原始日志和选择凭据。默认入口使用 `preparation_auto.yaml`，不绑定 4090。

## 实际结果

服务器 NVIDIA RTX 4090 24 GB、16 vCPU/120 GiB 容器，TensorFlow 2.20。数据及 checkpoint 来自已有 `preparation_v1`，测速不改变它们。

| 阶段 | eager 秒/步 | graph 秒/步 | 同线程稳态比 | 选中 CPU 线程数 | 校准总耗时 |
|---|---:|---:|---:|---:|---:|
| codec | 0.14915 | 0.02474 | 6.03× | 1 | 37.7 秒 |
| prior | 0.17190 | 0.01972 | 8.72× | 1 | 38.4 秒 |
| filter | 1.32517 | 0.06790 | 19.52× | 1 | 96.9 秒 |

以上为各候选五次更新的中位数，包含训练片段读取与同步，不包含完整阶段的加载、验证、checkpoint 写入和本表另列的校准开销。不是完整实验的加速倍数。graph 首次执行约 1.47/1.53/10.03 秒（分别 codec/prior/filter）；选择器计算剩余更新的成本，对不足 10% 的改善保留 eager。三个阶段的 1/4/8 线程候选均进行了测量，线程更多并不一定更快。

关闭 TF32 后三阶段通过相同的严格损失、全部模型参数与 Adam 状态检查。首轮 TF32 开启时，graph 虽快但未通过更新检查，自动回退 eager；这些失败记录同样保留。不能把短程数值一致性推论为完整训练收敛相同。

CASL 本批次选中 25 步 FP16，小样本独立确认相对原始同步热态约 2.53×，包含适配层约 2.17×；4 病例/96 帧，质量门槛通过。这属于通过当前小样本门槛的加速配置，不是完整 CASL 论文复现。

## 现有实验的处理

原 `preparation_v1` 在本次修改期间自行结束，未中断、未覆盖、未重跑。后续 CASL 分支和风险轨迹已使用选中的加速版本。`finished_with_gaps` 的原因是质量/资产门槛：filter 未优于插值基线，风险预测门槛未通过，TBIG 资产未提供，因此跳过对应后续实验；不是训练崩溃。

新代码部署于 `/root/autodl-tmp/cognitive-ultrasound-preparation-auto-v2`。新实验默认输出到 `outputs_casl/preparation_auto_v2`，旧目录及对应原仓库留档。GPU 测时保存在：

- `outputs_casl/hardware_calibration_20260919`：初轮 TF32 开启，graph 被门槛拒绝。
- `outputs_casl/hardware_calibration_fp32_20260919`：完整 FP32 的三阶段实测与选择。
- `outputs_casl/auto_training_smoke_20260919`：真实 GPU 训练入口与断点恢复检查，独立于研究实验。

详细用法与尚未覆盖的加速路径见 [默认加速与运行前检查](../docs/accelerated_experiments.md)。

## 最终验证

相关 24 项本地测试通过，Ruff 和 diff 空白检查通过。GPU 的真实 `execution: auto` 入口完成 filter 20 步后保存，随后独立进程从断点恢复到 40 步；两次均选择 graph，Adam iteration 也为 40，父 checkpoint 的 SHA-256 保持不变。训练/恢复检查为独立功能验证，不计入研究实验结果。

完整 GPU 校准凭据同步保存在本目录 `hardware_autotune_20260919.json`，包含候选耗时、框架构建信息、硬件指纹、原始数值检查和源文件哈希。服务器最终源码包与本地包校验后同步；原 v1 研究目录和源码没有覆盖。
