# 第二轮实验收尾故障与修复

2026-09-19 用户发现日志停止后，已登录 AutoDL 检查。9 个科学计算任务均有 `completed` 结果；没有 OOM、GPU 掉线或中途训练退出。任务累计墙钟约 785.6 秒（13.1 分钟，不含报告和归档）。

故障发生在最后生成报告时：旧版 BF 阶段输出使用 `model` 数组，新加入的冻结模型诊断使用 `online`、`reset_every_3` 等数组。报告绘图仍强制读取 `model`，触发 `KeyError: 'model is not a file in the archive'`，导致整轮状态标为 failed。

修复让绘图根据实际字段组织比较图，同时显式标记上一帧真值与完整输入 codec 为特权离线诊断。添加旧/新两种结果格式的报告回归测试，并检查绘图不修改原始 NPZ；相关 12 项测试及 Ruff 通过。

云端仅重建报告和结果包，未重新推理或训练。原实验源码与 identity.json 保留；修复用的渲染代码、原渲染代码、修复前 status、渲染 SHA-256、执行日志与恢复记录单独放在输出目录 `report_repair_20260919/`。科学任务 JSON/NPZ 的文件清单、尺寸和时间戳在生成报告前后核对一致。

## 已完成结果摘要

- 25 步 FP16：8 个新病例、两种子、896 帧次；同步热态提速 2.507×，病例均值 PSNR +0.220 dB、SSIM +0.00416、未观测 MAE 比 0.9798。扩展检查通过，仍是有限开发证据。
- 风险预测：训练 1984 对、development 248 对、confirmation 880 对；相对 development 选出的 current-uncertainty 简单参照，confirmation 预测 MAE 改善 7.215%，达到预设 5% 门槛。不能称为统计保证或已经证明主动采集收益。
- BF 冻结诊断：online 未测 MAE 0.4080；每 3 帧重置 0.2035；特权上一帧真值 latent 0.1418；插值 0.1169；插值经 codec 0.1351；完整输入 codec 0.0920。这支持优先检查递归状态漂移与训练/运行时历史分布差异，尚未证明原因只有这一项。
- 分支和小型闭环已实际执行。闭环仅 2 病例 × 2 种子、复用 confirmation，有训练策略到评估策略的分布变化，不应视为独立效能验证。

服务器输出：`/root/autodl-tmp/outputs_casl/preparation_followup_v3/`。

主报告：`REPORT.md`；完整结果包：同级 `preparation_followup_v3.results.tar.gz` 与 `.sha256`。

收尾已确认完成：修复进程正常结束，状态为 `completed`，归档重建及完整读取校验完成。最终结果包为 2,758,247,204 字节，SHA-256 为 `62a80ae5f1ba2010a3a0c80f808694e7092fd4e0b9601802b9e7afe9f30df6ed`。总控制台日志已追加 `REPORT_REPAIR_COMPLETED`，保留此前报错记录。未重跑 GPU 任务，未关闭实例。
