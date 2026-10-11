# 本轮缩减到136次更新

2026年10月11日，用户明确要求三项实验到136次停止，不继续264次。本操作先完成已有的136次开发评价，再停止原队列，不增加周期对照或确认集计算。

原模型模块、配置、数据、identity与已完成结果不改。Git更新仅增加独立收尾脚本、文档及启动器防误续跑检查，原模型源码哈希保持一致。

```bash
cd /root/autodl-tmp/cognitive-ultrasound
git pull --ff-only origin main
nohup /root/miniconda3/envs/casl/bin/python -u scripts/finish_task_budget_stages_136.py \
  --output /root/autodl-tmp/outputs_casl/task_budget_stages_v1 \
  >/root/autodl-tmp/outputs_casl/task_budget_stages_v1.finish136.log 2>&1 </dev/null &
```

程序是本次实验目录内的一次性后台进程，断开SSH后继续，完成后退出；没有安装服务器服务、修改SSH或创建系统定时任务。

它校验三项136检查点及16例×3模型开发评价提交后写STOP，等待原协调器与工作进程退出。若启动时已进入264阶段则立即停止并保留额外已提交更新；选定报告仍使用136检查点。不会删除或伪装多跑的结果。上游失败或校验失败会停止队列并记录失败，不能称为136收尾完成。

完成产物：

- `scope_closure_136.json`：用户指定缩减范围的完成状态、取消阶段、保留的更新数及原协调器状态。
- `SCOPE_136_REPORT.md`：72／136开发结果与证据边界。旧`REPORT_中文.md`仍保留其原完整计划描述，以本收尾报告说明实际执行范围。
- `scope_136_bundle_receipt.json`：校验后的轻量分析包位置／字节／SHA256，包不含完整帧、GRU／Adam或预训练权重。
- `.finish_136/status.json`、`task_budget_stages_v1.finish136.log`：收尾程序状态与日志。

原status可能显示stopped，这是原完整队列取消；scope_closure_136=completed才是缩减范围校验完成。不自动关实例。

启动器检测已收尾标记，默认拒绝start/resume进入原264计划。若用户以后明确决定继续原计划，可显式设置`TASK_STAGES_RESUME_ORIGINAL=1`；新算法仍需独立输出和身份。
