# 本轮只完成 7 条线确认及实验基座

用户于2026-09-28缩减本轮范围：保留正在运行的7条线五版本确认，不启动14/28条线确认；随后整理加速证据并完成已有实验基座。质量阈值不变，不宣称其他预算已经验证。

活动 checkout 固定在 `ae83eaf`。不要在其工作器运行期间 pull 或改源码。接续器作为独立 Git 版本脚本部署至该 checkout 的 `.cache/finish_compute_lab_b7.py`，不会改变原源码指纹。

已设置 `PAUSE_AFTER_JOB=confirmation_jax_25_bf16_b7`，原协调器完成最后一个7条线候选后释放锁。接续器检查协调器和工作器已退出、源码指纹与环境一致、五个确认任务均有终态结果，再记录 `scope_amendment.json`；不改写旧 identity。候选失败仍保留为失败，不扩大搜索。

接续器仅调用原有单帧性能剖析、closure收尾、CASL/codec/prior/filter的短训及恢复对照，不调用任何确认推理任务。单帧剖析使用已有14线debug缓存，不是新增14线确认集实验。最终生成 `ACCELERATION_SUMMARY.md`、原报告与事实CSV，以及校验过的结果包；Torch根因诊断证据随包保留。

运行与接续状态：

```bash
tail -n 30 -F /root/autodl-tmp/outputs_casl/compute_lab_v5.finish7.console.log
cat /root/autodl-tmp/outputs_casl/compute_lab_v5.finish7.status.json
cat /root/autodl-tmp/outputs_casl/compute_lab_v5/status.json
```

等待期间看原 v5 日志及当前工作器日志。接续完成状态可能是 `finished_with_gaps`，这保留先前Torch短测失败及其他未通过项，不代表所有实验成功。只有同级 `.bundle.json` 才表示结果包校验完成。

接续器尊重 `STOP`，不会自动清除用户停止请求。若异常停止，先查日志和真实进程状态，不要用原 `run_compute_lab.sh resume` 恢复全预算循环。
