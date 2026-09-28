# v5 的 finished_with_gaps 定向排查

2026-09-28实际服务器审计：五组7线确认全部完成；四类训练工作负载的 eager 和 graph 均完成连续200步、100+100恢复比较；closure 收尾完成。原批次仅有四个 Torch FP16/BF16、25/50步 SHORT 任务在 compiled/eager 内部检查失败。不要重新启动全预算协调器。

## 本轮修复操作

`scripts/repair_compute_lab_precision_gaps.py` 仅重跑这四个失败短测，要求原错误是已检查的数值不一致，禁止将成功任务、确认集或其他错误纳入重跑。原结果目录与 identity、锁定名单、配置均保留。

服务器 PyTorch 2.8 的 `torch._inductor.config` 说明：默认融合会省略低精度算子间的 downcast/upcast，导致与 eager 的中间舍入不同。本次显式设置 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`，验证保留这些舍入能否消除原失败。原 `atol=rtol=2e-4`、变输入检查、图捕获检查、精度、粒子数和500步冷启动均不改。选项和原失败哈希写入 `gap_repair/plan.json`，每个子结果也记录编译选项。

这不是自动放宽近似版的数值标准。重跑后仍未通过的候选保持拒绝；即使运行完成，也不等于跨框架等价或确认集质量通过。本轮不重新挑选确认候选，不补14/28线确认。

实际定位结果：保留中间舍入后四项仍失败，FP16步输出最大差异约0.000589，BF16约0.006034。单独开启此选项不能修复数值差异，记录保存在原 `gap_repair/`，不推荐作为已验证的修复。

后续修复分类错误：原流水线把B类有限数值差异和执行故障混为一谈，导致速度和质量完全缺失。`6f0dd0a` 将二者分开：仍按2e-4记录不等价，另测三组变化输入下的独立DPS梯度与融合采样更新，严格拒绝非有限、shape/dtype改变、逐位重复不稳定及梯度丢失。图捕获与直接编译执行的一致性检查不变。有限的B类差异允许继续原SHORT诊断，但 `internal_correctness` 不会因图捕获成功而覆盖为 true。第二次记录存 `gap_repair_classification/`，使用原编译选项（不保留额外舍入），不改变原确认名单或比较阈值。

冻结 checkout 通过 Git 获取独立脚本并部署至 `.cache/`，不会因更新主分支把正在运行的源代码替换掉。原始环境、参考缓存和结果继续使用原批次；新任务仅写入 `gap_repair/jobs/`。

## 导出

`scripts/export_compute_lab_gap_audit.py` 核对旧结果和科学配置的哈希，生成修复矩阵、训练摘要、事实追加CSV和 `FINAL_REPORT.md`。校验旧完整包后，另打修复附录包。最终 `.final_export.json` 同时绑定两包的SHA-256；原 `status.json` 保持历史值，不把旧失败改写成成功。

运行路径为：

```bash
python .cache/repair_compute_lab_precision_gaps.py --output /root/autodl-tmp/outputs_casl/compute_lab_v5
python .cache/export_compute_lab_gap_audit.py --output /root/autodl-tmp/outputs_casl/compute_lab_v5
```

必须使用已检查的原批次环境，且没有旧协调器或工作器占用目录。新修复入口已有结果时不自动重跑。最终是否完成看 `.final_export.json` 的 `experiments_terminal` 与 `bundles_verified`，不能以原来的 `finished_with_gaps` 判断新附录进度。

## 已确认的训练结论

8组“同执行模式”的连续/恢复比较均逐位一致，包括保存的梯度、参数、优化器状态、适用的EMA以及损失轨迹。codec/prior/filter的 graph/eager 比较通过容差标准；CASL未通过，保留 eager 参照。不能将CASL的这个负结果当成故障重跑至通过，也不能用200步证明完整训练收敛等价。

基础结果包保留被否定的候选、历史结果和质量门槛，修复附录只追加事实，不代填用户预测、Prediction Lock或核心研究判断。
