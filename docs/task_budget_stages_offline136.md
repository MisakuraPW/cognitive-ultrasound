# 全136之后：已有权重与缓存的离线分析

本批分析是用户授权的136次实验收尾。等待T1、Q2、T2全部完成136次更新和16个开发病例评价，原协调器与工作进程退出，并且136收尾包通过校验后才启动。不会进入264次训练，不增加GPU闭环推理或重新训练。

## 六项固定内容

| 内容 | 使用什么 | 能回答什么与限制 |
|---|---|---|
| 训练记录与同病例评价 | 1—136更新原始记录、16/72/136评价 | 任务误差、预算、梯度、奖励与Q2对偶变量如何变化；不同患者更新的曲线不等于固定患者收敛曲线 |
| 预算决策与配额约束 | 全136开发轨迹、1/16/72/136控制器 | 同一输入下未屏蔽与合法动作概率、期望预算、熵、被迫动作；区分网络变化、随机采样与配额限制 |
| 固定推理样例 | 开发名单前三个病例、每个视频四个时间块中心 | 原图、固定7+7、T1/Q2/T2缓存重建、绝对误差、稀疏观测、两阶段物理掩膜、已有双粒子方差、预算时间线；不挑最好病例 |
| 三个模型的参数变化 | 已有1/16/72/136参数 | 参数变化、共同输入上输出变化、与仅输出偏置的差异；第1次更新权重不冒充未训练的初始权重 |
| 表征与输入依赖 | GRU隐状态、已有可观测特征 | 每模型独立PCA、按病例的相关、当前特征局部敏感性、去历史/屏蔽资源特征重放；这些是诊断，不是心动周期理解或真实闭环消融证据 |
| 病例异质性与证据归档 | 全16开发病例结果与上述输出 | 同病例72→136变化、EF与重建预算权衡、异常与缺失清单、中文HTML报告和可追溯小结果包 |

只重新计算小型GRU的CPU前向，不调用CASL、EF任务模型或优化器。NumPy GRU公式与现有JAX实现核对，最新检查点在真实缓存状态上的概率与熵也必须通过`atol=rtol=2e-4`检查，否则失败并记录原因。早期权重在136状态上重放用于共同输入比较，不冒充早期模型自己的完整闭环。

训练图同时保留原始点与过去16次更新中位数。各样例重建共用固定[-1,1]色域，误差/方差使用共同色域。两粒子方差只是经验分歧，不是校准概率。缺失已有快照会标为未评估并形成`completed_with_gaps`，不会自动补GPU推理。

## 启动与队列

代码用Git部署。现有136收尾程序仍按`task_budget_stages_finish136.md`运行；新程序使用独立输出，不能写入原实验目录。

```bash
cd /root/autodl-tmp/cognitive-ultrasound
git pull --ff-only origin main
nohup /root/miniconda3/envs/casl/bin/python -u scripts/queue_task_budget_analysis_136.py \
  --source /root/autodl-tmp/outputs_casl/task_budget_stages_v1 \
  --output /root/autodl-tmp/outputs_casl/task_budget_stages_v1.offline136_v1 \
  >> /root/autodl-tmp/outputs_casl/task_budget_stages_v1.offline136_v1.queue.log 2>&1 < /dev/null &
```

**这条命令只启动一次。** `.queue.lock`、`.analysis.lock`及PID/进程创建时间检查阻止重复所有者。排队程序不加载数组或模型，等待时仅检查原进程和收尾状态。启动分析前校验136收尾包SHA256；分析检查原身份、模型源文件哈希、训练提交、评价覆盖、缓存哈希和控制器身份。

排队等待上限48小时，CPU分析上限2小时；达到上限或收尾失败会退出并登记失败，不扩展实验。运行时隐藏GPU，固定CPU库线程为2。分析不启动常驻服务或定时任务，完成后程序自行退出，不自动关闭实例。

```bash
cat /root/autodl-tmp/outputs_casl/task_budget_stages_v1.offline136_v1/queue_status.json
tail -n 40 -F /root/autodl-tmp/outputs_casl/task_budget_stages_v1.offline136_v1/analysis.console.log
```

`queued_after136`表示仍等原实验收尾；`running_cpu_analysis`表示开始；`completed`或`completed_with_gaps`表示已产生报告与校验包。原训练`status=stopped`仍是预期状态，用136的`scope_closure_136.json`判断缩减范围完成。

## 产物与续跑

输出含`REPORT_中文.html`、PNG/PDF图、训练/预算/参数/PCA/敏感性/按病例相关/评价CSV、`selected_samples.npz`、`analysis.json`、`provenance.json`、`complete.json`和`bundle_receipt.json`。默认包只有这些分析产物与少量选定帧，没有模型权重、优化器、整段视频缓存或旧包的重复副本。打包后逐项验证内容和SHA256，报告包路径在`bundle_receipt.json`中。

若异常退出，在原队列和分析子进程均已退出后，可执行相同命令；完成产物及输入、代码哈希仍一致时直接复用，不重复分析。缺失或损坏的原实验提交不会被自动修补。输入或代码变化会拒绝复用完成标记，需要另建分析输出目录，不能覆盖已完成证据。

## 本地验证

`tests/test_budget136_offline_analysis.py`覆盖CPU/JAX GRU一致性、无效历史填充、屏蔽/未屏蔽区别、病例等权PCA、136边界与包校验、重复输出所有者、孤立分析子进程、先等待后CPU启动、端到端图表/报告/包和完成结果复用。合成测试图明确标注`SYNTHETIC FUNCTIONAL ONLY`；不将其作为科研结果。
