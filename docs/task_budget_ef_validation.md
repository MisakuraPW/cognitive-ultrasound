# EF动态预算代码本地验收 · 2026-10-08

本轮完成代码与本地验证，未登录AutoDL、未启动科学训练或GPU测量。固定CASL和TBIG上游源码未修改。

## 已验证

- 新测试18项通过。覆盖因果状态、未观测像素变更不影响决策、唯一选线/排除已采线、合法预算与零补采跳过第二次感知、固定扩散随机键、GS硬前向二值及两头任务梯度、RL合法动作及优化器更新、划分交集与数据身份、视频级统计等。
- 另补充首帧EF窗口重复当前假设时的任务梯度累加检查；合计19项功能检查。E1/E2短视频更新恢复与gate的最终3项回归也通过，确认阶段安排在所有训练/开发工作之后。
- E1/E2训练工作器分别在合成工作负载上完成“连续两次更新”和“第一更新后STOP，再恢复第二更新”的实际保存/加载比较；最终控制器、Adam、step和奖励baseline逐位一致，病例/action seed/目标/梯度范数一致。这不是GPU训练收敛证据。
- 真实Torch R(2+1)D-18结构使用**随机功能测试权重**，经独立EF子进程读取包含NumPy标量的旧式checkpoint、严格加载DataParallel前缀、运行32帧输入及输入梯度，梯度有限且非零；服务退出已核对。不是预训练EF准确率验收。
- 冻结任务模型的输入梯度、极坐标坐标轴与中央crop、归一化和参数冻结在CPU测试中通过。
- GS GPU gate的拒绝语义测试通过：模拟任务梯度检查失败时E1登记blocked，E2保留为独立RL方法；不存在E1自动转成成本训练或RL的路径。
- 针对最终任务进程和gate改动的2项回归通过。Torch2.9本地环境有旧TF32 API弃用提示；不是计算失败。Windows的pyreadline退出析构提示发生在测试已结束后，不影响成功退出码。
- Ruff、Python编译与CLI帮助检查通过。两个Bash入口分别通过Git Bash `-n`语法检查；没有执行安装或服务器操作。

## 真实CASL官方权重的CPU反向功能检查

使用已有权重`model.weights.h5`，SHA256：

`3e9d7193a5aa8efaf0e59d4ce5e118d05147ee95a077dcdabb49d1ae77ed8bd2`

112×112、3通道、2粒子，测试仅使用**1个热启动步骤**，以降低本地CPU开销：

|检查|结果|
|---|---:|
|观测mask梯度范数|0.000957123|
|两阶段局部任务adjoint→第一预算头梯度范数|0.000317710|
|两阶段局部任务adjoint→第二预算头梯度范数|0.000648367|
|硬重放最大绝对差|0|

这里的task adjoint是合成可微任务，扩散模型权重是真实官方资产。它证明两阶段CASL局部VJP可运行，不证明EF质量、25步GPU内存/速度或完整视频训练表现。测试前后磁盘权重SHA256相同，优化器只接收预算控制器。

可重复入口：

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -X utf8 scripts/check_task_budget_ef_cpu.py
```

## 报告与包

合成报告fixture已生成并查看PNG，输出2000×800、约200dpi及对应PDF，来源表与置信区间含义保留。图上明确标记FUNCTIONAL SYNTHETIC；缺失种子/病例覆盖的工作点为空心，无插值或自动“成功”判定。

分析包逐成员校验与整体SHA256通过。该包只用于本地报告功能测试，不是待上传科研结果。事实CSV保留用户Prediction/Prediction Lock/Belief Update为空。

本地机器可读记录在`outputs/task_budget_local_validation/`：`junit.xml`、`targeted.xml`、`final_regression.xml`、`final_gradient.xml`、`official_cpu.json`与`synthetic_report/`，按项目约定不进入Git。

## 服务器首次执行仍需验证

- EF官方预训练权重下载、读取与原始TRAIN归一化统计；全观测原视频及极坐标转换后的EF误差。
- 真实JAX CASL与Torch EF两个GPU进程、确定性gather反向的可运行性和显存需求。
- 25步两阶段感知下GS硬重放与两个预算头的任务梯度；本地1步验证不能替代它。
- GPU训练更新边界恢复的数值复现、实际耗时及完整视频评估。
- EF—采集资源曲线和资源匹配对照的科学结果。

正式入口自动先执行GPU/梯度probe，失败照实登记。现有本地通过项不能填成上述科学结果，也不触发重新复现CASL或多任务训练。
