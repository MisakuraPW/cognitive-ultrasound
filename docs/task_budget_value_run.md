# 动态预算补采价值诊断运行说明

本入口执行已设计的P0—P5，不继续训练预算网络。旧v4目录只读，新输出默认`/root/autodl-tmp/outputs_casl/task_budget_value_v1`。首批使用v4已经锁定的8个开发视频，保留FP32、25步、2粒子和现有任务选线；不重新展开历史CASL复现或加速搜索。

## 一次启动

确认实例上已有原来的`casl`环境、数据及冻结权重，以及修复批次的`config.json`、`manifest.json`、`identity.json`、`status.json`。没有这些资产时入口会明确停止，不自动安装CUDA或下载整套数据。

新批次开始前检查输出盘至少24GiB可用空间，供完整帧与恢复状态缓存；运行中低于2GiB时停止并保留提交。续跑只需满足运行中的空间底线，不要求重新空出完整24GiB，也不会自动删除旧数据。

```bash
cd /root/autodl-tmp/cognitive-ultrasound
source /etc/network_turbo
git pull --ff-only origin main
bash scripts/run_task_budget_value.sh start
```

顺序：P0真实GPU／状态恢复检查 → P1三档固定预算 → P2缓存时间敏感性 → P3真实补采后缀 → 有信号时P4有界安排 → P5CPU可预测性筛查 → 中文报告及已校验的小分析包。

这次顶层日志也显示病例、条件、帧和EF读出窗口进度。查看：

```bash
tail -n 40 -F /root/autodl-tmp/outputs_casl/task_budget_value_v1.console.log
```

每阶段的详细日志位于`jobs/P0/console.log`至`jobs/P5/console.log`，不必打开另一个训练日志猜当前进度。

## 先短测与控制操作

如要先检查新硬件并估时：

```bash
bash scripts/run_task_budget_value.sh probe
# calibrate执行同一有界P0，不擅自增加参数搜索或改变精度
bash scripts/run_task_budget_value.sh calibrate
bash scripts/run_task_budget_value.sh status
# P0完成之后start会跳过已提交探测结果并继续
bash scripts/run_task_budget_value.sh start
```

P0记录实测暖帧时间及对应外推；外推不包括全部读出、状态I/O、分支及失败重算，不能当成硬保证。主输入张量／粒子数不因显存空闲擅自放大。

停止与续跑：

```bash
bash scripts/run_task_budget_value.sh stop
bash scripts/run_task_budget_value.sh resume
```

停止在工作负载安全检查点生效；每16帧提交重建、掩膜和完整状态。未提交的小块可能重算，完成的轨迹、窗口读出、病例和阶段跳过。重复启动同一活跃目录会被拒绝，不重复执行一轮。

源码、模型、数据、配置或硬件改变时身份检查要求独立输出，不能编辑旧identity绕过。CPU功能样例与真实GPU结果不可混用。

## 输出与解释

- `REPORT_中文.md`：已完成阶段、固定预算响应、真实边际收益、分配结果与证据边界。
- `summary.json`、`fixed_cases.csv`、`marginal_cases.csv`：病例级指标和筛查结果。
- `cached_sensitivity.png`：4个连续时间窗的修复／破坏响应；`fixed_response.png`及`marginal_response.png`分别展示预算响应与真实边际收益。
- `jobs/P0..P5/result.json`：完整阶段产物，包括18种缓存读出、锁定候选、按视频交叉验证。
- `trajectories/`：完整重建与掩膜、真实逐帧记录、四个锚点状态、原子小块恢复文件。
- `ef_cache/`：绑定当前权重／环境身份与输入内容的任务读出缓存。
- `bundle_receipt.json`：分析包路径、SHA256及校验状态。

分析包自动生成并逐文件校验，供手动下载，排除完整帧数组、恢复状态和模型权重。无需为了阅读结论下载整个缓存目录。

只重生成报告或分析包：

```bash
bash scripts/run_task_budget_value.sh report
bash scripts/run_task_budget_value.sh bundle
```

旧16个确认病例不进入新调参。P2拼接仅是下游敏感性；P3/P4才重新运行受影响闭环。P4候选来自事后信息，不是在线部署策略或严格上界。两种EF读出独立命名，不更改旧实验结果。少采线不等于GPU提速，所有筛查都不会自动替换正式baseline。

P0“无新增观测而追加DPS”对照锁定原低预算轨迹的采集位置，防止后验改变又改变后续选线而混入新的观测差异。正式P1／P3／P4仍正常根据各自状态重新排序。轨迹结果同时区分完整逻辑调用和新执行后缀调用，不能把复用前缀记成此次新增计算。

## 显式扩展16诊断病例

只有用户决定扩展时使用独立目录：

```bash
bash scripts/run_task_budget_value.sh start /root/autodl-tmp/outputs_casl/task_budget_value_extra16 --expand16
```

先检查首批诊断已完成。新病例从原始VAL与CASL未训练划分交集中选取，排除已知历史名单，固定EF分层和随机种子后锁定文件哈希。这是追加诊断，不是新的盲测临床验证；不会改变原8病例结果。扩展目录续跑时同样带`--expand16`。

若路径不同，可复制配置为本地YAML、修改`value_diagnostics.source_batch`等资产路径，再通过`TASK_VALUE_CONFIG=/path/to/config.yaml`指定。代码仍经Git交付，不使用源码压缩包。

## 本地验证范围

合成负载验证状态恢复、精确成本匹配、真实分支控制流、缓存完整性、数据隔离、P0—P5续跑和报告打包；不能证明真实权重质量、真实GPU数值一致性或科学收益。首次服务器运行P0负责真实模型检查，失败就停止。完整计划与诊断门槛见[实验计划](task_budget_value_diagnostics_plan_20261010.md)。
