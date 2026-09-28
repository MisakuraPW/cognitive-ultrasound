# 本地验收记录 · 2026-09-28

本轮按用户补充要求，仅完成代码与本地验证；没有登录/启动 AutoDL 实验。SSH 地址当时不可达，因此没有取回完整 closure-v4。新协议的 GPU 数值、完整闭环质量、显存与速度结果均待运行。

## 已执行

- 28 项通过、1 项跳过：新协议、病例隔离、病例级统计、质量失败拒绝、异步写入顺序/缓冲区、整段工作器与恢复、真实子进程 STOP/OOM 退出回收，以及已有官方权重的 Torch 算子与输入梯度测试。
- 跳过项是 CUDA Graph 的实际 GPU 捕获测试，本机无可用 CUDA GPU。新运行器在服务器首次图捕获时还会实际核对变化输入和输出缓冲区别名，失败则记录该模式失败。
- BF codec 用小网络与合成极坐标 HDF5 做了真实 CPU 优化器更新：连续200步，对比100步退出后在新进程恢复100步；最终参数、优化器、最终梯度和恢复段损失通过检查，冻结模块保持不变，初始快照未改变。
- BF prior/filter、CASL 的小网络 CPU eager/graph 短更新通过梯度、参数、优化器、冻结模块与 CASL EMA 检查。这些是实现功能测试，不是 EchoNet 科学实验或 GPU 性能结论。
- 官方 EMA 权重的网络、DPS 输入梯度、熵、选线通过已有 CPU 对照；Torch `aot_eager` 编译在两组变化输入下与 eager 对照通过。它不替代 CUDA Inductor 的服务器验收。
- 本地核查 mixed_bfloat16 下官方模型的卷积实际 compute dtype 为 bfloat16；GPU BF16 可运行性和质量仍未验证。
- 之后快速回归：21项通过、1项CUDA跳过。Ruff 检查、Python 编译检查、两个 Bash 入口的语法检查通过。
- 合成报告样例完成 PNG/PDF 渲染检查：通过/未通过以不同形状和颜色区分，各质量门槛、确认上界及显存分别显示。样例明确标记 SYNTHETIC，仅做版面验收。

测试命令：

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest -q tests/test_compute_lab.py tests/test_compute_lab_integration.py tests/test_compute_lab_training.py tests/test_torch_casl.py::test_official_ema_dps_entropy_and_actions tests/test_torch_casl.py::test_compiled_input_gradient_and_two_changed_inputs tests/test_torch_casl.py::test_zero_entropy_ties_preserve_upstream_behaviour
```

机器可读结果保存在本机 `outputs/compute_lab_validation_20260928/junit.xml` 和 `fast-regression.xml`。这些本地输出不上传 Git。测试中的小网络、合成样本、模拟OOM退出均不会写入正式科研结果目录。

## 等待服务器执行的验收

|项目|状态|
|---|---|
|完整 closure-v4 包取回、逐成员校验、原设计缺失项收尾|待服务器|
|42个未使用 validation 病例锁定与真实 HDF5 吞吐|待服务器；本地只测合成文件的协议|
|所有固定 JAX/Torch GPU 候选、冷启动及同步计时|待服务器|
|开发后至多4候选、32病例全视频/3种子/3预算确认|待服务器|
|原宽度、batch32 的 CASL 和真实 BF HDF5 200步/100＋100恢复|待服务器|
|真实 GPU OOM、Inductor/CUDA Graph、实际显存峰值|待服务器|
|正式速度—质量曲线、使用建议、结果包|待服务器；目前无新加速收益结论|

正式入口会先短测，不会自动把 FP16、BF16、减步或少 DPS 当成新 baseline。出现质量下降、非有限值、数值不等价或缺资产时照实保留，不自动扩大实验清单。
