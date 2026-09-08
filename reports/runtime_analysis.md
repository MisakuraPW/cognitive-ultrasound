# Runtime 状态

本地已经执行同步组件计时测试，原始记录在 `results/pretrained_profile_check.json`。这是合成输入、缩短步骤的 CPU 检查，不是正式推理性能结果。

真实 EchoNet/RTX 4090 时间均待运行。`casl-repro evaluate --profile --output results/profile` 会生成每帧计时与分模块报告。常规评估只给整体 JIT 时间和投影时间；缺失的组件时间不填 0。

profile 模式中 total = diffusion + entropy + action + projection + overhead。磁盘导出、指标、诊断熵和编译预热单独处理，不计入算法总时间。首帧单独报告。
