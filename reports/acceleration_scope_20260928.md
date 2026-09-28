# 历史加速证据适用范围追加

本文件追加解释，不修改旧报告。依据本机已取回的 `outputs/server_repair_20260922` 记录；完整 closure-v4 尚待服务器可用后取回校验。本轮用户明确选择先完成代码和本地验证。

|已有证据|可以说明|不能说明|
|---|---|---|
|`official25_fp16`|该组合在旧小样本协议下的时间/质量表现|减步、FP16各自的独立贡献；本轮严格确认通过|
|Torch eager 50/25：0.578/1.161 core FPS|当前端口实现的旧微基准速度|PyTorch 框架性能上限|
|Torch compile 50/25：3.422/6.826 core FPS|当前编译方式相对 eager 的收益|与原版完整闭环数值等价|
|Torch graph 50/25：10.030/20.074 core FPS|当前图捕获方式的旧核心速度|已达到32 FPS；完整端到端32 FPS；无损替代|
|旧 `comparison.json` 全部 `parity=false`、`jax_replay_passed=false`|旧轨迹数值门槛没有通过|模型导出每一层都错了；所有精度差异都来自 Torch|
|RNG eager/JIT 与权重数组完全一致，图边界诊断产生 max abs≈0.12397 的输出差异|融合边界是此诊断中的一个差异来源|所有后续案例的唯一差异原因|
|总协调器 closure `finished_with_gaps`|曾执行/跳过了已有收尾批次|没有完整包就能复核每项预算、预测、BF历史证据|

原文件定位：

- `outputs/server_repair_20260922/torch_casl_v3/comparison.json`
- `outputs/server_repair_20260922/torch_reference_diagnosis_v2/diagnosis.json`
- `outputs/server_repair_20260922/all_preparation_v3/status.json`

学长“4090可以32 FPS以上”的回复是待验证线索，并同时提到了精度下降。新协议不把这句话作为保证，也不把目前20 FPS作为能力上限。

新批次先固定参照和未使用的 validation 病例，再分别核对算子、共同输入重放与自己的历史闭环。修改类型 A 并不预设数值无损；B 即使质量接近也保留独立版本。确认集只用于确认，所有近似版本仍由用户明确决定是否采用。

本轮实现与操作入口：[compute_lab.md](../docs/compute_lab.md)。尚无本轮 GPU 速度—质量实验结论，生成的运行器和本地测试不能替代这些测量。
