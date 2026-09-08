# Diffusion training

模型来自固定版 `zea.models.diffusion.DiffusionModel`，不重写 U-Net 或 DPS。输入 `(B,112,112,3)`，官方极坐标 B-mode 强度归一化 `[-1,1]`，连续且重叠的三帧窗口，不跨患者。

cosine 信号/噪声 schedule 的 min/max signal rate 为 0.02/0.95。网络 widths=[32,64,96,128]，block_depth=2，embedding_dims=32，embedding_max_frequency=1000。训练优化 `mean(abs(epsilon - epsilon_theta))`；图像 MAE 是日志项。AdamW lr=1e-4，weight_decay=1e-4，EMA=0.999。

## 参数证据

论文与 checkpoint 均明确 W=3；上游 training YAML 的 n_frames=2 是不一致配置，已经纠正。500 epochs、batch size 32 来自上游 YAML。steps_per_epoch 的有效值为 5，旁边注释 10000；本项目正式配置用 10000，**不是声称论文明确披露了此值**。validation_steps=5 保留上游值。正式训练前应确认 epoch/step 定义与算力预算，并将变更保存到 manifest；不能用短程 smoke 训练冒充达到论文精度。

## 精度

默认 FP32，与论文训练一致。`--precision mixed_float16` 是独立精度实验：加入 FP32 loss、FP32 diffusion 代数计算和 `LossScaleOptimizer.scale_loss`，U-Net 使用混合精度，保留噪声 MAE、EMA、网络和 schedule。官方自定义 train_step 没有缩放 loss，denoise 中还会混用 half/float，所以不能只切换全局 policy。AMP 的 hub 导出为 FP32 推理配置，权重不变，能够加载到统一基线评估器。

## 保存与恢复

每个完整 epoch 保存网络权重、EMA、AdamW 优化器状态、epoch 计数；同时导出 `hub/config.json` 与 `hub/model.weights.h5`，可直接交给同一个 CASL 推理适配器。TensorBoard 与 CSV 在运行目录的 logs 中，无需 W&B 账号。

`save_weights_every` 控制独立 epoch 权重归档间隔，省略时保留每 epoch 归档；AutoDL 配置设为 50，最后一个 epoch 也归档。每个 epoch 的最新 hub 和最近 3 个优化器恢复点仍照常保存。`timing/epoch_*.json` 记录逐步墙钟时间和含验证/保存的 epoch 时间，可用真实数据 pilot 外推预算；不将合成 smoke 用作训练速度估计。

`--resume` 要求配置和 split hash 相同；中断时恢复上一个完成的 epoch。恢复优化器不等于逐位重现不中断的随机流：数据迭代位置和全局随机流重新初始化，该限制写入 manifest。正式复现应从固定 seed 完整训练并保留依赖锁、日志及数据哈希。

完整训练仅允许检测到 TensorFlow GPU 后启动。`--smoke` 使用合成数据、一个训练步与一个验证步，是工程验收；不会加载 EchoNet，也不证明训练收敛或达到论文水平。
