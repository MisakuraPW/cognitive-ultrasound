# CASL：论文与代码对应

核对来源：本地 CASL 正式论文（IEEE TMI 45(7), 2026, 4034–4046）；用户给定执行计划；`vendor/casl` 固定提交；官方权重配置见 `official_checkpoint_config.json`。中文毕业论文与汇报提供后续研究背景，本阶段不引入其中的 legacy LDM、快速信念滤波、10+4 选线或 WM/planning。

## 必须纠正的执行计划假设

| 执行计划描述 | 核实后的基线 |
|---|---|
| PyTorch diffusion | 官方 Keras U-Net；TensorFlow 训练，JAX 推理 |
| MSE 噪声目标举例 | 上游训练配置选择 MAE；优化噪声 L1，图像 L1 用于监控 |
| 每帧内采初始线，再循环补采至预算 | 每帧一次测量/恢复，用当前后验确定下一帧 K 条线 |
| Uniform 固定不动 | 官方 equispaced 逐帧向右滚动一列 |
| 使用 EchoNet 标准划分 | 官方极坐标筛选后重新按患者划分，6985/500/500 |
| 训练配置名称意味着 W=3 | 上游 YAML 实际是 n_frames=2；权重与论文均为 W=3，本项目设 3 |
| 每轮 5 步可当正式训练 | 上游是调试值；注释为 10000。本项目采用注释值并标为待核实训练参数 |

## 因果流程

```text
时刻 t 的 mask（由 t-1 的后验决定；首帧用官方初始选线）
    ↓
在 polar 图像 x_t 上取得 y_t = mask_t * x_t
    ↓
measurement_buffer.shift(y_t)，历史 mask 对齐
    ↓
官方 DPS / SeqDiff 从历史观测恢复 Np 个 W 帧后验张量
    ↓
最后一帧切片 = 当前 posterior particles
    ├── choose_first → hard_projection → reconstruction_t
    └── 高斯混合熵 → 沿深度求和 → K 次 greedy/reweight → mask_(t+1)
```

策略接口只接收后验粒子、当前已选线和种子；不接收未来帧或未观测真实像素。测量模拟器持有 GT 来生成观测，GT 仅在模拟测量和评估处出现。

## 源码映射

| 内容 | 官方位置 | 本项目入口 |
|---|---|---|
| EchoNet 筛选、极坐标转换 | zea/zea/data/convert/echonet.py | data.convert |
| 时序 U-Net / cosine schedule / MAE | zea/zea/models/{unet,diffusion}.py | training.train |
| 配置/权重加载/后验函数 | ulsa/agent.py: setup_agent | models/diffusion.py: CASLLoop |
| FIFO 测量和 mask | ulsa/buffer.py | 使用官方 AgentState/FrameBuffer |
| 整体恢复与下一帧动作 | ulsa/agent.py: recover | 使用原函数；仅包一层 Sampler 和计时 |
| 策略注册与封装 | ulsa/selection.py; ulsa/agent.py | acquisition/base.py |
| GMM 像素熵及邻线重权 | zea/zea/agent/selection.py: GreedyEntropy | 官方策略直接调用 |
| 可视化熵 | ulsa/entropy.py: pixelwise_entropy | 保留在轨迹，与算法内熵单独计时 |
| 掩码投影 | ulsa/agent.py: hard_projection | 原样调用 |
| LPIPS | zea/zea/metrics.py; zea/zea/models/lpips.py | evaluation/metrics.py |
| EchoNet 分割/scan conversion | ulsa/downstream_task.py | evaluation/segmentation.py |

## 张量与默认参数

数据 HDF5 为 `(T,112,112)`，键 `data/image`，范围 `[-60,0]`。训练及后验网络为 channels-last：`(B,112,112,3)`，末轴是时间，不是 RGB。归一化到 `[-1,1]`。状态后验为 `(Np,112,112,3)`；当前粒子为 `(Np,112,112,1)`。

权重输入 W=3，Np=2，omega=10，num_steps=500，initial_step=450，首帧从 step 0 开始；后续采用官方 SeqDiff 热启动。重建取第一个粒子（不是平均），FP32。论文的高速模式使用不同步数/精度，不混进质量基线。

GMM 熵为对粒子两两差异的高斯核求和再取对数，`entropy_sigma=1`。选线沿深度求和，并对被选线邻域重权；不能替换成简单 top-k variance。统一 Sampler 是协议/适配层，尚无 WM/planner 实现。

## 上游已知行为：保留并披露

1. `hard_projection` 用 `measurement != 0` 判断被观测像素，归一化后恰为 0 的已测量像素不会投影。本项目保留此行为，测试明确覆盖；未来修复须作为不同实验。
2. GreedyEntropy 将已选位置的熵设为 0；当全部熵为 0 时可重复选择同一索引，k-hot 的实际线数可能小于预算。本项目保存实际线数，报告预算不一致帧数，不偷偷补随机线。
3. 官方 inference README 的 `--config` 与当前解析器的 `--agent_config` 不一致；本项目独立 CLI 直接调用原函数。
4. 官方训练恢复 helper 存在把 `load_weights()` 返回值当作 model 的问题；本项目不调用该 helper，使用 TensorFlow checkpoint 管理网络/EMA/优化器。
