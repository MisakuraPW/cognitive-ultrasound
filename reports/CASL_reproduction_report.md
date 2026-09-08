# CASL 复现阶段报告

**当前结论：工程基座、核心复现代码和本地兼容性测试已完成；真实 EchoNet / GPU / 论文数值复现尚未完成。**

## 1. Environment

实际机器为 Windows、RTX 3060 Laptop 6GB；本次模型验证使用 CPU。主要库版本和实际硬件见 `environment.md`、`verification.json` 和 `requirements/cpu-verified.txt`。没有启动 AutoDL 或租用云资源。正式目标 RTX 4090 的 CUDA/显存/性能须在云端另行实测。

## 2. Dataset

已下载并核对官方患者划分：6985 train / 500 val / 500 test，极坐标输出应为 `(T,112,112)`，范围 `[-60,0]`。当前本地项目没有 EchoNet 医学视频；`dataset_statistics.md` 中 available=0 仅描述本地状态。用户已说明云端有历史 EchoNet 数据；已按 2026-09-08 交接文档准备只读检查、独立极坐标缓存和配置，尚未连接实例确认实际文件。

已实现官方转换入口、互斥患者清单检查、HDF5/数值校验。推理归一化与官方 pipeline 的一致性由测试核对。全部临时合成数据都标注为测试夹具。

## 3. Method

使用原版 CASL + zea 的 perception-action loop：Keras/TensorFlow 训练，Keras/JAX 推理。W=3，Np=2，omega=10，默认 500/450 steps，choose_first，hard_projection，FP32。当前帧后验选择下一帧的扫描线。Random、逐帧滚动 Uniform 和 GreedyEntropy 共享先验。

已修正执行计划与上游之间的假设差异，详见 `docs/CASL_architecture.md`。没有加入 WM、planning、快速滤波或新的 measurement model。

## 4. Reconstruction results

| Method | EchoNet PSNR | SSIM | LPIPS |
|---|---|---|---|
| Random | 待云端运行 | 待云端运行 | 待云端运行 |
| Uniform | 待云端运行 | 待云端运行 | 待云端运行 |
| CASL | 待云端运行 | 待云端运行 | 待云端运行 |

本地已使用官方预训练权重完成三种方法的合成连续帧测试，以及 PSNR/SSIM/官方 VGG-LPIPS 的完整评估和报告生成。该测试只有 2 个扩散步骤，结果仅检查接口和文件流，不能填入上述表格。

## 5. Sampling efficiency

代码支持预算 2/4/7/14/28/56/112；可自动生成预算-质量图和患者等权汇总。实际 EchoNet 曲线待运行。轨迹保存实际选中条数，退化导致预算不足时会披露，不替换官方动作。

## 6. Runtime

已验证同步计时和 diffusion/entropy/action/projection/overhead 分解。CPU 合成测试不能当作 RTX 4090 的性能数据。正式 JIT 总时延与逐模块 profile 模式分别记录；首帧与后续 SeqDiff 分开。

## 7. Training and downstream

FP32 和可选 mixed_float16 均已完成一次合成训练/验证，保存 checkpoint、优化器及 TensorBoard。FP32 checkpoint 已恢复检查；两种训练导出的 preset 均已在 JAX 中重载并产生有限值输出。这些权重未经过真实数据训练，不是可用于毕业设计结果的模型。

官方 EchoNet 分割权重、scan conversion 和接口已用相同合成输入检验，输出形状正确、self-Dice=1。它只证明接口一致，未验证医学精度；真实下游评估待运行。

兼容性处理：TF 后端要求 SeedGenerator；AMP 路径补充 loss scaling 和 FP32 diffusion 代数；tf2jax 0.3.6 使用了 JAX 0.6 已删除的 API，因此固定为 0.3.7。Windows TensorFlow 中文路径错误通过 ASCII 临时测试路径处理。官方算法文件未修改。

## 8. Reproduction gap / 下一阶段

1. 核实 AutoDL 上已有的共享 EchoNet 原始数据及挂载，运行只读 preflight、独立转换与完整统计核验，见 `docs/autodl_preparation.md`。
2. 在你指定的 AutoDL 环境安装依赖，确认 JAX/TensorFlow 可见 GPU，保存实际 lock 和环境报告。
3. 首先跑完整默认步数的 pretrained demo；再做正式测试集三方法预算对照。
4. 运行正式 diffusion training，并以训练所得 checkpoint 重复同一评估。
5. 与论文参考（例如 7 条线 PSNR 23.2 dB）比较，分析数据、参数、精度及训练差距。未满足这些条件之前不宣称论文复现完成。

代码入口与命令见项目 README；详细本地验证记录见 `verification.json`。

首次基座验收自动化测试：22 passed。代码静态检查通过，`pip check` 未发现依赖声明冲突。另已执行上文所述的真实模型合成集成测试；这些检查的通过不代替真实数据复现。

2026-09-08 数据交接准备后：37 passed（原有测试及新增的只读清单、路径保护、配置继承测试），Python 代码静态检查和格式检查通过。新增检查使用本地合成文件，不是云端数据验收；未重跑与此次配置改动无关的完整模型集成实验。

2026-09-09 操作指南准备后：44 passed；加入根目录 requirement.txt（复用系统 CUDA）、五预算完整测试配置、真实数据 pilot 计时与外推、结果导出及校验命令。新增训练计时和保存逻辑通过本地 CPU 合成训练检查；三策略官方权重合成推理验证了无轨迹输出、完整病例墙钟计时及续跑，产物在忽略目录 `results/runbook_synthetic_20260909`，不是医学实验。尚未在用户选择的 AutoDL 镜像上安装或验证系统 CUDA/cuDNN，不能将 Python 依赖文件称为已验证的云端 lock。完整云端步骤见 `docs/autodl_runbook.md`。
