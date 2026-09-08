# 实验协议与验收

## 数据

原始 EchoNet 为已成像 B-mode 视频，不是 RF/channel data。官方从 scan-converted 图像回到 polar，并排除几何不一致病例；本复现以该模拟 measurement model 为范围。不能据此证明真实设备采集帧率或声学性能。

EchoNet 本身已经有 train/val/test 划分，由 `FileList.csv` 的 `Split` 列指定，[EchoNet 官方加载器](https://github.com/echonet/dynamic/blob/master/echonet/datasets/echo.py) 直接读取该列。CASL 没有沿用这个分组：[论文 IV-A](https://arxiv.org/html/2508.08782v2#S4.SS1) 说明先排除无法一致转换为极坐标的扫描几何，再对剩余视频按患者重新划分为 6985/500/500；[官方代码说明](https://github.com/tue-bmd/casl#dataset) 提供作者使用的固定清单。本项目下载并固定作者清单，不自行重新随机分组。

保持该划分是为了匹配论文评测和官方先验训练集。若改用 EchoNet 原始 test 配合 CASL 预训练模型，其中某些病例可能已经属于 CASL train，需核对 ID 交集后才能声称未见病例测试。当前没有真实 FileList.csv，不能给出交集数量；云端 preflight 会输出交叉计数。以后在标准 EchoNet 协议上比较方法，应另设实验、核实或重新训练先验，让所有对比方法使用相同划分。

论文写排除 2044 段，而 EchoNet 总数 10030 与 CASL 清单合计 7985 相差 2045；这里不猜测一例差异的原因，以发布的逐病例清单与实际输入核验为准，不用总数相减生成名单。

固定 Hugging Face 划分 revision `534aa314fe5a54912483b7cab936ec023f934d10`：

| Split | 视频 | 官方清单帧数 | 少于 100 帧的视频 |
|---|---:|---:|---:|
| train | 6985 | 1222063 | 237 |
| val | 500 | 84953 | 15 |
| test | 500 | 88167 | 20 |

这些是发布清单统计，不表示视频已下载。论文描述每患者 100 帧；官方实现实际 `[:100]`，遇到短视频保留其所有帧，本项目一致，不补帧。先按帧算指标，再患者内平均，最后对患者等权平均。

转换使用固定版官方 H5Processor。脚本单进程运行，以便错误向上抛出；不改图像算法。完成后核对患者名称、互斥性、HDF5 key、形状、有限值和数值范围。大规模训练前运行完整 `audit-data`。

## 方法与预算

Random、滚动 Uniform、CASL 共用同一个官方 diffusion checkpoint、相同患者与预算；seed 通过固定全局 seed 与患者名 SHA256 派生，与遍历顺序无关。每个患者重置状态。各方法/预算的后验随机流遵循同一官方代码。

默认 2/4/7/14/28/56/112 条线。保留真实选中数量；若官方 greedy 退化导致实际数量不同，报告标出，不能在名义相同预算下直接声称公平获胜。

## 指标定义

- PSNR：在 polar 112×112 图像上，`[-1,1]→[0,255]` 后截断为 uint8，再转浮点做误差。完整图相等时是 +inf，不伪造有限值。
- SSIM：scikit-image，data_range=255，Gaussian sigma=1.5，population covariance。它是执行计划新增指标；不与其它 SSIM 默认参数混用。
- LPIPS：官方 zea 的 VGG 实现与官方 checkpoint，不替換为常见的 AlexNet 默认值。固定本地权重路径；每次运行保存哈希。
- 分割：官方 EchoNet 模型和 polar→Cartesian 转换。Dice 是重建图分割与完整图分割的**一致性**，不是人工标注精度。完整图连续 ≥5 帧出现多个连通分量的病例，仅在 Dice 汇总中排除；采用显式 8 邻域，保存排除标记及掩码，方便审计。

## 时间

正式质量/总耗时：完整 recover JIT，调用后同步设备。首帧完整 DPS 单独统计；稳定帧均值排除每位患者第 0 帧。先预热首帧与 SeqDiff 分支，再 reset seed/state。预热单独保存，磁盘、指标、可视化、额外诊断熵不进入算法总时间。

`--profile`：后验 JIT，恢复/选线不整体 JIT，通过同步时钟拆开 diffusion/entropy/action/projection/overhead。分模块之和构成 total。这种拆分影响融合和调度，不能冒充论文 JIT 吞吐速度，也不构成算法提速。

## 完成门槛

工程检查与科学复现分开。只有真实数据处理、完整官方权重推理、三策略同协议测试、PSNR/SSIM/LPIPS、轨迹及 runtime 实测完成后，才能进入论文结果比对。论文 7 条线 PSNR=23.2 dB 是参考值，不是验收的自动通过阈值；需要排查数据、权重、版本、精度及参数差异。

正式训练必须独立记录收敛和测试集结果。当前阶段没有宣称 GPU 环境已经验证、模型已经训练好或论文结果已经复现。
