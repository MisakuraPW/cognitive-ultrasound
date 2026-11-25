# 两条加速路线的学校作业

先提交 `bf-smoke` 到已验收的 `gpu-L20`。短测用 CASL 划分，训练 8 个视频，三个阶段各一轮。开发、前馈评估与共同测速使用互斥的验证病例，冻结扩散权重，同步比较 14 条线。前馈训练固定 10＋4，保留真实分组观测、同观测插值和连续递归的后段误差。学校批次暂用 CASL 公布的划分，未核对原始 EchoNet CSV，未做 EF 评估。作者模型、源码和权重未取得。

较大 `bf-train` 按既有 1024 视频训练设置运行。根据冒烟实测和当前队列决定是否提交。两种任务分别用独立输出目录，短测权重用于机制验收。

脚本优先使用 `CASL_SEU_TORCH_PYTHON` 或 `~/casl/envs/fast-belief/bin/python`，再检查已有 CASL 环境是否有可用 CUDA Torch。计算节点必须通过真实 CUDA 矩阵乘法。学校 CUDA/cuDNN 和官方代码检查沿用原入口。

Slurm 日志为 `~/casl/jobs/acceleration/casl-任务-作业号.log`。结果在 `~/casl/jobs/acceleration/任务-作业号/`，前馈联训权重在 `belief/filter/best.pt`，病例和帧级评估在 `benchmark/`。同时检查 Slurm 终态、退出码和 `SEU_ACCELERATION_OK`。
