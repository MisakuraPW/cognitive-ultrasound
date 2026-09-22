# 所有剩余准备实验：一次启动

2026-09-22。总入口为 `scripts/run_all_preparation.sh`，固定依次运行：

1. **closure-v4**：多预算加速确认、预测参照、公平动态预算闭环、学长论文启发 BF 的历史训练对照，以及条件 TBIG 检查；18 个既定任务，计算上限 3 小时。
2. **Torch CASL 对照**：原版 JAX → Torch eager → compile → CUDA Graph；冻结官方 EMA、无训练，计算上限 90 分钟。

不重跑已完成的 preparation_v1 / preparation_followup_v3，不重新转换数据，不启动完整 CASL 训练或 paper 全测试集评估。总入口没有增加实验或改变两个子协议。每批自己的硬件与数值预检保留。

## 本次故障修复与续跑

v1 的 JAX 中间图像记录错误已修复。v2 进一步揭示：即使权重与随机噪声逐元素完全一致，把 RNG 从编译图内移到图外也会改变 GPU 浮点融合边界。两帧诊断用 optimization_barrier 完全重现该路径，排除了抽样和权重不同的解释。现在将有限的数值偏离记录为实验负结果，非有限值仍然拒绝；不再由数值断言中止整个比较。原轨迹和同输入核心的结果分开保存，任何重放核对未通过都禁止认定等价加速或 32 FPS 成功，容差不变。

v1/v2 失败证据与诊断保留；新 Torch 和总入口使用 v3 目录，closure-v4 继续跳过。新批次最多运行 81 分钟，连同旧批次 112.19 秒和两次诊断的上限 240+180 秒，仍低于最初 90 分钟的子任务额度。

本实例基础 PyTorch 环境原缺 h5py，已补充；GPU 前向及输入梯度计算通过。首次使用其他镜像时，可在已有 Torch 环境安装 requirements/torch_benchmark_tools.txt 中的辅助依赖，不需要重装 Torch/CUDA。

## 一次粘贴启动

已有实例应保留 casl 环境、官方 vendor/checkpoints、转换数据、v1/v3 的原始输出目录。确认旧实验进程已经结束；**运行期间不更新源码**。若旧版本尚有未完成的任务，先按原版本续跑；代码/硬件变化会被子流水线的身份检查拒绝，不能强行混写旧输出。

```bash
(
set -euo pipefail
cd /root/autodl-tmp/cognitive-ultrasound
if [[ -n "$(git status --porcelain --untracked-files=normal --ignore-submodules=untracked)" ]]; then
    git status --short --ignore-submodules=untracked
    echo '工作区有修改，先保留处理后再更新；不要强制覆盖。'
    exit 2
fi
git pull --ff-only origin main
bash scripts/run_all_preparation.sh start
)
```

沿用已有 PyTorch 镜像时自动寻找可用的 GPU Torch Python；也可在启动命令前设置 `TORCH_PYTHON=/root/miniconda3/bin/python`。不自动重装 CUDA。没有 Torch 环境或缺外部 TBIG 资产，会在对应报告明确留档。

后台运行，断开 SSH 不影响任务。两批顺序等待，不通过两次后台 start 并行抢同一张 GPU。某批失败后仍尝试另一批；主动暂停则不启动后续批次。

## 进度、暂停与续跑

统一日志包含两个子协调器的 STAGE 输出：

```bash
tail -n 60 -F /root/autodl-tmp/outputs_casl/all_preparation_v3.console.log
```

具体任务的逐帧/训练输出仍在原目录：

```text
/root/autodl-tmp/outputs_casl/preparation_closure_v4/jobs/<任务名>/console.log
/root/autodl-tmp/outputs_casl/torch_casl_v3/{reference,eager,compile,graph}.log
```

```bash
cd /root/autodl-tmp/cognitive-ultrasound
bash scripts/run_all_preparation.sh status
bash scripts/run_all_preparation.sh stop
# 等当前子流水线停止后，再恢复；不用再次 git pull：
bash scripts/run_all_preparation.sh resume
```

resume 在确认没有旧协调器或 worker 存活后，清除总入口及这两个子批次的 STOP 标记。已完成的子批次自动跳过；缺包时只重建子报告/包。未完成子批次调用原有 resume，各自计时与恢复粒度保持不变，不重置计算额度。

`finished_with_gaps` 且报告完整也视为该批已收尾，避免对缺 TBIG、超限或负结果自动无限重复。暂停状态不算收尾。总入口的 report 只重新生成总索引，不训练。

## 跑完与下载

总索引：

```bash
cat /root/autodl-tmp/outputs_casl/all_preparation_v3/REPORT.md
```

下载两个新结果包及校验文件：

```text
/root/autodl-tmp/outputs_casl/preparation_closure_v4.results.tar.gz
/root/autodl-tmp/outputs_casl/preparation_closure_v4.results.tar.gz.sha256
/root/autodl-tmp/outputs_casl/torch_casl_v3.results.tar.gz
/root/autodl-tmp/outputs_casl/torch_casl_v3.results.tar.gz.sha256
```

合计 **4.5 小时是计算子任务上限，不是完成时间保证**。初始化、报告、校验打包另计；失败/超限会留档，实例不会自动关机。两批都生成终态和结果包后可手动关机，再下载/分析。

详细科学设置仍见 [准备收尾协议](preparation_closure.md) 和 [Torch 对照协议](torch_casl_benchmark.md)。本次只新增总编排，5 项本地协调测试通过：终态跳过、暂停保护、失败后顺序继续、正确传递恢复参数、拒绝重复存活进程；没有因此启动云端计算。
