# 准备阶段收尾包：本地验证记录

2026-09-20。准备代码与交接材料已完成，尚未在 AutoDL 启动 closure-v4；不把本地合成检查当作新增医学实验结果。

## 固定交付

- `configs/preparation_closure.yaml`：18 个固定任务的配置，计算累计上限 3 小时。
- `preparation/closure.py`：有界协调、历史病例排除、来源校验、任务状态、报告与 Excel 事实表。
- `budget_protocol.py`、`closure_budget.py`：混合预算数据、同历史候选标签、train-only 预测、可兑现的相同总额度、五策略对照。
- `closure_belief.py`：同初始 filter、同更新/帧数的两组历史对照；冻结 codec/prior，自动执行预检。
- `scripts/run_preparation_closure.sh`：沿用已有环境和资产的一键后台入口；停止/续跑不增加原预算。
- `docs/preparation_closure.md`：范围、协议、上传、启动、恢复、结果及 Excel 交接。

## 已通过的检查

41 项相关本地测试通过，分开进程运行 JAX 与 TensorFlow：

| 检查组 | 数量 | 内容 |
|---|---:|---|
| preparation_closure / preparation / followup / preflight | 26 | 精确总预算、无标签决策、train-only 拟合、预算覆盖、病例排除、有限任务清单、恢复、空/失败报告、原有上限与准入 |
| belief_kernels | 6 | eager/graph 的目标/更新、重置间隔目标一致性、策略边界 |
| belief_filter / preparation_belief | 7 | 三阶段功能、优化器恢复、因果观测；新续训重置 Adam；实际 TF 两组训练和三权重评估；冻结数组不变 |
| preparation_jax | 2 | 上游求解器数值对照及实际梯度调用；历史/随机状态复制隔离与显式采集掩膜 |

新增续训测试起初漏写合成 split 的 `test` 键，修正夹具后通过；没有放宽产品检查。完整 GPU 数值/质量与速度检查仍在云端运行时执行。

Ruff、Git diff 空白检查、两个 Bash 启动器语法检查和无 GPU 的 plan 命令通过。合成预算流水线完整生成报告、CSV、控制图和固定位置图；图像已目视核对。TF 合成流水线完整走过新 BF 训练/评估与图像生成。

## 完成标准

准备代码交付不要求所有科学猜测成立。本批把各任务完成、失败、超限、缺条件分别写入结果，不自动扩样、续训或搜索。TBIG 资产不足明确留档，不冒充完成。真实运行结束后，使用报告和事实表进入逐创新点的研究记录，Prediction/Lock/Surprise/Belief Update 由用户填写。

这次没有登录或启动云端实例，没有更改旧权重、旧实验目录和科研 Excel 的人工栏目，也没有触发全量转换、CASL 完整训练或全测试集评估。
