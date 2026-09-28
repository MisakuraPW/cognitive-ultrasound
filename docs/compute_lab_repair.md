# v2 修复与继续执行

v2 保持原样。新批次显式继承相同配置、环境、权重和病例下的兼容结果；核对原代码指纹、计算类 AST 和逐文件 SHA256。不会修改旧 identity，也不会将修复前的 compile/graph 失败复制为新批次终态。原失败日志另存 inherited_evidence。

修复模式开启完整 Torch 确定性算法，限制编译线程并使用 spawn，保留输入梯度；没有 FP32→FP16、TF32 或容差放宽。工程意图不等于与 JAX 等价，仍须单独通过各层数值和闭环检查。编译/Graph 的内部校验与跨框架等价性分开。

## 一次启动与阶段边界

```bash
cd /root/autodl-tmp/cognitive-ultrasound
git pull --ff-only origin main
bash scripts/setup_compute_lab.sh && \
COMPUTE_LAB_INHERIT=/root/autodl-tmp/outputs_casl/compute_lab_v2 \
COMPUTE_LAB_PHASE=torch \
bash scripts/run_compute_lab.sh start /root/autodl-tmp/outputs_casl/compute_lab_v4
```

`COMPUTE_LAB_PHASE=jax` 在固定 JAX DEV 后停稳；`torch` 在修复短测与允许投入的 FP32 DEV 后停稳；默认 `all` 继续其余任务。安装成功收据与启动时同步依赖核验双重阻断失败安装，即使手动另敲 start 也不会绕过。

确认处于 `paused_at_boundary` 后继续全部剩余任务：

```bash
bash scripts/run_compute_lab.sh resume /root/autodl-tmp/outputs_casl/compute_lab_v4
tail -n 60 -F /root/autodl-tmp/outputs_casl/compute_lab_v4.console.log
```

查看、边界暂停或立即停止（均显式指定批次）：

```bash
bash scripts/run_compute_lab.sh status /root/autodl-tmp/outputs_casl/compute_lab_v4
bash scripts/run_compute_lab.sh pause-after-current /root/autodl-tmp/outputs_casl/compute_lab_v4
bash scripts/run_compute_lab.sh stop /root/autodl-tmp/outputs_casl/compute_lab_v4
```

边界暂停在当前工作器结束后、下一工作器启动前检查；立即 stop 可能中断当前病例，病例完成收据仍可续用。不要编辑活动工作器使用的源码。源码或环境变化仍要求新批次；本次继承仅适用于已审查的 v2 修复范围，不能当作通用绕过指纹工具。

## 有限成本筛选

- 短测固定输入闭环时间超过官方四倍，不再投入未完成的开发组；四倍是计算投入上限，不是质量门槛。已完成兼容结果仍保留。
- Torch 仅在 compile/graph 内部正确性通过，且实测 DEV 闭环快于官方 JAX 时展开四个近似版本。不自动退回 eager。
- 训练先短测 eager/graph 两种模式。graph 预计成本超过 eager 四倍则不继续该路线，保留原因；eager 恢复对照仍执行。
- 每个训练模式有连续200步和独立100＋100恢复，共400次真实更新；4负载、2模式最多3200次，另有初始化、短测与容量探测。
- 质量失败、数值失败、实现故障、资产缺失和不值得投入分别记录；不会为了正结果增加病例、粒子或调参。

成本、已完成结果和剩余阶段分别见 `estimates.json`、`matrix.json`、`scheduling.json`、`remaining_tasks.md`。确认集不参与版本选择，默认 baseline 仍是官方版。

## 完成与下载

`status.json` 为 `completed` 或 `finished_with_gaps` 表示有限清单执行结束；后者必须检查失败/跳过项，不能当作全部验证成功。`archive_failed` 表示计算结束但打包未成功。结果包必须有同级 `.bundle.json` 才视为打包校验完成。

下载同级 `compute_lab_v4.results.tar.gz`、其 `.sha256` 和 `compute_lab_v4.bundle.json`。包内含报告、事实CSV、兼容性继承来源、旧失败证据、确认结果及训练验证；不包含可重建的大型参照缓存，不传源码包。

