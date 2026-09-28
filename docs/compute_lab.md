# 加速审计与计算基座：固定批次 v1

本入口落实 2026-09-28 的协议，替代新批次默认使用 `all_preparation` 的做法。历史入口、结果和报告保留。本轮不完整训练 CASL，不新增科学方向，不自动采纳近似版本。

## 一次启动（AutoDL）

确认旧实验已经结束，再在服务器执行。代码通过 Git 交付，结果包不含源码。

```bash
cd /root/autodl-tmp/cognitive-ultrasound
source /etc/network_turbo >/dev/null 2>&1
git status --short
git pull --ff-only origin main
bash scripts/setup_compute_lab.sh
bash scripts/run_compute_lab.sh start
tail -n 60 -F /root/autodl-tmp/outputs_casl/compute_lab_v1.console.log
```

如果 `git status` 显示本地修改，先核对修改，不要 reset、clean、强制覆盖或在运行中更新代码。`setup` 只给现有 Torch Python 补齐 HDF5/指标/绘图依赖，不安装 Torch、CUDA 或 NVIDIA 库。首次之后无需重复安装。换环境时修改独立 YAML 中的 `pythons`，通过 `COMPUTE_LAB_CONFIG` 指定。

默认无整轮 GPU 时间上限。实验候选固定，先对 2 个调试病例短测，生成 `estimates.json`，随后自动推进。估时不是保证；确认阶段包含完整视频，可能明显长于历史短片段实验。不会因效果差增加病例或搜索参数。

```bash
# 只探测环境 / 只短测估时，不继续开发与确认
bash scripts/run_compute_lab.sh probe
bash scripts/run_compute_lab.sh calibrate
# 查看当前任务、worker PID、运行秒数、RAM峰值（5秒更新）
bash scripts/run_compute_lab.sh status
# 主日志只打印阶段；逐帧进度在 status 中当前 stage 对应日志
tail -n 50 -F /root/autodl-tmp/outputs_casl/compute_lab_v1/jobs/当前stage/console.log
# 停止当前 worker 并回收子进程；之后明确恢复
bash scripts/run_compute_lab.sh stop
bash scripts/run_compute_lab.sh resume
```

`Ctrl+C` 退出 tail 不会终止实验。状态文件不等于进程存活证明，异常断电后用 resume 获取运行锁。完整病例和完整任务跳过；中断病例从该病例起点重算，最多损失该病例。短训独立输出中的 100→200 是真实跨进程恢复验收；它不改变旧 CASL trainer 仅恢复 epoch 边界状态的保证。失败候选保存失败结果，不自动反复重试；修复代码或改硬件后使用新目录，避免混合证据。

## 固定配置与分类

官方权重由 SHA256 锁定，FP32、TF32 关闭、2 粒子、窗口 3、冷启动 500、热启动 50、逐步 DPS。病例名单在读模型结果前冻结：历史使用记录中的 validation 病例全部排除，另取 debug 2 / development 8 / confirmation 32；不使用 test。

- A：JAX 预取、固定形状/常数的诊断熵核缓存、减少观测的主机往返、异步输出；先单项，再组合至少两项已经通过的工程项。官方主采样核原本已 JIT，粒子原本已 vmap，本轮不把它们宣称为新增优化。另测试 Torch FP32 eager / compile / CUDA Graph。
- B：JAX 50/25 步 × FP32/FP16/BF16；另有 FP32/50 步下 DPS 10/5 次；在开发集最快可运行的 Torch 模式上测试 50/25 步 × FP16/BF16。所有冷启动仍为 500 步，精度随版本记录。
- Torch 混合精度仅 autocast 网络，采样状态与误差范数仍为 FP32；DPS 输入梯度保留。JAX 使用对应 Keras mixed policy，并核验卷积实际 dtype。任何不支持、非有限输出/梯度均失败，不降级为另一种精度。
- 框架移植同时改变执行后端。Torch 近似版本在开发阶段另与同模式 Torch FP32/50 比较，避免把框架差异都归因于 FP16。

每个版本分别输出 `category`、`family` 和 `verdict`。逐位一致检查 dtype 和原始字节；容差一致为 `atol=rtol=2e-4`，选线/掩膜/实际预算轨迹必须完全相同。网络、梯度、熵探针，共同历史与显式噪声重放，以及自己的历史闭环分别记录。图内外 RNG 边界差异单列，不能把它混入同输入重放的误差来源。

A 的失败仍保留速度/质量结果，但不能当成等价替代。B 从不自动成为正式 baseline。`selection.json` 表示进入确认的候选，不表示采用。默认一直保留 official；后续由用户明确选用近似版本，并在新实验同配置下重新建立 baseline。

## 质量与时间口径

开发取每病例前 128 帧（不足全用）、种子 42/31415、预算 14。确认完整视频、种子再加 271828，分别验证 7/14/28。至多确认最佳合格工程、纯精度、纯采样、组合各一个；开发集按闭环实际耗时选择，确认结果不参与选择。

PSNR/SSIM 沿用项目现有极坐标 uint8 口径，另报浮点 MAE、双方共同未观测区域 MAE、重建增量相对真值增量的时序误差、选线变化和实际线数。帧先在种子内聚合，种子再在病例内聚合，最后病例等权。

严格门槛：病例平均下降≤0.1 dB PSNR / 0.002 SSIM / 1% 相对 MAE；确认集病例级配对 bootstrap 的单侧95%上界同样须过线。最差病例≤0.5 dB / 0.01 / 5%，各预算独立判定。缺少协议核验的 LPIPS、临床任务指标标为未评估。

分开记录 model load、首个冷/热签名编译、冷启动、同步 core、含传输的 closed_loop、完整 worker/task 时间。微基准同输入预热3次、计时20次；完整序列独立测量。完整任务包含配对诊断和轨迹写入，不能拿它与纯 core FPS 混用。所有版本写完整轨迹；仅参照全状态缓存持久保留，候选保留指标和固定3帧图像快照。JAX/Torch 的附加内部状态字段不同，也会影响总输出开销。

profiling 是独立任务，trace 不参与速度筛选。容量搜索为独立子进程的真实 CASL 单次优化器更新，固定候选 batch 1/8/16/32/64/128，遇失败停止，仅建议可行范围；科学 batch 始终32，不采用80%规则。梯度检查点没有默认启用，它是内存—计算取舍。

## 训练短测与已有探索收尾

CASL TensorFlow 和 BF codec/prior/filter 都从保存的同一初始快照、固定真实 HDF5 批次序列开始。每个 workload 的 eager / graph 先3步估时，再各200步；另各执行100步退出、跨新进程恢复到200。逐步检查有限损失/梯度、冻结模块，比较最终参数、优化器、EMA（CASL）、最终梯度和恢复段损失。BF 延续既有 preparation 的固定 uniform 观测工作负载；它与科学创新实验中的自适应训练政策不能混为一谈。

这些是隔离校准，不推进正式模型，不证明完整收敛等价。短训不自动引入 FP16/BF16 训练版本。本轮 B 类精度实验针对冻结权重推理；如以后改训练精度，另开训练版本。

先归档校验 closure-v4 完整结果；预算数据、预测参照、同总预算闭环、BF 历史对照按原设计收尾。完成项不重复算；缺失项仅在独立副本中补原任务，旧报告原样保留。原任务规格缺失则登记缺口，不猜测新实验。TBIG 缺官方资产继续登记，非主线阻塞。Excel 只生成事实 CSV，不代填预测、Prediction Lock 或核心认知。

## 输出与下载

目录 `/root/autodl-tmp/outputs_casl/compute_lab_v1`：

|产物|用途|
|---|---|
|`REPORT.md`, `matrix.json`, `excel_facts.csv`|解释、版本矩阵与事实导入|
|`manifest.json`, `identity.json`, `versions.json`|病例、资产、源码、环境与修改类型锁定|
|`probe/`, `throughput.json`, `estimates.json`, `capacity.json`|探测、吞吐、估时与容量建议|
|`selection.json`|开发集选择记录，默认 adopted=official|
|`training_report.json`, `training/`, `jobs/train_*`|快照、批次序列、200步/恢复结果|
|`closure_audit.json`, `closure_finished/REPORT.md`|历史校验与原设计收尾|
|`tradeoff_*.png/.pdf`, `figure_provenance.json`|每集合/预算分别绘图，保留未通过点|
|`jobs/*/console.log`, `result.json`|原始日志、逐帧/逐步记录、失败原因|
|`.cache/`|全状态官方参照缓存，留服务器续跑，不放下载包|

正常结束自动生成同级 `compute_lab_v1.results.tar.gz`、`.sha256`、`.bundle.json`。包逐成员校验再生成 SHA256。下载这三项即可取得报告、指标、快照、日志和完整 closure 历史包；原始极坐标数据和 `.cache` 不在包内。需要手动重新整理：

```bash
bash scripts/run_compute_lab.sh report
bash scripts/run_compute_lab.sh bundle
cd /root/autodl-tmp/outputs_casl
sha256sum -c compute_lab_v1.results.tar.gz.sha256
```

换卡/驱动/Python依赖/源码/科学配置必须换输出目录重新校准，例如 `bash scripts/run_compute_lab.sh start /root/autodl-tmp/outputs_casl/compute_lab_v2`。旧结果不混写，也不跨配置直接复用 baseline。

本地验收记录见 `docs/compute_lab_validation.md`。本轮用户选择仅完成代码和本地验证，GPU 测量与完整 closure-v4 取回尚未执行，不能将待跑结果描述为已经验证的加速收益。
