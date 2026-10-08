# TBIG 与 CASL 官方代码比较：共享感知框架，改变任务驱动选线

记录日期：2026-10-08。范围：官方仓库静态审计；本轮未安装运行环境、下载权重或数据、执行训练或推理。本文回答“除了下游目标，还有哪些实现区别”，不把仓库可克隆等同于实验可直接复现。

## 1. 结论

你的判断抓住了主要研究方向：TBIG 沿用了 CASL 的时序扩散感知框架，把采集目标转向下游定量任务。但“基本只有最后的下游目标不同”容易让人误以为它只是换了最后的评价指标。**真正的核心改动发生在闭环内部：每帧用下游测量对输入图像的梯度加权后验方差，再决定下一帧采哪些线。**因此，它会改变观测、历史状态及后续后验，而不是只改变最后输出的分数。

从这两个官方快照来看，TBIG 也不是另造一套扩散模型架构。它增加了 EchoNet-LVH 的关键点热图与可微长度测量链路、对应的数据和权重配置、任务时序评价与可视化。还有 Zea 版本、采样参数、仓库结构、RF 工具等工程差异，其中很多不能直接算成 TBIG 的方法贡献。

还需要修正一个认识：**本地 CASL 官方快照已经包含下游任务接口、分割模型封装和任务驱动选线原型。**所以“CASL 代码完全没有下游任务，TBIG 才首次加上这个接口”并不成立。TBIG 的任务敏感度计算与具体定量测量实现，和这个已有原型仍有实质差别。快照之间存在时间与分支差异，不能据此反推哪个想法最早提出。

## 2. 克隆位置、版本与核对范围

| 项目 | 本地位置 | 锁定提交 |
|---|---|---|
| CASL 官方源码 | [CASL/casl](G:/科研项目/毕设/CASL/casl) | 5f57aba668eb34054b8ced817c666b7d18a51ca2 |
| TBIG 官方源码 | [TBIG](G:/科研项目/毕设/TBIG) | d878652811093d8a234ff023228d46306db65948 |
| CASL 固定 Zea | CASL 的 Git 子模块指针；审计其已初始化的 vendor 副本中的提交对象 | 192c0bbd4e89061c38048048673beadcb8723a82 |
| TBIG 固定 Zea | [TBIG/zea](G:/科研项目/毕设/TBIG/zea) | 12aa3ca2981da9e1eff2ab253ed95f1471fa8760 |

TBIG 与 CASL 是同一层目录。CASL 原先就有一层 casl 子目录，本轮没有移动它。TBIG 主仓库和 Zea 子模块均克隆完成，主仓库 Git 对象连通性检查通过，工作区无本轮源码修改。

GitHub 直连本轮仍返回连接重置；实际下载使用公开只读转发路径 ghfast.top。TBIG 的 origin 已设回官方 https://github.com/tue-bmd/task-based-ulsa.git，Zea 的 origin 保持官方 git@github.com:tue-bmd/zea.git。本轮没有改全局代理，也没有把现有项目的认证信息发给转发站。这里只声明已取得并审计上述提交，不声明它们是此刻官方远端的最新提交。

CASL 主提交日期为 2026-05-22，TBIG 主提交日期为 2026-05-06；其 Zea 固定提交分别来自 2026-01-28 与 2025-08-13。README 可以比算法代码更新得更晚。比较对象是明确的代码快照，不是两个论文发表当天所有文件的历史还原。

逐文件映射、定义位置与 AST 核对记录见 [静态审计证据](G:/科研项目/毕设/cognitive-ultrasound/docs/tbig_casl_code_audit_20261008.json)。主要比较使用独立 CASL 官方工作区，避开我们在 cognitive-ultrasound 中添加的加速、记录及实验封装。

## 3. 闭环中保留的部分与改变的部分

两边的基本链路都是：当前稀疏观测及历史 → 时序扩散后验粒子 → 当前帧信念 → 为下一帧选线 → 更新观测历史。TBIG 的 recover 仍在采样完成后取当前帧粒子进行选线，并保存新的粒子、掩膜和历史状态。

| 环节 | CASL 官方代码 | TBIG 官方代码 | 判断 |
|---|---|---|---|
| 感知模型 | Zea DiffusionModel、DPS、时序输入 | 同类接口与感知流程 | 框架继承；不同数据的权重不等于同一份权重 |
| 时序与热启动 | 历史观测缓冲、上一轮后验初始化 | 保留上述机制 | 并非重新发明时序感知 |
| 粒子计算 | 每粒子 vectorized_map，避免 guidance 权重随 batch 改变 | 保留该实现思路 | 原本已有向量化 |
| 图像域选线 | 后验粒子构造图像熵，再 K 次贪心选线及邻域重权 | 仍可用 greedy_entropy 作 GIG 对照 | 比较基线保留 |
| 任务选线 | 带有早期分割梯度×方差原型 | 标量测量梯度的平均值平方×方差 | 核心算法实现发生变化 |
| 下游任务 | EchoNet-Dynamic 分割封装 | 保留旧封装，新增 LVH 热图及 LVID/LVPW/IVS 长度 | 任务链路扩展 |
| 重建与观测约束 | choose_first/mean 与 hard projection 等 | 保留同类机制 | 不是由任务损失替换 DPS |
| 评价 | 图像指标与相关可视化 | 增加关键点、测量长度 MAE 与时序图 | 科学评价目标改变 |

依据：[CASL setup_agent](G:/科研项目/毕设/CASL/casl/ulsa/agent.py:408)、[CASL recover](G:/科研项目/毕设/CASL/casl/ulsa/agent.py:610)、[TBIG setup_agent](G:/科研项目/毕设/TBIG/ulsa/agent.py:291)、[TBIG recover](G:/科研项目/毕设/TBIG/ulsa/agent.py:534)。两份 ulsa/entropy.py 内容完全相同；这只是一个组件的相同，不证明整套程序数值等价。

CASL 的常用策略 greedy_entropy 是以图像后验不确定性驱动选线，不能精确概括为“用 PSNR/重建质量直接驱动”。PSNR 是评价口径之一。TBIG 则让选线分数显式依赖任务函数；任务敏感度高但图像已确定的位置、或者图像不确定但不影响该测量的位置，都不应仅凭一个因素获高分。

## 4. 最重要的代码差别：任务选线究竟怎么算

### 4.1 CASL 快照已有的原型

[CASL DownstreamTaskSelection](G:/科研项目/毕设/CASL/casl/ulsa/selection.py:17) 继承 GreedyEntropy；其实际 sample 调用 compute_output_and_saliency_propagation_summed。它先把可微任务输出求和，再求这个标量对输入的梯度，对粒子取梯度绝对值平均，并乘图像后验方差。

对像素 i，其评分可写成：

- 令 g_i^(p) 为第 p 个粒子上“任务输出之和”的输入梯度。
- 评分 S_i = Var_p[X_i] × Mean_p[|g_i^(p)|]。

随后按深度方向汇总成每条扫描线的分数，沿用逐条贪心选择与邻域重权。这个原型已有任务梯度，但不是论文中一般向量任务 Jacobian Gram 矩阵的完整实现。文件另带未接入当前 sample 的 Hutchinson 草稿及 TODO；不能把这个备用草稿当成默认算法已经运行的证据。

### 4.2 TBIG 当前实现

[TBIG compute_output_and_saliency_propagation](G:/科研项目/毕设/TBIG/ulsa/selection.py:57) 明确要求任务返回标量。它对每个后验图像粒子独立求任务梯度，对梯度先求平均、再平方，并乘图像方差：

- g_i^(p) = ∂f(X^(p))/∂X_i。
- 代码评分 S_i = Var_p[X_i] × (Mean_p[g_i^(p)])²。

[TBIG sample](G:/科研项目/毕设/TBIG/ulsa/selection.py:86) 将像素评分沿深度求和，再按候选线分组，仍调用继承的 select_line_and_reweight_entropy 贪心选 K 条线。方法名保留 entropy 并不表示这里又计算了一遍图像熵。

因此 TBIG 的新增计算主要是“可微下游任务前向＋输入梯度＋任务敏感度加权”。它没有在这段实现里对每个候选动作重跑一次完整 DPS 预测，也没有训练一个独立的神经选线策略。

### 4.3 论文公式和实现应分别登记

TBIG 论文第 3.2 节公式 (2) 写的是 E[G_ii] × σ_i²，其中 G = JᵀJ。对标量任务，这对应 Mean_p[(g_i^(p))²] × Var_p[X_i]。但随后的文字描述又谈到平均 Jacobian，而当前代码确实实现了 (Mean_p[g_i^(p)])²。**平方后平均与平均后平方通常不同。**

一个纯数学例子：两粒子的梯度分别为 +1、−1，平均后平方为 0，平方后平均为 1。只有梯度在粒子之间足够一致等特殊情况下，两者才接近。对向量任务，还涉及所有输出分量平方梯度的求和，不能直接拿当前标量 helper 当成任意向量 Jacobian 的完整实现。

这是一项可定位的论文—文字—代码对应关系问题，不是对整篇工作的真实性判决。本轮没有替作者修改算法，也没有运行两种公式比较。以后引用时应明确“按官方提交实现”还是“按公式 (2) 实现”，不能静默切换然后混用同一个 baseline 名称。

## 5. 下游测量链路与评价含义

TBIG 新增的 [EchoNetLVHSegmentation](G:/科研项目/毕设/TBIG/ulsa/downstream_task.py:289) 包装四通道关键点热图模型；[EchoNetLVHMeasurement](G:/科研项目/毕设/TBIG/ulsa/downstream_task.py:610) 通过测量类型选热图，计算锚点坐标及两点间距离。

实际链路包含：极坐标图像 resize → scan conversion → 再 resize 到模型尺寸并归一化 → DeepLabV3+ 热图 → 热图平方及扇区过滤 → 归一化热图的质心 → 两锚点距离。LVID 对应中间两个通道，另外支持 LVPW、IVS。任务选线可微函数返回的是模型坐标中的标量距离，不在这个输入梯度函数内做病人特定的厘米换算。

[模型结构](G:/科研项目/毕设/TBIG/models/deeplabv3_segmenter.py:44) 是 Keras 实现的 ResNet50 backbone、空洞空间金字塔池化和解码输出；它不是新增的 CASL 扩散去噪网络。在线选线需要对任务函数求输入梯度；这和在线更新任务模型权重是两件事。检查到的选线路径没有 optimizer.step 或 fit，不需要为了每次选线重训该模型。

评价路径另有 [get_distance_in_cm](G:/科研项目/毕设/TBIG/ulsa/downstream_task.py:398)，结合 cone_parameters 与 MeasurementsList 中的尺寸、标定信息，把坐标距离换算为厘米。这里使用标注提供空间比例，并不意味着每一帧的比较参照都来自人工标注。

[apply_downstream_task](G:/科研项目/毕设/TBIG/active_sampling_temporal.py:141) 将任务模型分别应用于全采样目标图像与后验粒子，并在热图空间平均粒子输出后提取测量。对应 [测量 MAE 提取](G:/科研项目/毕设/TBIG/benchmark_active_sampling_ultrasound.py:749)。因此主要回答的是“稀疏采集能否保持全采样图像上的模型测量结果”，不能直接说成“达到人工真值的临床测量准确率”。同理，模型预测分割间的一致性不能冒充人工标注 Dice。

这也解释了为何既不能把已有 EchoNet-Dynamic 的 112×112 权重直接当成 LVH 的 256×256 三帧权重，也不能只把评价函数改成 LVID 就声称实现了任务驱动采集。

## 6. 配置与工程差别：不要都算成方法贡献

### 6.1 默认配置并不是公平比较协议

| 配置项 | CASL echonet_3_frames.yaml | TBIG echonetlvh_3_frames_downstream_task.yaml | 解释 |
|---|---|---|---|
| 数据/任务 | EchoNet-Dynamic；通常无下游任务 | EchoNet-LVH；LVID 测量 | 数据域及任务均变了 |
| 极坐标图像/候选线数 | 112×112 / 112 | 256×256 / 256 | 不是相同图像尺寸 |
| 后验粒子数 | 2 | 4 | 计算量与不确定性估计均受影响 |
| 当前默认每帧线数 | 7 | 5 | 预算不相同 |
| 扩散总步数 | 500 | 500 | 单看这一行不够 |
| initial_step | 450 | 50 | 下面详述实际循环语义 |
| DPS omega | 10 | 1；initial_omega 也为 1 | 引导配置不同 |
| 感知权重 | hf://zeahub/ulsa | LVH 三帧权重的作者本地路径 | 不能直接互换 |
| 选择策略 | greedy_entropy | downstream_task_selection | 主要研究变量 |

依据：[CASL 配置](G:/科研项目/毕设/CASL/casl/configs/echonet_3_frames.yaml)、[TBIG 配置](G:/科研项目/毕设/TBIG/configs/echonetlvh_3_frames_downstream_task.yaml)。CASL 的另一份任务原型配置又是 2 条线，因此不要把上表当成所有 CASL 配置的唯一设定。

TBIG 的 [LVH 评价脚本](G:/科研项目/毕设/TBIG/benchmarking_scripts/eval_downstream_task_echonetlvh.py:27) 从同一份 LVH 任务配置出发，对 TBIG、greedy_entropy/GIG、uniform_random 扫同样的 1/3/5 条线。GIG 对照覆盖选择器及其 kwargs，其他感知配置继承同一基础设置。这比拿两个仓库的默认 YAML 横比更接近有效的动作策略比较。论文也明确两种策略共享感知模型。

### 6.2 initial_step=50 不能读成“热启动只跑 50 步”

这次固定的 TBIG Zea 中，[posterior sampling 循环](G:/科研项目/毕设/TBIG/zea/zea/models/diffusion.py:646) 的边界是 initial_step 到 diffusion_steps。TBIG agent 直接传入配置，普通热启动分支没有把 50 自动转换成 450。因此配置 500/50 在这条路径上对应 **450 次循环**；CASL 的 500/450 对应 **50 次循环**。没有历史粒子时两边传 initial_step=0，执行冷启动。

论文使用“SeqDiff 初始化步为 50、总步数 500”的表述，这不等于承诺每帧只有 50 次去噪。这里登记的是已核对的循环语义，不据此宣称作者实验配置错误。后续若采用 Zea 新版或 notebook，必须重新核对参数含义。这个差别本身足以让直接横比 FPS 或成本产生严重误导；本轮没有实测 TBIG FPS。

### 6.3 仓库分叉和工具迁移

- CASL 的主入口已移到 ulsa 包，TBIG 仍有根目录 active_sampling_temporal.py、benchmark_active_sampling_ultrasound.py；入口位置不同不等于核心算法改变。
- CASL 的指标模块更多接入 zea.metrics，TBIG 保留一份较大的本地指标工具。不同 Zea 提交还影响 API 与算子归属，不能只比较两个 agent.py 就假定依赖相同。
- TBIG 还保留 RF 波束形成、pfield 到 transmit 映射等工具；CASL 较新快照也有自己的 RF/3D/发射时序工具。这些要按具体实验路径判断，不能统统归入 TBIG 的 LVH 图像域贡献。
- 两边都支持 JAX 编译 recover 或 posterior_sample。TBIG 官方选线直接用 JAX/Keras ops；没有在主闭环中引入本轮项目开发的 Torch 移植。
- TBIG buffer 中的函数名 lifo_shift 与 CASL fifo_shift 不同，但两者函数体都是丢弃最早通道、在末尾追加新帧。不能因命名不同就判断时间顺序反了。

## 7. 克隆完成后仍有哪些运行障碍

README 给出了官方 LVH 任务与扩散权重的 Hugging Face 页面，但当前任务类实际 load_weights 仍指向作者的 /mnt/z/.../segmenter_4.weights.h5，CSV 和扩散 run_dir 也包含作者路径。**所以“官方资产有发布入口”和“这个 checkout 默认即可运行”必须分开看。**此前准备记录中的 TBIG 资产状态需要保留当时日期；现在可追加“官方 README 已提供资产入口”，仍不能追加“已下载并验证兼容”。

未来实际接入时至少需要：指定权重及 CSV 路径、核对数据转换与 split、检查任务输入形状和数值范围、确认可微 scan conversion/关键点函数、记录扩散循环数。模型构造声明的输入通道与实际灰度预处理链也应先做形状冒烟检查，不能因为模型文件存在就跳过验证。

论文强调可以用于可微下游任务；但**当前任务评分 helper 的实现范围是标量输出**。扩展到分割图、多个测量或 EF 等向量/序列任务，要明确梯度聚合方式，不可仅替换 registry key 就认定已经完成一般任务实现。

本轮仅保留官方原样源码，不擅自修作者路径、替换模型、训练新权重或扩展实验清单。没有学长代码，也不能由 TBIG 的实现倒推出学长毕设用了相同训练方式。

## 8. 对你的研究主线意味着什么

可以复用的是时序粒子信念、DPS 感知、掩膜/历史更新、任务接口和固定预算选线框架。TBIG 可作为“任务敏感度驱动采集”的参考或 baseline，但其选线创新通常可以接在冻结感知权重后实现与评估；这并不自动要求完整重训 CASL。换到 LVH 等数据域时则另需合适的先验和任务模型资产，不能把“策略不需重训”扩张为“任何数据都不需适配”。

对动态预算而言，这两个检查到的默认选线器仍给定 n_actions，然后贪心选足 K 条线。TBIG 解决“把线放在哪里、围绕什么任务放”的问题；它没有在这条路径里自动决定每帧应该用多少条线，也没有完整的候选动作后验预测、世界模型或长程规划。因此你的后续预算研究与 TBIG 的位置选择可以组合，但需要届时单独固定质量目标、任务参照和公平预算协议。

当前最值得记住的不是“TBIG 和 CASL 差不多，可以忽略”，而是：**它复用成熟感知基座，在闭环动作准则上提出可定位的改动；评价时要控制感知权重、数据、采样语义和预算。**本轮不启动新的验证计算，也不改变已经结束的加速实验结论。

## 来源与定位

- 官方 TBIG 仓库： https://github.com/tue-bmd/task-based-ulsa 。对应本文锁定提交见第 2 节及 JSON。
- 官方 CASL 仓库： https://github.com/tue-bmd/casl 。本地源码定位见各节链接。
- TBIG 论文 v1（2026-01-28）： https://arxiv.org/html/2601.20711v1 ，重点为第 3.1、3.2、4 节。
- [静态审计 JSON](G:/科研项目/毕设/cognitive-ultrasound/docs/tbig_casl_code_audit_20261008.json)：逐文件对应、相同/变化定义、源码行号及 AST 哈希。AST 相同可确认所检查定义的静态一致；AST 变化不直接证明算法变化或运行差异。

以上是源码与论文对应关系的审计，不包含新的运行时速度、质量或兼容性结论。
