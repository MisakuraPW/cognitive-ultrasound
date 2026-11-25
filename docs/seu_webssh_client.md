# 从 Windows 后台管理学校 Slurm 作业

在毕设项目里可以直接说“查上次 GPU 冒烟为什么失败”“看这个作业号的日志”或“把这项任务提交到指定队列”。助手使用学校官方 WebSSH 和你自己的学校账号。专用 Edge 可以最小化，你可以同时使用自己的浏览器和其他程序。Slurm 已接收的作业在关闭本机浏览器后仍由集群管理。

在新的 Codex 会话里可用 `$seu-webssh` 点名这套工作方式。例如：“用 $seu-webssh 查上次学校 L20 的真实推理，给我作业号、退出码和结果目录。” 用户级安装可供本机同一 Windows 账号的其他会话使用。若从别的目录管理同一个 CASL 项目，技能安装目录的 `checkout.txt` 指向本地仓库。[技能源文件](../skills/seu-webssh/SKILL.md)

## 连接与跨会话复用

从仓库 `cognitive-ultrasound` 运行下面的命令。在上一级毕设工作区运行时，在脚本路径前加 `cognitive-ultrasound/`。

```powershell
node scripts/seu_portal/client.cjs session
node scripts/seu_portal/client.cjs login
node scripts/seu_portal/client.cjs probe
```

`session` 查看本机专用浏览器是否在线。`login` 在已有窗口时直接复用，首次使用时打开独立 Edge。用户在学校页面完成认证，进入「智算平台 → 远程终端」。`probe` 输出登录节点、提交器和学校仓库的短提交号，以 `SEU_PORTAL_READY` 结尾。[学校 WebSSH](https://sc.seu.edu.cn/webssh-connect)

同一 Windows 用户下的其他项目会话直接连接已有窗口。工作进程使用本机命名管道，终端输入和读出都限定在专用页面内。换一个 Codex 会话时，已打开的窗口和有效的学校登录可以继续使用。关闭专用窗口或重启电脑后需要重新打开，学校 SSO 决定是否要求密码或验证码。用户在学校页面完成认证。临时浏览器配置与项目仓库分开，本机 `%LOCALAPPDATA%\SEU-CASL\WebSSH` 存放本地连接信息、启动日志和提交回执。

读取作业时如果收不到完整回显，客户端会尝试重连原登录并重读一次。也可用 `reconnect` 打开新的 WebSSH 标签并验证连接。学校要求重新认证时，由用户在该窗口操作。平常保留专用窗口，`close` 用于主动结束本机连接。

## 提交、按号查询与历史排查

| 想做的事 | 仓库内的命令 |
| --- | --- |
| 看本人当前队列 | `node scripts/seu_portal/client.cjs queue` |
| 查近 14 天本人作业和现存 CASL 日志名 | `node scripts/seu_portal/client.cjs history 14` |
| 查看作业或历史退出码 | `node scripts/seu_portal/client.cjs status 123456` |
| 读取对应作业最后 50 行 | `node scripts/seu_portal/client.cjs logs 123456 50` |
| 读取指定作业已产生的 GPU 环境报告 | `node scripts/seu_portal/client.cjs gpu-report 123456` |
| 重连终端，复用当前学校登录 | `node scripts/seu_portal/client.cjs reconnect` |
| 更新学校工作区到已发布的 Git 提交 | `node scripts/seu_portal/client.cjs sync` |
| 按任务和队列提交 | `node scripts/seu_portal/client.cjs submit 任务名 队列名` |

新增任务沿用仓库的 `scripts/run_seu_casl.sh`。先在本地验证、发布并核对 Git 提交，再执行 `sync`，核对返回的学校仓库版本。用户要求提交时才调用 `submit`，保存学校返回的 `Submitted batch job` 作业号。若提交回显超时，先查 `queue` 和 `history`，避免重复排队。本地提交回执保存作业号或标明结果待核对。计算通过学校的 `sbatch_wrapper` 在计算节点执行。队列及资源选项见 [学校作业入口说明](seu_casl.md)。

`status` 优先查仍在调度器中的本人作业，再查历史会计记录。`PENDING` 阶段通常还没有输出文件。历史日志可按作业号在自己的 CASL 作业输出目录定位，`logs` 每次取 1–200 行。GPU 冒烟在环境检查后写 `gpu.json`，更早的启动错误应看 Slurm 日志。查到 `COMPLETED` 后再核对任务成功标记，例如 `CPU_CHECK_OK` 或 `GPU_SMOKE_OK`。

### 选择作业和队列

| 现有任务 | 默认分区 | 验收内容 |
| --- | --- | --- |
| `data-prep` | `6126-24C-192G` | 校验数据 TAR，在 SHARE 解包，复制到 NVMe 并逐文件比较，成功标记为 `DATA_PREP_OK`。 |
| `cpu-check` | `6126-24C-192G` | 校验官方权重、病例名单与两块盘的 HDF5 读取，成功标记为 `CPU_CHECK_OK`。 |
| `gpu-smoke` | `gpu-L20` | 先在 GPU 上验证两个框架，再用官方 FP32 权重对验证集一例的前 5 帧做 7 线闭环推理，成功标记为 `GPU_SMOKE_OK`。 |

这些任务可以按需要单独提交。相同数据与环境已有通过记录时，新会话可直接查询原记录。新的训练或评估要在提交入口登记名称、资源、输出目录和完成条件，并将执行逻辑放进计算作业脚本。登录节点负责 Git、提交、查询与轻量预检，训练、推理和数据预处理由调度器分配计算节点。[提交入口](../scripts/run_seu_casl.sh) · [计算作业脚本](../scripts/seu_casl_job.sh)

学校提交器会调整实际资源。已验证的 L20 作业申请 1 卡、3 核，提交器按该分区规则分配了 6 核。学校按 CPU 配套内存，项目脚本省略显式内存参数。其他队列的权限、配置、用量以提交时为准。排队位次可能变化，排队时长与 Slurm 的实际运行时长分开报告。[学校 GPU 作业手册](https://sc.seu.edu.cn/docs/ai/contents.html) · [学校 CPU 作业手册](https://sc.seu.edu.cn/docs/hpc/contents.html)

### 计算环境

项目共享目录默认为 `~/casl`，环境在 `~/casl/envs/casl`，学校仓库在 `~/casl/cognitive-ultrasound`。SHARE 保留数据和归档，真实推理读取 NVMe 上的极坐标 HDF5。需要新增依赖时，随 Git 记录用户环境或模块的变更，在分配到的计算节点验证实际运算。换到另一种卡、框架或依赖版本时，再针对发生变化的部分验证。

L20 实测使用 JAX 0.6.2、TensorFlow 2.20.0、学校的 CUDA 12.8 和 cuDNN 9.8。提交入口先检查 cuDNN、CUDA 工具和 `nvvm/libdevice/libdevice.10.bc`，将它们的路径传给 GPU 作业。计算节点运行时，`PATH` 优先使用这套 CUDA 的工具，`LD_LIBRARY_PATH` 优先使用已验证的 cuDNN，`XLA_FLAGS` 指定同一套 CUDA 的数据目录。Git 在计算节点还用于核对官方 CASL 和 zea 源码。项目已经通过单卡真实短推理，各实验的完整训练时间、显存需求和结果仍需针对对应作业测量。[实测代码版本](https://github.com/MisakuraPW/cognitive-ultrasound/commit/84e384880cef4a7a369311be4e0585c79d56244c)

### 作业回显怎么判断

| 看到的情况 | 接下来做什么 |
| --- | --- |
| `session` 有窗口，命令却报 `LOGIN_REQUIRED` 或 `NO_TERMINAL` | 检查专用窗口里的学校登录和远程终端。终端断开先用 `reconnect`，已有窗口可继续用于认证。 |
| `sync` 超时，退出码 `124` | 用 `probe` 比较学校实际提交号和本地已推送的提交。仍为旧版时重试同一个快进同步。遇到分支分叉时保留原分支和本地改动，再核对跟踪分支。 |
| `submit` 没返回作业号 | 依次查看 `queue`、`history` 和本机的 `submitted.jsonl`，按提交时间、任务名和队列定位。仍不能确定是否已受理时，先保留现场。 |
| `PENDING`，运行计时 `0:00` | 看 `Reason`。等待资源时留在原队列，沿用原作业号查询。此时的 `ExitCode=0:0` 不能用作完成依据。 |
| `RUNNING` 或已结束 | 运行中查看 `logs ID 60` 的新输出。结束后核对 Slurm 终态、退出码、任务标记和实际产物。要求持续跟踪时在会话中定期读取，之后回来也可按号续查。 |
| `FAILED`、`TIMEOUT` 或 `OUT_OF_MEMORY` | 保留旧作业号，读第一处导致退出的错误和失败阶段。GPU 诊断 JSON 已生成时再读 `gpu-report`。查过的实例包括计算节点缺 Git、误加载 cuDNN 9.1.1，以及 JAX 找不到 `libdevice.10.bc`。 |

### 文件传递与结果位置

**代码走 Git，单独的数据、权重和结果文件可以走平台文件管理。** 用户上传指定文件时先选目标目录，在上传任务面板等待“已完成”，再点文件列表面包屑旁的刷新图标，核对最终文件名和字节数。验证大文件时，可在 Windows 用 `Get-FileHash -Algorithm SHA256 -LiteralPath 文件路径`，在学校 WebSSH 用 `sha256sum -- 文件路径`，对照两端的哈希。取结果时在平台找到相应作业的目录，选择需要的文件下载。这个后台客户端提供代码同步和日志读取，通用文件上传、下载目前使用平台界面。[学校文件管理手册](https://sc.seu.edu.cn/docs/ai/contents.html)

GPU 冒烟的日志为 `~/casl/jobs/acceptance/casl-gpu-smoke-作业号.log`。同目录下的 `gpu-作业号/gpu.json` 记录框架检测，`gpu-作业号/evaluation/manifest.json` 记录配置与病例，`evaluation/casl/lines_007/病例编号/frames.csv` 记录实际线数、PSNR、SSIM 和逐帧耗时，`trajectory/` 保存配置选中的样例。新会话交接时，给出作业号、Git 提交、终态与退出码、关键日志和结果路径，就能从同一份证据继续工作。

**在另一个研究项目使用时**，学校账号的作业队列可以复用，当前 `sync` 和 `submit` 固定指向 CASL 仓库，`logs` 读取本人 CASL 作业目录。要为其他项目提交任务或读取其文件，先接入该项目的仓库、任务入口和日志目录。

## 本机运行条件

使用 Windows、Node.js、Edge（也支持本机 Chrome）和 Playwright Core。本机优先使用已安装的 `playwright-core` 或 Codex 自带的 Node 依赖，无须在学校服务器安装代理、浏览器或开放端口。运行客户端要使用启动专用窗口的 Windows 账号。Codex 若在隔离环境里看不到该会话，按权限提示从同一 Windows 用户环境调用这个客户端即可。

本机个人技能位于 Codex 用户目录下的 `skills/seu-webssh`。其中 `checkout.txt` 只含 CASL 仓库绝对路径。移动本地仓库时更新这一行即可。可移植的技能说明和显示信息随项目 Git 保存，机器路径留在个人安装目录。

本地回归测试为 `node scripts/seu_portal/client.test.cjs`。Bash 测试可通过 `GIT_BASH_EXE` 指定 Git for Windows 的 Bash 路径，全部作业及日志样本均在本地模拟。
