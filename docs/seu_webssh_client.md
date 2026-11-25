# 从 Windows 后台管理学校 Slurm 作业

在毕设项目里可以直接说“查上次 GPU 冒烟为什么失败”“看这个作业号的日志”或“把这项任务提交到指定队列”。助手使用学校官方 WebSSH 和你自己的学校账号。专用 Edge 可以最小化，你可以同时使用自己的浏览器和其他程序。Slurm 已接收的作业在关闭本机浏览器后仍由集群管理。

## 连接与跨会话复用

从仓库 `cognitive-ultrasound` 运行下面的命令。在上一级毕设工作区运行时，在脚本路径前加 `cognitive-ultrasound/`。

```powershell
node scripts/seu_portal/client.cjs session
node scripts/seu_portal/client.cjs login
node scripts/seu_portal/client.cjs probe
```

`session` 查看本机专用浏览器是否在线。`login` 在已有窗口时直接复用，首次使用时打开独立 Edge。用户在学校页面完成认证，进入「智算平台 → 远程终端」。`probe` 输出登录节点、提交器和学校仓库的短提交号，以 `SEU_PORTAL_READY` 结尾。[学校 WebSSH](https://sc.seu.edu.cn/webssh-connect)

同一 Windows 用户下的其他项目会话直接连接已有窗口。工作进程使用本机命名管道，终端输入和读出都限定在专用页面内。使用临时浏览器配置，专用窗口关闭或电脑重启后需要重新打开；学校 SSO 是否要求再次输入密码或验证码由当时的认证状态决定。项目不收集密码，也不导出浏览器 Cookie。本机用户目录的 `%LOCALAPPDATA%\SEU-CASL\WebSSH` 存放本地连接信息、启动日志和提交回执。

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

`status` 优先查仍在调度器中的本人作业，再查历史会计记录。`PENDING` 阶段通常还没有输出文件。历史日志可按作业号在自己的 CASL 作业输出目录定位；`logs` 每次取 1–200 行。GPU 冒烟在环境检查后写 `gpu.json`，更早的启动错误应看 Slurm 日志。查到 `COMPLETED` 后再核对任务成功标记，例如 `CPU_CHECK_OK` 或 `GPU_SMOKE_OK`。

## 本机运行条件

使用 Windows、Node.js、Edge（也支持本机 Chrome）和 Playwright Core。本机优先使用已安装的 `playwright-core` 或 Codex 自带的 Node 依赖，无须在学校服务器安装代理、浏览器或开放端口。运行客户端要使用启动专用窗口的 Windows 账号。Codex 若在隔离环境里看不到该会话，按权限提示从同一 Windows 用户环境调用这个客户端即可。

本地回归测试为 `node scripts/seu_portal/client.test.cjs`。Bash 测试可通过 `GIT_BASH_EXE` 指定 Git for Windows 的 Bash 路径，全部作业及日志样本均在本地模拟。
