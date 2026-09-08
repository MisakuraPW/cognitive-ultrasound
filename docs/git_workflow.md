# Git 工作方式

本项目是独立仓库，默认分支 `main`。提交身份采用你本机已有的 `Rimoonet / 2803937743@qq.com`，只写入项目级配置。原始 `../CASL/casl` 不参与本项目提交。

`vendor/casl` 是 Git 子模块，固定 CASL 提交，其 zea gitlink 也固定；`python scripts/bootstrap.py` 使用 HTTPS 填充两层依赖。上游代码保持干净，本项目的改动都放在 src/configs/scripts/tests/docs 中。官方仓库的远程地址只用于依赖获取。

2026-09-08 已将用户提供的 `https://github.com/MisakuraPW/cognitive-ultrasound.git` 关联为 `origin`，并推送 `main`、设置其跟踪 `origin/main`。仅上传 Git 跟踪的项目文件，不上传医学视频、模型或虚拟环境。

本地配置文件在 `G:/科研项目/毕设/cognitive-ultrasound/.git/config`（`.git` 是隐藏目录）。它属于自己的项目，不是相邻官方 CASL 仓库的配置。用下面的只读命令查看项目设置：

```powershell
git -C 'G:/科研项目/毕设/cognitive-ultrasound' config --local --list
git -C 'G:/科研项目/毕设/cognitive-ultrasound' remote -v
```

本地和远程仓库均已建立，无需再次 `git init` 或 `git remote add`。以后完成本地提交后，从项目目录推送：

```powershell
Set-Location 'G:/科研项目/毕设/cognitive-ultrasound'
git push
```

AutoDL 上可用 `git clone https://github.com/MisakuraPW/cognitive-ultrasound.git /root/autodl-tmp/cognitive-ultrasound` 取得本项目，再执行 `python scripts/bootstrap.py` 获取固定上游。私有仓库通过平台登录或 SSH 凭据访问，不把令牌写进远程 URL、脚本或配置文件。`.git/config` 不随代码推送；云端 clone 会自动设置自己的 origin。

本机 Windows Schannel 曾报 `SEC_E_NO_CREDENTIALS`，项目级 `http.sslBackend=openssl` 已解决 TLS 连接问题，证书验证保持开启；推送使用本机 Git 凭据管理器。这个本地设置无需复制到 Linux。

`.gitignore` 排除原始/处理数据、checkpoint、日志、结果、大型缓存、`.env`、本地配置与虚拟环境；固定患者清单、代码和报告进入版本控制。迁移机器时通过依赖安装和固定资产下载恢复计算环境。

以后每次实验保留配置、Git commit、模型/数据哈希与实际环境，结果放在单独 output 目录。改动基线算法前先创建分支；研究扩展应与已经验证的 CASL 基线分开。
