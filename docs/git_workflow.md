# Git 工作方式

本项目是独立仓库，默认分支 `main`。提交身份采用你本机已有的 `Rimoonet / 2803937743@qq.com`，只写入项目级配置。原始 `../CASL/casl` 不参与本项目提交。

`vendor/casl` 是 Git 子模块，固定 CASL 提交，其 zea gitlink 也固定；`python scripts/bootstrap.py` 使用 HTTPS 填充两层依赖。上游代码保持干净，本项目的改动都放在 src/configs/scripts/tests/docs 中。官方仓库的远程地址只用于依赖获取。

暂未为你的个人项目设置 `origin`。提供远程仓库地址后再添加和推送。当前没有创建 GitHub/Gitee 仓库，也没有上传代码、医学数据或模型。

本地配置文件在 `G:/科研项目/毕设/cognitive-ultrasound/.git/config`（`.git` 是隐藏目录）。它属于自己的项目，不是相邻官方 CASL 仓库的配置。用下面的只读命令查看项目设置：

```powershell
git -C 'G:/科研项目/毕设/cognitive-ultrasound' config --local --list
git -C 'G:/科研项目/毕设/cognitive-ultrasound' remote -v
```

本地仓库已经建立并有提交，无需再次 `git init`。为了之后在 AutoDL 克隆，建议在自己的 GitHub/Gitee 账号创建一个空仓库，例如 `cognitive-ultrasound`，可设为私有；不要初始化 README、License 或 .gitignore，以免产生不相关的初始历史。拿到地址后在本项目中关联并推送：

```powershell
Set-Location 'G:/科研项目/毕设/cognitive-ultrasound'
# 将下面的占位字符串换成你新建仓库的真实地址；不要填 EchoRVM 或官方 CASL。
git remote add origin '你的远程仓库地址'
git push -u origin main
```

这些是后续命令，本次没有执行关联和推送。私有仓库在 AutoDL 上通过 GitHub/Gitee 的登录或 SSH 凭据访问，不把令牌写进远程 URL、脚本或配置文件。`.git/config` 不随代码推送；云端 clone 会自动设置自己的 origin。

`.gitignore` 排除原始/处理数据、checkpoint、日志、结果、大型缓存、`.env`、本地配置与虚拟环境；固定患者清单、代码和报告进入版本控制。迁移机器时通过依赖安装和固定资产下载恢复计算环境。

以后每次实验保留配置、Git commit、模型/数据哈希与实际环境，结果放在单独 output 目录。改动基线算法前先创建分支；研究扩展应与已经验证的 CASL 基线分开。
