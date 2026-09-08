# Git 工作方式

本项目是独立仓库，默认分支 `main`。提交身份采用你本机已有的 `Rimoonet / 2803937743@qq.com`，只写入项目级配置。原始 `../CASL/casl` 不参与本项目提交。

`vendor/casl` 是 Git 子模块，固定 CASL 提交，其 zea gitlink 也固定；`python scripts/bootstrap.py` 使用 HTTPS 填充两层依赖。上游代码保持干净，本项目的改动都放在 src/configs/scripts/tests/docs 中。官方仓库的远程地址只用于依赖获取。

暂未为你的个人项目设置 `origin`。提供远程仓库地址后再添加和推送。当前没有创建 GitHub/Gitee 仓库，也没有上传代码、医学数据或模型。

`.gitignore` 排除原始/处理数据、checkpoint、日志、结果、大型缓存、`.env`、本地配置与虚拟环境；固定患者清单、代码和报告进入版本控制。迁移机器时通过依赖安装和固定资产下载恢复计算环境。

以后每次实验保留配置、Git commit、模型/数据哈希与实际环境，结果放在单独 output 目录。改动基线算法前先创建分支；研究扩展应与已经验证的 CASL 基线分开。
