# 东南大学智算平台 CASL 作业

用学校 WebSSH 进入已经通过 Git 同步的项目仓库。

```bash
cd ~/casl/cognitive-ultrasound
bash scripts/run_seu_casl.sh submit cpu-check
```

同一个入口支持下列任务。任选当前需要的任务提交即可。

| 命令 | 默认队列 | 资源与内容 | 完成标记 |
| --- | --- | --- | --- |
| `bash scripts/run_seu_casl.sh submit data-prep` | `6126-24C-192G` | 2 CPU，校验已上传的 TAR，在 SHARE 解包，复制到 NVMe 并比较文件 | `DATA_PREP_OK` |
| `bash scripts/run_seu_casl.sh submit cpu-check` | `6126-24C-192G` | 2 CPU，核对官方权重、两盘病例名单，抽读同一批 12 个 HDF5 | `CPU_CHECK_OK` |
| `bash scripts/run_seu_casl.sh submit gpu-smoke` | `gpu-L20` | 1 GPU，验证 JAX 和 TensorFlow，再用官方 FP32 配置跑验证集 1 例、5 帧、7 条线 | `GPU_SMOKE_OK` |

单卡测试也可以指定 `gpu-A6000`。

```bash
bash scripts/run_seu_casl.sh submit gpu-smoke gpu-A6000
```

提交后记录学校返回的作业号。把 `123456` 换成实际号码。

```bash
bash scripts/run_seu_casl.sh status 123456
bash scripts/run_seu_casl.sh logs 123456
```

日志在 `~/casl/jobs/data-prep/` 或 `~/casl/jobs/acceptance/`。GPU 检查 JSON、指标和样例图在 `~/casl/jobs/acceptance/gpu-作业号/`。脚本使用学校的 `sbatch_wrapper`，内存由学校按 CPU 资源配置自动分配。[学校 CPU 作业指南](https://sc.seu.edu.cn/docs/hpc/contents.html#id14) · [学校 GPU 作业指南](https://sc.seu.edu.cn/docs/ai/contents.html#id25)

默认使用 `~/casl/envs/casl` 和 `/seu_nvme/home/chenchuan/当前用户名/casl`。如需适配其他项目目录、环境或 NVMe 路径，可分别设置 `CASL_SEU_ROOT`、`CASL_SEU_ENV`、`CASL_SEU_NVME_ROOT`。
