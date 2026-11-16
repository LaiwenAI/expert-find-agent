# GPU 集群实验

用户提供的操作文档：https://docs.qq.com/doc/DSGpxVUNDeFVQaWV1 。已读取公开预览中的节点、存储、Slurm 提交和环境要求；文档完整数据接口返回 401，未将未读部分推测成规则。

登录入口为 `dxzhou@10.21.50.60`（ws0）。仅用于登录、Slurm 提交和状态查询。计算、模型校验、解包、安装环境均经 Slurm 在计算节点执行。已有 `/opt/miniconda3`，不安装另一套 Miniconda。

2026-11-16 探查：Slurm gpu 分区包含 ws1–ws6。ws1 空闲但只有根文件系统；ws2–ws5 的 GPU 均已分配。ws6 有一张可调度 48GB RTX 4090，数据盘 `/mnt/sda` 约 6.1TB 可用；根目录不可直接创建 zdx，但用户拥有 `/mnt/sda/dxzhou`。实际实验目录为：

```
/mnt/sda/dxzhou/zdx/expert-find-agent
```

独立 Python venv 位于该目录，基于现有 Miniconda 的 Python 3.13 创建。实测 PyTorch 2.15.0+cu132、Transformers 5.23.0、CUDA 13.2；驱动 595.84。作业仅看到其分配 GPU。集群状态随时变化，后续作业应重新查询。

从本地经登录节点提交：

```sh
ssh dxzhou@10.21.50.60 sbatch < experiments/cluster_qwen_embeddings.sbatch
```

脚本指定单节点、单 GPU、4 CPU、8GB 主存，最长 45 分钟，任务结束自动释放。模型和数据已传入计算节点大盘；代码在计算节点执行。模型权重先逐文件校验官方固定修订的摘要，然后执行 CUDA smoke，再依次跑 SIGIR 和 NIPS 的冻结 Qwen3 Embedding 基线。两种硬件后端使用不同结果目录，不能混用缓存。

首个正式实验 Job ID 为 39221，Slurm accounting 已确认 COMPLETED、ExitCode 0:0，总用时 77 秒；SIGIR 和 NIPS 编码与评分均完成，结果已回传本地。分配资源为单 GPU、4 CPU、8GB 主存。

日志位于 `logs/qwen-<jobid>.out` 与 `.err`。每 32 个输入保存一个带摘要的 NPZ 块，可在协议不变时续跑。查询向量缓存评分不包含在线编码时延。原始企业/教师资料及 `.env` 未上传本轮任务，输入仅为已有公开 CoF 归档和官方冻结模型。


后续授权研究继续在计算节点执行。作业39249在ws6运行需求先行论文检索，COMPLETED、ExitCode 0:0、10秒：读取此前已建立的完整归属来源向量，仅选择54409篇论文，对原固定20条真实企业需求编码及召回；本轮包含原始需求文本和教师论文前缀，不再是仅公开CoF输入。GPU仍由Slurm分配，未在ws0计算。结果及运行环境已回传本地，未传输密钥或.env。提交脚本为 `experiments/cluster_demand_papers.sbatch`。


全尺度冻结嵌入及来源融合：KDD作业39252在ws6完成（0:0，9分04秒），处理99594个来源及174查询；SciRepEval作业39253处理73257来源及107查询后因DOI文件名写出失败，编码块均已保留。修复为URL编码文件名后，39260在21秒内复用缓存完成评分/融合，状态COMPLETED、0:0；不把恢复时间当作编码耗时。完整向量留在计算节点，结果JSON和运行信息回传本地；BM25分数快照哈希本地/远端一致。所有模型编码只在Slurm计算节点执行。
