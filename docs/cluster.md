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

2026-11-17重新探查：ws6四张GPU均已分配；ws5可分配RTX5090（约32GB），大盘为`/media/Ws5_4TB`。在`/media/Ws5_4TB/zdx/expert-find-agent`创建环境，依赖安装作业39285完成；官方模型逐文件摘要验证后，39289用68秒完成原始EvidenceBench测试集来源向量，39292用7秒完成固定评价，两者0:0。本地仅回传结果与日志，完整向量留计算节点。两次直接tar回传遇到Slurm二进制流截断，最终改用计算节点gzip+base64回传；解压结果JSON的SHA256与计算节点一致，未重跑计算作业。

全库来源归属检索实验：39309在ws5用RTX5090重新编码固定20条完整需求（4秒，0:0）；39313在ws6单CPU/8GB完成96749来源的精确/ANN对照（61秒，0:0）；39326单CPU/8GB在同一图上比较efSearch下限64和512（63秒，0:0）。ws6未请求GPU，因为四张GPU当时已被分配。faiss-cpu1.16.0安装通过计算节点Slurm执行；模型权重按原固定官方摘要逐文件验证。只回传查询向量、协议、评价和日志，完整来源向量仍留大盘。两组结果的协议/向量/评价/摘要已核对本地与计算节点SHA256。


双语检索诊断：39435在ws6单张RTX4090、4CPU、12GB配额下完成，0:0，用时2分51秒。固定1000个construction真实需求衍生意图，每类200个，对应2000个中英文视图；核验原始来源、向量与官方模型文件摘要后批量编码，无截断。比较相同96749来源和9861专家上的精确/ef64/ef512，CPU计时不含批量编码或索引构建。结果经独立Slurm CPU步骤gzip+base64回传，本地检查CRC、相对路径和协议/查询向量/评价摘要。脚本为`experiments/cluster_bilingual_retrieval.sbatch`。检索稳定性和精确排序保留率不等于专家相关性精度。
