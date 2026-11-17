# GPU Cluster Experiments

User-provided operating documentation: https://docs.qq.com/doc/DSGpxVUNDeFVQaWV1. The node, storage, Slurm submission, and environment requirements were read from the public preview; the document’s complete data interface returned 401, so unread portions were not inferred as rules.

The login entry point is `dxzhou@10.21.50.60` (ws0). It is used only for login, Slurm submission, and status queries. Computation, model validation, unpacking, and environment installation are all performed on compute nodes via Slurm. `/opt/miniconda3` already exists; no additional Miniconda installation is needed.

2026-11-16 inspection: the Slurm gpu partition includes ws1–ws6. ws1 was idle but had only its root filesystem; GPUs on ws2–ws5 were all allocated. ws6 had one schedulable 48GB RTX 4090, with approximately 6.1TB available on data disk `/mnt/sda`; zdx could not be created directly in the root directory, but the user owns `/mnt/sda/dxzhou`. The actual experiment directory is:

```
/mnt/sda/dxzhou/zdx/expert-find-agent
```

An independent Python venv is located in that directory, created with Python 3.13 from the existing Miniconda installation. Tested versions: PyTorch 2.15.0+cu132, Transformers 5.23.0, CUDA 13.2; driver 595.84. Jobs see only their allocated GPU. Cluster status can change at any time, so it should be queried again for later jobs.

Submit from the local machine through the login node:

```sh
ssh dxzhou@10.21.50.60 sbatch < experiments/cluster_qwen_embeddings.sbatch
```

The script specifies one node, one GPU, 4 CPUs, 8GB of main memory, and a maximum runtime of 45 minutes; resources are released automatically when the job ends. The model and data have been transferred to the compute node’s large disk; code runs on the compute node. Model weights are first checked file by file against the official fixed-revision digests, followed by a CUDA smoke test, then the frozen Qwen3 Embedding baselines for SIGIR and NIPS are run in sequence. The two hardware backends use different result directories; their caches must not be mixed.

The first formal experiment, Job ID 39221, was confirmed by Slurm accounting as COMPLETED, ExitCode 0:0, with a total runtime of 77 seconds; encoding and scoring for SIGIR and NIPS both completed, and the results were returned locally. Allocated resources were one GPU, 4 CPUs, and 8GB of main memory.

Logs are located at `logs/qwen-<jobid>.out` and `.err`. One NPZ chunk with a digest is saved for every 32 inputs, allowing a run to resume as long as the protocol remains unchanged. Scoring with cached query vectors does not include online encoding latency. The original company/teacher materials and `.env` were not uploaded for this round; the inputs consisted only of the existing public CoF archive and the official frozen model.


Subsequent authorized research continued to run on compute nodes. Job 39249 ran a requirements-first paper search on ws6 and completed with COMPLETED, ExitCode 0:0, in 10 seconds: it read the previously established complete attribution-source vectors, selected only 54409 papers, and encoded and retrieved against the original fixed set of 20 real company requirements. This round included the original requirement texts and teacher-paper prefixes, and was no longer limited to public CoF inputs. The GPU was still allocated by Slurm; no computation was performed on ws0. Results and runtime environment were returned locally; no keys or .env files were transferred. The submission script is `experiments/cluster_demand_papers.sbatch`.


Full-scale frozen embeddings and source fusion: KDD Job 39252 completed on ws6 (0:0, 9 minutes 04 seconds), processing 99594 sources and 174 queries. SciRepEval Job 39253 processed 73257 sources and 107 queries, then failed when writing a DOI filename; all encoding chunks were retained. After fixing the issue by URL-encoding filenames, Job 39260 reused the cache to complete scoring/fusion in 21 seconds, with status COMPLETED, 0:0; recovery time is not counted as encoding time. Full vectors remain on the compute node, while result JSON and runtime information were returned locally. The BM25 score snapshot hashes match locally and remotely. All model encoding was performed only on Slurm compute nodes.

2026-11-17 re-inspection: all four GPUs on ws6 were allocated; an RTX5090 (approximately 32GB) was available on ws5, with large disk `/media/Ws5_4TB`. An environment was created in `/media/Ws5_4TB/zdx/expert-find-agent`, and dependency installation Job 39285 completed. After verifying the official model file digests one by one, Job 39289 completed source-vector generation for the original EvidenceBench test set in 68 seconds, and Job 39292 completed the fixed evaluation in 7 seconds; both exited 0:0. Only results and logs were returned locally; full vectors remain on the compute node. Two direct tar transfers back encountered Slurm binary-stream truncation, so the final transfer used gzip+base64 on the compute node. The SHA256 of the decompressed result JSON matches the compute-node copy; the compute jobs were not rerun.

Full-corpus source-attribution retrieval experiment: Job 39309 re-encoded the fixed 20 complete requirements on ws5 using an RTX5090 (4 seconds, 0:0); Job 39313 completed an exact/ANN comparison over 96749 sources on ws6 with one CPU and 8GB (61 seconds, 0:0); Job 39326 compared efSearch lower bounds of 64 and 512 on the same graph with one CPU and 8GB (63 seconds, 0:0). No GPU was requested on ws6 because all four GPUs were allocated at the time. Installing faiss-cpu1.16.0 was performed via Slurm on a compute node; model weights were verified file by file against the original fixed official digests. Only query vectors, protocol, evaluation, and logs were returned; full source vectors remain on the large disk. The protocol, vectors, evaluation, and digests for both result sets were checked against local and compute-node SHA256 values.


Bilingual retrieval diagnostics: Job 39435 completed on ws6 with one RTX4090, a 4CPU, 12GB allocation, exiting 0:0 in 2 minutes 51 seconds. It used 1000 fixed construction real requirements to derive intents, 200 per category, corresponding to 2000 Chinese and English views. After verifying the original sources, vectors, and official model-file digests, it performed batch encoding without truncation. It compared exact/ef64/ef512 over the same 96749 sources and 9861 experts; CPU timings exclude batch encoding and index construction. Results were returned via an independent Slurm CPU step using gzip+base64. Locally, CRC, relative paths, and protocol/query-vector/evaluation digests were checked. The script is `experiments/cluster_bilingual_retrieval.sbatch`. Retrieval stability and exact-ranking retention rate are not equivalent to expert relevance precision.
