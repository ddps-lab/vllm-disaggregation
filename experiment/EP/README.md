# EP Online-Serving Test (2-4x g6.xlarge, 1x L4 24GiB each)

vLLM 0.26 expert-parallel smoke test: DP=N attention + EP=N experts across
N single-GPU nodes, Ray backend, OpenAI-compatible online serving.

## Verified model candidates (2026-07)

Per-GPU weight = replicated dense part + routed experts / N. Budget: 21.6 GiB
(24 x 0.9), "fits" requires >= 3 GiB left for KV cache.

| Model | per-GPU @2 | per-GPU @4 | Experts | Note |
|---|---|---|---|---|
| **deepseek-ai/DeepSeek-V2-Lite-Chat** (default) | 15.9 | 9.2 | 64 + 2 shared | MLA -> tiny KV, DeepSeek arch |
| Qwen/Qwen1.5-MoE-A2.7B-Chat | 15.1 | 9.3 | 60 + 1 shared | classic MoE baseline |
| moonshotai/Moonlight-16B-A3B-Instruct | 16.3 | 9.6 | 64 | DeepSeek-V3 arch |
| openai/gpt-oss-20b | 8.1 | 5.7 | 32 | MXFP4; needs vLLM Marlin path (OK on sm_89) |
| allenai/OLMoE-1B-7B-0924-Instruct | 6.9 | 3.9 | 64 | fastest download; good first smoke |
| ibm-granite/granite-3.1-3b-a800m-instruct | 3.3 | 1.9 | 40 | tiny |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | 29.9 (X) | 16.4 | 128 | **4 nodes only** |
| baidu/ERNIE-4.5-21B-A3B-PT | 21.9 (X) | 12.4 | 64 + 2 shared | **4 nodes only** |

All architectures verified to use FusedMoE (EP-capable) in this source tree.
Default all2all backend `allgather_reducescatter` works over plain TCP —
no EFA/RDMA needed (DeepEP is not usable on this fleet).

## Usage

```bash
# 0. local -> remote sync (repeat for every node)
./sync.sh <node_ip>

# 1. Ray cluster (env vars must be set at `ray start` time, not at serve time)
./ray_cluster.sh head                 # on head node
./ray_cluster.sh worker <head_ip>     # on each worker node

# 2. serve (head node only; Ray places one engine per GPU node)
./serve_ep.sh                                         # DP=2, DeepSeek-V2-Lite
DP=4 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 ./serve_ep.sh

# 3. client (anywhere that can reach head:8000)
python client.py --base-url http://<head_ip>:8000/v1 -n 8
```

## Fleet caveats (g6.xlarge)

- **Security group**: self-referencing all-TCP rule within the cluster SG
  (Ray uses random object/node-manager ports; NCCL uses ephemeral ports).
- **NCCL over ENA**: `NCCL_SOCKET_IFNAME=ens5` is exported inside
  `ray_cluster.sh` so Ray workers inherit it; exporting it only in the serve
  shell does NOT reach remote engines.
- **16 GiB host RAM is tight**: `ray_cluster.sh` caps the Ray object store at
  2 GB. Add an 8 GB swapfile if loading >= 30 GB checkpoints. (`--swap-space`
  no longer exists in v0.26 — the v1 engine removed CPU KV swap.)
- **Performance expectation**: EP does allgather+reducescatter across nodes
  for every MoE layer every step; over 10 Gbps TCP decode is
  communication-bound (~order of magnitude slower than single-node). This
  setup validates mechanics, not throughput.
