#!/bin/bash
# =============================================================================
# run.sh  –  launch Cupcake on each node (run this script on every node)
#
# Usage:
#   NODE_RANK=0 MASTER_ADDR=192.168.0.11 bash run.sh   # node 0
#   NODE_RANK=1 MASTER_ADDR=192.168.0.11 bash run.sh   # node 1
#   NODE_RANK=2 MASTER_ADDR=192.168.0.11 bash run.sh   # node 2
#   NODE_RANK=3 MASTER_ADDR=192.168.0.11 bash run.sh   # node 3
#
# Results land in ~/results/straggler/metrics.json
#                  ~/results/baseline/metrics.json
# =============================================================================

WORKER_HOSTS="192.168.0.11:1,192.168.0.12:1,192.168.0.13:1,192.168.0.14:1"

# ── Straggler run (comm straggler on rank 0 via tc) ───────────────────────────
docker run --rm --gpus all --network host --cap-add NET_ADMIN \
    -e NODE_RANK="${NODE_RANK}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e WORKER_HOSTS="${WORKER_HOSTS}" \
    -e MODE=straggler \
    -v ~/data_cache:/app/data_cache \
    -v ~/results:/app/results \
    cupcake:latest


# ── Baseline run (no straggler) ───────────────────────────────────────────────
# docker run --rm --gpus all --network host --cap-add NET_ADMIN \
#     -e NODE_RANK="${NODE_RANK}" \
#     -e MASTER_ADDR="${MASTER_ADDR}" \
#     -e WORKER_HOSTS="${WORKER_HOSTS}" \
#     -e MODE=baseline \
#     -v ~/data_cache:/app/data_cache \
#     -v ~/results:/app/results \
#     cupcake:latest
