#!/bin/bash
# =============================================================================
# entrypoint.sh  –  Docker entrypoint for bare-metal multi-node Cupcake (DDP)
# =============================================================================
# Required on all nodes:
#   NODE_RANK    – this node's rank (0, 1, 2, or 3)
#   MASTER_ADDR  – IP of node 0
#
# Optional (all map directly to CupcakeConfig fields via _env()):
#   MASTER_PORT          (default: 20015)
#   MODE                 baseline | straggler  (default: straggler)
#   MODEL                resnet18 | resnet50 | resnet152  (default: resnet50)
#   EPOCHS               (default: 100)
#   BATCH_SIZE           (default: 256)
#   COMPRESSOR           (default: efsignsgd)
#   COMM                 (default: allreduce)
#   MEMORY               (default: residual)
#   FUSION_NUM           (default: 2)
#   TC_INTERFACE         (default: enp8s0)
#   SLEEP_TARGET_RANK    (default: 0)
#   SLEEP_PROB_ON        (default: 0.30)
#   SLEEP_PROB_OFF       (default: 0.20)
#   TC_BANDWIDTH_MAX     (default: 1000mbit)
#   COMP_INJECT_SLEEP    (default: 0)
# =============================================================================
set -euo pipefail

: "${NODE_RANK:?ERROR: NODE_RANK is required (0-3)}"
: "${MASTER_ADDR:?ERROR: MASTER_ADDR is required}"

MASTER_PORT="${MASTER_PORT:-20015}"
NNODES=4
NPROC_PER_NODE=1
MODE="${MODE:-straggler}"
TC_INTERFACE="${TC_INTERFACE:-enp8s0}"

# ── Export training config as env vars (CupcakeConfig reads via _env()) ──────
export MODEL="${MODEL:-resnet101}"
export DATA_ROOT="${DATA_ROOT:-/app/data_cache}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-0.0125}"
export COMPRESSOR="${COMPRESSOR:-efsignsgd}"
export COMM="${COMM:-allreduce}"
export MEMORY="${MEMORY:-residual}"
export FUSION_NUM="${FUSION_NUM:-2}"
export COMPRESS="${COMPRESS:-true}"
export TC_INTERFACE="${TC_INTERFACE}"
export RESULTS_DIR="/app/results/${MODE}"

# ── Straggler config (straggler mode only) ────────────────────────────────────
if [ "${MODE}" = "straggler" ]; then
    export INJECT_TC=1
    export SLEEP_TARGET_RANK="${SLEEP_TARGET_RANK:-0}"
    export SLEEP_PROB_ON="${SLEEP_PROB_ON:-0.30}"
    export SLEEP_PROB_OFF="${SLEEP_PROB_OFF:-0.20}"
    export SLEEP_CHECK_INTERVAL="${SLEEP_CHECK_INTERVAL:-10}"
    export SLEEP_START_AFTER="${SLEEP_START_AFTER:-15}"
    export TC_BANDWIDTH_MAX="${TC_BANDWIDTH_MAX:-1000mbit}"

    if [ "${COMP_INJECT_SLEEP:-0}" = "1" ]; then
        export COMP_INJECT_SLEEP=1
        export COMP_SLEEP_TARGET_RANK="${COMP_SLEEP_TARGET_RANK:--1}"
        export COMP_SLEEP_PROB_ON="${COMP_SLEEP_PROB_ON:-0.20}"
        export COMP_SLEEP_PROB_OFF="${COMP_SLEEP_PROB_OFF:-0.20}"
        export COMP_SLEEP_DURATION_RATIO="${COMP_SLEEP_DURATION_RATIO:-0.4}"
    fi
fi

# ── NCCL config ───────────────────────────────────────────────────────────────
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${TC_INTERFACE}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

echo "============================================================"
echo " NODE_RANK   : ${NODE_RANK}"
echo " MASTER_ADDR : ${MASTER_ADDR}:${MASTER_PORT}"
echo " MODE        : ${MODE}"
echo " MODEL       : ${MODEL}  EPOCHS=${EPOCHS}  BATCH=${BATCH_SIZE}"
echo " NIC         : ${TC_INTERFACE}"
echo " RESULTS_DIR : ${RESULTS_DIR}"
echo "============================================================"

# No CLI flags passed — CupcakeConfig reads everything from env vars
exec torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --rdzv_backend=static \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    /app/examples/cifar10_benchmark/cifar10_benchmark.py
