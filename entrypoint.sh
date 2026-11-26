#!/bin/bash
# =============================================================================
# entrypoint.sh  –  Docker entrypoint for bare-metal multi-node Cupcake
# =============================================================================
# Required on all nodes:
#   NODE_RANK    – this node's rank (0, 1, 2, or 3)
#   MASTER_ADDR  – IP of node 0
#
# Required on node 0 only:
#   WORKER_HOSTS – horovodrun host string, e.g.
#                  "192.168.0.11:1,192.168.0.12:1,192.168.0.13:1,192.168.0.14:1"
#
# Optional (all map directly to CupcakeConfig fields via _env()):
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
#   SLEEP_CHECK_INTERVAL (default: 10)
#   SLEEP_START_AFTER    (default: 15)
#   TC_BANDWIDTH_MAX     (default: 1000mbit)
#   COMP_INJECT_SLEEP    (default: 0)
#   COMP_SLEEP_PROB_ON   (default: 0.20)
#   COMP_SLEEP_PROB_OFF  (default: 0.20)
# =============================================================================
set -euo pipefail

: "${NODE_RANK:?ERROR: NODE_RANK is required (0-3)}"
: "${MASTER_ADDR:?ERROR: MASTER_ADDR is required}"

MASTER_PORT="${MASTER_PORT:-12355}"
NNODES=4
MODE="${MODE:-straggler}"

# ── Export all training config as env vars (CupcakeConfig reads via _env()) ──
export MODEL="${MODEL:-resnet50}"
export EPOCHS="${EPOCHS:-100}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export LR="${LR:-0.0125}"
export COMPRESSOR="${COMPRESSOR:-efsignsgd}"
export COMM="${COMM:-allreduce}"
export MEMORY="${MEMORY:-residual}"
export FUSION_NUM="${FUSION_NUM:-2}"
export COMPRESS="${COMPRESS:-true}"
export TC_INTERFACE="${TC_INTERFACE:-enp8s0}"
export RESULTS_DIR="/app/results/${MODE}"

# ── Straggler config (only applied for straggler mode) ───────────────────────
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

# ── Start SSH server on every node ───────────────────────────────────────────
/usr/sbin/sshd

echo "============================================================"
echo " NODE_RANK   : ${NODE_RANK}"
echo " MASTER_ADDR : ${MASTER_ADDR}:${MASTER_PORT}"
echo " MODE        : ${MODE}"
echo " MODEL       : ${MODEL}  EPOCHS=${EPOCHS}  BATCH=${BATCH_SIZE}"
echo " NIC         : ${TC_INTERFACE}"
echo " RESULTS_DIR : ${RESULTS_DIR}"
echo "============================================================"

# Workers just serve SSH; horovodrun SSHes in and launches the script
if [ "${NODE_RANK}" != "0" ]; then
    echo "Worker node ${NODE_RANK} ready (SSH server running)."
    exec tail -f /dev/null
fi

# ── Master node (NODE_RANK=0) ─────────────────────────────────────────────────
: "${WORKER_HOSTS:?ERROR: WORKER_HOSTS required on node 0}"

echo "Master: waiting for SSH on all workers..."
IFS=',' read -ra HOSTLIST <<< "${WORKER_HOSTS}"
for entry in "${HOSTLIST[@]}"; do
    host="${entry%:*}"
    echo -n "  ${host} ..."
    until ssh -o ConnectTimeout=3 "${host}" exit 2>/dev/null; do
        sleep 2; echo -n "."
    done
    echo " OK"
done
echo "All workers reachable."

# No CLI flags passed to the script — CupcakeConfig reads everything from env vars
exec horovodrun \
    -np "${NNODES}" \
    -H  "${WORKER_HOSTS}" \
    --network-interface "${TC_INTERFACE}" \
    python /app/examples/cifar10_benchmark/cifar10_benchmark.py
