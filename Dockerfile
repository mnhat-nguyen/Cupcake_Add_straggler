# =============================================================================
# Dockerfile  –  Cupcake multi-node DDP training image (bare-metal, no Horovod)
# =============================================================================
# Build:
#   docker build -t cupcake:latest .
#
# Run on each node (substitute NODE_RANK and MASTER_ADDR):
#   docker run --rm --gpus all --network host --cap-add NET_ADMIN \
#       -e NODE_RANK=0 \
#       -e MASTER_ADDR=192.168.0.11 \
#       -e MODE=straggler \
#       -v ~/data_cache:/app/data_cache \
#       -v ~/results:/app/results \
#       cupcake:latest
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies (tc for straggler injection)
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 net-tools iputils-ping procps \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project source
COPY mergeComp_dl/            mergeComp_dl/
COPY straggler_injector.py    .
COPY comp_sleep_injector.py   .
COPY examples/cifar10_benchmark/ examples/cifar10_benchmark/

# Runtime entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
