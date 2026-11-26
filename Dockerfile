# =============================================================================
# Dockerfile  –  Cupcake multi-node Horovod training image (bare-metal)
# =============================================================================
# Build:
#   docker build -t cupcake:latest .
#
# Run on each node (substitute NODE_RANK, MASTER_ADDR, WORKER_HOSTS):
#   docker run --rm --gpus all --network host --cap-add NET_ADMIN \
#       -e NODE_RANK=0 \
#       -e MASTER_ADDR=192.168.0.11 \
#       -e WORKER_HOSTS=192.168.0.11:1,192.168.0.12:1,192.168.0.13:1,192.168.0.14:1 \
#       -e MODE=straggler \
#       -v ~/data_cache:/app/data_cache \
#       -v ~/results:/app/results \
#       cupcake:latest
# =============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# System dependencies (MPI + SSH for Horovod multi-node + tc for straggler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenmpi-dev openmpi-bin \
        openssh-server openssh-client \
        iproute2 net-tools iputils-ping procps \
    && rm -rf /var/lib/apt/lists/*

# Passwordless SSH between nodes (all nodes share the same keypair)
RUN mkdir -p /root/.ssh /run/sshd && \
    ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N '' && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    printf "StrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n" \
        >> /root/.ssh/config

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build Horovod with NCCL + MPI (compiled from source — takes a few minutes)
RUN HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_WITH_PYTORCH=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_MXNET=1 \
    pip install --no-cache-dir horovod[pytorch]

# Project source
COPY mergeComp_dl/            mergeComp_dl/
COPY straggler_injector.py    .
COPY comp_sleep_injector.py   .
COPY examples/cifar10_benchmark/ examples/cifar10_benchmark/

# Runtime entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
