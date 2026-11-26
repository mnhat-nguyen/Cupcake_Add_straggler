import torch.distributed as dist
from mergeComp_dl.torch import Communicator


class PoolAllreduce(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.name = "PoolAllReduce"
        self.injector = None

    def async_send(self, tensor, name):
        """
        Called per-parameter by the optimizer hook with raw gradient tensor.
        Accumulates gradient into the pool via memory.pool_compensate().
        When this parameter triggers the pool (name in compress_name),
        all-reduces the raw pool tensor. Otherwise returns (None, ctx).
        ctx is always (name, numel) so receive_step can read ctx[0] = name.
        """
        # Accumulate gradient in pool + residual compensation
        pool_tensor, _ = self.memory.pool_compensate(tensor, name)

        ctx = (name, tensor.numel())

        if pool_tensor is None:
            # Pool not full yet — nothing to send for this parameter
            return None, ctx

        if self.injector is not None:
            self.injector.inject()

        # Clone pool tensor before in-place all_reduce (pool_tensor is a view
        # into memory.velocities and must not be modified in-place)
        t = pool_tensor.clone()
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        return (work, t), ctx

    def wait_receive(self, handle, ctx):
        """
        Waits for the all_reduce to complete and returns the SUM pool tensor.
        memory.reduce() in receive_step will divide by world_size.
        Returns None if pool was not triggered (handle is None).
        """
        if handle is None:
            return None
        work, t = handle
        work.wait()
        return t
