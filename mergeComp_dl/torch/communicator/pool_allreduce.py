import torch.distributed as dist
from mergeComp_dl.torch import Communicator


class PoolAllreduce(Communicator):
    def __init__(self, compressor, memory):
        super().__init__(compressor, memory)
        self.name = "PoolAllReduce"
        self.injector = None

    def async_send(self, tensors_compressed, ctx):
        if self.injector is not None:
            self.injector.inject()
        tensor = tensors_compressed[0]
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
        return (work, tensor)

    def wait_receive(self, handle, ctx):
        work, tensor = handle
        work.wait()
        tensor.div_(dist.get_world_size())
        return [self.compressor.decompress([tensor], ctx)]
