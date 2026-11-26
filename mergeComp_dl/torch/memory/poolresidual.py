import torch
import torch.distributed as dist

from .memory_pool import MemoryPool


class PoolResidualMemory(MemoryPool):
    def __init__(self, named_parameters, fusion_num=2, beta=0.9, gamma=1.0):
        self.beta = beta
        self.gamma = gamma
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        super().__init__(named_parameters, fusion_num)


    def compensate(self, tensor, name):
        """vec stores the residuals"""
        grad = self.get_grad(name)
        residual = self.get_velocity(name)
        #residual.add_(grad)
        residual.mul_(self.beta).add_(self.gamma*grad)


    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = self.get_velocity(name)
        residual.copy_(tensor.view(-1) - tensor_decompressed)


    def reduce(self, tensor, name):
        reduction = self.get_reduction(name)
        reduction.set_(tensor/self.world_size)