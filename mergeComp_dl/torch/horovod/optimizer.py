import os
import warnings
from contextlib import contextmanager

import torch
import torch.distributed as dist


def split_list(l, n):
    d, r = divmod(len(l), n)
    return [l[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]


class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression, compress_config=(1, 0),
                 backward_passes_per_step=1, num_groups=0, name=None):
        super(self.__class__, self).__init__(params)
        self._compression = compression
        self.name = name

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [('allreduce.noname.%s' % i, v)
                                for param_group in self.param_groups
                                for i, v in enumerate(param_group['params'])]

        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of tuples (name, parameter).')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids  = {id(v) for param_group in self.param_groups for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.backward_passes_per_step = backward_passes_per_step
        self._allreduce_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles        = {}
        self._grad_accs      = []
        self._requires_update = set()
        self._synchronized    = False
        self._should_synchronize = True
        self._num_groups      = num_groups
        self._p_to_group      = {}
        self._group_counts    = {}

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            self._register_hooks()

        self.iterations = 0
        self.compression_start_iteration = max(compress_config[0], 1)
        self.uncompression_frequency     = max(compress_config[1], 0)

    def load_state_dict(self, *args, **kwargs):
        self._handles = {}
        self._synchronized = False
        self._should_synchronize = True
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step
        super(self.__class__, self).load_state_dict(*args, **kwargs)

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._allreduce_delay:
            self._allreduce_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        if self._num_groups > 0:
            p_list = [p for param_group in self.param_groups
                        for p in param_group['params'] if p.requires_grad]

            p_list_names = [self._parameter_names.get(p) for p in p_list]
            container = [p_list_names]
            dist.broadcast_object_list(container, src=0)
            p_list_names = container[0]
            p_list = sorted(p_list, key=lambda p: p_list_names.index(
                self._parameter_names.get(p)))

            p_groups = split_list(p_list, self._num_groups)
            p_groups = [tuple(p) for p in p_groups]
            for group in p_groups:
                for p in group:
                    self._p_to_group[p] = group
                self._group_counts[group] = 0

        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _compression_iteration(self):
        if self.iterations < self.compression_start_iteration:
            return False
        if self.uncompression_frequency > 0 and self.iterations % self.uncompression_frequency == 0:
            return False
        return True

    def _communicate_grad_async(self, p):
        name   = self._parameter_names.get(p)
        tensor = p.grad
        handle, ctx = None, None

        if self._compression_iteration():
            handle, ctx = self._compression.send_step(tensor, name)
        else:
            if self.iterations == 0:
                self._compression.memory.add_tensor(name, tensor.numel())
            work   = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
            handle = (work, tensor)
        return handle, ctx

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._allreduce_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call to step().")
            assert not p.grad.requires_grad
            assert self._allreduce_delay[p] > 0
            handle, ctx = None, None
            self._allreduce_delay[p] -= 1
            if self._allreduce_delay[p] == 0:
                handle, ctx = self._communicate_grad_async(p)
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):
        completed = set()
        for x in self._handles.keys():
            completed.update(x) if isinstance(x, tuple) else completed.add(x)
        missing_p = self._requires_update - completed
        for p in missing_p:
            handle, ctx = self._communicate_grad_async(p)
            self._handles[p] = (handle, ctx)

        for p, (handle, ctx) in self._handles.items():
            if handle is None:
                handle, ctx = self._communicate_grad_async(p)
                self._handles[p] = (handle, ctx)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        for p, (handles, ctx) in self._handles.items():
            self._allreduce_delay[p] = self.backward_passes_per_step
            if handles:
                if self._compression_iteration():
                    self._compression.receive_step(handles, ctx)
                else:
                    work, tensor = handles
                    work.wait()
                    tensor.div_(world_size)
        self._handles.clear()
        self._synchronized = True

    @contextmanager
    def skip_synchronize(self):
        self._should_synchronize = False
        try:
            yield
        finally:
            self._should_synchronize = True

    def step(self, closure=None):
        if self._should_synchronize:
            if self._synchronized:
                warnings.warn("optimizer.step() called without optimizer.skip_synchronize() "
                              "context after optimizer.synchronize(). This can cause training "
                              "slowdown. You may want to consider using "
                              "optimizer.skip_synchronize() context if you use "
                              "optimizer.synchronize() in your code.")
            self.synchronize()
        self._synchronized = False
        if self._compression.memory.fusion_num <= 1:
            super(self.__class__, self).step(closure)
        self.iterations += 1

    def zero_grad(self):
        if self._handles:
            raise AssertionError("optimizer.zero_grad() was called after loss.backward() "
                                 "but before optimizer.step() or optimizer.synchronize(). "
                                 "This is prohibited as it can cause a race condition.")
        return super(self.__class__, self).zero_grad()


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=None,
                         compress_config=(0, 1),
                         backward_passes_per_step=1,
                         num_groups=0):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters, compression,
               compress_config, backward_passes_per_step, num_groups)
