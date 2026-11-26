import sys
import torch


def grace_from_params(params):
    comp = params.get('compressor', 'none')
    mem = params.get('memory', 'none')
    comm = params.get('communicator', 'allreduce')
    model_params = params.get('params', 'none')
    ratio = params.get('ratio', 0.01)
    if model_params == 'none':
        sys.exit("No model parameters for grace_from_params()")
    fusion_num = params.get('fusion_num', 2)
    momentum = params.get('momentum', 0.9)
    qsgd_quantum = params.get('quantum', 64)
    print("[Compression Setup] compressor: {}\n\tmemory: {}\n\tcommunicator: {}\n\tsparsity ratio: {}\n\tfusion num: {}".format(
        comp,
        mem,
        comm,
        ratio,
        fusion_num
    ))


    if comp == 'dgc':
        from mergeComp_dl.torch.compressor.pooldgc import PoolDgcCompressor
        compressor = PoolDgcCompressor(compress_ratio=ratio)
    elif comp == 'efsignsgd':
        from mergeComp_dl.torch.compressor.poolefsignsgd import PoolEFSignSGDCompressor
        compressor = PoolEFSignSGDCompressor()
    elif comp == 'fp16':
        from mergeComp_dl.torch.compressor.poolfp16 import PoolFP16Compressor
        compressor = PoolFP16Compressor()
    elif comp == 'none':
        from mergeComp_dl.torch.compressor.poolnone import PoolNoneCompressor
        compressor = PoolNoneCompressor()
    elif comp == 'onebit':
        from mergeComp_dl.torch.compressor.poolonebit import PoolOneBitCompressor
        compressor = PoolOneBitCompressor()
    elif comp == 'qsgd':
        from mergeComp_dl.torch.compressor.poolqsgd import PoolQSGDCompressor
        compressor = PoolQSGDCompressor(quantum_num=qsgd_quantum)
    elif comp == 'randomk':
        from mergeComp_dl.torch.compressor.poolrandomk import PoolRandomKCompressor
        compressor = PoolRandomKCompressor(compress_ratio=ratio)
    elif comp == 'signsgd':
        from mergeComp_dl.torch.compressor.poolsignsgd import PoolSignSGDCompressor
        compressor = PoolSignSGDCompressor()
    elif comp == 'signum':
        from mergeComp_dl.torch.compressor.poolsignum import PoolSignumCompressor
        compressor = PoolSignumCompressor(momentum=momentum)
    elif comp == 'terngrad':
        from mergeComp_dl.torch.compressor.poolterngrad import PoolTernGradCompressor
        compressor = PoolTernGradCompressor()
    elif comp == 'topk':
        from mergeComp_dl.torch.compressor.pooltopk import PoolTopKCompressor
        compressor = PoolTopKCompressor(compress_ratio=ratio)
    else:
        raise NotImplementedError(comp)

    if fusion_num > 0:
        if mem == 'dgc':
            from mergeComp_dl.torch.memory.pooldgc import PoolDgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = PoolDgcMemory(model_params, momentum=0, fusion_num=fusion_num, momentum_masking=False)
            else:
                memory = PoolDgcMemory(model_params, fusion_num=fusion_num)
        elif mem == 'none':
            from mergeComp_dl.torch.memory.poolnone import PoolNoneMemory
            memory = PoolNoneMemory(model_params, fusion_num=fusion_num)
        elif mem == 'residual':
            from mergeComp_dl.torch.memory.poolresidual import PoolResidualMemory
            memory = PoolResidualMemory(model_params, fusion_num=fusion_num)
        else:
            raise NotImplementedError(mem)
    elif fusion_num == 0:
        if mem == 'dgc':
            from mergeComp_dl.torch.memory.dgc import DgcMemory
            if comp == 'topk' or comp == 'randomk':
                memory = DgcMemory(model_params, momentum=0, momentum_masking=False)
            else:
                memory = DgcMemory(model_params)
        elif mem == 'none':
            from mergeComp_dl.torch.memory.none import NoneMemory
            memory = NoneMemory(model_params)
        elif mem == 'residual':
            from mergeComp_dl.torch.memory.residual import ResidualMemory
            memory = ResidualMemory(model_params)
        else:
            raise NotImplementedError(mem)

    if comm == 'allreduce':
        from mergeComp_dl.torch.communicator.pool_allreduce import PoolAllreduce
        return PoolAllreduce(compressor, memory)
    elif comm == 'allgather':
        from mergeComp_dl.torch.communicator.DDP_allgather import DDPAllgather
        return DDPAllgather(compressor, memory)
    elif comm == 'ps':
        from mergeComp_dl.torch.communicator.pool_ps import PoolPS
        return PoolPS(compressor, memory)
    else:
        raise NotImplementedError(comm)



def add_parser_arguments(parser):
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                        help='apply gradient predivide factor in optimizer (default: 1.0)')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')
    parser.add_argument('--compress', action='store_true', default=False,
                        help='use gradient compression')
    parser.add_argument('--compressor', type=str, default='efsignsgd',
                        help='compress algorithm')
    parser.add_argument('--compress-ratio', type=float, default=0.01,
                        help='compress ratio for sparsification')
    parser.add_argument('--memory', type=str, default='residual',
                        help='compress algorithm')
    parser.add_argument('--fusion-num', type=int, default=2,
                        help='the number of merged tensors')
    parser.add_argument('--comm', type=str, default='allgather',
                        help='communication for compression')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use Adam optimizer')
    parser.add_argument('--speed-test', action='store_true', default=False,
                        help='test the training speed')

    # ── Communication straggler (tc) ─────────────────────────────────────────
    parser.add_argument('--inject-tc', action='store_true', default=False,
                        help='enable tc HTB bandwidth throttling on straggler rank')
    parser.add_argument('--inject-sleep', action='store_true', default=False,
                        help='enable sleep-based communication straggler injection')
    parser.add_argument('--sleep-target-rank', type=int, default=0,
                        help='rank that acts as communication straggler (-1 = all ranks)')
    parser.add_argument('--sleep-prob-on', type=float, default=0.30,
                        help='Markov probability of entering slow state')
    parser.add_argument('--sleep-prob-off', type=float, default=0.20,
                        help='Markov probability of leaving slow state')
    parser.add_argument('--sleep-check-interval', type=int, default=10,
                        help='Markov chain re-roll every N inject() calls')
    parser.add_argument('--sleep-start-after', type=int, default=15,
                        help='number of steps before straggler injection begins')
    parser.add_argument('--sleep-duration-ratio', type=float, default=0.5,
                        help='sleep duration as fraction of recent step duration (sleep mode)')
    parser.add_argument('--sleep-seed', type=int, default=42,
                        help='RNG seed for communication straggler Markov chain')
    parser.add_argument('--tc-interface', type=str, default='enp8s0',
                        help='network interface for tc bandwidth control')
    parser.add_argument('--throttle-dst-ip', type=str, default='',
                        help='destination IP to throttle (empty = all traffic)')
    parser.add_argument('--tc-bandwidth-max', type=str, default='1000mbit',
                        help='baseline (fast) bandwidth for tc HTB')
    parser.add_argument('--tc-bandwidth-ratios', type=float, nargs='+', default=[1.0, 0.1],
                        help='bandwidth ratios for tc presets (first=fast, rest=slow)')

    # ── Computation straggler (sleep) ─────────────────────────────────────────
    parser.add_argument('--comp-inject-sleep', action='store_true', default=False,
                        help='enable sleep-based computation straggler injection')
    parser.add_argument('--comp-sleep-target-rank', type=int, default=-1,
                        help='rank that acts as computation straggler (-1 = all ranks)')
    parser.add_argument('--comp-sleep-prob-on', type=float, default=0.20,
                        help='Markov probability of entering sleep state')
    parser.add_argument('--comp-sleep-prob-off', type=float, default=0.20,
                        help='Markov probability of leaving sleep state')
    parser.add_argument('--comp-sleep-check-interval', type=int, default=10,
                        help='Markov chain re-roll every N batches')
    parser.add_argument('--comp-sleep-duration-ratio', type=float, default=0.4,
                        help='sleep duration as fraction of recent forward-pass time')
    parser.add_argument('--comp-sleep-seed', type=int, default=43,
                        help='RNG seed for computation straggler Markov chain')

    parser.add_argument('--scheduler', action='store_true', default=False,
                        help='use scheduler partition')
    parser.add_argument('--scheduler-epoch', type=int, default=20,
                        help='scheduler AVERAGE for epoch')
    parser.add_argument('--scheduler-step', type=int, default=2,
                        help='scheduler add step for search')
    parser.add_argument('--scheduler-warmup', type=int, default=5,
                        help='warmup iterations for scheduler')
    parser.add_argument('--scheduler-baseline', action='store_true', default=False,
                        help='use scheduler baseline based on tensor number')
    return parser



def wrap_compress_optimizer(model, optimizer, args):
    if args.compress:
        params = {
            'compressor': args.compressor,
            'memory': args.memory,
            'communicator': args.comm,
            'params': model.named_parameters(),
            'ratio': args.compress_ratio,
            'fusion_num': args.fusion_num,
        }

        grc = grace_from_params(params)
        compress_config = (1, 0)

        from .horovod.optimizer import DistributedOptimizer
        optimizer = DistributedOptimizer(optimizer, compression=grc,
                                         named_parameters=model.named_parameters(),
                                         compress_config=compress_config)
        return optimizer, grc
    else:
        # No compression — gradients must be synced externally (e.g. DDP or manual dist.all_reduce)
        return optimizer, None
