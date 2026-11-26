'''Train CIFAR10 with PyTorch + DDP + Cupcake compression.'''
import time
import json
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist

from models import *
import torchvision
import torchvision.transforms as transforms

_script_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir   = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _script_dir)   # models, utils, config
sys.path.insert(0, _root_dir)     # mergeComp_dl, straggler_injector, comp_sleep_injector

from utils import progress_bar
from mergeComp_dl.torch.helper import wrap_compress_optimizer
from straggler_injector import StragglerInjector
from comp_sleep_injector import CompSleepInjector
from config import CupcakeConfig


# ── Training ──────────────────────────────────────────────────────────────────

def train(epoch, model, optimizer, criterion, train_loader, grc, comm_injector, comp_injector):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct    = 0
    total      = 0
    step_times = []
    last_x_t   = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # non-compressed path: advance Markov chain once per batch
        if comm_injector is not None and grc is None:
            comm_injector.inject()

        torch.cuda.synchronize()
        t_step = time.perf_counter()

        optimizer.zero_grad()

        # forward-pass timer — comp_injector sleep inside so it scales off x_t
        t_fwd = time.perf_counter()
        if comp_injector is not None:
            comp_injector.maybe_sleep(batch_idx, last_x_t)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        torch.cuda.synchronize()
        x_t = (time.perf_counter() - t_fwd) * 1000.0  # ms

        # backward + allreduce (comm_injector fires per allreduce via grc.injector)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t_step) * 1000.0)
        last_x_t = x_t

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    p50, p95, p99 = np.percentile(step_times, [50, 95, 99])
    return train_loss / len(train_loader), 100.0 * correct / total, p50, p95, p99


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if dist.get_rank() == 0:
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    correct_t = torch.tensor(correct, dtype=torch.float32, device='cuda')
    total_t   = torch.tensor(total,   dtype=torch.float32, device='cuda')
    dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
    return test_loss / len(test_loader), 100.0 * correct_t.item() / total_t.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config: CupcakeConfig) -> None:
    rank       = int(os.environ.get('RANK',       0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group('nccl')

    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    torch.set_num_threads(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    print('==> Building model..')
    if config.model == 'resnet18':
        model = ResNet18()
    elif config.model == 'resnet50':
        model = ResNet50()
    elif config.model == 'resnet101':
        model = ResNet101()
    elif config.model == 'resnet152':
        model = ResNet152()
    else:
        sys.exit('Unknown model: ' + config.model)

    model.cuda()

    # Broadcast parameters from rank 0 to ensure all workers start identically
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    best_acc    = 0
    start_epoch = 0

    # ── Data ──────────────────────────────────────────────────────────────────
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_root = os.path.join(config.data_root, 'cifar10-data-%d' % rank)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=1, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.test_batch_size, sampler=test_sampler,
        num_workers=1, pin_memory=True)

    # ── Checkpoint resume ──────────────────────────────────────────────────────
    if config.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        ckpt        = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(ckpt['model'])
        best_acc    = ckpt['acc']
        start_epoch = ckpt['epoch']

    # ── Optimizer ─────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    lr_scaler = world_size
    if config.adam:
        optimizer = optim.Adam(model.parameters(), lr=config.lr * world_size)
    else:
        optimizer = optim.SGD(model.parameters(), momentum=config.momentum,
                              lr=config.lr * lr_scaler, weight_decay=config.weight_decay)

    # config passed directly — wrap_compress_optimizer uses duck typing on attrs
    optimizer, grc = wrap_compress_optimizer(model, optimizer, config)

    # ── Straggler injectors ────────────────────────────────────────────────────
    comm_injector = None
    comp_injector = None

    is_comm_target = (config.sleep_target_rank == -1 or rank == config.sleep_target_rank)
    if (config.inject_tc or config.inject_sleep) and is_comm_target:
        comm_injector = StragglerInjector(
            prob_on         = config.sleep_prob_on,
            prob_off        = config.sleep_prob_off,
            check_interval  = config.sleep_check_interval,
            start_after     = config.sleep_start_after,
            duration_ratio  = config.sleep_duration_ratio,
            seed            = config.sleep_seed,
            inject_tc       = config.inject_tc,
            iface           = config.tc_interface,
            throttle_dst_ip = config.throttle_dst_ip,
            bandwidth_max   = config.tc_bandwidth_max,
            bandwidth_ratios= config.tc_bandwidth_ratios,
        )
        mode = 'tc' if config.inject_tc else 'sleep'
        print(f'[straggler] {mode} injection ENABLED on rank {rank} '
              f'prob_on={config.sleep_prob_on} prob_off={config.sleep_prob_off} '
              f'iface={config.tc_interface} bw_max={config.tc_bandwidth_max}')

    if grc is not None and comm_injector is not None:
        grc.injector = comm_injector

    is_comp_target = (config.comp_sleep_target_rank == -1
                      or rank == config.comp_sleep_target_rank)
    if config.comp_inject_sleep and is_comp_target:
        comp_injector = CompSleepInjector(
            prob_on        = config.comp_sleep_prob_on,
            prob_off       = config.comp_sleep_prob_off,
            check_interval = config.comp_sleep_check_interval,
            duration_ratio = config.comp_sleep_duration_ratio,
            seed           = config.comp_sleep_seed + rank,
        )
        print(f'[comp-straggler] sleep injection ENABLED on rank {rank} '
              f'prob_on={config.comp_sleep_prob_on} prob_off={config.comp_sleep_prob_off}')

    # ── Main loop ──────────────────────────────────────────────────────────────
    os.makedirs(config.results_dir, exist_ok=True)
    metrics            = []
    cumulative_train_s = 0.0

    for epoch in range(start_epoch, start_epoch + config.epochs):
        train_sampler.set_epoch(epoch)

        t0 = time.perf_counter()
        train_loss, train_acc, p50, p95, p99 = train(
            epoch, model, optimizer, criterion, train_loader,
            grc, comm_injector, comp_injector)
        cumulative_train_s += time.perf_counter() - t0

        if not config.speed_test:
            test_loss, test_acc = evaluate(model, criterion, test_loader)

            metrics.append({
                'epoch':              epoch,
                'cumulative_train_s': round(cumulative_train_s, 3),
                'train_loss':         round(train_loss, 6),
                'train_acc':          round(train_acc,  4),
                'test_loss':          round(test_loss,  6),
                'test_acc':           round(test_acc,   4),
                'p50_ms':             round(p50, 2),
                'p95_ms':             round(p95, 2),
                'p99_ms':             round(p99, 2),
            })

            if rank == 0:
                with open(os.path.join(config.results_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)

                if test_acc > best_acc:
                    print('Saving..')
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save({'model': model.state_dict(),
                                'acc':   test_acc,
                                'epoch': epoch},
                               './checkpoint/ckpt.pth')
                    best_acc = test_acc

            if config.compress:
                grc.memory.update_lr(optimizer.param_groups[0]['lr'])

    if world_size > 1:
        dist.destroy_process_group()


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 Training (Cupcake + DDP)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model',           type=str)
    parser.add_argument('--batch_size',      type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--epochs',          type=int)
    parser.add_argument('--lr',              type=float)
    parser.add_argument('--momentum',        type=float)
    parser.add_argument('--weight_decay',    type=float)
    parser.add_argument('--data_root',       type=str)
    parser.add_argument('--results_dir',     type=str)

    parser.add_argument('--compress',        type=lambda x: x.lower() == 'true')
    parser.add_argument('--compressor',      type=str)
    parser.add_argument('--memory',          type=str)
    parser.add_argument('--comm',            type=str)
    parser.add_argument('--fusion_num',      type=int)
    parser.add_argument('--compress_ratio',  type=float)

    parser.add_argument('--inject_tc',           action='store_true', default=None)
    parser.add_argument('--sleep_target_rank',   type=int)
    parser.add_argument('--sleep_prob_on',       type=float)
    parser.add_argument('--sleep_prob_off',      type=float)
    parser.add_argument('--tc_interface',        type=str)
    parser.add_argument('--tc_bandwidth_max',    type=str)

    parser.add_argument('--comp_inject_sleep',   action='store_true', default=None)
    parser.add_argument('--comp_sleep_prob_on',  type=float)
    parser.add_argument('--comp_sleep_prob_off', type=float)

    parser.add_argument('--speed_test', action='store_true', default=None)
    parser.add_argument('--resume',     action='store_true', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    config = CupcakeConfig()
    for key, val in vars(args).items():
        if val is not None and hasattr(config, key):
            setattr(config, key, val)
    main(config)
