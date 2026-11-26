"""
config.py -- all hyperparameters in one place.

Every field can be overridden by an environment variable of the same name
(upper-cased):

    EPOCHS=50 BATCH_SIZE=128 MODE=straggler bash run.sh
"""
import os
from dataclasses import dataclass, field


def _env(key: str, default):
    val = os.environ.get(key.upper())
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ('1', 'true', 'yes', 'on')
    try:
        return type(default)(val)
    except (ValueError, TypeError):
        return default


@dataclass
class CupcakeConfig:
    # ── Model / Dataset ───────────────────────────────────────────────────────
    model:           str = field(default_factory=lambda: _env('MODEL',      'resnet101'))
    data_root:       str = field(default_factory=lambda: _env('DATA_ROOT',  './data'))
    results_dir:     str = field(default_factory=lambda: _env('RESULTS_DIR', './results'))

    # ── Training ──────────────────────────────────────────────────────────────
    epochs:          int   = field(default_factory=lambda: _env('EPOCHS',       100))
    batch_size:      int   = field(default_factory=lambda: _env('BATCH_SIZE',   256))
    test_batch_size: int   = field(default_factory=lambda: _env('TEST_BATCH_SIZE', 1000))
    # lr is multiplied by hvd.size() inside the script (linear scaling rule).
    # Default 0.0125 × 4 workers = 0.05 effective — matches Tinker.
    lr:              float = field(default_factory=lambda: _env('LR',           0.0125))
    momentum:        float = field(default_factory=lambda: _env('MOMENTUM',     0.9))
    weight_decay:    float = field(default_factory=lambda: _env('WEIGHT_DECAY', 5e-4))
    adam:            bool  = field(default_factory=lambda: _env('ADAM',         False))

    # ── Gradient compression ──────────────────────────────────────────────────
    compress:                   bool  = field(default_factory=lambda: _env('COMPRESS',            True))
    compressor:                 str   = field(default_factory=lambda: _env('COMPRESSOR',          'efsignsgd'))
    memory:                     str   = field(default_factory=lambda: _env('MEMORY',              'residual'))
    comm:                       str   = field(default_factory=lambda: _env('COMM',                'allreduce'))
    fusion_num:                 int   = field(default_factory=lambda: _env('FUSION_NUM',          2))
    compress_ratio:             float = field(default_factory=lambda: _env('COMPRESS_RATIO',      0.01))
    # ── Communication straggler simulation (StragglerInjector, tc mode) ───────
    inject_tc:            bool  = field(default_factory=lambda: _env('INJECT_TC',            True))
    inject_sleep:         bool  = field(default_factory=lambda: _env('INJECT_SLEEP',         False))
    sleep_target_rank:    int   = field(default_factory=lambda: _env('SLEEP_TARGET_RANK',    0))
    sleep_prob_on:        float = field(default_factory=lambda: _env('SLEEP_PROB_ON',        0.30))
    sleep_prob_off:       float = field(default_factory=lambda: _env('SLEEP_PROB_OFF',       0.20))
    sleep_check_interval: int   = field(default_factory=lambda: _env('SLEEP_CHECK_INTERVAL', 10))
    sleep_start_after:    int   = field(default_factory=lambda: _env('SLEEP_START_AFTER',    15))
    sleep_duration_ratio: float = field(default_factory=lambda: _env('SLEEP_DURATION_RATIO', 0.5))
    sleep_seed:           int   = field(default_factory=lambda: _env('SLEEP_SEED',           42))

    # ── TC bandwidth control ──────────────────────────────────────────────────
    tc_interface:        str   = field(default_factory=lambda: _env('TC_INTERFACE',     'enp8s0'))
    throttle_dst_ip:     str   = field(default_factory=lambda: _env('THROTTLE_DST_IP', ''))
    tc_bandwidth_max:    str   = field(default_factory=lambda: _env('TC_BANDWIDTH_MAX', '1000mbit'))
    tc_bandwidth_ratios: tuple = field(default_factory=lambda: tuple(
        float(x) for x in os.environ.get('TC_BANDWIDTH_RATIOS', '1.0,0.1').split(',')))

    # ── Computation straggler simulation (CompSleepInjector) ──────────────────
    comp_inject_sleep:         bool  = field(default_factory=lambda: _env('COMP_INJECT_SLEEP',         True))
    comp_sleep_target_rank:    int   = field(default_factory=lambda: _env('COMP_SLEEP_TARGET_RANK',    -1))
    comp_sleep_prob_on:        float = field(default_factory=lambda: _env('COMP_SLEEP_PROB_ON',        0.20))
    comp_sleep_prob_off:       float = field(default_factory=lambda: _env('COMP_SLEEP_PROB_OFF',       0.20))
    comp_sleep_check_interval: int   = field(default_factory=lambda: _env('COMP_SLEEP_CHECK_INTERVAL', 10))
    comp_sleep_duration_ratio: float = field(default_factory=lambda: _env('COMP_SLEEP_DURATION_RATIO', 0.4))
    comp_sleep_seed:           int   = field(default_factory=lambda: _env('COMP_SLEEP_SEED',           43))

    # ── Misc ──────────────────────────────────────────────────────────────────
    speed_test: bool = field(default_factory=lambda: _env('SPEED_TEST', False))
    resume:     bool = field(default_factory=lambda: _env('RESUME',     False))
