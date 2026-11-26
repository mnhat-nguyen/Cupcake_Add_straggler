"""
straggler_injector.py

Simulates a straggler (slow) worker on one rank's training loop. Two modes:

  inject_sleep=True  : injects time.sleep() before each send (legacy).
  inject_tc=True     : squeezes bandwidth on the egress NIC via Linux tc HTB.
                       Requires CAP_NET_ADMIN (--cap-add NET_ADMIN in Docker).

Markov-chain scheduling:
  Every `check_interval` steps, re-roll whether to flip state:
    Normal -> Slow  with probability prob_on
    Slow   -> Normal with probability prob_off

  TC mode: on each transition to Slow, a bandwidth preset is chosen
  independently at random from tc_bandwidth_ratios[1:], so successive
  slow periods can differ in severity (e.g. 570mbit, then 285mbit).
  TC class change is issued only on state transitions, not every step.

  Sleep mode: while Slow, injects time.sleep() before each send, scaled
  to recent average step duration * duration_ratio.

  Teardown (tc qdisc del) is registered with atexit and SIGTERM.
"""

import atexit
import random
import signal
import subprocess
import sys
import time
from collections import deque


# ---------------------------------------------------------------------------
# bandwidth helpers
# ---------------------------------------------------------------------------

def _parse_bw(s):
    """Parse a tc rate string to bits per second. '1gbit' -> 1e9."""
    s = s.lower().strip()
    for suffix, mult in (('gbit', 1e9), ('mbit', 1e6), ('kbit', 1e3)):
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


def _format_bw(bps):
    """Format bits per second back to a tc rate string."""
    if bps >= 1e9:
        v = bps / 1e9
        return f"{v:g}gbit"
    if bps >= 1e6:
        v = bps / 1e6
        return f"{v:g}mbit"
    v = bps / 1e3
    return f"{v:g}kbit"


def _tc(*args):
    r = subprocess.run(["tc", *args], capture_output=True)
    if r.returncode != 0:
        print(f"[tc] WARN: tc {' '.join(args)} failed: {r.stderr.decode().strip()}", file=sys.stderr)


# ---------------------------------------------------------------------------
# injector
# ---------------------------------------------------------------------------

class StragglerInjector:
    """Per-rank Markov-chain scheduler that toggles between slow and normal.

    See module docstring for mode and scheduling details.
    """

    def __init__(
        self,
        prob_on=0.30,
        prob_off=0.30,
        check_interval=10,
        start_after=15,
        duration_ratio=0.80,
        history_len=20,
        seed=42,
        # TC mode
        inject_tc=False,
        iface='enp8s0',
        throttle_dst_ip='',
        bandwidth_max='1gbit',
        bandwidth_ratios=(1.0, 0.5, 0.2, 0.1),
    ):
        self.prob_on        = prob_on
        self.prob_off       = prob_off
        self.check_interval = check_interval
        self.start_after    = start_after
        self.duration_ratio = duration_ratio
        self._rng           = random.Random(seed)

        self._is_slow       = False
        self._step_count    = 0
        self.compress_info_fn = None   # optional () -> str, appended to [tc] prints

        self._recent_durations      = deque(maxlen=history_len)
        self._recent_comm_durations = deque(maxlen=history_len)
        self._cumulative_injected_s = 0.0

        self.inject_tc = inject_tc
        if inject_tc:
            self._iface           = iface
            self._throttle_dst_ip = throttle_dst_ip
            bps_max = _parse_bw(bandwidth_max)
            self._rate_strings = tuple(
                _format_bw(bps_max * r) for r in bandwidth_ratios
            )
            self._fast_rate = self._rate_strings[0]
            self._slow_rate = self._rate_strings[-1]   # placeholder until first toggle
            self._tc_setup()
            atexit.register(self._tc_teardown)
            signal.signal(signal.SIGTERM, lambda *_: (self._tc_teardown(), exit(0)))

    # -- Markov-chain toggle -------------------------------------------------

    def _maybe_toggle(self):
        self._step_count += 1
        if self._step_count <= self.start_after:
            return
        if self._step_count % self.check_interval != 0:
            return
        if self._is_slow:
            if self._rng.random() < self.prob_off:
                self._is_slow = False
                if self.inject_tc:
                    info = self.compress_info_fn() if self.compress_info_fn is not None else ""
                    print(f"[tc] normal: prob_off={self.prob_off}{info}", file=sys.stderr)
                    self._tc_set_fast()
        else:
            if self._rng.random() < self.prob_on:
                self._is_slow = True
                if self.inject_tc:
                    idx = self._rng.randint(1, len(self._rate_strings) - 1)
                    self._slow_rate = self._rate_strings[idx]
                    info = self.compress_info_fn() if self.compress_info_fn is not None else ""
                    print(f"[tc] slow: rate={self._slow_rate} prob_on={self.prob_on}{info}", file=sys.stderr)
                    self._tc_set_slow()

    # -- tc lifecycle --------------------------------------------------------

    def _tc_setup(self):
        iface = self._iface
        subprocess.run(["tc", "qdisc", "del", "dev", iface, "root"], capture_output=True)
        _tc("qdisc", "add", "dev", iface, "root", "handle", "1:", "htb", "default", "10")
        _tc("class", "add", "dev", iface, "parent", "1:", "classid", "1:10",
            "htb", "rate", self._fast_rate)
        _tc("class", "add", "dev", iface, "parent", "1:", "classid", "1:20",
            "htb", "rate", self._fast_rate)
        if self._throttle_dst_ip:
            _tc("filter", "add", "dev", iface, "protocol", "ip", "parent", "1:0",
                "prio", "1", "u32", "match", "ip", "dst",
                f"{self._throttle_dst_ip}/32", "flowid", "1:20")
        print(
            f"[tc] setup: iface={iface} fast={self._fast_rate}"
            f" dst_filter={self._throttle_dst_ip or '(none)'}",
            file=sys.stderr,
        )

    def _tc_teardown(self):
        _tc("qdisc", "del", "dev", self._iface, "root")

    def _tc_set_slow(self):
        _tc("class", "change", "dev", self._iface, "parent", "1:",
            "classid", "1:20", "htb", "rate", self._slow_rate)

    def _tc_set_fast(self):
        _tc("class", "change", "dev", self._iface, "parent", "1:",
            "classid", "1:20", "htb", "rate", self._fast_rate)

    # -- public interface ----------------------------------------------------

    @property
    def is_slow(self):
        return self._is_slow

    def record_step_duration(self, duration_s):
        """Feed in the actual step duration so sleep-mode delays scale with it."""
        self._recent_durations.append(duration_s)

    def record_comm_duration(self, duration_s):
        """Feed in a clean send duration (only when not slow) for sleep baseline."""
        self._recent_comm_durations.append(duration_s)

    def inject(self):
        """Advance the Markov scheduler and apply the simulated delay.

        TC mode  : state transitions trigger tc class change; returns 0.0.
        Sleep mode: sleeps when slow; returns seconds slept (0.0 if fast).
        """
        self._maybe_toggle()

        if self.inject_tc:
            return 0.0

        if not self._is_slow:
            return 0.0
        if self._recent_comm_durations:
            base = sum(self._recent_comm_durations) / len(self._recent_comm_durations)
        elif self._recent_durations:
            base = sum(self._recent_durations) / len(self._recent_durations)
        else:
            return 0.0
        delay = base * self.duration_ratio
        time.sleep(delay)
        self._cumulative_injected_s += delay
        return delay

    def pop_cumulative_injected(self):
        """Return total sleep injected since last call, then reset to 0."""
        total = self._cumulative_injected_s
        self._cumulative_injected_s = 0.0
        return total
