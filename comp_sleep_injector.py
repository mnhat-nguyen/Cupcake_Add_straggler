"""
comp_sleep_injector.py
----------------------
Simulates computation stragglers by injecting sleep inside the forward-pass
timer in train_step(). The ComputationDetector sees the inflated x_t and
activates AMP to compensate.

Two classes:
  CompSleepInjector       -- live Markov-chain scheduling
  CompReplaySleepInjector -- replays a recorded pattern for fair comparison
"""
import time
import random
from collections import deque


class CompSleepInjector:
    """
    Markov-chain sleep injector for computation straggler simulation.

    Every `check_interval` batches, re-rolls the sleeping state:
      Normal → Sleeping  with probability prob_on
      Sleeping → Normal  with probability prob_off

    While sleeping: sleeps for `duration_ratio` × recent average x_t.
    Skips batch 0 (cold-start outlier; last_iter_ms == 0.0 on that call).
    """

    def __init__(
        self,
        prob_on:         float = 0.20,
        prob_off:        float = 0.20,
        check_interval:  int   = 10,
        duration_ratio:  float = 0.80,
        seed:            int   = 43,
    ) -> None:
        self.prob_on   = prob_on
        self.prob_off  = prob_off
        self.interval  = check_interval
        self.ratio     = duration_ratio
        self.rng       = random.Random(seed)

        self.sleeping        = False
        self._recent: deque  = deque(maxlen=20)
        self.last_sleep_ms   = 0.0

    def maybe_sleep(self, batch_idx: int, last_iter_ms: float) -> bool:
        """
        Call this inside the forward-pass timer, before the forward pass.

        Parameters
        ----------
        batch_idx    : current batch index within the epoch
        last_iter_ms : x_t from the previous batch (ms); 0.0 on cold-start

        Returns True if a sleep was injected.
        """
        self.last_sleep_ms = 0.0

        if last_iter_ms <= 0:
            return False

        self._recent.append(last_iter_ms)

        if batch_idx % self.interval == 0:
            if self.sleeping:
                if self.rng.random() < self.prob_off:
                    self.sleeping = False
            else:
                if self.rng.random() < self.prob_on:
                    self.sleeping = True

        if self.sleeping and self._recent:
            avg_ms    = sum(self._recent) / len(self._recent)
            sleep_sec = (avg_ms * self.ratio) / 1000.0
            time.sleep(sleep_sec)
            self.last_sleep_ms = sleep_sec * 1000.0
            return True

        return False

    @property
    def is_sleeping(self) -> bool:
        return self.sleeping


class CompReplaySleepInjector:
    """
    Replays a recorded CompSleepInjector pattern for fair comparison runs.
    Pattern dict: {"epoch:batch": sleep_ms}. Call set_epoch(e) each epoch.
    """

    def __init__(self, pattern: dict) -> None:
        self.pattern       = pattern
        self.epoch         = 0
        self.sleeping      = False
        self.last_sleep_ms = 0.0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def maybe_sleep(self, batch_idx: int, last_iter_ms: float) -> bool:
        self.last_sleep_ms = 0.0
        self.sleeping      = False

        if last_iter_ms <= 0:
            return False

        ms = self.pattern.get(f"{self.epoch}:{batch_idx}", 0.0)
        if ms > 0:
            time.sleep(ms / 1000.0)
            self.sleeping      = True
            self.last_sleep_ms = ms
            return True

        return False

    @property
    def is_sleeping(self) -> bool:
        return self.sleeping
