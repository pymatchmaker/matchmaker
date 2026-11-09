#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Dannenberg 1985 dynamic-programming polyphonic score follower.

Re-implementation of the dynamic grouping matcher from:

    Bloch, J. & Dannenberg, R.B. (1985). Real-Time Computer Accompaniment
    of Keyboard Performances. Proc. ICMC, 279-289.

The matcher processes MIDI note-ons individually through a DP matrix
whose rows are score compound events (cevts = pitch sets at each onset)
and whose columns are individual performance notes. The algorithm finds
the association that maximises a rating function penalising wrong and
missing notes (Sec. 3, 6). Epsilon and epsilon-fraction tests (Sec. 5)
group incoming notes into performance cevts for confidence checking.

Classes:
  DPDannenberg -- windowed DP matcher (event-level, MIDI only)
"""

from typing import Any, List, Optional, Set

import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.io.queue import RECVQueue


class DPDannenberg(OnlineAlignment):
    """Windowed DP matcher (Bloch & Dannenberg 1985, Sec. 6).

    Rows of the DP matrix are score compound events (one pitch set per
    unique onset beat); columns are individual performance MIDI notes.
    Each incoming note updates one column by considering three
    predecessors -- same row (extend), diagonal (advance and match), and
    vertical (skip note) -- choosing the one that maximises the rating.

    Parameters
    ----------
    reference_features : list of set of int
        MIDI pitch set for each score onset (one per ``score_positions``
        entry).
    score_positions : np.ndarray
        Beat position of each score onset.
    queue : optional
        Input queue for streaming.
    c_m : float
        Cost per missing score note when advancing past a score cevt
        (Sec. 8, default 0.5).
    c_e : float
        Cost per extra performance note not in the current score cevt
        (Sec. 8, default 0.0).
    epsilon : float
        Absolute time threshold (seconds) for grouping notes into the
        same performance cevt (Sec. 5, default 0.090 s).
    epsilon_fraction : float
        Fraction of predicted next-cevt time for the relative grouping
        test (Sec. 5, default 0.25).
    window : int
        Number of score cevts ahead of the current best match to
        consider (default 50).
    tempo : float
        Initial tempo in BPM, used to seed the epsilon-fraction test
        and position interpolation (default 120.0).
    smoothing : float
        Exponential smoothing coefficient for tempo tracking
        (default 0.5).
    """

    score_cevts: List[Set[int]]
    n_states: int
    c_m: float
    c_e: float
    epsilon: float
    epsilon_fraction: float
    window: int
    smoothing: float

    _values: NDArray[np.float64]
    _used: List[Set[int]]
    _prev_values: NDArray[np.float64]
    _prev_used: List[Set[int]]

    _best_value: float
    _best_row: int

    _prev_note_time: Optional[float]

    _beats_per_second: float
    _anchor_beat: float
    _anchor_time: float
    _prev_match_time: Optional[float]
    _prev_match_beat: Optional[float]

    def __init__(
        self,
        reference_features: List[Set[int]],
        score_positions: NDArray[np.floating[Any]],
        queue: Optional[RECVQueue] = None,
        c_m: float = 0.5,
        c_e: float = 0.0,
        epsilon: float = 0.090,
        epsilon_fraction: float = 0.25,
        window: int = 50,
        tempo: float = 120.0,
        smoothing: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.score_cevts = reference_features
        self.n_states = len(reference_features)
        self.c_m = c_m
        self.c_e = c_e
        self.epsilon = epsilon
        self.epsilon_fraction = epsilon_fraction
        self.window = window
        self.smoothing = smoothing

        self._values = np.zeros(self.n_states, dtype=np.float64)
        self._used = [set() for _ in range(self.n_states)]
        self._prev_values = np.zeros(self.n_states, dtype=np.float64)
        self._prev_used = [set() for _ in range(self.n_states)]

        self._best_value = float("-inf")
        self._best_row = 0

        self._prev_note_time = None

        self._beats_per_second = tempo / 60.0
        self._anchor_beat = float(score_positions[0])
        self._anchor_time = 0.0
        self._prev_match_time = None
        self._prev_match_beat = None

    def _is_new_cevt(self, note_time: float) -> bool:
        """Epsilon and epsilon-fraction tests (Sec. 5, p. 283-284)."""
        if self._prev_note_time is None:
            return True
        dt = note_time - self._prev_note_time
        if dt > self.epsilon:
            return True
        if (
            self._beats_per_second > 0
            and self._best_row + 1 < self.n_states
        ):
            beat_gap = float(
                self.score_positions[self._best_row + 1]
                - self.score_positions[self._best_row]
            )
            predicted_dt = beat_gap / self._beats_per_second
            if predicted_dt > 0 and dt > self.epsilon_fraction * predicted_dt:
                return True
        return False

    def step(self, features: Any) -> None:
        pitch = int(np.atleast_1d(features)[0])
        note_time = self.current_perf_time

        if self._is_new_cevt(note_time):
            pass
        self._prev_note_time = note_time

        lo = max(0, self._best_row - 10)
        hi = min(self.n_states, self._best_row + self.window)

        # Shift columns: save current as previous (Sec. 6, p. 285).
        for r in range(lo, hi):
            self._prev_values[r] = self._values[r]
        for r in range(lo, hi):
            self._prev_used[r] = self._used[r].copy()

        # DP update: one column per performance note (Fig. 6-2, p. 285).
        for r in range(lo, hi):
            B = self.score_cevts[r]

            y_val = float(self._prev_values[r])
            y_used = self._prev_used[r]

            z_val = y_val
            z_used = y_used.copy()

            if pitch not in B:
                z_val -= self.c_e

            if pitch in B:
                if pitch not in y_used:
                    z_val = y_val + 1
                    z_used = y_used | {pitch}

                if r > lo:
                    A = self.score_cevts[r - 1]
                    w_val = float(self._prev_values[r - 1])
                    w_used = self._prev_used[r - 1]
                    missing_in_A = len(A - w_used)
                    d = w_val + 1 - self.c_m * missing_in_A
                    if d > z_val:
                        z_val = d
                        z_used = {pitch}

            if r > lo:
                A_skip = self.score_cevts[r - 1]
                x_val = float(self._values[r - 1])
                x_used = self._used[r - 1]
                missing_in_A_skip = len(A_skip - x_used)
                v = x_val - self.c_m * missing_in_A_skip
                if v >= z_val:
                    z_val = v
                    z_used = set()

            self._values[r] = z_val
            self._used[r] = z_used

        best_r = lo + int(np.argmax(self._values[lo:hi]))
        best_v = float(self._values[best_r])

        if best_v > self._best_value:
            self._best_value = best_v
            self._best_row = best_r
            self.current_index = best_r

            beat = float(self.score_positions[best_r])
            if self._prev_match_time is not None:
                elapsed = note_time - self._prev_match_time
                advanced = beat - self._prev_match_beat
                if elapsed > 0.05 and 0 < advanced < 16:
                    rate = advanced / elapsed
                    self._beats_per_second = (
                        self.smoothing * self._beats_per_second
                        + (1 - self.smoothing) * rate
                    )

            self._anchor_beat = beat
            self._anchor_time = note_time
            self._prev_match_time = note_time
            self._prev_match_beat = beat

    def get_current_position(self) -> float:
        drift = self._beats_per_second * (
            self.current_perf_time - self._anchor_time
        )
        pos = self._anchor_beat + drift
        if self.current_index + 1 < self.n_states:
            cap = float(self.score_positions[self.current_index + 1])
            pos = min(pos, cap)
        return pos

    def is_still_following(self) -> bool:
        return self.current_index < self.n_states - 1
