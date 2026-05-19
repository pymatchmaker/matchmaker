#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Time Warping (OLTWDixon)

OLTW with backward-forward search, based on:
  Dixon (2005) "An On-Line Time Warping Algorithm for Tracking Musical
               Performances" (IJCAI)
  Dixon (2005) "Live Tracking of Musical Performances Using On-Line Time
               Warping" (DAFx)

Classes:
  OnlineTimeWarpingDixon      — base class (common properties, run loop)
  OnlineTimeWarpingDixonFrame — frame-level variant for audio (backward-forward search)
  OnlineTimeWarpingDixonEvent — event-level variant for MIDI (onset-by-onset)
"""

import time
from enum import IntEnum
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats


class Direction(IntEnum):
    REF = 0
    TARGET = 1
    BOTH = 2


MAX_RUN_COUNT: int = 3
FRAME_PER_SEG = 1
WINDOW_SIZE = 10  # seconds for frame, events for event


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class OnlineTimeWarpingDixon(OnlineAlignment):
    """Base class for Dixon-style OLTW.

    Subclasses must implement ``step()``, ``reset()``, ``is_still_following()``.

    Parameters
    ----------
    reference_features : np.ndarray
        Feature matrix for the reference (score) sequence.
    queue : RECVQueue or None
        Input queue for streaming.
    max_run_count : int
        Max consecutive same-direction advances.
    distance_func : str
        Distance metric.
    score_positions : np.ndarray or None
        Score positions (beats).
    """

    DEFAULT_DISTANCE_FUNC: str = "euclidean"

    def __init__(
        self,
        reference_features,
        score_positions,
        queue=None,
        max_run_count=MAX_RUN_COUNT,
        distance_func=DEFAULT_DISTANCE_FUNC,
        **kwargs,
    ):
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.N_ref = self.reference_features.shape[0]
        self.max_run_count = max_run_count
        self.distance_func = (
            distance_func if isinstance(distance_func, str) else "euclidean"
        )
        self.current_index = 0
        self.input_index = 0


# ---------------------------------------------------------------------------
# Frame-level subclass (audio)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingDixonFrame(OnlineTimeWarpingDixon):
    """Frame-level OLTW with backward-forward search for audio input.

    Window size is in seconds (converted to frames via frame_rate).
    """

    def __init__(
        self,
        reference_features,
        score_positions,
        queue=None,
        window_size=WINDOW_SIZE,
        distance_func=OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
        max_run_count=MAX_RUN_COUNT,
        frame_per_seg=FRAME_PER_SEG,
        frame_rate=FRAME_RATE,
        ref_frame_to_beat: NDArray = None,
        **kwargs,
    ):
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Dixon requires `ref_frame_to_beat` (per-frame beat mapping)."
            )
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
            max_run_count=max_run_count,
            distance_func=distance_func,
            **kwargs,
        )
        self.frame_rate = frame_rate
        self._ref_frame_to_beat = ref_frame_to_beat
        self.w = int(window_size * self.frame_rate)
        self.frame_per_seg = frame_per_seg
        self.queue_timeout = QUEUE_TIMEOUT
        self.reset()

    def reset(self):
        self.input_features = []
        self.acc_dist_matrix = np.full((self.w, self.w), np.inf, dtype=np.float32)
        self.wp = np.array([[0, 0]]).T
        self.ref_pointer = 0
        self.input_pointer = 0
        self.run_count_ref = 0
        self.run_count_input = 0
        self.best_ref = 0
        self.best_input = 0
        self.current_index = 0
        self.input_index = 0
        self.current_perf_time: float = 0.0
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self._initialized = False

    def _frame_to_beat(self, frame: int) -> float:
        """Per-frame beat value (frame-precision)."""
        return float(
            self._ref_frame_to_beat[min(frame, len(self._ref_frame_to_beat) - 1)]
        )

    def _frame_to_score_idx(self, frame: int) -> int:
        """Project a reference-frame index to the score_positions index."""
        beat = self._frame_to_beat(frame)
        idx = int(np.searchsorted(self.score_positions, beat, side="right") - 1)
        return max(0, min(idx, len(self.score_positions) - 1))

    def get_current_position(self) -> float:
        # Frame-precision beat via _ref_frame_to_beat lookup.
        return self._frame_to_beat(self.best_ref)

    @property
    def alignment_path(self) -> NDArray:
        return self.wp

    def is_still_following(self):
        return self.ref_pointer <= (self.N_ref - self.frame_per_seg)

    # Frame Dixon's step() can append to wp multiple times via save_history()
    # during catch-up loops, so the base __call__'s one-append-per-call contract
    # doesn't fit
    def __call__(self, observation: Any, perf_time: float) -> float:
        self.current_perf_time = perf_time
        t0 = time.time()
        self.step(observation)
        self.current_position = self.get_current_position()
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return self.current_position

    # ---- internal methods ----

    def _ref_offset(self):
        return max(0, self.ref_pointer - self.w)

    def _input_offset(self):
        return max(0, self.input_pointer - self.w)

    def _ref_win_size(self):
        return min(self.w, self.ref_pointer)

    def _input_win_size(self):
        return min(self.w, self.input_pointer)

    def offset(self):
        return np.array([self._ref_offset(), self._input_offset()])

    def _compute_cell(self, r_win, i_win):
        abs_ref = self._ref_offset() + r_win
        abs_input = self._input_offset() + i_win
        ref_feature = self.reference_features[abs_ref]
        input_feature = self.input_features[abs_input]
        local_dist = scipy.spatial.distance.cdist(
            ref_feature.reshape(1, -1),
            input_feature.reshape(1, -1),
            metric=self.distance_func,
        )[0, 0]

        if r_win == 0 and i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = local_dist
        elif r_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win, i_win - 1] + local_dist
            )
        elif i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win - 1, i_win] + local_dist
            )
        else:
            self.acc_dist_matrix[r_win, i_win] = min(
                self.acc_dist_matrix[r_win - 1, i_win] + local_dist,
                self.acc_dist_matrix[r_win, i_win - 1] + local_dist,
                self.acc_dist_matrix[r_win - 1, i_win - 1] + 2 * local_dist,
            )

    def _accept_input(self, input_features):
        self.input_features.append(input_features)
        self.input_pointer += self.frame_per_seg

    def _compute_input_column(self):
        if self.input_pointer > self.w:
            self.acc_dist_matrix[:, :-1] = self.acc_dist_matrix[:, 1:]
            self.acc_dist_matrix[:, -1] = np.inf
        input_col = self._input_win_size() - 1
        for r in range(self._ref_win_size()):
            self._compute_cell(r, input_col)

    def _advance_ref(self):
        self.ref_pointer += self.frame_per_seg
        if self.ref_pointer > self.w:
            self.acc_dist_matrix[:-1, :] = self.acc_dist_matrix[1:, :]
            self.acc_dist_matrix[-1, :] = np.inf
        ref_row = self._ref_win_size() - 1
        for c in range(self._input_win_size()):
            self._compute_cell(ref_row, c)

    def _determine_direction(self):
        suggested = self._get_expand_direction()
        if self.run_count_ref >= self.max_run_count:
            return Direction.TARGET
        elif self.run_count_input >= self.max_run_count:
            return Direction.REF
        return suggested

    def _get_expand_direction(self):
        ref_ws = self._ref_win_size()
        input_ws = self._input_win_size()
        ref_off = self._ref_offset()
        input_off = self._input_offset()
        row = ref_ws - 1
        col = input_ws - 1
        best_cost = np.inf
        best_row, best_col = row, col

        for r in range(ref_ws):
            cost = self.acc_dist_matrix[r, col] / (1 + ref_off + r + input_off + col)
            if cost < best_cost:
                best_cost = cost
                best_row, best_col = r, col

        for c in range(input_ws):
            cost = self.acc_dist_matrix[row, c] / (1 + ref_off + row + input_off + c)
            if cost < best_cost:
                best_cost = cost
                best_row, best_col = row, c

        self.best_ref = ref_off + best_row
        self.best_input = input_off + best_col

        if best_row == row and best_col == col:
            return Direction.BOTH
        elif best_row < row:
            return Direction.TARGET
        return Direction.REF

    def save_history(self):
        # best_input is the input-buffer index of the best match; convert to seconds
        perf_time = self.best_input / float(self.frame_rate)
        beat = self.get_current_position()
        new_point = np.array([[beat], [perf_time]])
        self.wp = np.concatenate((self.wp, new_point), axis=1)

    def step(self, input_features):
        if not self._initialized:
            self._accept_input(input_features)
            self.ref_pointer += self.frame_per_seg
            self._compute_cell(0, 0)
            self._initialized = True
            self.input_index += 1
            return

        direction = None
        while self.is_still_following():
            direction = self._determine_direction()
            if direction != Direction.REF:
                break
            self._advance_ref()
            self.run_count_ref += 1
            self.run_count_input = 0
            self.current_index = self._frame_to_score_idx(self.best_ref)
            self.save_history()

        if not self.is_still_following():
            return

        self._accept_input(input_features)
        self._compute_input_column()

        if direction is not Direction.TARGET:
            self._advance_ref()

        if direction == Direction.TARGET:
            self.run_count_input += 1
            self.run_count_ref = 0
        else:
            self.run_count_ref = 1
            self.run_count_input = 0

        self.current_index = self._frame_to_score_idx(self.best_ref)
        self.save_history()
        self.input_index += 1

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))


# ---------------------------------------------------------------------------
# Event-level subclass (MIDI)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingDixonEvent(OnlineTimeWarpingDixon):
    """Event-level OLTW for MIDI input.

    Each step processes one onset event. Uses sparse cost matrix with
    path-length normalization and reference catch-up based on argmin
    of normalized path cost.

    Parameters
    ----------
    c : int
        Search width in number of events.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        queue: Optional[RECVQueue] = None,
        c: int = 30,
        max_run_count: int = MAX_RUN_COUNT,
        distance_func: str = "euclidean",
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
            max_run_count=max_run_count,
            distance_func=distance_func,
            **kwargs,
        )
        self.ref = self.reference_features.astype(np.float32)
        self.c = c
        self.reset()

    def reset(self) -> None:
        self._D: Dict = {}
        self._L: Dict = {}
        self._inputs = []
        self.t = -1
        self.j = -1
        self.run_count_input = 0
        self.run_count_ref = 0
        self.current_index = 0
        self.input_index = 0
        self._alignment_path = []
        self.last_queue_update = time.time()
        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

    # ---- cost matrix (sparse) ----

    def _local_cost(self, ref_idx, input_idx):
        return scipy.spatial.distance.cdist(
            self.ref[ref_idx].reshape(1, -1),
            self._inputs[input_idx].reshape(1, -1),
            metric=self.distance_func,
        )[0, 0]

    def _get_D(self, i, j):
        return self._D.get((i, j), np.inf)

    def _get_L(self, i, j):
        return self._L.get((i, j), 0)

    def _evaluate(self, ref_i, inp_j):
        if ref_i < 0 or inp_j < 0 or ref_i >= self.N_ref:
            return
        d = self._local_cost(ref_i, inp_j)
        if ref_i == 0 and inp_j == 0:
            self._D[(ref_i, inp_j)] = d
            self._L[(ref_i, inp_j)] = 1
            return
        best_cost, best_len = np.inf, 0
        for prev_i, prev_j, w in [
            (ref_i - 1, inp_j, 1),
            (ref_i, inp_j - 1, 1),
            (ref_i - 1, inp_j - 1, 2),
        ]:
            prev = self._D.get((prev_i, prev_j), np.inf)
            if prev < np.inf:
                c = prev + w * d
                if c < best_cost:
                    best_cost = c
                    best_len = self._L.get((prev_i, prev_j), 0) + 1
        if best_cost < np.inf:
            self._D[(ref_i, inp_j)] = best_cost
            self._L[(ref_i, inp_j)] = best_len

    def _norm_cost(self, ref_i, inp_j):
        L = self._get_L(ref_i, inp_j)
        return self._get_D(ref_i, inp_j) / L if L > 0 else np.inf

    def _argmin_path_cost(self):
        best_cost, best_ref, best_inp = np.inf, self.j, self.t
        for k in range(max(0, self.t - self.c + 1), self.t + 1):
            nc = self._norm_cost(self.j, k)
            if nc < best_cost:
                best_cost, best_ref, best_inp = nc, self.j, k
        for k in range(max(0, self.j - self.c + 1), self.j + 1):
            nc = self._norm_cost(k, self.t)
            if nc < best_cost:
                best_cost, best_ref, best_inp = nc, k, self.t
        return best_ref, best_inp

    def _compute_input_column(self):
        for k in range(max(0, self.j - self.c + 1), self.j + 1):
            self._evaluate(k, self.t)

    def _compute_ref_row(self):
        for k in range(max(0, self.t - self.c + 1), self.t + 1):
            self._evaluate(self.j, k)

    def step(self, input_feat: NDArray[np.float32]) -> None:
        self._inputs.append(np.asarray(input_feat, dtype=np.float32))
        self.t += 1
        self._compute_input_column()

        if self.t < self.c and self.j + 1 < self.N_ref:
            self.j += 1
            self._compute_ref_row()
            self.run_count_input = 0
            self.run_count_ref = 0
        else:
            self.run_count_input += 1
            self.run_count_ref = 0
            while self.j + 1 < self.N_ref:
                best_ref, best_inp = self._argmin_path_cost()
                if best_ref >= self.j and best_inp < self.t:
                    self.j += 1
                    self._compute_ref_row()
                    self.run_count_ref += 1
                    self.run_count_input = 0
                    if self.run_count_ref >= self.max_run_count:
                        break
                else:
                    break

        best_ref, _ = self._argmin_path_cost()
        self.current_index = min(best_ref, self.N_ref - 1)
        self.input_index += 1

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))


if __name__ == "__main__":
    pass  # pragma: no cover
