#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Time Warping (OLTWDixon)

Incremental alignment with a bounded search band, based on:
  Dixon (2005) "An On-Line Time Warping Algorithm for Tracking Musical
               Performances" (IJCAI)
  Dixon (2005) "Live Tracking of Musical Performances Using On-Line Time
               Warping" (DAFx)

Classes:
  OnlineTimeWarpingDixon      — base class (common properties)
  OnlineTimeWarpingDixonFrame — frame-level variant for audio
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
from matchmaker.io.stream import STREAM_END
from matchmaker.utils.misc import set_latency_stats


class Direction(IntEnum):
    REF = 0
    TARGET = 1
    BOTH = 2


MAX_RUN_COUNT: int = 3
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
        self.distance_func = distance_func
        self.current_index = 0
        self.input_index = 0


# ---------------------------------------------------------------------------
# Frame-level subclass (audio)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingDixonFrame(OnlineTimeWarpingDixon):
    """Frame-level OLTW for audio, following the pseudocode of Dixon (2005).

    Mapping to the paper:
      ``evaluate_path_cost``  <->  EvaluatePathCost(t, j)
      ``get_inc``             <->  GetInc(t, j)
      ``run`` / ``step``       <->  the main loop (advance input, reference, or both)
      ``self.w``               <->  search width c (seconds * frame_rate)
    """

    # Predecessor offsets and step weights of the DTW recursion, keyed by the
    # direction that reaches cell (i, j). Iteration order defines tie-breaking.
    STEP_WEIGHTS = {
        Direction.BOTH: ((-1, -1), 2.0),
        Direction.REF: ((-1, 0), 1.0),
        Direction.TARGET: ((0, -1), 1.0),
    }

    def __init__(
        self,
        reference_features,
        score_positions,
        queue=None,
        window_size=WINDOW_SIZE,
        distance_func=OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
        max_run_count=MAX_RUN_COUNT,
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
        self.queue_timeout = QUEUE_TIMEOUT
        self.reset()

    def reset(self):
        self.input_features = []
        self.accumulated_costs = {}  # (ref_index, input_index) -> accumulated path cost
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
        self._pending = []  # input frames received but not yet consumed

    def _frame_to_beat(self, frame: int) -> float:
        return float(
            self._ref_frame_to_beat[min(frame, len(self._ref_frame_to_beat) - 1)]
        )

    def _frame_to_score_idx(self, frame: int) -> int:
        beat = self._frame_to_beat(frame)
        idx = int(np.searchsorted(self.score_positions, beat, side="right") - 1)
        return max(0, min(idx, len(self.score_positions) - 1))

    def get_current_position(self) -> float:
        return self._frame_to_beat(self.best_ref)

    @property
    def alignment_path(self) -> NDArray:
        return self.wp

    def save_history(self):
        perf_time = self.best_input / float(self.frame_rate)
        beat = self.get_current_position()
        new_point = np.array([[perf_time], [beat]])
        self.wp = np.concatenate((self.wp, new_point), axis=1)

    def is_still_following(self):
        return self.ref_pointer < self.N_ref

    def __call__(self, observation: Any, perf_time: float) -> float:
        self.current_perf_time = perf_time
        t0 = time.time()
        self.step(observation)
        self.current_position = self.get_current_position()
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return self.current_position

    def _distances(self, A, b):
        """Distances from each row of band A (k, dim) to single frame b (dim,)."""
        return scipy.spatial.distance.cdist(
            A, b.reshape(1, -1), metric=self.distance_func
        )[:, 0]

    def evaluate_path_cost(self, i, j, local_dist):
        """EvaluatePathCost(t, j): DTW recursion at (ref=i, input=j).

        Diagonal (BOTH) steps are weighted twice the local distance, straight
        steps once, following Sakoe & Chiba (1978) as adopted by Dixon (2005,
        DAFx): every monotone path then covers equal extent at equal weight,
        which makes the length normalization in ``path_cost`` meaningful.
        """
        if i == 0 and j == 0:
            self.accumulated_costs[(0, 0)] = local_dist
            return
        best = None
        for offset, weight in self.STEP_WEIGHTS.values():
            predecessor = self.accumulated_costs.get((i + offset[0], j + offset[1]))
            if predecessor is None:
                continue
            cost = predecessor + weight * local_dist
            if best is None or cost <= best:
                best = cost
        if best is not None:
            self.accumulated_costs[(i, j)] = best

    def path_cost(self, i, j):
        """Length-normalized path cost at (i, j); None if outside the band."""
        v = self.accumulated_costs.get((i, j))
        if v is None:
            return None
        return v / (1 + i + j)

    def get_inc(self):
        """GetInc(t, j): pick the next expansion direction.

        The minimum normalized path cost over the border row and column of the
        search band decides the direction; ``best_ref``/``best_input`` (the
        argmin cell) is the reported alignment position. A side that advanced
        ``max_run_count`` times in a row is forced to yield to the other.
        """
        i = self.ref_pointer - 1
        j = self.input_pointer - 1
        best = self.path_cost(i, j)
        if best is None:
            best = np.inf
        best_row, best_col = i, j

        for r in range(max(0, i - self.w + 1), i + 1):
            cost = self.path_cost(r, j)
            if cost is not None and cost < best:
                best = cost
                best_row, best_col = r, j

        for c in range(max(0, j - self.w + 1), j + 1):
            cost = self.path_cost(i, c)
            if cost is not None and cost < best:
                best = cost
                best_row, best_col = i, c

        self.best_ref = best_row
        self.best_input = best_col

        if self.run_count_ref >= self.max_run_count:
            return Direction.TARGET
        if self.run_count_input >= self.max_run_count:
            return Direction.REF
        if best_row == i and best_col == j:
            return Direction.BOTH
        if best_row < i:
            return Direction.TARGET
        return Direction.REF

    def update_run_counts(self, adv_ref, adv_input):
        if adv_ref and adv_input:
            self.run_count_ref = 0
            self.run_count_input = 0
        elif adv_ref:
            self.run_count_ref += 1
            self.run_count_input = 0
        else:
            self.run_count_input += 1
            self.run_count_ref = 0

    # ---- band expansion ----

    def advance_ref(self):
        """Add reference row i, evaluated against input columns [j-w+1, j]."""
        self.ref_pointer += 1
        i = self.ref_pointer - 1
        j = self.input_pointer - 1
        lo = max(0, j - self.w + 1)
        input_band = np.asarray(self.input_features[lo : j + 1])
        dists = self._distances(input_band, self.reference_features[i])
        for k, c in enumerate(range(lo, j + 1)):
            self.evaluate_path_cost(i, c, dists[k])

    def advance_input(self, input_features):
        """Add input column j, evaluated against reference rows [i-w+1, i]."""
        self.input_features.append(np.asarray(input_features).reshape(-1))
        self.input_pointer += 1
        j = self.input_pointer - 1
        i = self.ref_pointer - 1
        lo = max(0, i - self.w + 1)
        dists = self._distances(
            self.reference_features[lo : i + 1], self.input_features[j]
        )
        for k, r in enumerate(range(lo, i + 1)):
            self.evaluate_path_cost(r, j, dists[k])
        if self.input_pointer % self.w == 0:
            self._prune()

    def _prune(self):
        """Drop cells far behind the band frontier to bound memory."""
        cutoff = self.ref_pointer - 2 * self.w
        if cutoff > 0:
            self.accumulated_costs = {
                k: v for k, v in self.accumulated_costs.items() if k[0] >= cutoff
            }

    def initialize(self, observation):
        """Initialize the DP with the first input frame at cell (0, 0)."""
        self.input_features.append(np.asarray(observation).reshape(-1))
        self.input_pointer += 1
        self.ref_pointer += 1
        d0 = self._distances(self.reference_features[0:1], self.input_features[0])[0]
        self.evaluate_path_cost(0, 0, d0)
        self._initialized = True
        self.input_index += 1

    def step(self, input_features):
        if not self._initialized:
            self.initialize(input_features)
            return

        self._pending.append(input_features)
        while self.is_still_following():
            direction = self.get_inc()
            adv_ref = direction != Direction.TARGET  # REF or BOTH
            adv_input = direction != Direction.REF  # TARGET or BOTH
            if adv_input and not self._pending:
                break  # needs an input frame that has not arrived yet
            if adv_ref:
                self.advance_ref()
            if adv_input:
                self.advance_input(self._pending.pop(0))
            self.update_run_counts(adv_ref, adv_input)
            self.save_history()

        self.current_index = self._frame_to_score_idx(self.best_ref)
        self.save_history()
        self.input_index += 1

    def _pull_input(self):
        """Block for the next input item from the queue; None at stream end."""
        while True:
            item = self.queue.get(timeout=self.queue_timeout)
            if item is STREAM_END:
                return None
            if item is not None:
                return item

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        first = self._pull_input()
        if first is None:
            return self.alignment_path
        observation, self.current_perf_time = first
        self.initialize(observation)
        self.current_index = self._frame_to_score_idx(self.best_ref)
        yield self.get_current_position()

        while self.is_still_following():
            t0 = time.time()
            direction = self.get_inc()
            adv_ref = direction != Direction.TARGET
            adv_input = direction != Direction.REF
            if adv_input:
                item = self._pull_input()
                if item is None:
                    break
                observation, self.current_perf_time = item
            if adv_ref:
                self.advance_ref()
            if adv_input:
                self.advance_input(observation)
            self.update_run_counts(adv_ref, adv_input)
            self.current_index = self._frame_to_score_idx(self.best_ref)
            self.save_history()
            if adv_input:
                self.latency_stats = set_latency_stats(
                    time.time() - t0, self.latency_stats, self.input_index
                )
                self.input_index += 1
            yield self.get_current_position()
        return self.alignment_path


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
        distance_func: str = OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
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
