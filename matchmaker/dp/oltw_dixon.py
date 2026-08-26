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
from matchmaker.utils.misc import set_latency_stats


class Direction(IntEnum):
    REF = 0
    INPUT = 1
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
        self.distance_func = (
            distance_func if isinstance(distance_func, str) else "euclidean"
        )
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
      ``self.w``               <->  search width c (window_size seconds * frame_rate)
    """

    # Predecessor offsets and step weights of the DTW recursion, keyed by the
    # direction that reaches cell (i, j). Iteration order defines tie-breaking.
    STEP_WEIGHTS = {
        Direction.BOTH: ((-1, -1), 2.0),
        Direction.REF: ((-1, 0), 1.0),
        Direction.INPUT: ((0, -1), 1.0),
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
        """Return one score position for every consumed performance frame."""
        if self.input_pointer == 0:
            return np.empty((2, 0), dtype=float)

        score_positions = np.full(self.input_pointer, np.nan, dtype=float)
        perf_frames = np.rint(self.wp[0] * self.frame_rate).astype(int)
        for perf_frame, score_position in zip(perf_frames, self.wp[1]):
            if 0 <= perf_frame < self.input_pointer:
                score_positions[perf_frame] = score_position

        previous_position = self._frame_to_beat(0)
        for perf_frame in range(self.input_pointer):
            if np.isnan(score_positions[perf_frame]):
                score_positions[perf_frame] = previous_position
            else:
                previous_position = score_positions[perf_frame]

        perf_times = np.arange(self.input_pointer, dtype=float) / self.frame_rate
        return np.vstack((perf_times, score_positions))

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
        """Distances from each feature in band A to frame b."""
        return scipy.spatial.distance.cdist(
            A, b.reshape(1, -1), metric=self.distance_func
        )[:, 0]

    def evaluate_path_cost(self, ref_idx, input_idx, local_dist):
        """EvaluatePathCost(t, j): DTW recursion at (ref_idx, input_idx).

        Diagonal (BOTH) steps are weighted twice the local distance, straight
        steps once, following Sakoe & Chiba (1978) as adopted by Dixon (2005,
        DAFx): every monotone path then covers equal extent at equal weight,
        which makes the length normalization in ``path_cost`` meaningful.
        """
        if ref_idx == 0 and input_idx == 0:
            self.accumulated_costs[(0, 0)] = local_dist
            return
        best = None
        for offset, weight in self.STEP_WEIGHTS.values():
            predecessor = self.accumulated_costs.get(
                (ref_idx + offset[0], input_idx + offset[1])
            )
            if predecessor is None:
                continue
            cost = predecessor + weight * local_dist
            if best is None or cost <= best:
                best = cost
        if best is not None:
            self.accumulated_costs[(ref_idx, input_idx)] = best

    def path_cost(self, ref_idx, input_idx):
        """Return normalized cost at (ref_idx, input_idx), if available."""
        v = self.accumulated_costs.get((ref_idx, input_idx))
        if v is None:
            return None
        return v / (1 + ref_idx + input_idx)

    def _update_best_alignment(self):
        """Report the frontier cell with the minimum normalized path cost."""
        current_ref = self.ref_pointer - 1
        current_input = self.input_pointer - 1
        best = self.path_cost(current_ref, current_input)
        if best is None:
            best = np.inf
        best_ref, best_input = current_ref, current_input

        for ref_idx in range(max(0, current_ref - self.w + 1), current_ref + 1):
            cost = self.path_cost(ref_idx, current_input)
            if cost is not None and cost < best:
                best = cost
                best_ref, best_input = ref_idx, current_input

        for input_idx in range(max(0, current_input - self.w + 1), current_input + 1):
            cost = self.path_cost(current_ref, input_idx)
            if cost is not None and cost < best:
                best = cost
                best_ref, best_input = current_ref, input_idx

        self.best_ref = best_ref
        self.best_input = best_input
        return best_ref, best_input

    def get_inc(self):
        """Pick expansion direction from the frontier argmin."""
        current_ref = self.ref_pointer - 1
        current_input = self.input_pointer - 1
        best_ref, best_input = self._update_best_alignment()

        # Preserve the frontier argmin while BOTH initializes the c x c band.
        if self.ref_pointer < self.w or self.input_pointer < self.w:
            return Direction.BOTH

        if self.run_count_ref >= self.max_run_count:
            return Direction.INPUT
        if self.run_count_input >= self.max_run_count:
            return Direction.REF
        if best_ref == current_ref and best_input == current_input:
            return Direction.BOTH
        if best_ref < current_ref:
            return Direction.INPUT
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
        """Advance REF and evaluate it against the active INPUT range."""
        self.ref_pointer += 1
        ref_idx = self.ref_pointer - 1
        current_input = self.input_pointer - 1
        lo = max(0, current_input - self.w + 1)
        input_band = np.asarray(self.input_features[lo : current_input + 1])
        dists = self._distances(input_band, self.reference_features[ref_idx])
        for k, input_idx in enumerate(range(lo, current_input + 1)):
            self.evaluate_path_cost(ref_idx, input_idx, dists[k])

    def advance_input(self, input_features):
        """Advance INPUT and evaluate it against the active REF range."""
        self.input_features.append(np.asarray(input_features).reshape(-1))
        self.input_pointer += 1
        input_idx = self.input_pointer - 1
        current_ref = self.ref_pointer - 1
        lo = max(0, current_ref - self.w + 1)
        dists = self._distances(
            self.reference_features[lo : current_ref + 1],
            self.input_features[input_idx],
        )
        for k, ref_idx in enumerate(range(lo, current_ref + 1)):
            self.evaluate_path_cost(ref_idx, input_idx, dists[k])
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
            input_direction = self.get_inc()
            adv_input = input_direction != Direction.REF
            if adv_input and not self._pending:
                break
            if adv_input:
                self.advance_input(self._pending.pop(0))

            ref_direction = self.get_inc()
            adv_ref = ref_direction != Direction.INPUT
            if adv_ref:
                self.advance_ref()

            self._update_best_alignment()
            self.update_run_counts(adv_ref, adv_input)
            self.save_history()

        self.current_index = self._frame_to_score_idx(self.best_ref)
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
    window_size : int
        Search width in number of events.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        queue: Optional[RECVQueue] = None,
        window_size: int = 30,
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
        self.w = window_size
        self.reset()

    def reset(self) -> None:
        self._D: Dict = {}
        self._inputs = []
        self._pending = []
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

    def _evaluate(self, ref_i, inp_j):
        if ref_i < 0 or inp_j < 0 or ref_i >= self.N_ref:
            return
        d = self._local_cost(ref_i, inp_j)
        if ref_i == 0 and inp_j == 0:
            self._D[(ref_i, inp_j)] = d
            return
        best_cost = np.inf
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
        if best_cost < np.inf:
            self._D[(ref_i, inp_j)] = best_cost

    def _norm_cost(self, ref_i, inp_j):
        return self._get_D(ref_i, inp_j) / (1 + ref_i + inp_j)

    def _argmin_path_cost(self):
        best_cost, best_ref, best_inp = np.inf, self.j, self.t
        for k in range(max(0, self.t - self.w + 1), self.t + 1):
            nc = self._norm_cost(self.j, k)
            if nc < best_cost:
                best_cost, best_ref, best_inp = nc, self.j, k
        for k in range(max(0, self.j - self.w + 1), self.j + 1):
            nc = self._norm_cost(k, self.t)
            if nc < best_cost:
                best_cost, best_ref, best_inp = nc, k, self.t
        return best_ref, best_inp

    def _compute_input_column(self):
        for k in range(max(0, self.j - self.w + 1), self.j + 1):
            self._evaluate(k, self.t)

    def _compute_ref_row(self):
        for k in range(max(0, self.t - self.w + 1), self.t + 1):
            self._evaluate(self.j, k)

    def get_inc(self):
        if self.j >= self.N_ref - 1:
            return Direction.INPUT
        if self.t + 1 < self.w or self.j + 1 < self.w:
            return Direction.BOTH
        if self.run_count_ref >= self.max_run_count:
            return Direction.INPUT
        if self.run_count_input >= self.max_run_count:
            return Direction.REF

        best_ref, best_input = self._argmin_path_cost()
        if best_ref == self.j and best_input == self.t:
            return Direction.BOTH
        if best_ref < self.j:
            return Direction.INPUT
        return Direction.REF

    def _update_run_counts(self, adv_ref, adv_input):
        if adv_ref and adv_input:
            self.run_count_ref = 0
            self.run_count_input = 0
        elif adv_ref:
            self.run_count_ref += 1
            self.run_count_input = 0
        else:
            self.run_count_input += 1
            self.run_count_ref = 0

    def step(self, input_feat: NDArray[np.float32]) -> None:
        self._pending.append(np.asarray(input_feat, dtype=np.float32))
        if self.t < 0:
            self._inputs.append(self._pending.pop(0))
            self.t = 0
            self.j = 0
            self._evaluate(0, 0)
        else:
            while True:
                input_direction = self.get_inc()
                adv_input = input_direction != Direction.REF
                if adv_input and not self._pending:
                    break
                if adv_input:
                    self._inputs.append(self._pending.pop(0))
                    self.t += 1
                    self._compute_input_column()

                ref_direction = self.get_inc()
                adv_ref = ref_direction != Direction.INPUT
                if adv_ref:
                    self.j += 1
                    self._compute_ref_row()

                self._update_run_counts(adv_ref, adv_input)

        best_ref, _ = self._argmin_path_cost()
        self.current_index = min(best_ref, self.N_ref - 1)
        self.input_index += 1

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))


if __name__ == "__main__":
    pass  # pragma: no cover
