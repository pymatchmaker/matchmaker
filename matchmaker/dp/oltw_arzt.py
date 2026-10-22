#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Time Warping (OLTWArzt)

OLTW with step-size constraint and adaptive window, based on:
  Arzt & Widmer (2010) "Simple Tempo Models for Real-Time Music Tracking"
  Arzt & Widmer (2015) "Real-Time Music Tracking Using Multiple Performances
                        as a Reference"

Classes:
  OnlineTimeWarpingArzt      — base class (common properties, step-size clamp, run loop)
  OnlineTimeWarpingArztFrame — frame-level variant for audio (Cython-accelerated)
  OnlineTimeWarpingArztEvent — event-level variant for MIDI (onset-by-onset)
"""

import time
from typing import Any, Callable, Dict, Generator, NamedTuple, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.direction import Direction
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
    distances,
)
from matchmaker.utils.distances import Metric, vdist
from matchmaker.utils.errors import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
from matchmaker.utils.misc import set_latency_stats
from matchmaker.utils.tempo_models import ArztTempoModel, make_tempo_model

STEP_SIZE: int = 3
START_WINDOW_SIZE: Union[float, int] = 0.1
WINDOW_SIZE: int = 10


class BackpointerColumn(NamedTuple):
    """Predecessor directions in one input frame's reference window."""

    window_start: int
    predecessor_directions: NDArray[np.int8]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArzt(OnlineAlignment):
    """Base class for Arzt-style OLTW with step-size constraint.

    Parameters
    ----------
    reference_features : np.ndarray
        Feature matrix for the reference (score) sequence.
    score_positions : np.ndarray
        Score beat positions for unique onsets.
    window_size : int
        Search window size (interpretation depends on subclass).
    step_size : int
        Maximum position advance per input step.
    start_window_size : int
        Window size during warmup.
    queue : RECVQueue or None
        Input queue for streaming.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size=WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        start_window_size=START_WINDOW_SIZE,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.N_ref: int = self.reference_features.shape[0]
        self.step_size: int = step_size
        self.reset()

    def reset(self) -> None:
        self.current_index = 0
        self._alignment_path: list = []
        self.global_cost_matrix: NDArray[np.float32] = np.full(
            (self.N_ref + 1, 2), np.inf, dtype=np.float32
        )
        self.input_index: int = 0

    @property
    def window_index(self) -> int:
        return self.current_index

    def get_window(self) -> Tuple[int, int]:
        """Return (start, end) of the search window."""
        raise NotImplementedError

    def step(self, input_features: NDArray[np.float32]) -> None:
        """Process one input element and update self.current_index."""
        raise NotImplementedError

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))


# ---------------------------------------------------------------------------
# Frame-level subclass (audio)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArztFrame(OnlineTimeWarpingArzt):
    """Frame-level OLTW for audio input.

    Uses Cython-accelerated ``oltw_arzt_loop`` and Matchmaker distance
    functions. Window size is in seconds (converted to frames via frame_rate).
    """

    DEFAULT_DISTANCE_FUNC: str = "Manhattan"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: NDArray = None,
        tempo_model=None,
        backward_forward: Optional[bool] = None,
        max_run_count: Optional[int] = None,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Arzt requires `ref_frame_to_beat` (per-frame beat mapping)."
            )
        self.frame_rate = frame_rate
        self._ref_frame_to_beat = ref_frame_to_beat
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(start_window_size * frame_rate))
        self.queue_timeout = QUEUE_TIMEOUT

        self.tempo_model = make_tempo_model(tempo_model)
        self.backward_forward = bool(backward_forward)
        self.max_run_count = max_run_count

        if self.tempo_model is not None:
            self._orig_features = np.array(reference_features, copy=True)
            self._orig_f2b = np.asarray(ref_frame_to_beat, dtype=np.float64).copy()

        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            **kwargs,
        )
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self._init_distance_func(distance_func)

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        self.input_features.append(observation)
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return beat

    def reset(self) -> None:
        if self.tempo_model is not None:
            self.reference_features = self._orig_features.copy()
            self._ref_frame_to_beat = self._orig_f2b.copy()
            self.N_ref = len(self.reference_features)
            on = np.searchsorted(self._ref_frame_to_beat, self.score_positions)
            self.is_onset = np.zeros(self.N_ref, dtype=bool)
            self.is_onset[on[on < self.N_ref]] = True
        if self.tempo_model is not None or self.backward_forward:
            self._backpointers = {}
            self._last_window_end = 0
            self._min_x_needed = 0
        self.run_count_ref = 0
        self.run_count_input = 0
        if self.backward_forward:
            K = 4 * self._window_size
            self._cost_band = np.full(
                (K, 2 * self._window_size + 64), np.inf, np.float32
            )
            self._cost_start = np.zeros(K, dtype=np.int64)
            self._cost_len = np.zeros(K, dtype=np.int64)
            self._n_bf = 0
        super().reset()
        self.input_features: list = []
        self._current_frame = 0
        if self.tempo_model is not None:
            self.tempo_model.reset(self)

    @property
    def current_relative_tempo(self) -> float:
        return 1.0 if self.tempo_model is None else self.tempo_model.current_tempo

    @property
    def n_deleted(self) -> int:
        return self.tempo_model.n_deleted if self.tempo_model else 0

    @property
    def n_inserted(self) -> int:
        return self.tempo_model.n_inserted if self.tempo_model else 0

    @property
    def use_tempo_model(self) -> bool:
        return self.tempo_model is not None

    @property
    def N(self) -> int:
        return len(self.reference_features)

    @property
    def c(self) -> int:
        return self._window_size

    @property
    def V(self) -> np.ndarray:
        return self.reference_features

    # Interface for the tempo model (TempoModel).

    def tempo_last_input(self) -> int:
        return self.input_index - 1

    def tempo_current_cell(self) -> Tuple[int, int]:
        return self.input_index - 1, self._current_frame

    def tempo_backward_path(self, start_input_index: int, start_ref_index: int):
        input_index, ref_index = start_input_index, start_ref_index
        while input_index > 0 and ref_index > 0:
            column = self._backpointers.get(input_index)
            if column is None:
                break
            offset = ref_index - column.window_start
            if offset < 0 or offset >= len(column.predecessor_directions):
                break
            direction = column.predecessor_directions[offset]
            if direction == Direction.REF:
                ref_index -= 1
            elif direction == Direction.INPUT:
                input_index -= 1
            elif direction == Direction.BOTH:
                input_index -= 1
                ref_index -= 1
            else:
                break
            yield input_index, ref_index

    def tempo_note_path_end(self, x: int, y: int) -> None:
        self._min_x_needed = x if not self._min_x_needed else min(self._min_x_needed, x)

    def tempo_last_score_frame(self) -> int:
        return self._current_frame

    def tempo_score(self):
        return self.reference_features, self._ref_frame_to_beat, self.is_onset

    def tempo_set_score(self, features, frame_to_beat, is_onset) -> None:
        n_frames = len(features)
        if len(frame_to_beat) != n_frames or len(is_onset) != n_frames:
            raise ValueError("Score features, beat mapping and onset flags must agree.")
        if abs(n_frames - self.N_ref) > 1:
            raise ValueError("A tempo alteration must insert or delete a single frame.")

        cost_matrix = self.global_cost_matrix  # row k + 1 = reference frame k
        if n_frames < self.N_ref:
            deleted_ref_index = self._current_frame + 2
            if not 0 <= deleted_ref_index < self.N_ref:
                raise ValueError("The reference frame to delete is out of bounds.")
            self.global_cost_matrix = np.ascontiguousarray(
                np.delete(cost_matrix, deleted_ref_index + 1, axis=0)
            )
            for input_index, column in list(self._backpointers.items()):
                if column.window_start > deleted_ref_index:
                    self._backpointers[input_index] = column._replace(
                        window_start=column.window_start - 1
                    )
                else:
                    offset = deleted_ref_index - column.window_start
                    if offset < len(column.predecessor_directions):
                        self._backpointers[input_index] = column._replace(
                            predecessor_directions=np.delete(
                                column.predecessor_directions, offset
                            )
                        )
            if self.backward_forward:
                self._band_alter(deleted_ref_index, -1)
        elif n_frames > self.N_ref:
            inserted_ref_index = self._current_frame + 1
            if not 0 <= inserted_ref_index <= self.N_ref:
                raise ValueError("The reference insertion position is out of bounds.")
            if inserted_ref_index + 1 < len(cost_matrix):
                interpolated_cost = 0.5 * (
                    cost_matrix[inserted_ref_index]
                    + cost_matrix[inserted_ref_index + 1]
                )
            else:
                interpolated_cost = np.full(2, np.inf, dtype=np.float32)
            self.global_cost_matrix = np.ascontiguousarray(
                np.insert(
                    cost_matrix, inserted_ref_index + 1, interpolated_cost, axis=0
                )
            )
            for input_index, column in list(self._backpointers.items()):
                if column.window_start >= inserted_ref_index:
                    self._backpointers[input_index] = column._replace(
                        window_start=column.window_start + 1
                    )
                else:
                    offset = inserted_ref_index - column.window_start
                    if offset < len(column.predecessor_directions):
                        self._backpointers[input_index] = column._replace(
                            predecessor_directions=np.insert(
                                column.predecessor_directions, offset, Direction.REF
                            ).astype(np.int8)
                        )
            if self.backward_forward:
                self._band_alter(inserted_ref_index, 1)
        self.reference_features = features
        self._ref_frame_to_beat = frame_to_beat
        self.is_onset = is_onset
        self.N_ref = n_frames

    # Strategy 1 (Arzt, Widmer & Dixon 2008, Sec. 3.2).

    def _band_alter(self, ref_index, change):
        for slot in range(len(self._cost_band)):
            window_start = self._cost_start[slot]
            window_length = self._cost_len[slot]
            offset = ref_index - window_start
            if change < 0:
                if window_start > ref_index:
                    self._cost_start[slot] = window_start - 1
                elif offset < window_length:
                    self._cost_band[slot, offset : window_length - 1] = self._cost_band[
                        slot, offset + 1 : window_length
                    ].copy()
                    self._cost_band[slot, window_length - 1] = np.inf
                    self._cost_len[slot] = window_length - 1
            else:
                if window_start >= ref_index:
                    self._cost_start[slot] = window_start + 1
                elif offset < window_length:
                    new_length = min(window_length + 1, self._cost_band.shape[1])
                    self._cost_band[slot, offset + 1 : new_length] = self._cost_band[
                        slot, offset : new_length - 1
                    ].copy()
                    self._cost_band[slot, offset] = np.inf
                    self._cost_len[slot] = new_length

    def _cost_row(self, ref_index, input_start, input_end):
        input_indices = np.arange(input_start, input_end + 1)
        capacity = len(self._cost_band)
        slots = input_indices % capacity
        offsets = ref_index - self._cost_start[slots]
        valid = (
            (input_indices > self.input_index - capacity)
            & (input_indices <= self.input_index)
            & (offsets >= 0)
            & (offsets < self._cost_len[slots])
        )
        costs = np.full(len(input_indices), np.inf)
        costs[valid] = self._cost_band[slots[valid], offsets[valid]]
        return costs

    def _cost_col(self, input_index, ref_start, ref_end):
        costs = np.full(ref_end - ref_start + 1, np.inf)
        capacity = len(self._cost_band)
        if (
            input_index > self.input_index
            or input_index <= self.input_index - capacity
            or input_index < 0
        ):
            return costs
        slot = input_index % capacity
        window_start = self._cost_start[slot]
        window_length = self._cost_len[slot]
        overlap_start = max(ref_start, window_start)
        overlap_end = min(ref_end, window_start + window_length - 1)
        if overlap_start <= overlap_end:
            costs[overlap_start - ref_start : overlap_end - ref_start + 1] = (
                self._cost_band[
                    slot, overlap_start - window_start : overlap_end - window_start + 1
                ]
            )
        return costs

    def _cost(self, input_index, ref_index):
        return float(self._cost_col(input_index, ref_index, ref_index)[0])

    def update_run_counts(self, adv_ref, adv_input):
        """Count actual single-axis advances, following Dixon's convention."""
        if adv_ref and adv_input:
            self.run_count_ref = 0
            self.run_count_input = 0
        elif adv_ref:
            self.run_count_ref += 1
            self.run_count_input = 0
        elif adv_input:
            self.run_count_input += 1
            self.run_count_ref = 0

    def get_inc(self, input_index, ref_index) -> Direction:
        if self.max_run_count is not None:
            if self.run_count_ref >= self.max_run_count:
                return Direction.INPUT
            if self.run_count_input >= self.max_run_count:
                return Direction.REF
        window_size = self._window_size
        best_cost = self._cost(input_index, ref_index) / (input_index + ref_index + 1.0)
        best_input, best_ref = input_index, ref_index
        input_start = max(0, input_index - window_size + 1)
        ref_start = max(0, ref_index - window_size + 1)
        row_costs = self._cost_row(ref_index, input_start, input_index) / (
            np.arange(input_start, input_index + 1) + ref_index + 1.0
        )
        best_offset = int(np.argmin(row_costs))
        if row_costs[best_offset] < best_cost:
            best_cost = row_costs[best_offset]
            best_input, best_ref = input_start + best_offset, ref_index
        column_costs = self._cost_col(input_index, ref_start, ref_index) / (
            input_index + np.arange(ref_start, ref_index + 1) + 1.0
        )
        best_offset = int(np.argmin(column_costs))
        if column_costs[best_offset] < best_cost:
            best_input, best_ref = input_index, ref_start + best_offset
        if best_input < input_index:
            return Direction.REF
        if best_ref < ref_index:
            return Direction.INPUT
        return Direction.BOTH

    def _strategy1(self, input_end, ref_index):
        backward_distance = 50 if self._n_bf % 5 == 4 else 10
        self._n_bf += 1
        input_index, target_ref = input_end, ref_index - backward_distance
        for input_index, ref_index in self.tempo_backward_path(input_end, ref_index):
            if ref_index <= target_ref:
                break
        self.tempo_note_path_end(input_index, ref_index)
        ref_end = self._last_window_end - 1
        self.run_count_ref = 0
        self.run_count_input = 0
        while input_index < input_end and ref_index < ref_end:
            previous_input, previous_ref = input_index, ref_index
            if self.get_inc(input_index, ref_index) != Direction.REF:
                if not np.isfinite(
                    self._cost(input_index + 1, ref_index)
                ) and not np.isfinite(self._cost(input_index + 1, ref_index + 1)):
                    break
                input_index += 1
            if (
                input_index <= input_end
                and ref_index < ref_end
                and self.get_inc(input_index, ref_index) != Direction.INPUT
            ):
                ref_index += 1
            if not np.isfinite(self._cost(input_index, ref_index)):
                break
            self.update_run_counts(
                ref_index > previous_ref, input_index > previous_input
            )
        return ref_index

    @property
    def window_index(self) -> int:
        return self._current_frame

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
        if self.tempo_model is not None and self.input_index > 5:
            x, y = self.input_index - 1, self._current_frame
            path = [(x, y)]
            for bx, by in self.tempo_backward_path(x, y):
                path.append((bx, by))
                if len(path) >= 3:
                    break
            if len(path) >= 2:
                bx, by = path[-1]
                dx = x - bx
                tempo = self.tempo_model.current_tempo
                pred_y = by + dx * tempo
                return float(
                    self._frame_to_beat(
                        int(round(np.clip(pred_y, 0, len(self._ref_frame_to_beat) - 1)))
                    )
                )
        return self._frame_to_beat(self._current_frame)

    def _init_distance_func(self, distance_func):
        if not (isinstance(distance_func, (str, tuple)) or callable(distance_func)):
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="distance_func",
                required_parameter_type=(str, tuple, Callable),
                actual_parameter_type=type(distance_func),
            )

        if isinstance(distance_func, str):
            if distance_func not in CYTHONIZED_METRICS_WO_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_WO_ARGUMENTS,
                    value=distance_func,
                )
            self.distance_func = getattr(distances, distance_func)()
        elif isinstance(distance_func, tuple):
            if distance_func[0] not in CYTHONIZED_METRICS_W_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_W_ARGUMENTS,
                    value=distance_func[0],
                )
            self.distance_func = getattr(distances, distance_func[0])(
                **distance_func[1]
            )
        elif callable(distance_func):
            self.distance_func = distance_func

        if isinstance(self.distance_func, Metric):
            self.vdist = vdist
        else:
            self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(
                np.float32
            )

    def get_window(self) -> Tuple[int, int]:
        w = self._window_size
        if self.window_index < self._start_window_size:
            w = self._start_window_size
        start = max(self.window_index - w, 0)
        end = min(self.window_index + w, self.N_ref)
        return start, end

    def step(self, input_features: NDArray[np.float32]) -> None:
        if self.tempo_model is not None and self.input_index > 0:
            self.tempo_model.before_step(self)

        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            input_features.squeeze(),
            self.distance_func,
        )

        store = (self.tempo_model is not None) or self.backward_forward
        backpointers = None
        if store:
            backpointers = np.full(
                window_end - window_start, Direction.NONE, dtype=np.int8
            )

        self.global_cost_matrix, min_index, min_costs = oltw_arzt_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
            backpointers=backpointers,
        )

        if self.backward_forward:
            k = self.input_index % len(self._cost_band)
            n = min(window_end - window_start, self._cost_band.shape[1])
            self._cost_band[k, :n] = self.global_cost_matrix[
                window_start + 1 : window_start + 1 + n, 0
            ]
            self._cost_start[k], self._cost_len[k] = window_start, n

        if store:
            self._backpointers[self.input_index] = BackpointerColumn(
                window_start=window_start, predecessor_directions=backpointers
            )
            self._last_window_end = window_end
            if (self.input_index + 1) % self._window_size == 0:
                keep = min(self.input_index - 2 * self._window_size, self._min_x_needed)
                for x in [x for x in list(self._backpointers.keys()) if x < keep - 1]:
                    del self._backpointers[x]
                self._min_x_needed = 0

        if self.input_index > 0:
            cf = self._current_frame
            min_pos = max(0, cf - 1) if self.tempo_model is not None else cf
            pos = int(
                np.clip(
                    min_index,
                    min_pos,
                    cf + self.step_size,
                )
            )
            if self.backward_forward and (self.input_index + 1) % 2 == 0:
                pos = self._strategy1(self.input_index, pos)
                pos = int(np.clip(pos, min_pos, cf + self.step_size))
            self._current_frame = int(pos)
        else:
            self._current_frame = min_index

        self.current_index = self._frame_to_score_idx(self._current_frame)
        self.input_index += 1


# ---------------------------------------------------------------------------
# Event-level subclass (MIDI)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArztEvent(OnlineTimeWarpingArzt):
    """Event-level OLTW for MIDI input.

    Each step processes one onset event. Uses scipy for distance computation
    and a pure-Python rolling DP loop. Window size is in number of events.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size: int = 30,
        step_size: int = 5,
        start_window_size: int = 5,
        distance_func: str = "cosine",
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            **kwargs,
        )
        self._window_size = window_size
        self._start_window_size = start_window_size
        self._scipy_metric = distance_func

    def get_window(self) -> Tuple[int, int]:
        w = self._window_size
        if self.input_index < self._start_window_size:
            w = self._start_window_size
        start = max(self.window_index - self.step_size, 0)
        end = min(start + w, self.N_ref)
        return start, end

    def step(self, input_features: NDArray[np.float32]) -> None:
        feat = np.asarray(input_features, dtype=np.float32).squeeze()

        window_start, window_end = self.get_window()
        window_cost = scipy.spatial.distance.cdist(
            self.reference_features[window_start:window_end],
            feat.reshape(1, -1),
            metric=self._scipy_metric,
        ).flatten()

        # Rolling 2-column DP (same logic as oltw_arzt_loop)
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        for idx_w, score_index in enumerate(range(window_start, window_end)):
            si = score_index + 1  # 1-indexed in cost matrix

            if score_index == 0 and self.input_index == 0:
                self.global_cost_matrix[1, 1] = window_cost[idx_w]
                min_costs = window_cost[idx_w]
                min_index = 0
                continue

            local_dist = window_cost[idx_w]
            d1 = self.global_cost_matrix[si - 1, 1] + local_dist
            d2 = self.global_cost_matrix[si, 0] + local_dist
            d3 = self.global_cost_matrix[si - 1, 0] + local_dist
            best = min(d1, d2, d3)
            self.global_cost_matrix[si, 1] = best

            norm_cost = best / (self.input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        # Shift columns
        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        self.global_cost_matrix[:, 1] = np.inf

        if self.input_index > 0:
            self.current_index = int(
                np.clip(
                    min_index,
                    self.current_index,
                    self.current_index + self.step_size,
                )
            )
        else:
            self.current_index = min_index
        self.input_index += 1


if __name__ == "__main__":
    pass  # pragma: no cover
