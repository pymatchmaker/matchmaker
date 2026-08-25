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
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
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

STEP_SIZE: int = 3
START_WINDOW_SIZE: Union[float, int] = 0.1
WINDOW_SIZE: int = 10


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
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Arzt requires `ref_frame_to_beat` (per-frame beat mapping)."
            )
        self.frame_rate = frame_rate
        self._ref_frame_to_beat = ref_frame_to_beat
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(start_window_size * frame_rate))
        self.queue_timeout = QUEUE_TIMEOUT
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
        super().reset()
        self.input_features: list = []
        self._current_frame = 0

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
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            input_features.squeeze(),
            self.distance_func,
        )

        self.global_cost_matrix, min_index, min_costs = oltw_arzt_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
        )

        if self.input_index > 0:
            self._current_frame = int(
                np.clip(
                    min_index,
                    self._current_frame,
                    self._current_frame + self.step_size,
                )
            )
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
