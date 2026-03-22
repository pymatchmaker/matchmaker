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
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import progressbar
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END
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

    Subclasses must implement ``step()`` and may override ``reset()``,
    ``get_window()``, and ``_init_distance_func()``.

    Parameters
    ----------
    reference_features : np.ndarray
        Feature matrix for the reference (score) sequence.
    window_size : int
        Search window size (interpretation depends on subclass).
    step_size : int
        Maximum position advance per input step.
    start_window_size : int
        Window size during warmup.
    queue : RECVQueue or None
        Input queue for streaming.
    state_space : np.ndarray or None
        Score positions (beats) for each reference element.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size=WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        start_window_size=START_WINDOW_SIZE,
        queue: Optional[RECVQueue] = None,
        state_space=None,
        **kwargs,
    ) -> None:
        super().__init__(reference_features=reference_features)
        self.N_ref: int = self.reference_features.shape[0]
        self.queue = queue or RECVQueue()
        self.step_size: int = step_size
        self.state_space = state_space
        self._ref_frame_to_beat: Optional[NDArray] = kwargs.get(
            "ref_frame_to_beat", None
        )
        # Subclasses set self._window_size and self._start_window_size
        # after calling super().__init__
        self.init_position: int = kwargs.get("current_position", 0)
        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        # Legacy attributes (used by some callers)
        self.state_to_ref_time_map = kwargs.get("state_to_ref_time_map", None)
        self.ref_to_state_time_map = kwargs.get("ref_to_state_time_map", None)

        self.reset()

    def reset(self) -> None:
        self.current_position: int = self.init_position
        self.positions: List[int] = []
        self._warping_path: List = []
        self.global_cost_matrix: NDArray[np.float32] = np.full(
            (self.N_ref + 1, 2), np.inf, dtype=np.float32
        )
        self.input_index: int = 0
        self.input_features = None

    @property
    def current_beat(self) -> float:
        if self._ref_frame_to_beat is not None:
            idx = min(self.current_position, len(self._ref_frame_to_beat) - 1)
            return float(self._ref_frame_to_beat[idx])
        if self.state_space is not None:
            idx = min(self.current_position, len(self.state_space) - 1)
            return float(self.state_space[idx])
        return float(self.current_position)

    @property
    def warping_path(self) -> NDArray[np.float32]:
        if self._warping_path:
            return np.array(self._warping_path).T
        return np.empty((2, 0))

    @property
    def window_index(self) -> int:
        return self.current_position

    def __call__(self, input: NDArray[np.float32]) -> int:
        self.step(input)
        return self.current_position

    def is_still_following(self) -> bool:
        return self.current_position <= self.N_ref - 2

    def get_window(self) -> Tuple[int, int]:
        """Return (start, end) of the search window."""
        raise NotImplementedError

    def step(self, input_features: NDArray[np.float32]) -> None:
        """Process one input element and update position."""
        raise NotImplementedError

    def _clamp_position(self, min_index: int) -> None:
        """Apply step-size constraint: no backwards, max step_size forward."""
        if self.input_index > 0:
            self.current_position = int(
                np.clip(
                    min_index,
                    self.current_position,
                    self.current_position + self.step_size,
                )
            )
        else:
            self.current_position = min_index

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        """Run alignment from queue.

        Yields
        ------
        float
            Current score position in beats.

        Returns
        -------
        np.ndarray
            Warping path (2, T).
        """
        self.reset()

        if verbose:
            n = len(self.state_space) if self.state_space is not None else self.N_ref
            pbar = progressbar.ProgressBar(
                max_value=n,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            pbar.start()

        while self.is_still_following():
            item = self.queue.get(timeout=QUEUE_TIMEOUT)
            if item is STREAM_END:
                break
            if item is None:
                continue
            features = item[0] if isinstance(item, tuple) else item
            self.last_queue_update = time.time()

            # Frame subclass accumulates input_features for plotting
            if hasattr(self, "_accumulate_input") and self._accumulate_input:
                self.input_features = (
                    np.concatenate((self.input_features, features))
                    if self.input_features is not None
                    else features
                )

            self.step(features)

            if verbose:
                pbar.update(
                    int(np.searchsorted(self.state_space, self.current_beat))
                    if self.state_space is not None
                    else min(self.current_position, self.N_ref - 1)
                )

            latency = time.time() - self.last_queue_update
            self.latency_stats = set_latency_stats(
                latency, self.latency_stats, self.input_index
            )
            yield self.current_beat

        if verbose:
            pbar.finish()

        return self.warping_path


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
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        current_position: int = 0,
        frame_rate: int = FRAME_RATE,
        queue: Optional[RECVQueue] = None,
        state_to_ref_time_map=None,
        ref_to_state_time_map=None,
        state_space=None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            state_space=state_space,
            current_position=current_position,
            state_to_ref_time_map=state_to_ref_time_map,
            ref_to_state_time_map=ref_to_state_time_map,
            **kwargs,
        )
        self.frame_rate = frame_rate
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(start_window_size * frame_rate))
        self._accumulate_input = True

        # Set distance function (Cython or callable)
        self._init_distance_func(distance_func)

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

        self._clamp_position(min_index)
        self._warping_path.append((self.current_beat, self.input_index))
        self.input_index += 1


# ---------------------------------------------------------------------------
# Event-level subclass (MIDI)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArztEvent(OnlineTimeWarpingArzt):
    """Event-level OLTW for MIDI input.

    Each step processes one onset event. Uses scipy for distance computation
    and a pure-Python rolling DP loop. Window size is in number of events.

    When queue provides ``(feature, timestamp)`` tuples, the warping path
    records ``(score_beat, perf_time_sec)`` instead of ``(beat, input_index)``.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size: int = 30,
        step_size: int = 5,
        start_window_size: int = 5,
        distance_func: str = "cosine",
        queue: Optional[RECVQueue] = None,
        state_space=None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            state_space=state_space,
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

    def step(
        self, input_features: NDArray[np.float32], perf_time: float = None
    ) -> None:
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

        self._clamp_position(min_index)
        t = perf_time if perf_time is not None else self.input_index
        self._warping_path.append((self.current_beat, t))
        self.input_index += 1

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        """Run from queue. Extracts timestamp from (feature, time) tuples."""
        self.reset()
        if verbose:
            pbar = progressbar.ProgressBar(
                max_value=self.N_ref,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            pbar.start()

        while self.is_still_following():
            item = self.queue.get(timeout=QUEUE_TIMEOUT)
            if item is STREAM_END:
                break
            if item is None:
                continue
            if isinstance(item, tuple):
                features, perf_time = item[0], float(item[1])
            else:
                features, perf_time = item, None
            self.last_queue_update = time.time()
            self.step(features, perf_time=perf_time)

            if verbose:
                pbar.update(min(self.current_position, self.N_ref - 1))
            latency = time.time() - self.last_queue_update
            self.latency_stats = set_latency_stats(
                latency, self.latency_stats, self.input_index
            )
            yield self.current_beat

        if verbose:
            pbar.finish()
        return self.warping_path


if __name__ == "__main__":
    pass  # pragma: no cover
