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
from matchmaker.utils.tempo_models import ArztTempoModel, make_tempo_model

STEP_SIZE: int = 3
START_WINDOW_SIZE: Union[float, int] = 0.1
WINDOW_SIZE: int = 10

_ADV_IN, _ADV_SC, _BOTH = (
    0,
    1,
    2,
)  # Dixon (2005) GetInc: Row (input), Column (score), Both


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
        self._run_count, self._run_prev = 0, None
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

    def tempo_backward_path(self, start_x: int, start_y: int):
        x, y = start_x, start_y
        while x > 0 and y > 0:
            entry = self._backpointers.get(x)
            if entry is None:
                break
            ws, codes = entry
            offset = y - ws
            if offset < 0 or offset >= len(codes):
                break
            code = codes[offset]
            if code == 1:  # vertical: (x, y-1)
                y -= 1
            elif code == 2:  # horizontal: (x-1, y)
                x -= 1
            elif code == 3:  # diagonal: (x-1, y-1)
                x -= 1
                y -= 1
            else:
                break
            yield x, y

    def tempo_note_path_end(self, x: int, y: int) -> None:
        self._min_x_needed = x if not self._min_x_needed else min(self._min_x_needed, x)

    def tempo_last_score_frame(self) -> int:
        return self._current_frame

    def tempo_score(self):
        return self.reference_features, self._ref_frame_to_beat, self.is_onset

    def tempo_set_score(self, features, frame_to_beat, is_onset) -> None:
        n = len(features)
        ps = self._current_frame
        G = self.global_cost_matrix  # row k + 1 = score frame k
        if n < self.N_ref:
            r = ps + 2
            self.global_cost_matrix = np.ascontiguousarray(np.delete(G, r + 1, axis=0))
            for x, (s, c) in list(self._backpointers.items()):
                if s > r:
                    self._backpointers[x] = (s - 1, c)
                elif r - s < len(c):
                    self._backpointers[x] = (s, np.delete(c, r - s))
            if self.backward_forward:
                self._band_alter(r, -1)
        elif n > self.N_ref:
            r = ps + 1
            if r + 1 < len(G):
                interp = 0.5 * (G[r] + G[r + 1])
            else:
                interp = np.full(2, np.inf, dtype=np.float32)
            self.global_cost_matrix = np.ascontiguousarray(
                np.insert(G, r + 1, interp, axis=0)
            )
            for x, (s, c) in list(self._backpointers.items()):
                if s >= r:
                    self._backpointers[x] = (s + 1, c)
                elif r - s < len(c):
                    self._backpointers[x] = (s, np.insert(c, r - s, 1).astype(np.int8))
            if self.backward_forward:
                self._band_alter(r, 1)
        self.reference_features = features
        self._ref_frame_to_beat = frame_to_beat
        self.is_onset = is_onset
        self.N_ref = n

    # Strategy 1 (Arzt, Widmer & Dixon 2008, Sec. 3.2).

    def _band_alter(self, r, sign):
        st, ln, band = self._cost_start, self._cost_len, self._cost_band
        for k in range(len(band)):
            s, n = st[k], ln[k]
            if sign < 0:
                if s > r:
                    st[k] = s - 1
                elif r - s < n:
                    band[k, r - s : n - 1] = band[k, r - s + 1 : n].copy()
                    band[k, n - 1] = np.inf
                    ln[k] = n - 1
            else:
                if s >= r:
                    st[k] = s + 1
                elif r - s < n:
                    m = min(n + 1, band.shape[1])
                    band[k, r - s + 1 : m] = band[k, r - s : m - 1].copy()
                    band[k, r - s] = np.inf
                    ln[k] = m

    def _cost_row(self, y, x0, x1):
        xs = np.arange(x0, x1 + 1)
        K = len(self._cost_band)
        sl = xs % K
        k = y - self._cost_start[sl]
        ok = (
            (xs > self.input_index - K)
            & (xs <= self.input_index)
            & (k >= 0)
            & (k < self._cost_len[sl])
        )
        out = np.full(len(xs), np.inf)
        out[ok] = self._cost_band[sl[ok], k[ok]]
        return out

    def _cost_col(self, x, y0, y1):
        out = np.full(y1 - y0 + 1, np.inf)
        K = len(self._cost_band)
        if x > self.input_index or x <= self.input_index - K or x < 0:
            return out
        sl = x % K
        s, n = self._cost_start[sl], self._cost_len[sl]
        a, b = max(y0, s), min(y1, s + n - 1)
        if a <= b:
            out[a - y0 : b - y0 + 1] = self._cost_band[sl, a - s : b - s + 1]
        return out

    def _cost(self, x, y):
        return float(self._cost_col(x, y, y)[0])

    def _update_runs(self, move: int) -> None:
        if move == self._run_prev:
            self._run_count += 1
        else:
            self._run_count, self._run_prev = 1, move

    def _get_inc(self, x, y):
        M = self.max_run_count
        if M is not None and self._run_count > M:
            return _ADV_SC if self._run_prev == _ADV_IN else _ADV_IN
        c = self._window_size
        best = self._cost(x, y) / (x + y + 1.0)
        bx, by = x, y
        x0, y0 = max(0, x - c + 1), max(0, y - c + 1)
        r = self._cost_row(y, x0, x) / (np.arange(x0, x + 1) + y + 1.0)
        k = int(np.argmin(r))
        if r[k] < best:
            best, bx, by = r[k], x0 + k, y
        cl = self._cost_col(x, y0, y) / (x + np.arange(y0, y + 1) + 1.0)
        k = int(np.argmin(cl))
        if cl[k] < best:
            best, bx, by = cl[k], x, y0 + k
        if bx < x:
            return _ADV_SC
        if by < y:
            return _ADV_IN
        return _BOTH

    def _strategy1(self, T, y):
        b = 50 if self._n_bf % 5 == 4 else 10
        self._n_bf += 1
        x, target = T, y - b
        for x, y in self.tempo_backward_path(T, y):
            if y <= target:
                break
        self.tempo_note_path_end(x, y)
        J = self._last_window_end - 1
        self._run_count, self._run_prev = 0, None
        while x < T and y < J:
            if self._get_inc(x, y) != _ADV_SC:
                if not np.isfinite(self._cost(x + 1, y)) and not np.isfinite(
                    self._cost(x + 1, y + 1)
                ):
                    break
                x += 1
            if x <= T and y < J and self._get_inc(x, y) != _ADV_IN:
                y += 1
            if not np.isfinite(self._cost(x, y)):
                break
            if self.max_run_count is not None:
                self._update_runs(self._get_inc(x, y))
        return y

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
            backpointers = np.zeros(window_end - window_start, dtype=np.int8)

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
            self._backpointers[self.input_index] = (window_start, backpointers)
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
