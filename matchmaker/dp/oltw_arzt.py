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
  OnlineTimeWarpingArztTempoFrame — Arzt, Widmer & Dixon (2008) tracker with the
                        tempo model of Arzt & Widmer (2010); built on
                        OnlineTimeWarpingDixonFrame, not on the classes above
  OnlineTimeWarpingArztEvent — event-level variant for MIDI (onset-by-onset)
"""

import time
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.dp.oltw_dixon import Direction, OnlineTimeWarpingDixonFrame
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
        self.reset()

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


class OnlineTimeWarpingArztTempoFrame(OnlineTimeWarpingDixonFrame):
    """Frame-level OLTW with the simple tempo model of Arzt & Widmer (2010).

    The tracker is the one Arzt & Widmer (2010) build on, Dixon's (2005) on-line
    time warping with the changes of Arzt, Widmer & Dixon (2008, Sec. 3.2):
    straight steps weighted 1.3 and diagonal steps 2, MaxRunCount 6, and an
    initial diagonal phase of 1 s instead of the search width c (here
    ``window_size`` seconds, 500 frames at the default 10 s and 50 fps). The
    path cost is normalised by ``1 + i + j`` as in :class:`OnlineTimeWarpingDixonFrame`.

    Tempo model (Arzt & Widmer, 2010, Sec. 4): after every incoming frame and
    before the path computation, the relative tempo t is estimated from the
    backward path through the current position. The score time of each onset
    is taken on the notated reference (not the stretched one), so t stays the
    tempo relative to the score. The ``tempo_n_onsets`` most recent onsets at
    least ``tempo_min_past`` seconds in the past each give the slope of the
    onset-rectified path over a ``tempo_window_size``-second window, and t is
    their linearly weighted mean (Eq. 1). With probability 1 - 1/t (t > 1) the
    score frames ps+1 and ps+2 are replaced by their mean; with probability
    1 - t (t < 1) the mean of ps and ps+1 is inserted between them. ps is the
    last processed score frame of the forward path, so only frames not yet in
    the cost matrix change. An insertion that would duplicate an onset frame is
    postponed to the next frame; at most 3 alterations happen in a row. The
    model starts once the initial diagonal phase is over.

    Differences from the papers: the features are the library's ``lse``
    (Dixon 2005: 84 linear-log bins, half-wave rectified difference, then L2
    normalisation per frame) at 50 fps with Euclidean distance. Arzt et al.
    (2008) normalise the spectrum to sum to 1 before the difference; on
    synthesised score audio against piano recordings that variant runs ahead of
    the performance and loses most pieces, so it is not used. The
    backward-forward and onset strategies of Arzt et al. (2008) are not
    implemented. With ``use_tempo_model=False`` this is the 2008 tracker alone.

    This class does not share code with :class:`OnlineTimeWarpingArztFrame`
    (the ``arzt`` method), which follows a different on-line DTW.
    """

    # Predecessor offsets and step weights (Arzt, Widmer & Dixon, 2008, Eq. 1).
    STEP_WEIGHTS = {
        Direction.BOTH: ((-1, -1), 2.0),
        Direction.REF: ((-1, 0), 1.3),
        Direction.INPUT: ((0, -1), 1.3),
    }
    # Initial phase in which both sequences advance together (2008: 1 s).
    INIT_SECONDS: float = 1.0

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        ref_frame_to_beat: NDArray = None,
        max_run_count: int = 6,
        tempo_window_size: float = 3.0,
        tempo_min_past: float = 1.0,
        tempo_n_onsets: int = 20,
        random_seed: Optional[int] = 1984,
        use_tempo_model: bool = True,
        **kwargs,
    ) -> None:
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Arzt tempo requires `ref_frame_to_beat` "
                "(per-frame beat mapping)."
            )
        self.use_tempo_model = use_tempo_model
        self.tempo_window_size = tempo_window_size
        self.tempo_min_past = tempo_min_past
        self.tempo_n_onsets = tempo_n_onsets
        self.random_seed = random_seed
        self._original_reference_features = np.array(reference_features, copy=True)
        self._original_ref_frame_to_beat = np.array(ref_frame_to_beat, copy=True)
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            ref_frame_to_beat=ref_frame_to_beat,
            max_run_count=max_run_count,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        # The tempo model stretches the reference on the fly; restore the
        # pristine copy so each run starts from the notated score.
        self.reference_features = self._original_reference_features.copy()
        self._ref_frame_to_beat = self._original_ref_frame_to_beat.copy()
        self.N_ref = self.reference_features.shape[0]
        onset_frames = np.searchsorted(self._ref_frame_to_beat, self.score_positions)
        self.is_onset_frame = np.zeros(self.N_ref, dtype=bool)
        self.is_onset_frame[onset_frames[onset_frames < self.N_ref]] = True
        self._backpointers: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._rng = np.random.RandomState(self.random_seed)
        self._consecutive_alterations: int = 0
        self._insert_pending: bool = False
        self.current_relative_tempo: float = 1.0
        self.n_inserted = 0
        self.n_deleted = 0

    # ---- tracker (Arzt, Widmer & Dixon, 2008) ----

    def get_inc(self):
        """GetInc with the 1 s initial phase of Arzt et al. (2008)."""
        current_ref = self.ref_pointer - 1
        current_input = self.input_pointer - 1
        best_ref, best_input = self._update_best_alignment()
        init = int(self.INIT_SECONDS * self.frame_rate)
        if self.ref_pointer < init or self.input_pointer < init:
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

    def evaluate_path_cost(self, ref_idx, input_idx, local_dist):
        """DTW recursion as in the parent, also recording the chosen predecessor."""
        if ref_idx == 0 and input_idx == 0:
            self.accumulated_costs[(0, 0)] = local_dist
            return
        best, best_offset = None, None
        for offset, weight in self.STEP_WEIGHTS.values():
            predecessor = self.accumulated_costs.get(
                (ref_idx + offset[0], input_idx + offset[1])
            )
            if predecessor is None:
                continue
            cost = predecessor + weight * local_dist
            if best is None or cost <= best:
                best, best_offset = cost, offset
        if best is not None:
            self.accumulated_costs[(ref_idx, input_idx)] = best
            self._backpointers[(ref_idx, input_idx)] = best_offset

    def _prune(self):
        super()._prune()
        cutoff = self.ref_pointer - 2 * self.w
        if cutoff > 0:
            self._backpointers = {
                k: v for k, v in self._backpointers.items() if k[0] >= cutoff
            }

    def _get_backward_path(self, max_history_frames: int) -> list:
        """Backward path from the current position, oldest cell first."""
        ref_idx, input_idx = self.best_ref, self.best_input
        min_input = max(0, input_idx - max_history_frames)
        path = [(ref_idx, input_idx)]
        while input_idx > min_input:
            offset = self._backpointers.get((ref_idx, input_idx))
            if offset is None:
                break
            ref_idx, input_idx = ref_idx + offset[0], input_idx + offset[1]
            path.append((ref_idx, input_idx))
        path.reverse()
        return path

    # ---- tempo model (Arzt & Widmer, 2010, Sec. 4) ----

    def _compute_relative_tempo(self) -> float:
        """Relative tempo from the onset-rectified backward path (Sec. 4.1)."""
        fr = self.frame_rate
        if self.input_pointer < int(fr * 1.5):
            return 1.0
        cur_perf_time = (self.input_pointer - 1) / fr
        path = self._get_backward_path(int(fr * 10))
        if len(path) < 2:
            return 1.0
        score_frames = np.array([p[0] for p in path])
        perf_frames = np.array([p[1] for p in path])
        s_min, s_max = int(score_frames[0]), int(score_frames[-1])
        if s_max <= s_min:
            return 1.0
        local_onsets = np.where(self.is_onset_frame[s_min : s_max + 1])[0] + s_min
        if len(local_onsets) < 2:
            return 1.0
        # Score time on the notated reference, so t is the tempo relative to the
        # score rather than what is left after the alterations already made.
        original = self._original_ref_frame_to_beat
        idx = np.searchsorted(score_frames, local_onsets)
        onset_beats = self._ref_frame_to_beat[local_onsets]
        t_s = np.interp(onset_beats, original, np.arange(len(original))) / fr
        t_p = perf_frames[np.minimum(idx, len(perf_frames) - 1)] / fr
        valid = np.where(t_p <= cur_perf_time - self.tempo_min_past)[0]
        if len(valid) == 0:
            return 1.0
        valid = valid[-self.tempo_n_onsets :]
        half = self.tempo_window_size / 2.0
        tempi = []
        for k in valid:
            w_start, w_end = t_s[k] - half, t_s[k] + half
            if w_start >= t_s[0] and w_end <= t_s[-1]:
                delta_p = float(np.interp(w_end, t_s, t_p)) - float(
                    np.interp(w_start, t_s, t_p)
                )
                tempi.append((w_end - w_start) / delta_p if delta_p > 1e-4 else 1.0)
        if not tempi:
            return 1.0
        weights = np.arange(1, len(tempi) + 1)
        t = np.sum(np.array(tempi) * weights) / np.sum(weights)
        return float(np.clip(t, 0.33, 3.0))

    def _alter_score_representation(self, t: float) -> None:
        """Stretch or compress the score ahead of the forward path (Sec. 4.2)."""
        ps = self.ref_pointer - 1  # last processed frame of the score representation
        if ps < 0 or ps + 2 >= self.N_ref:
            return
        if t >= 1.0:
            self._insert_pending = False
        if self._consecutive_alterations >= 3:
            self._consecutive_alterations = 0
            return
        r = self._rng.uniform(0.0, 1.0)
        if t > 1.0 and r > 1.0 / t:
            features = self.reference_features
            beats, onset = self._ref_frame_to_beat, self.is_onset_frame
            features[ps + 1] = 0.5 * (features[ps + 1] + features[ps + 2])
            # the merged frame keeps an onset's beat and flag
            if onset[ps + 2] and not onset[ps + 1]:
                beats[ps + 1] = beats[ps + 2]
            elif not onset[ps + 1]:
                beats[ps + 1] = 0.5 * (beats[ps + 1] + beats[ps + 2])
            onset[ps + 1] |= onset[ps + 2]
            self.reference_features = np.delete(features, ps + 2, axis=0)
            self._ref_frame_to_beat = np.delete(beats, ps + 2)
            self.is_onset_frame = np.delete(onset, ps + 2)
            self.N_ref -= 1
            self.n_deleted += 1
            self._consecutive_alterations += 1
        elif t < 1.0 and (self._insert_pending or r > t):
            # onset vectors are not duplicated: postpone to the next frame
            if self.is_onset_frame[ps] or self.is_onset_frame[ps + 1]:
                self._insert_pending = True
                self._consecutive_alterations = 0
                return
            self._insert_pending = False
            mean = 0.5 * (self.reference_features[ps] + self.reference_features[ps + 1])
            self.reference_features = np.insert(
                self.reference_features, ps + 1, mean, axis=0
            )
            beat = 0.5 * (self._ref_frame_to_beat[ps] + self._ref_frame_to_beat[ps + 1])
            self._ref_frame_to_beat = np.insert(self._ref_frame_to_beat, ps + 1, beat)
            self.is_onset_frame = np.insert(self.is_onset_frame, ps + 1, False)
            self.N_ref += 1
            self.n_inserted += 1
            self._consecutive_alterations += 1
        else:
            self._consecutive_alterations = 0

    def step(self, input_features: NDArray[np.float32]) -> None:
        # after every incoming frame and before the path computation, once
        # GetInc steers the forward path
        if (
            self.use_tempo_model
            and self._initialized
            and self.input_pointer >= int(self.INIT_SECONDS * self.frame_rate)
        ):
            self.current_relative_tempo = self._compute_relative_tempo()
            self._alter_score_representation(self.current_relative_tempo)
        super().step(input_features)


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
        self.reset()

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
