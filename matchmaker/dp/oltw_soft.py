#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Soft Online Time Warping (Soft-OLTW) for Score Following

Uses soft-minimum instead of hard minimum for smoother alignment.
Includes velocity-adaptive directional weighting to prevent racing
on repeated harmonic patterns, and Kalman-based tempo tracking.

References:
- Cuturi & Blondel (2017), "Soft-DTW"
- Weighted Soft-DTW: groupmm/weightedSDTW (ICASSP 2024)
"""

import time
from queue import Empty
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import progressbar
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END
from matchmaker.utils import distances
from matchmaker.utils.distances import Metric, vdist
from matchmaker.utils.errors import MatchmakerInvalidParameterTypeError
from matchmaker.utils.misc import set_latency_stats

STEP_SIZE: int = 3
START_WINDOW_SIZE: Union[float, int] = 0.1
DEFAULT_GAMMA: float = 1.0
WINDOW_SIZE: int = 10

# Prior / Kalman defaults
DEFAULT_PRIOR_LAMBDA: float = 0.05
DEFAULT_PRIOR_SIGMA: float = 50.0
DEFAULT_PROCESS_VAR: float = 0.5
DEFAULT_OBS_VAR: float = 5.0


# ---------------------------------------------------------------------------
# Core loop functions
# ---------------------------------------------------------------------------


def softmin(a: float, b: float, c: float, gamma: float) -> float:
    """Soft minimum via log-sum-exp. gamma→0 gives hard min."""
    min_val = min(a, b, c)
    if min_val == np.inf:
        return np.inf
    exp_sum = (
        np.exp(-(a - min_val) / gamma)
        + np.exp(-(b - min_val) / gamma)
        + np.exp(-(c - min_val) / gamma)
    )
    return min_val - gamma * np.log(exp_sum + 1e-10)


def soft_oltw_loop(
    global_cost_matrix,
    window_cost,
    window_start,
    window_end,
    input_index,
    min_costs,
    min_index,
    gamma=DEFAULT_GAMMA,
):
    """Soft-OLTW main loop (unweighted)."""
    N = global_cost_matrix.shape[0]
    idx = 0
    score_index = window_start

    if score_index == input_index == 0:
        global_cost_matrix[1, 1] = np.sum(window_cost)
        min_costs = global_cost_matrix[1, 1]
        min_index = 0

    while score_index < window_end:
        if not (score_index == input_index == 0):
            local_dist = window_cost[idx]
            dist1 = global_cost_matrix[score_index, 1] + local_dist
            dist2 = global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = global_cost_matrix[score_index, 0] + local_dist
            soft_min_dist = softmin(dist1, dist2, dist3, gamma)
            global_cost_matrix[score_index + 1, 1] = soft_min_dist
            norm_cost = soft_min_dist / (input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index
        idx += 1
        score_index += 1

    for i in range(N):
        global_cost_matrix[i, 0] = global_cost_matrix[i, 1]
        global_cost_matrix[i, 1] = np.inf

    return global_cost_matrix, min_index, min_costs


def weighted_soft_oltw_loop(
    global_cost_matrix,
    window_cost,
    window_start,
    window_end,
    input_index,
    min_costs,
    min_index,
    gamma=DEFAULT_GAMMA,
    w_vertical=1.0,
    w_horizontal=1.0,
    w_diagonal=1.0,
):
    """Weighted Soft-OLTW: directional weights on local cost."""
    N = global_cost_matrix.shape[0]
    idx = 0
    score_index = window_start

    if score_index == input_index == 0:
        global_cost_matrix[1, 1] = np.sum(window_cost)
        min_costs = global_cost_matrix[1, 1]
        min_index = 0

    while score_index < window_end:
        if not (score_index == input_index == 0):
            local_dist = window_cost[idx]
            dist1 = global_cost_matrix[score_index, 1] + w_vertical * local_dist
            dist2 = global_cost_matrix[score_index + 1, 0] + w_horizontal * local_dist
            dist3 = global_cost_matrix[score_index, 0] + w_diagonal * local_dist
            soft_min_dist = softmin(dist1, dist2, dist3, gamma)
            global_cost_matrix[score_index + 1, 1] = soft_min_dist
            norm_cost = soft_min_dist / (input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index
        idx += 1
        score_index += 1

    for i in range(N):
        global_cost_matrix[i, 0] = global_cost_matrix[i, 1]
        global_cost_matrix[i, 1] = np.inf

    return global_cost_matrix, min_index, min_costs


# ---------------------------------------------------------------------------
# Base Soft-OLTW class
# ---------------------------------------------------------------------------


class OnlineTimeWarpingSoftBase(OnlineAlignment):
    """Base Soft-OLTW (internal). Use SoftOnlineTimeWarping instead."""

    DEFAULT_DISTANCE_FUNC: str = "Manhattan"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        gamma: float = DEFAULT_GAMMA,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: int = START_WINDOW_SIZE,
        current_position: int = 0,
        frame_rate: int = FRAME_RATE,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(reference_features=reference_features)

        self._ref_frame_to_beat = kwargs.get("ref_frame_to_beat", None)
        self.gamma = gamma
        self.input_features: List[NDArray[np.float32]] = None
        self.queue = queue or RECVQueue()

        # Distance function setup
        if isinstance(distance_func, str):
            self.distance_func = getattr(distances, distance_func)()
        elif isinstance(distance_func, tuple):
            self.distance_func = getattr(distances, distance_func[0])(
                **distance_func[1]
            )
        elif callable(distance_func):
            self.distance_func = distance_func
        else:
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="distance_func",
                required_parameter_type=(str, tuple, Callable),
                actual_parameter_type=type(distance_func),
            )

        if isinstance(self.distance_func, Metric):
            self.vdist = vdist
        else:
            self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(
                np.float32
            )

        # State
        self.N_ref: int = self.reference_features.shape[0]
        self.frame_rate = frame_rate
        self.window_size: int = window_size * self.frame_rate
        self.step_size: int = step_size
        self.start_window_size: int = int(np.round(start_window_size * frame_rate))
        self.init_position: int = current_position
        self.current_position: int = current_position
        self.positions: List[int] = []
        self._warping_path: List = []
        self.global_cost_matrix: NDArray[np.float32] = (
            np.ones((reference_features.shape[0] + 1, 2), dtype=np.float32) * np.inf
        )
        self.input_index: int = 0
        self.is_following: bool = False
        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

    @property
    def current_beat(self) -> float:
        if self._ref_frame_to_beat is not None:
            idx = min(self.current_position, len(self._ref_frame_to_beat) - 1)
            return float(self._ref_frame_to_beat[idx])
        return float(self.current_position)

    @property
    def warping_path(self) -> NDArray[np.float32]:
        return np.array(self._warping_path).T

    def __call__(self, input: NDArray[np.float32]) -> int:
        self.step(input)
        return self.current_position

    def run(self, verbose: bool = True) -> Generator[int, None, NDArray[np.float32]]:
        self.reset()
        if verbose:
            pbar = progressbar.ProgressBar(max_value=self.N_ref, redirect_stdout=True)
        while self.is_still_following():
            try:
                item = self.queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                break
            if item is STREAM_END:
                break
            features, f_time = item
            self.last_queue_update = time.time()
            self.input_features = (
                np.concatenate((self.input_features, features))
                if self.input_features is not None
                else features
            )
            self.step(features)
            if verbose:
                pbar.update(int(self.current_position))
            latency = time.time() - self.last_queue_update
            self.latency_stats = set_latency_stats(
                latency, self.latency_stats, self.input_index
            )
            yield self.current_beat
        if verbose:
            pbar.finish()
        return self.warping_path

    def is_still_following(self):
        return self.current_position <= self.N_ref - 2

    def reset(self) -> None:
        self.current_position = self.init_position
        self.positions = []
        self._warping_path = []
        self.global_cost_matrix = (
            np.ones((self.reference_features.shape[0] + 1, 2), dtype=np.float32)
            * np.inf
        )
        self.input_index = 0

    def get_window(self) -> Tuple[int, int]:
        w_size = self.window_size
        if self.window_index < self.start_window_size:
            w_size = self.start_window_size
        window_start = max(self.window_index - w_size, 0)
        window_end = min(self.window_index + w_size, self.N_ref)
        return window_start, window_end

    @property
    def window_index(self) -> int:
        return self.current_position

    def step(self, input_features: NDArray[np.float32]) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Kalman filter for position-tempo tracking
# ---------------------------------------------------------------------------


class PositionTempoKalman:
    """2D Kalman filter tracking (position, tempo)."""

    def __init__(
        self,
        init_position=0.0,
        init_tempo=1.0,
        process_var=DEFAULT_PROCESS_VAR,
        obs_var=DEFAULT_OBS_VAR,
    ):
        self.state = np.array([init_position, init_tempo], dtype=np.float64)
        self.P = np.diag([10.0, 1.0])
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Q = np.array(
            [[process_var * 0.25, process_var * 0.5], [process_var * 0.5, process_var]]
        )
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[obs_var]])

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()

    def update(self, observed_position: float):
        z = np.array([observed_position])
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.state[1] = np.clip(self.state[1], 0.1, 10.0)
        return self.state.copy()

    @property
    def position(self):
        return self.state[0]

    @property
    def tempo(self):
        return self.state[1]

    @property
    def position_uncertainty(self):
        return np.sqrt(self.P[0, 0])

    def reset(self, position=0.0, tempo=1.0):
        self.state = np.array([position, tempo], dtype=np.float64)
        self.P = np.diag([10.0, 1.0])


# ---------------------------------------------------------------------------
# Main class: Soft-OLTW with velocity-adaptive directional weighting
# ---------------------------------------------------------------------------


class SoftOnlineTimeWarping(OnlineTimeWarpingSoftBase):
    """
    Soft Online Time Warping with velocity-adaptive directional weighting.

    Key features:
    - Soft-min DTW with configurable temperature (gamma)
    - Velocity-adaptive w_horizontal: penalizes score-direction advance
      when the tracker races ahead of expected velocity
    - Uninformative frame detection: freezes position on silence/noise
    - Optional Kalman prior on local cost (disabled by default)
    - Adaptive Kalman observation noise based on cost informativeness
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        gamma: float = DEFAULT_GAMMA,
        distance_func: Union[str, Callable, Tuple[str, Dict[str, Any]]] = "Manhattan",
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        current_position: int = 0,
        frame_rate: int = FRAME_RATE,
        queue: Optional[RECVQueue] = None,
        prior_lambda: float = DEFAULT_PRIOR_LAMBDA,
        prior_sigma: float = DEFAULT_PRIOR_SIGMA,
        process_var: float = DEFAULT_PROCESS_VAR,
        obs_var: float = DEFAULT_OBS_VAR,
        w_horizontal: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            window_size=window_size,
            step_size=step_size,
            gamma=gamma,
            distance_func=distance_func,
            start_window_size=start_window_size,
            current_position=current_position,
            frame_rate=frame_rate,
            queue=queue,
            **kwargs,
        )
        self.prior_lambda = prior_lambda
        self.prior_sigma = prior_sigma
        self.kalman = PositionTempoKalman(
            init_position=0.0,
            init_tempo=1.0,
            process_var=process_var,
            obs_var=obs_var,
        )
        self._obs_var = obs_var
        self.w_horizontal = w_horizontal
        self._pos_history = []
        self._music_started = False

    def reset(self) -> None:
        super().reset()
        self.kalman.reset(position=0.0, tempo=1.0)
        self._pos_history = []

    def step(self, input_features: NDArray[np.float32]) -> None:
        feat = input_features.squeeze()

        # 0. Uninformative frame detection (silence/noise)
        feat_abs = np.abs(feat)
        peakiness = float(np.max(feat_abs)) / (float(np.mean(feat_abs)) + 1e-10)
        is_uninformative = peakiness < 2.0

        # Skip DTW on pre-music silence only (once music starts, never skip).
        self._real_frame = getattr(self, "_real_frame", 0)
        if not self._music_started:
            if is_uninformative:
                self._pos_history.append(self.current_position)
                self._warping_path.append((self.current_beat, self._real_frame))
                self._real_frame += 1
                return
            self._music_started = True

        # 1. Kalman prediction
        predicted_state = self.kalman.predict()
        predicted_pos = predicted_state[0]
        pos_uncertainty = self.kalman.position_uncertainty
        effective_sigma = self.prior_sigma + pos_uncertainty

        # 2. Compute window and audio costs
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)
        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            feat,
            self.distance_func,
        )

        # 3. Cost informativeness
        cost_range = (
            float(np.max(window_cost) - np.min(window_cost))
            if len(window_cost) > 1
            else 0.0
        )
        cost_med = float(np.median(window_cost)) if len(window_cost) > 0 else 1.0
        informativeness = cost_range / (cost_med + 1e-10)

        # 4. Cost-adaptive Gaussian prior
        warmup_done = self.input_index > 30
        kalman_confident = pos_uncertainty < effective_sigma * 0.5
        if (
            self.prior_lambda > 0
            and warmup_done
            and kalman_confident
            and informativeness > 0.5
        ):
            for idx in range(len(window_cost)):
                score_idx = window_start + idx
                deviation = score_idx - predicted_pos
                prior_cost = (deviation**2) / (2.0 * effective_sigma**2)
                window_cost[idx] += self.prior_lambda * prior_cost

        # 5. Tempo-adaptive directional weighting
        # expected_tempo = 1.0 (score and performance at same frame rate).
        # TODO: use score BPM marking for better initialization
        expected_tempo = 1.0
        recent_tempo = expected_tempo
        if len(self._pos_history) >= 10:
            lookback = min(30, len(self._pos_history))
            recent_tempo = (
                self.current_position - self._pos_history[-lookback]
            ) / lookback
        racing_amount = np.clip(
            (recent_tempo - expected_tempo) / max(expected_tempo * 0.35, 0.12), 0.0, 1.0
        )
        effective_wh = 1.0 + (self.w_horizontal - 1.0) * racing_amount

        self.global_cost_matrix, min_index, min_costs = weighted_soft_oltw_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
            gamma=self.gamma,
            w_vertical=1.0,
            w_horizontal=effective_wh,
            w_diagonal=1.0,
        )

        # 6. Update position
        if self.input_index == 0:
            self.current_position = min(
                max(self.current_position, min_index),
                self.current_position,
            )
        else:
            self.current_position = min(
                max(self.current_position, min_index),
                self.current_position + self.step_size,
            )
        self._pos_history.append(self.current_position)

        # 7. Kalman update with adaptive observation noise
        adaptive_R = self._obs_var * np.clip(2.0 / (informativeness + 0.5), 0.2, 10.0)
        self.kalman.R = np.array([[adaptive_R]])
        self.kalman.update(float(self.current_position))

        self._warping_path.append((self.current_beat, self._real_frame))
        self.input_index += 1
        self._real_frame += 1
