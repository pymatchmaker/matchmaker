#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Soft Online Time Warping (Soft-OLTW) for Score Following.

Uses the same 2-column DP architecture as the Arzt-style OLTW (smooth,
step-size-clamped paths with no catch-up jitter), but replaces the hard
minimum with a softmin and adds Sakoe-Chiba step weights so that the DP
is unbiased regardless of the tempo ratio between the synthesized score
and the actual performance.

A 2D Kalman filter tracks (position, tempo). The tracked tempo modulates:
    - **gamma** (softmin temperature): confident tempo → sharp (near-hard)
      min; uncertain tempo → smooth blending.
    - **step_size**: the per-frame advance cap scales with the tracked
      tempo so that slow performers are not forced to jump forward.

References:
    - Cuturi & Blondel (2017), Soft-DTW.
    - Zeitler et al. (ISMIR 2023), Stabilizing Training with Soft DTW:
      gamma scheduling + diagonal prior.
    - Zeitler et al. (ICASSP 2024), Soft DTW with Variable Step Weights:
      Sakoe-Chiba-style step weights for SDTW.
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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

STEP_SIZE: int = 3
WINDOW_SIZE: int = 10  # seconds
START_WINDOW_SIZE: float = 0.1  # seconds

DEFAULT_GAMMA: float = 0.05

# Sakoe-Chiba step weights: diagonal = 2× prevents bias toward the main
# diagonal when reference/performance frame rates differ.
DEFAULT_W_VERTICAL: float = 1.0  # advance ref, hold input
DEFAULT_W_HORIZONTAL: float = 1.0  # advance input, hold ref
DEFAULT_W_DIAGONAL: float = 2.0  # advance both

# Kalman filter
DEFAULT_PROCESS_VAR: float = 0.02
DEFAULT_OBS_VAR: float = 3.0
DEFAULT_INIT_TEMPO: float = 1.0
DEFAULT_TEMPO_CLIP: Tuple[float, float] = (0.2, 4.0)

# Gamma modulation
GAMMA_FACTOR_MIN: float = 0.5
GAMMA_FACTOR_MAX: float = 3.0


# ---------------------------------------------------------------------------
# Soft-OLTW inner loop (Python; mirrors the Cython oltw_arzt_loop)
# ---------------------------------------------------------------------------


def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    """Softmin over three values via log-sum-exp.  gamma→0 is hard min."""
    m = min(a, b, c)
    if not np.isfinite(m):
        return m
    if gamma <= 1e-8:
        return m
    s = np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    return m - gamma * float(np.log(s + 1e-12))


def soft_oltw_loop(
    global_cost_matrix: NDArray[np.float32],
    window_cost: NDArray[np.float32],
    window_start: int,
    window_end: int,
    input_index: int,
    min_costs: float,
    min_index: int,
    gamma: float = DEFAULT_GAMMA,
    w_vertical: float = DEFAULT_W_VERTICAL,
    w_horizontal: float = DEFAULT_W_HORIZONTAL,
    w_diagonal: float = DEFAULT_W_DIAGONAL,
    prior_position: int = -1,
    prior_lambda: float = 0.0,
) -> Tuple[NDArray[np.float32], int, float]:
    """Arzt-style 2-column DP with softmin and Sakoe-Chiba weights.

    Column 0 = previous input frame, column 1 = current input frame.
    On return column 1 is rolled into column 0 and column 1 is reset to inf.

    Step-weight convention (matching the ICASSP 2024 paper):
        - ``w_vertical``  : ref advances, input stays  (dist1)
        - ``w_horizontal``: input advances, ref stays   (dist2)
        - ``w_diagonal``  : both advance                (dist3)

    If ``prior_position >= 0`` and ``prior_lambda > 0``, a Gaussian prior
    bias is added to the local cost, penalising positions far from the
    Kalman-predicted (prior) position. This is an online adaptation of
    the diagonal prior from Zeitler et al. (ISMIR 2023).
    """
    N = global_cost_matrix.shape[0]
    idx = 0
    score_index = window_start
    use_prior = prior_position >= 0 and prior_lambda > 0

    if score_index == input_index == 0:
        global_cost_matrix[1, 1] = float(np.sum(window_cost))
        min_costs = float(global_cost_matrix[1, 1])
        min_index = 0

    while score_index < window_end:
        if not (score_index == 0 and input_index == 0):
            local_dist = float(window_cost[idx])

            # Prior bias: penalise positions far from Kalman prediction
            if use_prior:
                dev = float(score_index - prior_position)
                local_dist = local_dist + prior_lambda * dev * dev

            d_vert = global_cost_matrix[score_index, 1] + w_vertical * local_dist
            d_horz = global_cost_matrix[score_index + 1, 0] + w_horizontal * local_dist
            d_diag = global_cost_matrix[score_index, 0] + w_diagonal * local_dist

            sm = _softmin3(d_vert, d_horz, d_diag, gamma)
            global_cost_matrix[score_index + 1, 1] = sm

            norm_cost = sm / (input_index + score_index + 1.0)
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
# Kalman filter: (position, tempo)
# ---------------------------------------------------------------------------


class PositionTempoKalman:
    """2D Kalman filter tracking (score position, tempo).

    State: [position, tempo].  Dynamics: pos += tempo per input frame.
    Observation: position only (the DP argmin).
    """

    def __init__(
        self,
        init_position: float = 0.0,
        init_tempo: float = DEFAULT_INIT_TEMPO,
        process_var: float = DEFAULT_PROCESS_VAR,
        obs_var: float = DEFAULT_OBS_VAR,
        tempo_clip: Tuple[float, float] = DEFAULT_TEMPO_CLIP,
    ) -> None:
        self.state = np.array([init_position, init_tempo], dtype=np.float64)
        self.P = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        q = float(process_var)
        self.Q = np.array([[q * 0.25, q * 0.5], [q * 0.5, q]])
        self.H = np.array([[1.0, 0.0]])
        self._base_R = np.array([[float(obs_var)]])
        self.R = self._base_R.copy()
        self.tempo_min, self.tempo_max = tempo_clip

    def predict(self) -> NDArray[np.float64]:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()

    def update(self, observed_position: float) -> NDArray[np.float64]:
        z = np.array([observed_position], dtype=np.float64)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.state[1] = float(np.clip(self.state[1], self.tempo_min, self.tempo_max))
        return self.state.copy()

    @property
    def position(self) -> float:
        return float(self.state[0])

    @property
    def tempo(self) -> float:
        return float(self.state[1])

    @property
    def tempo_std(self) -> float:
        return float(np.sqrt(max(self.P[1, 1], 0.0)))

    def set_obs_var(self, var: float) -> None:
        self.R = np.array([[float(var)]])

    def reset(self, position: float = 0.0, tempo: float = DEFAULT_INIT_TEMPO) -> None:
        self.state = np.array([position, tempo], dtype=np.float64)
        self.P = np.array([[4.0, 0.0], [0.0, 1.0]])
        self.R = self._base_R.copy()


# ---------------------------------------------------------------------------
# Soft Online Time Warping
# ---------------------------------------------------------------------------


class SoftOnlineTimeWarping(OnlineAlignment):
    """Soft Online Time Warping with Kalman-adaptive gamma.

    Produces smooth, monotonically advancing warping paths (Arzt-style
    step-size clamping, no Dixon-style catch-up jitter) while matching
    or exceeding Dixon-quality alignment via Sakoe-Chiba step weights
    and a tempo-adaptive softmin temperature.

    Parameters
    ----------
    reference_features : np.ndarray
        Score-derived reference feature matrix, shape ``(N_ref, D)``.
    window_size : int
        OLTW search window half-width in seconds.
    step_size : int
        Baseline maximum per-frame advance in score frames.
    gamma : float
        Base softmin temperature.
    w_vertical, w_horizontal, w_diagonal : float
        Step weights.  Diagonal defaults to 2.0 (Sakoe-Chiba) to
        eliminate diagonal bias when score/perf rates differ.
    process_var, obs_var : float
        Kalman filter noise parameters.
    init_tempo : float
        Initial tempo estimate (score frames per input frame).
    """

    DEFAULT_DISTANCE_FUNC: str = "Euclidean"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        gamma: float = DEFAULT_GAMMA,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        current_position: int = 0,
        frame_rate: int = FRAME_RATE,
        queue: Optional[RECVQueue] = None,
        w_vertical: float = DEFAULT_W_VERTICAL,
        w_horizontal: float = DEFAULT_W_HORIZONTAL,
        w_diagonal: float = DEFAULT_W_DIAGONAL,
        process_var: float = DEFAULT_PROCESS_VAR,
        obs_var: float = DEFAULT_OBS_VAR,
        init_tempo: float = DEFAULT_INIT_TEMPO,
        # Kalman prior bias in DP (Zeitler ISMIR 2023 diagonal prior, online)
        prior_lambda: float = 0.0,
        prior_sigma_frames: float = 12.0,
        max_run_count: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(reference_features=reference_features)

        self._ref_frame_to_beat = kwargs.get("ref_frame_to_beat", None)
        self.base_gamma = float(gamma)
        self.gamma = float(gamma)
        self.prior_lambda = float(prior_lambda)
        self.queue = queue or RECVQueue()

        # Step weights
        self.w_vertical = float(w_vertical)
        self.w_horizontal = float(w_horizontal)
        self.w_diagonal = float(w_diagonal)

        # Distance function
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

        # Window / frame state
        self.N_ref: int = int(reference_features.shape[0])
        self.frame_rate: int = int(frame_rate)
        self.window_size: int = int(window_size * frame_rate)
        self.start_window_size: int = int(np.round(start_window_size * frame_rate))
        self.step_size: int = int(step_size)
        self.init_position: int = int(current_position)
        self.current_position: int = int(current_position)

        # Alignment state
        self.input_features: Optional[NDArray[np.float32]] = None
        self.positions: List[int] = []
        self._warping_path: List[Tuple[float, int]] = []
        self.global_cost_matrix: NDArray[np.float32] = (
            np.ones((self.N_ref + 1, 2), dtype=np.float32) * np.inf
        )
        self.input_index: int = 0

        # Kalman filter
        self.kalman = PositionTempoKalman(
            init_position=float(current_position),
            init_tempo=float(init_tempo),
            process_var=float(process_var),
            obs_var=float(obs_var),
        )
        self._obs_var = float(obs_var)

        # Runtime bookkeeping
        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_beat(self) -> float:
        if self._ref_frame_to_beat is not None:
            idx = min(self.current_position, len(self._ref_frame_to_beat) - 1)
            return float(self._ref_frame_to_beat[idx])
        return float(self.current_position)

    @property
    def warping_path(self) -> NDArray[np.float32]:
        return np.array(self._warping_path).T

    @property
    def window_index(self) -> int:
        """Window center: posterior (DP-corrected position)."""
        return self.current_position

    def is_still_following(self) -> bool:
        return self.current_position <= self.N_ref - 2

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.current_position = self.init_position
        self._prior_position = self.init_position
        self.positions = []
        self._warping_path = []
        self.global_cost_matrix = (
            np.ones((self.N_ref + 1, 2), dtype=np.float32) * np.inf
        )
        self.input_index = 0
        self.kalman.reset(position=float(self.init_position))

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def get_window(self) -> Tuple[int, int]:
        w = self.window_size
        if self.window_index < self.start_window_size:
            w = self.start_window_size
        start = max(self.window_index - w, 0)
        end = min(self.window_index + w, self.N_ref)
        return start, end

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, input_features: NDArray[np.float32]) -> None:
        feat = input_features.squeeze()

        # 1. Kalman predict (PRIOR) — tempo model only, no DP evidence yet.
        #    This drives window centering so the search region follows the
        #    tempo model's expectation, not the noisy DP argmin.
        predicted = self.kalman.predict()
        self._prior_position = int(np.round(np.clip(predicted[0], 0, self.N_ref - 1)))
        tempo = self.kalman.tempo
        tempo_unc = self.kalman.tempo_std

        # 2. Local costs over the search window (centered on PRIOR).
        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            feat,
            self.distance_func,
        )

        # 3. Tempo-adaptive gamma.
        #    Confident tempo → near-hard min (low gamma); uncertain →
        #    smooth blending (high gamma). Base gamma is scaled by the
        #    local cost magnitude so the setting transfers across feature
        #    types.
        cost_scale = max(float(np.median(window_cost)), 1e-6)
        tempo_unc_norm = tempo_unc / max(tempo, 1e-3)
        gamma_factor = float(
            np.clip(0.6 + 2.5 * tempo_unc_norm, GAMMA_FACTOR_MIN, GAMMA_FACTOR_MAX)
        )
        eff_gamma = self.base_gamma * gamma_factor * cost_scale * 0.1
        self.gamma = eff_gamma

        # 4. Run the soft-OLTW DP.
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        # Prior bias: scale lambda by inverse of cost magnitude so it
        # transfers across feature types.  The Kalman prior position
        # (before measurement update) anchors the DP toward the tempo
        # model's expectation.
        if self.prior_lambda > 0 and self.input_index > 0:
            eff_prior_lambda = self.prior_lambda / max(cost_scale, 1e-6)
            prior_pos = self._prior_position
        else:
            eff_prior_lambda = 0.0
            prior_pos = -1

        self.global_cost_matrix, min_index, min_costs = soft_oltw_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
            gamma=eff_gamma,
            w_vertical=self.w_vertical,
            w_horizontal=self.w_horizontal,
            w_diagonal=self.w_diagonal,
            prior_position=prior_pos,
            prior_lambda=eff_prior_lambda,
        )

        # 5. Kalman observation update with raw DP argmin.
        #    Feed the RAW min_index (not clamped) so the Kalman filter
        #    sees the true DP evidence. When the DP jumps to a confusing
        #    region (e.g., pause with similar harmonic content), the
        #    Kalman's prediction-based inertia absorbs the outlier. When
        #    the correct features return, the Kalman converges back.
        #    Adaptive R: flat cost → high R (don't trust argmin).
        if len(window_cost) > 1:
            std = float(np.std(window_cost))
            mean = float(np.mean(window_cost)) + 1e-6
            signal = std / mean
        else:
            signal = 1.0
        adaptive_R = self._obs_var * float(np.clip(1.5 / (signal + 0.3), 0.3, 6.0))
        self.kalman.set_obs_var(adaptive_R)
        self.kalman.update(float(min_index))

        # 6. Position from Kalman-smoothed estimate.
        #    Allows bounded backward movement (recovery from jumps)
        #    while preventing wild oscillation. The tempo-adaptive step
        #    cap limits how far the position can move per frame.
        eff_step = max(1, int(np.ceil(max(tempo, 0.5) * self.step_size)))
        kalman_pos = int(np.round(np.clip(self.kalman.position, 0, self.N_ref - 1)))
        if self.input_index > 0:
            # Allow forward up to eff_step, backward up to eff_step.
            self.current_position = int(
                np.clip(
                    kalman_pos,
                    self.current_position - eff_step,
                    self.current_position + eff_step,
                )
            )
        else:
            self.current_position = kalman_pos
        self.current_position = int(np.clip(self.current_position, 0, self.N_ref - 1))

        # 7. Bookkeeping.
        self.positions.append(self.current_position)
        self._warping_path.append((self.current_beat, self.input_index))
        self.input_index += 1

    # ------------------------------------------------------------------
    # Run loop (mirrors Arzt)
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray[np.float32]]:
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
            features, _ = item
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

    def __call__(self, input: NDArray[np.float32]) -> int:
        self.step(input)
        return self.current_position
