#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Soft Online Time Warping (Soft-OLTW) for Score Following.

Builds on Dixon-style frame-level OLTW (window-local DP with backward-
forward search and ref catch-up) by replacing the hard ``min`` in the DP
recursion with a soft minimum. The softmin temperature ``gamma`` and the
per-frame maximum catch-up budget are modulated by a Kalman filter that
tracks the performer's tempo, i.e., the ratio of score frames advanced
per input frame.

Key ideas drawn from the references:
    - Cuturi & Blondel (2017), Soft-DTW: differentiable softmin recursion.
    - Zeitler et al. (ISMIR 2023), Stabilizing Training with Soft DTW:
      adaptively schedule gamma (high when uncertain, low when confident)
      and add a Gaussian diagonal prior to the cost matrix.
    - Zeitler et al. (ICASSP 2024), Soft DTW with Variable Step Weights:
      step weights (w_h, w_v, w_d) shape the allowed warping and make the
      DP robust to repetitive or noisy input frames.

The class inherits the window management, ref-catch-up loop, and
run()/warping_path plumbing from ``OnlineTimeWarpingDixonFrame``. Only
the cell recursion and the direction selection are overridden so that
the softmin temperature can respond to the tracked tempo.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from matchmaker.dp.oltw_dixon import (
    MAX_RUN_COUNT,
    WINDOW_SIZE,
    Direction,
    OnlineTimeWarpingDixonFrame,
)
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.queue import RECVQueue

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GAMMA: float = 0.1
DEFAULT_W_HORIZONTAL: float = 1.0  # hold: input advances, ref stays
DEFAULT_W_VERTICAL: float = 1.0  # catch-up: ref advances, input stays
DEFAULT_W_DIAGONAL: float = 2.0  # both advance (Sakoe-Chiba weight)

# Kalman-centered Gaussian position prior on the local cost row.
DEFAULT_PRIOR_LAMBDA: float = 0.0
DEFAULT_PRIOR_SIGMA_FRAMES: float = 12.0

# Kalman filter.
DEFAULT_PROCESS_VAR: float = 0.02
DEFAULT_OBS_VAR: float = 3.0
DEFAULT_INIT_TEMPO: float = 1.0
DEFAULT_TEMPO_CLIP: Tuple[float, float] = (0.2, 4.0)

# Gamma modulation by tempo uncertainty.
GAMMA_FACTOR_MIN: float = 0.5
GAMMA_FACTOR_MAX: float = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmin3(a: float, b: float, c: float, gamma: float) -> float:
    """Softmin over three values via log-sum-exp. gamma→0 is hard min."""
    m = min(a, b, c)
    if not np.isfinite(m):
        return m
    if gamma <= 1e-8:
        return m
    s = np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    return m - gamma * float(np.log(s + 1e-12))


class PositionTempoKalman:
    """2D Kalman filter tracking (position, tempo).

    State dynamics: position advances by tempo each input frame; tempo is
    (near-)constant with process noise. We observe position (the DP argmin).
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
        # Generous initial tempo uncertainty so the filter adapts fast.
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
    def position_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))

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
# Soft OLTW class
# ---------------------------------------------------------------------------


class SoftOnlineTimeWarping(OnlineTimeWarpingDixonFrame):
    """Dixon-style frame-level OLTW with soft minimum recursion and
    tempo-adaptive gamma.

    Parameters
    ----------
    reference_features : np.ndarray
        Score-derived feature matrix, shape ``(N_ref, D)``.
    queue : RECVQueue, optional
        Input queue for streaming audio features.
    window_size : int
        OLTW search window half-width in seconds.
    max_run_count : int
        Maximum consecutive same-direction advances (Dixon's safety net).
    gamma : float
        Base softmin temperature. The effective temperature scales with
        both the local cost magnitude and the Kalman tempo uncertainty.
    w_horizontal, w_vertical, w_diagonal : float
        Step weights. Diagonal defaults to 2.0 (Sakoe-Chiba) to avoid a
        diagonal bias when the two feature rates do not match.
    prior_lambda : float
        Strength of the optional Kalman-centered Gaussian position prior
        on the local cost. 0 disables the prior.
    prior_sigma_frames : float
        Width of the Gaussian prior in score frames.
    process_var, obs_var : float
        Kalman process and observation noise.
    init_tempo : float
        Initial tempo estimate (score frames per input frame).
    """

    DEFAULT_DISTANCE_FUNC: str = "euclidean"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        queue: Optional[RECVQueue] = None,
        window_size: int = WINDOW_SIZE,
        max_run_count: int = MAX_RUN_COUNT,
        gamma: float = DEFAULT_GAMMA,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        w_horizontal: float = DEFAULT_W_HORIZONTAL,
        w_vertical: float = DEFAULT_W_VERTICAL,
        w_diagonal: float = DEFAULT_W_DIAGONAL,
        prior_lambda: float = DEFAULT_PRIOR_LAMBDA,
        prior_sigma_frames: float = DEFAULT_PRIOR_SIGMA_FRAMES,
        process_var: float = DEFAULT_PROCESS_VAR,
        obs_var: float = DEFAULT_OBS_VAR,
        init_tempo: float = DEFAULT_INIT_TEMPO,
        frame_rate: int = FRAME_RATE,
        step_size: Optional[int] = None,  # accepted for API compat, unused
        **kwargs,
    ) -> None:
        dfn = (
            distance_func
            if isinstance(distance_func, str)
            else self.DEFAULT_DISTANCE_FUNC
        )
        super().__init__(
            reference_features=reference_features,
            queue=queue,
            window_size=window_size,
            distance_func=dfn,
            max_run_count=max_run_count,
            frame_rate=frame_rate,
            **kwargs,
        )
        self.base_gamma = float(gamma)
        self.gamma = float(gamma)
        self.w_horizontal = float(w_horizontal)
        self.w_vertical = float(w_vertical)
        self.w_diagonal = float(w_diagonal)
        self.prior_lambda = float(prior_lambda)
        self.prior_sigma_frames = float(prior_sigma_frames)
        self._obs_var = float(obs_var)
        self.kalman = PositionTempoKalman(
            init_position=0.0,
            init_tempo=float(init_tempo),
            process_var=float(process_var),
            obs_var=float(obs_var),
        )
        # Cache the latest local cost for prior / gamma decisions.
        self._last_local_dist: float = 1.0

    # ------------------------------------------------------------------
    # Public hooks for inspection
    # ------------------------------------------------------------------

    @property
    def tracked_tempo(self) -> float:
        return self.kalman.tempo

    @property
    def tracked_position(self) -> float:
        return self.kalman.position

    # ------------------------------------------------------------------
    # Reset: keep Dixon state and reset Kalman.
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "kalman"):
            self.kalman.reset(
                position=0.0,
                tempo=self.kalman.tempo,
            )

    # ------------------------------------------------------------------
    # Override cell computation: hard min → soft min with step weights.
    # ------------------------------------------------------------------

    def _compute_cell(self, r_win: int, i_win: int) -> None:
        abs_ref = self._ref_offset() + r_win
        abs_input = self._input_offset() + i_win
        ref_feature = self.reference_features[abs_ref]
        input_feature = self.input_features[abs_input]
        import scipy.spatial.distance  # local import keeps base module light

        local_dist = float(
            scipy.spatial.distance.cdist(
                ref_feature.reshape(1, -1),
                input_feature.reshape(1, -1),
                metric=self.distance_func,
            )[0, 0]
        )
        self._last_local_dist = local_dist

        # Tempo-adaptive effective gamma: Kalman tempo uncertainty drives
        # the softness of the softmin. Confident tempo -> near-hard min;
        # uncertain tempo -> smoother blending across the three step
        # options. base_gamma is interpreted as a fraction of the local
        # cost so the setting carries over across feature scales.
        tempo_unc_norm = self.kalman.tempo_std / max(self.kalman.tempo, 1e-3)
        gamma_factor = float(
            np.clip(0.6 + 2.5 * tempo_unc_norm, GAMMA_FACTOR_MIN, GAMMA_FACTOR_MAX)
        )
        eff_gamma = self.base_gamma * gamma_factor * max(local_dist, 1e-6) * 0.1
        self.gamma = eff_gamma

        if r_win == 0 and i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = local_dist
            return
        if r_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win, i_win - 1] + self.w_horizontal * local_dist
            )
            return
        if i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win - 1, i_win] + self.w_vertical * local_dist
            )
            return

        d_vert = self.acc_dist_matrix[r_win - 1, i_win] + self.w_vertical * local_dist
        d_horz = self.acc_dist_matrix[r_win, i_win - 1] + self.w_horizontal * local_dist
        d_diag = (
            self.acc_dist_matrix[r_win - 1, i_win - 1] + self.w_diagonal * local_dist
        )
        self.acc_dist_matrix[r_win, i_win] = _softmin3(
            d_vert, d_horz, d_diag, eff_gamma
        )

    # ------------------------------------------------------------------
    # Override direction selection to incorporate the Kalman prior.
    # ------------------------------------------------------------------

    def _get_expand_direction(self) -> Direction:
        ref_ws = self._ref_win_size()
        input_ws = self._input_win_size()
        ref_off = self._ref_offset()
        input_off = self._input_offset()
        row = ref_ws - 1
        col = input_ws - 1

        pred_pos = self.kalman.position
        pred_std = max(self.kalman.position_std, 1.0)
        sigma = max(self.prior_sigma_frames, pred_std + 2.0)
        two_s2 = 2.0 * sigma * sigma
        lam = self.prior_lambda
        local_scale = max(self._last_local_dist, 1e-6)

        def _prior(abs_ref_idx: int) -> float:
            if lam <= 0:
                return 0.0
            dev = float(abs_ref_idx) - pred_pos
            p = (dev * dev) / two_s2
            return lam * local_scale * p

        best_cost = np.inf
        best_row, best_col = row, col

        # Last input column (find best score row for current input frame).
        for r in range(ref_ws):
            abs_ref = ref_off + r
            raw = self.acc_dist_matrix[r, col]
            cost = (raw + _prior(abs_ref)) / (1 + ref_off + r + input_off + col)
            if cost < best_cost:
                best_cost = cost
                best_row, best_col = r, col

        # Last ref row (find best input frame for current ref row).
        for c in range(input_ws):
            abs_ref = ref_off + row
            raw = self.acc_dist_matrix[row, c]
            cost = (raw + _prior(abs_ref)) / (1 + ref_off + row + input_off + c)
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

    # ------------------------------------------------------------------
    # Override step to run Kalman predict/update around Dixon's step.
    # ------------------------------------------------------------------

    def step(self, input_features: NDArray[np.float32]) -> None:
        # Kalman predict before DP; tempo uncertainty feeds into gamma.
        self.kalman.predict()
        super().step(input_features)
        # Update observation noise based on local cost informativeness:
        # when the window cost is essentially flat we should trust the
        # argmin less, so R grows.
        try:
            ws_r = self._ref_win_size()
            ws_i = self._input_win_size()
            if ws_r > 1 and ws_i > 0:
                col_costs = self.acc_dist_matrix[:ws_r, ws_i - 1]
                finite = col_costs[np.isfinite(col_costs)]
                if len(finite) > 1:
                    std = float(np.std(finite))
                    mean = float(np.mean(finite)) + 1e-6
                    signal = std / mean
                    adaptive_R = self._obs_var * float(
                        np.clip(1.5 / (signal + 0.3), 0.3, 6.0)
                    )
                    self.kalman.set_obs_var(adaptive_R)
        except Exception:
            pass
        self.kalman.update(float(self.current_position))
