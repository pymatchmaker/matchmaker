#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tempo Models

Includes:
* ReactiveTempoModel: Baseline raw IOI ratio model.
* MovingAverageTempoModel: Smoothed moving average IOI model.
* KalmanTempoModel: Linear Kalman filter tempo tracker (ACCompanion).
* LinearTempoModel: Sensorimotor synchronization linear model.
* JointAdaptationAnticipationModel: Joint adaptation & anticipation model.
* ArztTempoModel: Simple tempo model of Arzt & Widmer (SMC 2010) with backward
  path estimation and dynamic score alterations for OLTW, also usable on onsets.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


class TempoModel(object):
    """
    Base class for tempo models.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.

    Attributes
    ----------
    beat_period : float
        The current tempo in beats per second
    prev_score_onset: float
        Latest covered score onset (in beats)
    prev_perf_onset : float
        Last performed onset time in seconds
    asynchrony : float
        Asynchrony of the estimated onset time and the actually performed onset time.
    has_tempo_expectations : bool
        Whether the model includes tempo expectations
    counter: int
        The number of times that the model has been updated. Useful for debugging
        purposes.
    """

    beat_period: float
    prev_score_onset: float
    prev_perf_onset: float
    est_onset: float
    asynchrony: float
    has_tempo_expectations: bool
    counter: int
    score_onsets: np.ndarray

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        score_onsets: Optional[np.ndarray] = None,
    ) -> None:
        self.beat_period = init_beat_period
        self.prev_score_onset = init_score_onset
        self.prev_perf_onset = None
        self.est_onset = None
        self.asynchrony = 0.0
        self.has_tempo_expectations = False
        self.score_onsets = score_onsets
        # Count how many times has the tempo model been
        # called
        self.counter = 0

    def __call__(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Update beat period and compute estimated onset time

        Parameters
        ----------
        performed_onset : float
            Latest performed onset
        score_onset: float
            Latest score onset corresponding to the performed onset

        Returns
        -------
        beat_period : float
            Tempo in beats per second
        est_onsete : float
            Estimated onset given the current beat period
        """
        self.update_beat_period(performed_onset, score_onset, *args, **kwargs)
        self.counter += 1
        return self.beat_period, self.est_onset

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Internal method for updating the beat period.
        Needs to be implemented for each variant of the model
        """
        raise NotImplementedError


class ReactiveTempoModel(TempoModel):
    """
    Reactive tempo model.

    This sync model computes the tempo as the direct (raw) value of the performed
    ioi divided by the notated ioi. This method is mostly intended for as a baseline
    and is generally a poor choice of a tempo model.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.
    """

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0.0,
        update_on_valid_onsets_only: bool = False,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.update_on_valid_onsets_only = update_on_valid_onsets_only

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        See documentation in SyncModel above.
        """

        self.est_onset = performed_onset
        if self.prev_perf_onset:
            s_ioi = score_onset - self.prev_score_onset
            p_ioi = performed_onset - self.prev_perf_onset

            if s_ioi > 0 and p_ioi > 0:
                self.beat_period = p_ioi / s_ioi
            else:
                self.beat_period *= 0.5

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


class MovingAverageTempoModel(TempoModel):
    """
    Moving average tempo model

    This sync model computes the tempo as moving average value of the raw tempo
    (performed ioi divided by the notated ioi). This method is mostly intended as a
    baseline and is generally a poor choice of a tempo model.

    Parameters
    ----------
    init_beat_period: float
        Initial tempo in seconds per beat
    init_score_onset: float
        Initial score onset time in beats.
    alpha: float
        Smoothing factor (must be between 0 and 1).
        A value closer to 1 changes the MA value very slowly, while a
        value closer to 0 "forgets" the previous estimate and
        takes always the most recent value.
    predict_onset: bool
        If True, computes the expected next performed onset time using the current
        tempo estimation. Otherwise, it takes the observed performed onset.
        This option is for testing purposes.
    """

    alpha: float
    predict_onset: bool

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        alpha: float = 0.5,
        predict_onset: bool = True,
    ):
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.alpha = alpha
        self.predict_onset = predict_onset

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        See documentation in TempoModel above.
        """
        if self.prev_perf_onset:
            s_ioi = score_onset - self.prev_score_onset
            p_ioi = performed_onset - self.prev_perf_onset

            if s_ioi > 0:
                beat_period = p_ioi / s_ioi
            else:
                beat_period = self.beat_period

            if self.predict_onset:
                self.est_onset = self.est_onset + self.beat_period * s_ioi
            else:
                self.est_onset = performed_onset
            self.beat_period = (
                self.alpha * self.beat_period + (1 - self.alpha) * beat_period
            )
        else:
            self.est_onset = performed_onset

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


class KalmanTempoModel(TempoModel):
    """
    A Tempo model using a linear Kalman filter.
    """

    trans_par: float
    trans_var: float
    obs_var: float
    var_est: float

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        trans_par: float = 1,
        trans_var: float = 0.03,
        obs_var: float = 0.0213,  # values from old ACCompanion
        init_var: float = 1,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        # Assign the parameters:
        self.trans_par = trans_par
        self.trans_var = trans_var
        self.obs_var = obs_var

        # Assing the initial values:
        self.var_est = init_var

    # A function to compute one step of the predict-update cycle:
    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """
        Updates the model, when a new IOI observation is made. Computes
        one step of the predict-update cycle in the Kalman Filter.

        Parameters
        ----------
        performed_onset: float
            The latest performed onset

        score_onset : float
           Latest score onset corresponding to the performed onset
        """

        if self.counter == 0:
            score_ioi = 0
            performed_ioi = 0
            self.est_onset = performed_onset
            self.prev_perf_onset = performed_onset
            self.prev_score_onset = score_onset
            self.counter += 1
            return  # skip update on first observation
        else:
            performed_ioi = abs(performed_onset - self.prev_perf_onset)

            score_ioi = abs(score_onset - self.prev_score_onset)

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset
        # First, compute the prediction step:
        period_pred = self.beat_period * self.trans_par
        var_pred = (self.trans_par**2) * self.var_est + self.trans_var

        # Compute the error (innovation), between the estimation and obs:
        err = performed_ioi - score_ioi * period_pred
        # Compute the Kalman gain with the new predictions:
        kalman_gain = float(var_pred * score_ioi) / (
            (score_ioi**2) * var_pred + self.obs_var
        )
        # Compute the estimations after the update step:
        self.beat_period = period_pred + kalman_gain * err
        self.var_est = (1 - kalman_gain * score_ioi) * var_pred
        self.est_onset += score_ioi * self.beat_period
        # print("tempo", self.beat_period)
        self.counter += 1


class LinearTempoModel(TempoModel):
    """
    Linear synchronization model.

    The sensorimotor synch model to use if there are no tempo expectations.

    Parameters
    ----------
    init_beat_period : float
        Initial beat period in seconds
    init_score_onset : float
        Initial score onset in beats (can be negative)
    eta_t : float
        Learning rate for the tempo. This parameter serves a similar function
        to the alpha parameter in the MASM.
    eta_o : float
        Learning rate for the onset. This parameter serves a similar function to
        the alpha parameter in the MASM.
    """

    eta_t: float
    eta_p: float

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        eta_t: float = 0.3,
        eta_p: float = 0.7,
        min_beat_period: float = 0.25,
        max_beat_period: float = 3,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.eta_t = eta_t
        self.eta_p = eta_p
        self.min_beat_period = min_beat_period
        self.max_beat_period = max_beat_period

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        if self.prev_perf_onset:
            s_ioi = abs(score_onset - self.prev_score_onset)
            self.est_onset = (
                self.est_onset + self.beat_period * s_ioi - self.eta_p * self.asynchrony
            )
            self.asynchrony = self.est_onset - performed_onset

        else:
            s_ioi = 0
            self.est_onset = performed_onset

        tempo_correction_term = (
            self.asynchrony if self.asynchrony != 0 and s_ioi != 0 else 0
        )

        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        if tempo_correction_term < 0:
            beat_period = self.beat_period - self.eta_t * tempo_correction_term
        else:
            beat_period = self.beat_period - 2 * self.eta_t * tempo_correction_term

        if beat_period >= self.min_beat_period and beat_period <= self.max_beat_period:
            self.beat_period = beat_period


class JointAdaptationAnticipationModel(TempoModel):
    """
    Tempo Model with Joint Adaptation and Anticipation.
    """

    alpha: float
    beat: float
    delta: float
    omdelta: float
    gamma: float
    rng_motor: np.random.RandomState
    rng_timekeeper: np.random.RandomState

    def __init__(
        self,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0.0,
        alpha: float = 0.5,
        beta: float = 0.2,
        delta: float = 0.8,
        gamma: float = 0.1,
        rng_motor: np.random.RandomState = np.random.RandomState(1984),
        rng_timekeeper: np.random.RandomState = np.random.RandomState(1984),
    ):
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
        )
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.omdelta = 1 - delta
        self.gamma = gamma

        self.score_iois = []
        self.perf_iois = []
        # Have an initial value for beat periods
        self.instant_beat_periods = [1]
        self.cnt = 0
        self.motor_noise = rng_motor
        self.timekeeper_noise = rng_timekeeper
        self.tkns = []
        self.mns = []

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        if self.cnt > 0:
            # perf_ioi = performed_onset - self.prev_perf_onset
            self.perf_iois.append(performed_onset - self.prev_perf_onset)
            self.score_iois.append(score_onset - self.prev_score_onset)
            ibp = self.perf_iois[-1] / self.score_iois[-1]

            if ibp >= 0.25 and ibp <= 3:
                # CC: add checks for instant beat period
                self.instant_beat_periods.append(ibp)
            else:
                self.instant_beat_periods.append(self.instant_beat_periods[-1])

            if self.cnt > 1:
                # ADAPTATION MODULE
                # TODO: add timekeeper noise
                self.ad_est_onset = (
                    self.est_onset
                    + self.beat_period * self.score_iois[-1]
                    - self.alpha * self.asynchrony
                    + self.tkns[-1]
                )
                # update beat period
                self.beat_period = self.beat_period - self.beta * (
                    self.asynchrony  # / self.score_iois[-2]
                )

                # ANTICIPATION MODULE  # could be any prediction model (NN, LSTM, etc.) - non-linear

                # add timekeeper noise
                self.an_onset = (
                    self.prev_perf_onset
                    + (
                        self.delta * self.extrapolated_interval
                        + self.omdelta * self.preserverated_interval
                    )
                    * self.score_iois[-1]
                    + self.tkns[-1]
                )

                self.extrapolated_interval = (
                    2 * self.instant_beat_periods[-1] - self.instant_beat_periods[-2]
                )

                self.preserverated_interval = self.instant_beat_periods[-1]

                # JOINT MODULE

                self.joint_asynchrony = self.ad_est_onset - self.an_onset
                self.est_onset = (
                    self.ad_est_onset
                    - self.gamma * self.joint_asynchrony
                    + self.mns[-1]
                    - self.mns[-2]
                )

            else:
                # For the first IOI wi do not really have
                # an asynchrony yet
                self.est_onset = self.est_onset + self.beat_period * self.score_iois[-1]

                # perhaps use instant beat period instead?
                # self.beat_period = ibp
                self.extrapolated_interval = (
                    2 * self.instant_beat_periods[-1] - self.beat_period
                )
                self.preserverated_interval = self.beat_period
        else:
            self.est_onset = performed_onset

        # update global
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset
        self.asynchrony = self.est_onset - performed_onset
        self.cnt += 1
        # Check other noise types
        self.tkns.append(self.timekeeper_noise.normal(0, 0.005))
        self.mns.append(self.motor_noise.normal(0, 0.0025))


class ArztTempoModel(TempoModel):
    """
    Simple tempo model of Arzt & Widmer (2010).

    References
    ----------
    A. Arzt & G. Widmer (2010), "Simple Tempo Models for Real-Time Music
    Tracking", Sound and Music Computing Conference (SMC), Sec. 4.

    When used with an on-line time warping tracker (e.g.,
    ``OnlineTimeWarpingArztFrame``), after every input frame, the relative
    tempo t is the Eq. 1-weighted mean of the local tempi at the ``n_onsets``
    most recent score onsets at least ``min_past`` seconds in the past, each
    over a ``window_size``-second window of the rectified backward path (as in
    Mueller et al. (2009), ISMIR, Sec. 3.1 / 3.3: path rectified at onsets,
    extended with slope 1 past its ends). The score representation is then
    altered at the last processed score frame ps: with probability 1 - 1/t
    (t > 1) frames ps+1 and ps+2 are merged into their mean, with probability
    1 - t (t < 1) the mean of ps and ps+1 is inserted; onset frames are not
    duplicated (the insertion is postponed), and at most 3 alterations are
    made in a row.

    When used as a standalone tempo model (e.g. with discrete performed and score
    onsets), ``update_beat_period`` calculates local tempi over recent onset intervals
    and computes the Eq. 1 weighted average.

    Parameters
    ----------
    window_size : float
        Window around an onset for its local tempo, in seconds (default: 3.0).
    min_past : float
        Onsets used must be at least this many seconds in the past (default: 1.0).
    n_onsets : int
        Number of recent onsets averaged (default: 20).
    random_seed : int or None
        Seed for the random score alterations (default: 1984).
    tempo_min : float
        Lower bound on relative tempo factor (default: 0.85).
    tempo_max : float
        Upper bound on relative tempo factor (default: 1.20).
    min_advance : int
        Minimum score frames between successive alterations (default: 10).
    init_beat_period : float
        Initial tempo in seconds per beat (default: 0.5).
    init_score_onset : float
        Initial score onset time in beats (default: 0.0).
    score_onsets : Optional[np.ndarray]
        Score onsets array in beats.
    """

    window_size: float
    min_past: float
    n_onsets: int
    random_seed: Optional[int]
    tempo_min: float
    tempo_max: float
    min_advance: int
    n_inserted: int
    n_deleted: int
    n_postponed: int

    def __init__(
        self,
        window_size: float = 3.0,
        min_past: float = 1.0,
        n_onsets: int = 20,
        random_seed: Optional[int] = 1984,
        tempo_min: float = 0.85,
        tempo_max: float = 1.20,
        min_advance: int = 10,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0.0,
        score_onsets: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            init_beat_period=init_beat_period,
            init_score_onset=init_score_onset,
            score_onsets=score_onsets,
        )
        self.init_beat_period = init_beat_period
        self.window_size = window_size
        self.min_past = min_past
        self.n_onsets = n_onsets
        self.random_seed = random_seed
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.min_advance = min_advance
        self._current_tempo = 1.0
        self.n_inserted = self.n_deleted = self.n_postponed = 0
        self._last_altered_ps: int = -1
        self._consecutive = 0
        self._insert_pending = False
        self._w = 0
        self._min_past = 0
        self._rng = (
            np.random.RandomState(random_seed)
            if random_seed is not None
            else np.random.RandomState()
        )
        self._history_perf: List[float] = []
        self._history_score: List[float] = []

    def reset(self, tracker=None) -> None:
        """Reset the tempo model state."""
        self.counter = 0
        self.prev_perf_onset = None
        self.est_onset = None
        self.asynchrony = 0.0
        self.beat_period = getattr(self, "init_beat_period", 0.5)
        self._consecutive = 0
        self._insert_pending = False
        self._current_tempo = 1.0
        self.n_inserted = self.n_deleted = self.n_postponed = 0
        self._last_altered_ps = -1
        self._history_perf = []
        self._history_score = []
        if tracker is not None:
            fr = tracker.frame_rate
            self._w = int(round(self.window_size * fr))
            self._min_past = int(round(self.min_past * fr))
            self._rng = (
                np.random.RandomState(self.random_seed)
                if self.random_seed is not None
                else np.random.RandomState()
            )
        elif self.random_seed is not None:
            self._rng = np.random.RandomState(self.random_seed)

    @property
    def current_tempo(self) -> float:
        """Current relative tempo (performance / score representation)."""
        return self._current_tempo

    # --- Tracker integration for OLTW ---

    def before_step(self, tracker) -> None:
        """Estimate the tempo and alter the tracker's score representation."""
        t = self.relative_tempo(tracker)
        if t is not None:
            self._current_tempo = t
            base_bp = getattr(self, "init_beat_period", 0.5)
            self.beat_period = base_bp / max(1e-4, t)
        self.alter(tracker, t)

    def relative_tempo(self, tracker) -> Optional[float]:
        """Sec. 4.1: relative tempo from the rectified backward path."""
        half = (self._w - 1) // 2
        limit = tracker.tempo_last_input() - self._min_past
        is_onset = tracker.tempo_score()[2]
        x, y = tracker.tempo_current_cell()
        y_start = y
        phi = {}  # row -> min input index on the backward path
        onsets = []  # onset rows on the path, most recent first
        last_row = y
        phi[y] = x
        need_below = None
        for x, y in tracker.tempo_backward_path(x, y):
            phi[y] = x
            if y != last_row:
                if is_onset[last_row] and phi[last_row] <= limit:
                    onsets.append(last_row)
                last_row = y
                if need_below is None and len(onsets) >= self.n_onsets:
                    need_below = float(onsets[self.n_onsets - 1]) - half
                if need_below is not None and float(y) < need_below and is_onset[y]:
                    break
        if is_onset[last_row] and phi[last_row] <= limit and last_row not in onsets:
            onsets.append(last_row)
        tracker.tempo_note_path_end(x, y)
        if not onsets:
            return None
        # anchors: onsets on the path plus both ends (Mueller et al. 2009, Sec. 3.3)
        rows = sorted(set([y, y_start] + [r for r in phi if is_onset[r]]))
        g = np.array(rows, dtype=np.float64)
        p = np.array([phi[r] for r in rows], dtype=np.float64)

        def phi_r(n):
            if n <= g[0]:
                return p[0] - (g[0] - n)  # extended with slope 1
            if n >= g[-1]:
                return p[-1] + (n - g[-1])
            return float(np.interp(n, g, p))

        sel = onsets[: self.n_onsets][::-1]  # oldest first (Eq. 1 weights 1..n)
        tempi = []
        for o in sel:
            go = float(o)
            n1, n2 = go - half, go + half
            tempi.append((n2 - n1 + 1) / (phi_r(n2) - phi_r(n1) + 1))
        w = np.arange(1, len(tempi) + 1)
        return float(np.sum(np.array(tempi) * w) / np.sum(w))

    def alter(self, tracker, t: Optional[float]) -> None:
        """Sec. 4.2: alter the score representation at the last processed frame."""
        if t is None:
            self._consecutive = 0
            return
        t = max(self.tempo_min, min(self.tempo_max, t))
        if t >= 1.0:
            self._insert_pending = False
        ps = tracker.tempo_last_score_frame()
        if ps <= self._last_altered_ps + self.min_advance:
            return
        V, f2b, on = tracker.tempo_score()
        if ps < 0 or ps + 2 >= len(V):
            return
        if self._consecutive >= 3:  # at most 3 alterations in a row
            self._consecutive = 0
            return
        r = self._rng.uniform(0.0, 1.0)
        if t > 1.0 and r > 1.0 / t:
            V[ps + 1] = 0.5 * (V[ps + 1] + V[ps + 2])
            if on[ps + 2] and not on[ps + 1]:
                f2b[ps + 1] = f2b[ps + 2]
            elif not on[ps + 1]:
                f2b[ps + 1] = 0.5 * (f2b[ps + 1] + f2b[ps + 2])
            on[ps + 1] |= on[ps + 2]
            tracker.tempo_set_score(
                np.delete(V, ps + 2, axis=0),
                np.delete(f2b, ps + 2),
                np.delete(on, ps + 2),
            )
            self.n_deleted += 1
            self._consecutive += 1
            self._last_altered_ps = ps
        elif t < 1.0 and (self._insert_pending or r > t):
            if on[ps] or on[ps + 1]:  # onset vectors are not duplicated
                self._insert_pending = True
                self.n_postponed += 1
                self._consecutive = 0
                return
            self._insert_pending = False
            tracker.tempo_set_score(
                np.insert(V, ps + 1, 0.5 * (V[ps] + V[ps + 1]), axis=0),
                np.insert(f2b, ps + 1, 0.5 * (f2b[ps] + f2b[ps + 1])),
                np.insert(on, ps + 1, False),
            )
            self.n_inserted += 1
            self._consecutive += 1
            self._last_altered_ps = ps
        else:
            self._consecutive = 0

    # --- Standalone onset update (Eq. 1) ---

    def update_beat_period(
        self,
        performed_onset: float,
        score_onset: float,
        *args,
        **kwargs,
    ) -> None:
        """Update beat period using the Eq. 1 weighted average of recent local tempi."""
        self.est_onset = performed_onset
        self._history_perf.append(performed_onset)
        self._history_score.append(score_onset)
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        if len(self._history_perf) < 2:
            return

        perf = np.array(self._history_perf)
        score = np.array(self._history_score)
        n = min(self.n_onsets, len(perf) - 1)
        tempi = []
        for i in range(1, n + 1):
            dp = perf[-1] - perf[-1 - i]
            ds = score[-1] - score[-1 - i]
            if dp > 0 and ds > 0:
                tempi.append(dp / ds)

        if tempi:
            tempi = tempi[::-1]
            w = np.arange(1, len(tempi) + 1)
            est_bp = float(np.sum(np.array(tempi) * w) / np.sum(w))
            base_bp = getattr(self, "init_beat_period", 0.5)
            rel_t = base_bp / max(1e-4, est_bp)
            rel_t = max(self.tempo_min, min(self.tempo_max, rel_t))
            self._current_tempo = rel_t
            self.beat_period = base_bp / rel_t


# Aliases for backward compatibility
ArztWidmer2010TempoModel = ArztTempoModel


def make_tempo_model(tempo_model, **kwargs) -> Optional[TempoModel]:
    """None, a ``TempoModel`` instance, or a class (built with ``kwargs``)."""
    if tempo_model is None or isinstance(tempo_model, TempoModel):
        return tempo_model
    if isinstance(tempo_model, type) and issubclass(tempo_model, TempoModel):
        return tempo_model(**kwargs)
    raise TypeError(
        f"tempo_model must be None, a TempoModel or its class, got {tempo_model!r}"
    )


if __name__ == "__main__":
    pass
