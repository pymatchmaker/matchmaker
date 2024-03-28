#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tempo Models

TODO
----
* Adapt models from ACCompanion
"""
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d


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
        score_onsets: Optional[np.ndarray] = None
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
    ) -> None:
        """
        See documentation in SyncModel above.
        """

        
        self.est_onset = performed_onset
        if self.prev_perf_onset:
            s_ioi = abs(score_onset - self.prev_score_onset)
            p_ioi = abs(performed_onset - self.prev_perf_onset)
            self.beat_period = p_ioi / s_ioi

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


if __name__ == "__main__":
    pass
