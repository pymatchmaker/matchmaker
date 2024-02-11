#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements the specific HMM for the score follower for Accompanion.

TODO
----
* The definition of the HMMs need to be updated to work with audio
"""
from typing import Optional

import numpy as np
import scipy.spatial.distance as sp_dist
from hiddenmarkov import (ConstantTransitionModel, HiddenMarkovModel,
                          ObservationModel)
from scipy.stats import gumbel_l

from matchmaker.base import OnlineAlignment
from matchmaker.utils.tempo_models import TempoModel


class PitchIOIObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed notes at the current moment/state.

    Parameters
    ----------
    _pitch_profiles : numpy array
        he pre-computed pitch profiles, for each separate possible pitch
        in the MIDI range. Used in calculating the pitch observation
        probabilities.

    _ioi_matrix : numpy array
        The pre-computed score IOI values in beats, from each unique state
        to all other states, stored in a matrix.

    _ioi_precision : float
        The precision parameter for computing the IOI observation probability.

    _ioi_norm_term : float
        The normalization term of the Gaussian distribution used for the
        computation of the IOI probabilities.

    TODO
    ----
    * Implement log probabilities
    """

    def __init__(self, pitch_profiles, ioi_matrix, ioi_precision):
        """
        The initialization method.

        Parameters
        ----------
        pitch_profiles : numpy array
            he pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.

        ioi_matrix : numpy array
            The pre-computed score IOI values in beats, from each unique state
            to all other states, stored in a matrix.

        ioi_precision : float
            The precision parameter for computing the IOI observation
            probability.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self._pitch_profiles = pitch_profiles
        self._ioi_matrix = ioi_matrix
        self._ioi_precision = ioi_precision
        # Compute the IOI normalization term:
        self._ioi_norm_term = np.sqrt(0.5 * self._ioi_precision / np.pi)
        self.current_state = None

    def compute_pitch_observation_probability(self, pitch_obs):
        """
        Compute the pitch observation probability.

        Parameters
        ----------
        pitch_obs : numpy array
            All the MIDI pitch values in an observation.

        Returns
        -------
        pitch_obs_prob : numpy array
            The computed pitch observation probabilities for all states.
        """
        # Use Bernouli distribution to compute the prob:
        # Binary piano-roll observation:
        pitch_prof_obs = np.zeros((1, 128))
        pitch_prof_obs[0, pitch_obs.astype(int)] = 1

        # Compute Bernoulli probability:
        pitch_prob = (self._pitch_profiles**pitch_prof_obs) * (
            (1 - self._pitch_profiles) ** (1 - pitch_prof_obs)
        )

        # Return the values:
        return np.prod(pitch_prob, 1)

    def compute_ioi_observation_probability(self, ioi_obs, current_state, tempo_est):
        """
        Compute the IOI observation probability.

        Parameters
        ----------
        ioi_obs : numpy array
            All the observed IOI.

        current_state : int
            The current state of the Score HMM.

        tempo_est : float
            The tempo estimation.

        Returns
        -------
        ioi_obs_prob : numpy array
            The computed IOI observation probabilities for each state.
        """
        # Use Gaussian distribution:
        ioi_idx = current_state if current_state is not None else 0
        # Compute the expected argument:
        exp_arg = (
            -0.5
            * ((tempo_est * self._ioi_matrix[ioi_idx] - ioi_obs) ** 2)
            * self._ioi_precision
        )

        # Return the value:
        return self._ioi_norm_term * np.exp(exp_arg)

    def get_score_ioi(self, current_state):
        """
        Get the score inter onset interval (IOI) between the current state and
        the previous state in beats.

        Parameters
        ----------
        current_state : int
            The current state of the Score HMM.

        Returns
        -------
        state_ioi : numpy.int
            The IOI in beats.
        """
        # Return the specific value:
        return self._ioi_matrix[current_state]

    def __call__(self, observation):
        pitch_obs, ioi_obs, tempo_est = observation
        observation_prob = self.compute_pitch_observation_probability(
            pitch_obs
        ) * self.compute_ioi_observation_probability(
            ioi_obs=ioi_obs, current_state=self.current_state, tempo_est=tempo_est
        )
        return observation_prob


class PitchIOIHMM(HiddenMarkovModel):
    """
    Implements the bahaviour of a HiddenMarkovModel, specifically designed for
    the task of score following.

    Parameters
    ----------
    _transition_matrix : numpy.ndarray
        Matrix for computations of state transitions within the HMM.

    _observation_model : ObservationModel
        Object responsible for computing the observation probabilities for each
        state of the HMM.

    initial_distribution : numpy array
        The initial distribution of the model. If not given, it is asumed to
        be uniform.

    forward_variable : numpy array
        The current (latest) value of the forward variable.

    _variation_coeff : float
        The normalized coefficient of variation of the current (latest) forward
        variable. Used to determine the confidence of the prediction of the HMM.

    current_state : int
        The index of the current state of the HMM.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        pitch_profiles: np.ndarray,
        ioi_matrix: np.ndarray,
        score_onsets: np.ndarray,
        tempo_model: TempoModel,
        ioi_precision: float = 1,
        initial_probabilities: Optional[np.ndarray] = None,
        has_insertions=True,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        transition_matrix : numpy array
            The Tranistion probability matrix of HMM.

        pitch_profiles : numpy array
            The pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.

        ioi_matrix : numpy array
            The pre-computed score IOI values in beats, from each unique state
            to all other states, stored in a matrix.

        ioi_precision : float
            The precision parameter for computing the IOI observation
            probability.

        score_onsets : numpy array
            TODO

        initial_distribution : numpy array
            The initial distribution of the model. If not given, it is asumed to
            be uniform.
            Default = None.
        """
        # reference_features = (transition_matrix, pitch_profiles, ioi_matrix)

        observation_model = PitchIOIObservationModel(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=ioi_precision,
        )

        super().__init__(
            observation_model=observation_model,
            transition_model=ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            ),
            state_space=score_onsets,
            # reference_features=reference_features,
        )

        self.tempo_model = tempo_model
        self.has_insertions = has_insertions

    def __call__(self, input):
        self.current_state = self.forward_algorithm_step(
            observation=input + (self.tempo_model.beat_period,), log_probabilities=False
        )
        return self.state_space[self.current_state]

    @property
    def current_state(self):
        return self.observation_model.current_state

    @current_state.setter
    def current_state(self, state):
        self.observation_model.current_state = state


def gumbel_transition_matrix(
    n_states,
    mp_trans_state=1,
    scale=0.5,
    inserted_states=True,
):
    """
    Compute a transiton matrix, where each row follows a normalized Gumbel
    distribution.

    Parameters
    ----------
    n_states : int
        The number of states in the Hidden Markov Model (HMM), which is required
        for the size of the matrix.

    mp_trans_state : int
        Which state should have the largest probability to be transitioned into
        from the current state the model is in.
        Default = 1, which means that the model would prioritize transitioning
        into the state that is next in line, e.g. from State 3 to State 4.

    scale : float
        The scale parameter of the distribution.
        Default = 0.5

    inserted_states : boolean
        Indicates whether the HMM includes inserted states (intermediary states
        between chords for errors and insertions in the score following).
        Default = True

    Returns
    -------
    transition_matrix : numpy array
        The computed transition matrix for the HMM.
    """
    # Initialize transition matrix:
    transition_matrix = np.zeros((n_states, n_states), dtype="f8")

    # Compute transition matrix:
    for i in range(n_states):
        if inserted_states:
            if np.mod(i, 2) == 0:
                transition_matrix[i] = gumbel_l.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2, scale=scale
                )
            else:
                transition_matrix[i] = gumbel_l.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
                )
        else:
            transition_matrix[i] = gumbel_l.pdf(
                np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
            )

    # Normalize transition matrix (so that it is a proper stochastic matrix):
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # Return the computed transition matrix:
    return transition_matrix


def gumbel_init_dist(n_states: int, loc: int = 0, scale: float = 10):
    """
    Compute the initial probabilites for all states in the Hidden Markov Model
    (HMM), which follow a Gumbel distribution.

    Parameters
    ----------
    n_states : int
        The number of states in the Hidden Markov Model (HMM), which is required
        for the size of the initial probabilites vector.

    Returns
    -------
    init_probs : numpy array
        The computed initial probabilities in the form of a vector.
    """

    prob_scale: float = scale if scale < n_states else n_states / 10

    init_probs: np.ndarray = gumbel_l.pdf(
        np.arange(n_states),
        loc=loc,
        scale=prob_scale,
    )

    return init_probs


def compute_ioi_matrix(unique_onsets, inserted_states=False):

    # Construct unique onsets with skips:
    if inserted_states:
        unique_onsets_s = np.insert(
            unique_onsets,
            np.arange(1, len(unique_onsets)),
            (unique_onsets[:-1] + 0.5 * np.diff(unique_onsets)),
        )
        ioi_matrix = sp_dist.squareform(sp_dist.pdist(unique_onsets_s.reshape(-1, 1)))

    # ... or without skips:
    else:
        unique_onsets_s = unique_onsets
        ioi_matrix = sp_dist.squareform(sp_dist.pdist(unique_onsets.reshape(-1, 1)))

    return ioi_matrix


def compute_pitch_profiles(
    chord_pitches,
    profile=np.array([0.02, 0.02, 1, 0.02, 0.02]),
    eps=0.01,
    piano_range=False,
    normalize=True,
    inserted_states=True,
):
    """
    Pre-compute the pitch profiles used in calculating the pitch
    observation probabilities.

    Parameters
    ----------
    chord_pitches : array-like
        The pitches of each chord in the piece.

    profile : numpy array
        The probability "gain" of how probable are the closest pitches to
        the one in question.

    eps : float
        The epsilon value to be added to each pre-computed pitch profile.

    piano_range : boolean
        Indicates whether the possible MIDI pitches are to be restricted
        within the range of a piano.

    normalize : boolean
        Indicates whether the pitch profiles are to be normalized.

    inserted_states : boolean
        Indicates whether the HMM uses inserted states between chord states.

    Returns
    -------
    pitch_profiles : numpy array
        The pre-computed pitch profiles.
    """
    # Compute the high and low contexts:
    low_context = profile.argmax()
    high_context = len(profile) - profile.argmax()

    # Get the number of states, based on the presence of inserted states:
    if inserted_states:
        n_states = 2 * len(chord_pitches) - 1
    else:
        n_states = len(chord_pitches)
    # Initialize the numpy array to store the pitch profiles:
    pitch_profiles = np.zeros((n_states, 128))

    # Compute the profiles:
    for i in range(n_states):
        # Work on chord states (even indices), not inserted (odd indices):
        if np.mod(i, 2) == 0:
            chord = chord_pitches[i // 2]
            for pitch in chord:
                lowest_pitch = pitch - low_context
                highest_pitch = pitch + high_context
                # Compute the indices which are to be updated:
                idx = slice(np.maximum(lowest_pitch, 0), np.minimum(highest_pitch, 128))
                # Add the values:
                pitch_profiles[i, idx] += profile

        # Add the extra value:
        pitch_profiles[i] += eps

    # Check whether to trim and normalize:
    if piano_range:
        pitch_profiles = pitch_profiles[:, 21:109]
    if normalize:
        pitch_profiles /= pitch_profiles.sum(1, keepdims=True)

    # Return the profiles:
    return pitch_profiles


class PitchIOIKHMM(HiddenMarkovModel):
    """
    Implements the bahaviour of a HiddenMarkovModel, specifically designed for
    the task of score following.

    Parameters
    ----------
    _transition_matrix : numpy.ndarray
        Matrix for computations of state transitions within the HMM.

    _observation_model : ObservationModel
        Object responsible for computing the observation probabilities for each
        state of the HMM.

    initial_distribution : numpy array
        The initial distribution of the model. If not given, it is asumed to
        be uniform.

    forward_variable : numpy array
        The current (latest) value of the forward variable.

    _variation_coeff : float
        The normalized coefficient of variation of the current (latest) forward
        variable. Used to determine the confidence of the prediction of the HMM.

    current_state : int
        The index of the current state of the HMM.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        pitch_profiles: np.ndarray,
        ioi_matrix: np.ndarray,
        score_onsets: np.ndarray,
        init_beat_period: float = 0.5,
        init_score_onset: float = 0,
        trans_par: float = 1,
        trans_var: float = 0.03,
        obs_var: float = 0.0213,  # values from old ACCompanion
        init_var: float = 1,
        ioi_precision: float = 1,
        initial_probabilities: Optional[np.ndarray] = None,
        has_insertions=True,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        transition_matrix : numpy array
            The Tranistion probability matrix of HMM.

        pitch_profiles : numpy array
            The pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.

        ioi_matrix : numpy array
            The pre-computed score IOI values in beats, from each unique state
            to all other states, stored in a matrix.

        ioi_precision : float
            The precision parameter for computing the IOI observation
            probability.

        score_onsets : numpy array
            TODO

        initial_distribution : numpy array
            The initial distribution of the model. If not given, it is asumed to
            be uniform.
            Default = None.
        """

        from accompanion.accompanist.tempo_models import KalmanTempoTempoModel

        # reference_features = (transition_matrix, pitch_profiles, ioi_matrix)

        observation_model = PitchIOIObservationModel(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=ioi_precision,
        )

        super().__init__(
            observation_model=observation_model,
            transition_model=ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            ),
            state_space=score_onsets,
            # reference_features=reference_features,
        )

        self.prev_perf_onset = None

        self.tempo_model = KalmanTempoTempoModel(
            init_beat_period=init_beat_period,
            init_score_onset=self.state_space[0],
            trans_par=trans_par,
            trans_var=trans_var,
            obs_var=obs_var,
            init_var=init_var,
        )
        self.has_insertions = has_insertions

    def __call__(self, input):
        self.current_state = self.forward_algorithm_step(
            observation=input + (self.tempo_model.beat_period,), log_probabilities=False
        )

        if self.prev_perf_onset is None:
            perf_onset = 0
        else:
            # previous onset + last observed ioi
            perf_onset = self.prev_perf_onset + input[1]
        self.prev_perf_onset = perf_onset
        # Update tempo model for real notes (not insertions)
        if self.current_state % 2 == 0 and self.has_insertions:
            current_score_onset = self.state_space[self.current_state]

            self.tempo_model.update_beat_period(
                performed_onset=perf_onset,
                score_onset=current_score_onset,
            )
            self.tempo_model.counter += 1

        return self.state_space[self.current_state]

    @property
    def current_state(self):
        return self.observation_model.current_state

    @current_state.setter
    def current_state(self, state):
        self.observation_model.current_state = state


if __name__ == "__main__":

    rng = np.random.RandomState(1984)

    n_onsets = 100

    onsets = 100 * rng.rand(100)

if __name__ == "__main__":
    pass
