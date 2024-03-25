#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements Hidden Markov Models for score following
"""
from typing import Optional, Union, Tuple, Dict, Any, Callable, List

import numpy as np
import scipy.spatial.distance as sp_dist
from hiddenmarkov import (
    ConstantTransitionModel,
    HiddenMarkovModel,
    ObservationModel,
    TransitionModel,
)
from numpy.typing import NDArray
from scipy.stats import gumbel_l

from matchmaker.base import OnlineAlignment
from matchmaker.utils.tempo_models import TempoModel
from matchmaker.utils.misc import RECVQueue, get_window_indices

# Alias for typing arrays
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


class BaseHMM(OnlineAlignment, HiddenMarkovModel):
    """
    Base class for Hidden Markov Model alignment methods.

    Parameters
    ----------
    observation_model: ObservationModel
        An observation (data) model for computing the observation probabilities.

    transition_model: TransitionModel
        A transition model for computing the transition probabilities.

    state_space: np.ndarray
        The hidden states (positions in reference time).

    tempo_model: Optional[TempoModel]
        A tempo model instance

    has_insertions: bool
        A boolean indicating whether the state space consider inserted notes.
    """

    observation_model: ObservationModel
    transition_model: TransitionModel
    state_space: Union[NDArrayFloat, NDArrayInt]
    tempo_model: Optional[TempoModel]
    has_insertions: bool
    _warping_path: List[Tuple[int, int]]
    queue: Optional[RECVQueue]

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        state_space: Optional[Union[NDArrayFloat, NDArrayInt]] = None,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
        queue: Optional[RECVQueue] = None,
        patience: int = 10,
    ) -> None:

        HiddenMarkovModel.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=state_space,
        )
        OnlineAlignment.__init__(
            self,
            reference_features=observation_model,
        )

        self.tempo_model = tempo_model
        self.has_insertions = has_insertions
        self.input_index = 0
        self._warping_path = []
        self.queue = queue
        self.patience = patience
        self.current_state = None

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)

    def __call__(self, input: NDArrayFloat) -> float:

        current_state = self.forward_algorithm_step(
            observation=input,
            log_probabilities=False,
        )
        self._warping_path.append((current_state, self.input_index))
        self.input_index += 1
        self.current_state = current_state

        return self.state_space[current_state]

    def run(self) -> NDArrayInt:
        if self.queue is not None:

            prev_state = self.current_state
            same_state_counter = 0
            while self.is_still_following():
                target_feature = self.queue.get()

                current_state = self(target_feature)

                if current_state == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0

            return self.warping_path

    def is_still_following(self) -> bool:
        if self.current_state is not None:

            return self.current_state <= self.n_states

        return False


class PitchHMM(BaseHMM):
    """
    A simple HMM that uses pitch information (symbolic or spectrograms)
    as input. This model does not include temporal information.

    This model is meant to be used as a baseline only,
    and is not expected to have a good performance other
    than in very simple idealized cases.
    """

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_matrix: NDArrayFloat,
        score_onsets: NDArrayFloat,
        initial_probabilities: Optional[NDArrayFloat] = None,
        has_insertions: bool = False,
    ) -> None:

        transition_model = ConstantTransitionModel(
            transition_probabilities=transition_matrix,
            init_probabilities=initial_probabilities,
        )
        BaseHMM.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=score_onsets,
            tempo_model=None,
            has_insertions=has_insertions,
        )


def gumbel_transition_matrix(
    n_states: int,
    mp_trans_state: int = 1,
    scale: float = 0.5,
    inserted_states: bool = False,
) -> NDArrayFloat:
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


def gumbel_init_dist(
    n_states: int,
    loc: int = 0,
    scale: float = 10,
) -> NDArrayFloat:
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


def compute_continous_pitch_profiles(
    spectral_features: NDArrayFloat,
    spectral_feature_times: NDArrayFloat,
    onset_times: NDArrayFloat,
    eps: float = 0.01,
    context: int = 3,
    normalize=True,
    inserted_states=True,
) -> NDArrayFloat:

    onset_idxs_in_features = np.searchsorted(
        a=spectral_feature_times,
        v=onset_times,
        side="left",
    )

    window_indices = get_window_indices(
        indices=onset_idxs_in_features,
        context=context,
    )

    mask = (window_indices >= 0)[:, :, np.newaxis]

    _pitch_profiles = (spectral_features[window_indices] * mask).sum(1)

    if inserted_states:
        pitch_profiles = np.ones(2 * len(onset_times)) * eps

        pitch_profiles[np.arange(len(onset_times)) * 2] += _pitch_profiles
    else:
        pitch_profiles = _pitch_profiles

    if normalize:
        pitch_profiles /= pitch_profiles.sum(1, keepdims=True)

    return pitch_profiles


def compute_discrete_pitch_profiles(
    chord_pitches: NDArrayFloat,
    profile: NDArrayFloat = np.array([0.02, 0.02, 1, 0.02, 0.02], dtype=np.float32),
    eps: float = 0.01,
    piano_range: bool = False,
    normalize: bool = True,
    inserted_states: bool = True,
) -> NDArrayFloat:
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

        if not inserted_states or (inserted_states and np.mod(i, 2) == 0):

            chord = chord_pitches[i // 2] if inserted_states else chord_pitches[i]

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
    return pitch_profiles.astype(np.float32)


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


def compute_bernoulli_pitch_probabilities(
    pitch_obs: NDArrayFloat,
    pitch_profiles: NDArrayFloat,
) -> NDArrayFloat:
    """
    Compute pitch observation probabilities
    """

    # Compute Bernoulli probability:
    pitch_prob = (pitch_profiles**pitch_obs) * ((1 - pitch_profiles) ** (1 - pitch_obs))

    obs_prob = np.prod(pitch_prob, 1)

    return obs_prob


def compute_gaussian_ioi_observation_probability(
    ioi_obs: float,
    ioi_score: NDArrayFloat,
    tempo_est: float,
    ioi_precision: float,
    norm_term: float,
) -> NDArrayFloat:
    """
    Compute the IOI observation probability as a zero mean
    Gaussian

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
    obs_prob : numpy array
        The computed IOI observation probabilities for each state.
    """
    # Compute the expected argument:
    exp_arg = -0.5 * ((tempo_est * ioi_score - ioi_obs) ** 2) * ioi_precision

    obs_prob = norm_term * np.exp(exp_arg)
    return obs_prob


class BernoulliPitchObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed notes at the current moment/state.

    Parameters
    ----------
    pitch_profiles : NDArrayFloat
        Pre-computed pitch profiles, for each separate possible pitch
        in the MIDI range. Used in calculating the pitch observation
        probabilities.
    """

    def __init__(self, pitch_profiles: NDArrayFloat):
        """
        The initialization method.

        Parameters
        ----------
        pitch_profiles : NDArrayFloat
            he pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self.pitch_profiles = pitch_profiles

    def __call__(self, observation: NDArrayFloat) -> NDArrayFloat:

        return compute_bernoulli_pitch_probabilities(
            pitch_obs=observation,
            pitch_profiles=self.pitch_profiles,
        )


class PitchIOIObservationModel(ObservationModel):

    def __init__(
        self,
        pitch_obs_prob_func: Callable[..., NDArrayFloat],
        ioi_obs_prob_func: Callable[..., NDArrayFloat],
        ioi_matrix: NDArrayFloat,
        pitch_prob_args: Optional[Dict[str, Any]] = None,
        ioi_prob_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(use_log_probabilities=False)

        # TODO: Add log probabilities
        self.pitch_obs_prob_func = pitch_obs_prob_func
        self.ioi_obs_prob_func = ioi_obs_prob_func
        self.pitch_prob_args = pitch_prob_args
        self.ioi_prob_args = ioi_prob_args
        self.ioi_matrix = ioi_matrix

    def __call__(self, observation: Any, *args, **kwargs) -> NDArrayFloat:

        pitch_obs, ioi_obs, tempo_est = observation
        ioi_idx = self.current_state if self.current_state is not None else 0

        ioi_score = self.ioi_matrix[ioi_idx]
        obs_prob = self.pitch_obs_prob_func(
            pitch_obs=pitch_obs,
            **self.pitch_obs_prob_args,
        ) * self.ioi_obs_prob_func(
            ioi_obs=ioi_obs,
            ioi_score=ioi_score,
            tempo_est=tempo_est,
            **self.ioi_prob_args,
        )
        return obs_prob


class BernoulliGaussianPitchIOIObservationModel(PitchIOIObservationModel):
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

        pitch_prob_args = dict(
            pitch_profiles=pitch_profiles,
        )
        ioi_prob_args = dict(
            ioi_precision=ioi_precision,
            norm_term=np.sqrt(0.5 * self.ioi_precision / np.pi),
        )
        PitchIOIObservationModel.__init__(
            self,
            pitch_obs_prob_func=compute_bernoulli_pitch_probabilities,
            ioi_obs_prob_func=compute_gaussian_ioi_observation_probability,
            ioi_matrix=ioi_matrix,
            pitch_prob_args=pitch_prob_args,
            ioi_prob_args=ioi_prob_args,
        )


class PitchIOIHMM(HiddenMarkovModel, OnlineAlignment):
    """
    Implements the behavior of a HiddenMarkovModel, specifically designed for
    the task of score following.

    Parameters
    ----------
    _transition_matrix : numpy.ndarray
        Matrix for computations of state transitions within the HMM.

    _observation_model : ObservationModel
        Object responsible for computing the observation probabilities for each
        state of the HMM.

    initial_distribution : numpy array
        The initial distribution of the model. If not given, it is assumed to
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

        HiddenMarkovModel.__init__(
            self,
            observation_model=observation_model,
            transition_model=ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            ),
            state_space=score_onsets,
        )

        OnlineAlignment.__init__(
            self,
            reference_features=observation_model,
        )

        self.tempo_model = tempo_model
        self.has_insertions = has_insertions

    def __call__(self, input):
        self.current_state = self.forward_algorithm_step(
            observation=input + (self.tempo_model.beat_period,),
            log_probabilities=False,
        )
        return self.state_space[self.current_state]

    @property
    def current_state(self):
        return self.observation_model.current_state

    @current_state.setter
    def current_state(self, state):
        self.observation_model.current_state = state
