#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements Hidden Markov Models for score following
"""

import time
import warnings
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import progressbar
import scipy.spatial.distance as sp_dist
from hiddenmarkov import (
    ConstantTransitionModel,
    HiddenMarkovModel,
    ObservationModel,
    TransitionModel,
)
from numpy.typing import NDArray
from scipy.signal import convolve
from scipy.stats import gumbel_l, norm

from matchmaker.base import OnlineAlignment
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END
from matchmaker.utils.errors import MatchmakerMissingParameterError
from matchmaker.utils.misc import (
    get_window_indices,
    interleave_with_constant,
    set_latency_stats,
)
from matchmaker.utils.tempo_models import (
    KalmanTempoModel,
    LinearTempoModel,
    MovingAverageTempoModel,
    ReactiveTempoModel,
    TempoModel,
)

# Alias for typing arrays
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

DEFAULT_GAUSSIAN_AUDIO_PRECISION = 2.0

DEFAULT_GAUSSIAN_AUDIO_IOI_PRECISION = 2.0
DEFAULT_GUMBEL_AUDIO_SCALE = 0.05
QUEUE_TIMEOUT = 10


class BaseHMM(HiddenMarkovModel, OnlineAlignment):
    """
    Base class for Hidden Markov Model alignment methods.

    Parameters
    ----------
    observation_model: ObservationModel
        An observation (data) model for computing the observation probabilities.

    transition_model: TransitionModel
        A transition model for computing the transition probabilities.

    score_positions: np.ndarray
        The hidden states (positions in reference time).

    tempo_model: Optional[TempoModel]
        A tempo model instance

    has_insertions: bool
        A boolean indicating whether the state space consider inserted notes.
    """

    observation_model: ObservationModel
    transition_model: TransitionModel
    score_positions: Union[NDArrayFloat, NDArrayInt]
    tempo_model: Optional[TempoModel]
    has_insertions: bool
    _alignment_path: List[Tuple[int, int]]
    queue: Optional[RECVQueue]

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        score_positions: Optional[Union[NDArrayFloat, NDArrayInt]] = None,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
        queue: Optional[RECVQueue] = None,
        patience: int = 10,
        **kwargs,
    ) -> None:
        HiddenMarkovModel.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=score_positions,
        )
        OnlineAlignment.__init__(
            self,
            score_positions=score_positions,
            queue=queue,
        )
        self.tempo_model = tempo_model
        self.has_insertions = has_insertions
        self.patience = patience
        self.input_index = 0

    def step(self, features) -> None:
        current_state = self.forward_algorithm_step(
            observation=features,
            log_probabilities=False,
        )
        self.current_index = current_state
        self.input_index += 1
        self.observation_model.current_state = current_state


class PitchHMM(BaseHMM):
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
        reference_features: np.ndarray,  # snote_array
        queue: Optional[RECVQueue] = None,
        transition_model: Optional[TransitionModel] = None,
        observation_model: Optional[ObservationModel] = None,
        transition_matrix: Optional[NDArrayFloat] = None,
        pitch_obs_prob_func: Optional[Callable[..., NDArrayFloat]] = None,
        pitch_prob_args: Optional[Dict[str, Any]] = None,
        initial_probabilities: Optional[np.ndarray] = None,
        has_insertions: bool = True,
        piano_range: bool = True,
        **kwargs,
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

        initial_distribution : numpy array
            The initial distribution of the model. If not given, it is asumed to
            be uniform.
            Default = None.
        """
        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )
        self.reference_features = reference_features
        self.perf_onset = None
        (
            observation_model,
            transition_matrix,
            initial_probabilities,
            unique_onsets,
        ) = self._build_hmm_modules(
            inserted_states=has_insertions,
            piano_range=piano_range,
        )

        if transition_model is not None and transition_matrix is not None:
            warnings.warn(
                "Both `transition_model` and `transition_matrix` were "
                "provided. Only `transition_model` will be used."
            )
        obs_model_params_given = [
            pitch_obs_prob_func is not None,
            pitch_prob_args is not None,
        ]
        if observation_model is not None and any(obs_model_params_given):
            warnings.warn(
                "`observation_model` and params were provided. "
                "Only `observation_model` will be used."
            )

        if observation_model is None and not all(obs_model_params_given):
            missing_params = [
                pn
                for pn, given in zip(
                    [
                        "pitch_obs_prob_func",
                        "pitch_prob_args",
                    ],
                    obs_model_params_given,
                )
                if not given
            ]
            raise ValueError(missing_params)

        if transition_model is None:
            transition_model = ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            )

        BaseHMM.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            score_positions=unique_onsets,
            has_insertions=has_insertions,
            queue=queue,
        )

    def _build_hmm_modules(
        self,
        piano_range: bool = True,
        inserted_states: bool = True,
    ):
        snote_array = self.reference_features
        unique_sonsets = np.unique(snote_array["onset_beat"])
        unique_sonset_idxs = [
            np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
        ]
        chord_pitches = [snote_array["pitch"][uix] for uix in unique_sonset_idxs]
        pitch_profiles = compute_discrete_pitch_profiles(
            chord_pitches=chord_pitches,
            piano_range=piano_range,
            inserted_states=inserted_states,
        )

        # observation model
        observation_model = BernoulliPitchObservationModel(
            pitch_profiles=pitch_profiles,
        )

        if inserted_states:
            unique_onsets_s = np.insert(
                unique_sonsets,
                np.arange(1, len(unique_sonsets)),
                (unique_sonsets[:-1] + 0.5 * np.diff(unique_sonsets)),
            )
        else:
            unique_onsets_s = unique_sonsets

        transition_matrix = stable_transition_matrix(
            n_states=len(unique_onsets_s),
            dist=gumbel_l,
            scale=1.0,  # 0.5,
            inserted_states=inserted_states,
        )
        initial_probabilities = init_dist(
            n_states=len(unique_onsets_s),
            dist=gumbel_l,
        )

        return (
            observation_model,
            transition_matrix,
            initial_probabilities,
            unique_onsets_s,
        )


def jiang_transition_matrix(
    n_states: int,
    trans_prob: float = 0.8,
    # frame_rate, sigma, transition_variance: float
) -> NDArrayFloat:
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            if j <= i:
                transition_matrix[i, j] = 0  # (1 - p1 if we want to go back)
            elif j == i:
                transition_matrix[i, j] = trans_prob
            else:
                transition_matrix[i, j] = (1 - trans_prob) / (n_states - j)

    return transition_matrix


def gaussian_transition_matrix(n_states: int) -> NDArrayFloat:
    # Use broadcasting to compute the Gaussian PDF for each pair of states
    # transition_matrix = norm.pdf(states[:, np.newaxis], loc=states, scale=1)
    transition_matrix = np.eye(n_states)
    transition_matrix = np.roll(transition_matrix, 1, axis=1)

    return transition_matrix


def jiang_transition_matrix_from_sequence(sequence, frame_rate, sigma):
    """Compute the transition matrix from a given sequence for the Jiang HMM model.

    Args:
        sequence (list): List of note indices representing the sequence from score.
            ex. [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9]

        frame_rate (float): Frame rate in Hz.
        sigma (float): Standard deviation for note duration.

    Returns:
        np.array: A dense matrix representing the transition probabilities.
    """
    total_frames = len(sequence)
    notes = list(set(sequence))
    n_notes = len(notes)

    # Define state space (note, age)
    score_positions = []
    for note in notes:
        ages = [i for i, x in enumerate(sequence) if x == note]
        score_positions.extend([(note, age) for age in ages])

    n_states = len(score_positions)
    transition_matrix = np.zeros((n_states, n_states))
    delta = 1 / frame_rate

    for i in range(n_states - 1):
        current_note, current_age = score_positions[i]
        next_note, next_age = score_positions[i + 1]

        # Calculate the mean duration based on the note index
        mean_duration = (current_note + 1) * delta
        stddev_duration = sigma

        if stddev_duration == 0:
            stddev_duration = 1e-6  # Prevent division by zero

        # Calculate the Gaussian CDF terms for transition probability
        phi_current = norm.cdf((current_age * delta - mean_duration) / stddev_duration)
        phi_next = norm.cdf(
            ((current_age + 1) * delta - mean_duration) / stddev_duration
        )

        if (1 - phi_current) == 0:
            p1 = 1  # Prevent division by zero
        else:
            p1 = (1 - phi_next) / (1 - phi_current)

        # Ensure p1 is between 0 and 1
        p1 = max(0, min(p1, 1))

        if current_note == next_note:
            # Staying in the same note, increasing the age
            transition_matrix[i, i + 1] = p1
        else:
            # Moving to the next note
            transition_matrix[i, i + 1] = 1 - p1

    # Ensure the last state transitions to itself
    transition_matrix[n_states - 1, n_states - 1] = 1.0

    return transition_matrix, score_positions


def simple_transition_matrix(
    n_states, trans_prob: float = 0.6, stay_prob: float = 0.3
) -> NDArrayFloat:
    if not (0 <= stay_prob <= 1) or not (0 <= trans_prob <= 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    if stay_prob + trans_prob > 1:
        raise ValueError("stay_prob + trans_prob must be <= 1.")

    matrix = np.zeros((n_states, n_states), dtype=np.float32)

    for i in range(n_states):
        if i == n_states - 1:
            matrix[i, i] = 1.0  # Absorbing state
            continue

        matrix[i, i] = stay_prob
        matrix[i, i + 1] = trans_prob

        # Distribute remaining probability over future states beyond i+1
        remaining_prob = 1.0 - stay_prob - trans_prob
        if i + 2 < n_states:
            num_other_states = n_states - (i + 2)
            uniform_prob = (
                remaining_prob / num_other_states if num_other_states > 0 else 0
            )
            matrix[i, i + 2 :] = uniform_prob
        else:
            # If no future states beyond i+1, give all remaining prob to i+1
            matrix[i, i + 1] += remaining_prob

    return matrix


def kalman_transition_matrix(n_states: int, transition_variance: float) -> NDArrayFloat:
    """
    Create a transition matrix based on a Kalman filter model.
    """
    transition_matrix = np.zeros((n_states, n_states))

    for i in range(n_states - 1):
        transition_matrix[i, i] = 1 - transition_variance
        transition_matrix[i, i + 1] = transition_variance

    transition_matrix[-1, -1] = 1  # Last state is an absorbing state

    return transition_matrix


def gumbel_transition_matrix(  # TODO check works for audio (parameter)
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


def stable_transition_matrix(  # TODO check works for audio (parameter)
    n_states: int,
    mp_trans_state: int = 1,
    dist=gumbel_l,
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
                transition_matrix[i] = dist.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2, scale=scale
                )
            else:
                transition_matrix[i] = dist.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
                )
        else:
            transition_matrix[i] = dist.pdf(
                np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
            )

    # Normalize transition matrix (so that it is a proper stochastic matrix):
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # Return the computed transition matrix:
    return transition_matrix


def init_dist(
    n_states: int,
    dist=gumbel_l,
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

    init_probs: np.ndarray = dist.pdf(
        np.arange(n_states),
        loc=loc,
        scale=prob_scale,
    )

    return init_probs


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


def compute_continous_pitch_profiles(  # TODO check separately
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


def compute_chord_matrix(chord_pitches: List[NDArrayInt]) -> NDArrayFloat:
    num_rows = len(chord_pitches)
    matrix = np.zeros((num_rows, 128), dtype=float)

    # Flatten the chord pitches and create corresponding row indices
    row_indices = np.repeat(np.arange(num_rows), [len(p) for p in chord_pitches])
    col_indices = np.concatenate(chord_pitches)

    # Set the specified indices to 1
    matrix[row_indices, col_indices] = 1

    return matrix


def compute_discrete_pitch_profiles(
    chord_pitches: NDArrayFloat,
    profile: NDArrayFloat = np.array([0.02, 0.02, 1, 0.02, 0.02]),
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
    chord_pitches : numpy array
        A 2D array of size (n_chords, 128) with the pitch of each chord.

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

    Returns
    -------
    pitch_profiles : numpy array
        The pre-computed pitch profiles.
    """

    chord_matrix = compute_chord_matrix(chord_pitches=chord_pitches)

    pitch_profiles = convolve(chord_matrix, profile[None, :], mode="same")

    if inserted_states:
        pitch_profiles = interleave_with_constant(array=pitch_profiles)
        pitch_profiles = pitch_profiles[:-1]
    # Add extra value
    pitch_profiles += eps

    # Check whether to trim and normalize:
    if piano_range:
        pitch_profiles = pitch_profiles[:, 21:109]
    if normalize:
        pitch_profiles /= pitch_profiles.sum(1, keepdims=True)

    return pitch_profiles


# Old version, to be deprecated.
def compute_discrete_pitch_profiles_old(
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


def compute_gaussian_audio_probabilities(
    audio_obs: NDArrayFloat,
    audio_features: NDArrayFloat,
    precision: float,
    norm_term: float,
) -> NDArrayFloat:
    """
    Compute Gaussian observations for audio features.
    """
    diff = audio_features - audio_obs

    exp_arg = -0.5 * precision * np.sum(diff**2, axis=1)

    obs_prob = norm_term * np.exp(exp_arg)

    return obs_prob


def compute_exponential_cosine_audio_probabilities(
    audio_obs: NDArrayFloat,
    audio_features: NDArrayFloat,
    rate: float,
    epsilon: float = 1e-10,
) -> NDArrayFloat:
    # Compute norms with epsilon to avoid division by zero
    obs_norm = np.linalg.norm(audio_obs.flatten()) + epsilon
    features_norm = np.linalg.norm(audio_features, axis=1, keepdims=True) + epsilon

    # Normalize the vectors
    audio_obs_norm = audio_obs.flatten() / obs_norm
    audio_features_norm = audio_features / features_norm

    # Cosine similarity
    cos_sim = np.dot(audio_features_norm, audio_obs_norm)
    # Compute cosine similarity
    cos_sim = np.dot(audio_features_norm, audio_obs_norm)

    # Convert to cosine distance
    cos_dist = 1.0 - cos_sim

    # Apply exponential decay
    obs_prob = rate * np.exp(-rate * cos_dist)

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


class GaussianAudioPitchObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed audio frame at the current moment/state.

    Parameters
    ----------
    audio_features : NDArrayFloat
        Audio features from the reference (score). Used in calculating
        the pitch observation probabilities.
    """

    def __init__(
        self,
        audio_features: NDArrayFloat,
        precision: float = 1,
    ):
        """
        The initialization method.

        Parameters
        ----------
        audio_features : NDArrayFloat
            he pre-computed audio features for the reference (e.g., score).
            Used in calculating the pitch observation probabilities.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self.audio_features = audio_features
        self.precision = precision
        self.norm_term = np.sqrt(0.5 * precision / np.pi)

    def __call__(self, observation: NDArrayFloat) -> NDArrayFloat:
        return compute_gaussian_audio_probabilities(
            audio_obs=observation,
            audio_features=self.audio_features,
            precision=self.precision,
            norm_term=self.norm_term,
        )


class CosineExpAudioPitchObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed audio frame at the current moment/state.

    Parameters
    ----------
    audio_features : NDArrayFloat
        Audio features from the reference (score). Used in calculating
        the pitch observation probabilities.
    """

    def __init__(
        self,
        audio_features: NDArrayFloat,
        rate: float = 1,
    ):
        """
        The initialization method.

        Parameters
        ----------
        audio_features : NDArrayFloat
            he pre-computed audio features for the reference (e.g., score).
            Used in calculating the pitch observation probabilities.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self.audio_features = audio_features
        self.rate = rate

    def __call__(self, observation: NDArrayFloat) -> NDArrayFloat:
        return compute_exponential_cosine_audio_probabilities(
            audio_obs=observation,
            audio_features=self.audio_features,
            rate=self.rate,
        )


class GaussianAudioPitchTempoObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed audio frame at the current moment/state.

    Parameters
    ----------
    audio_features : NDArrayFloat
        Audio features from the reference (score). Used in calculating
        the pitch observation probabilities.
    """

    def __init__(
        self,
        audio_features: NDArrayFloat,
        pitch_precision: float = 1,
        ioi_precision: float = 1,
    ):
        """
        The initialization method.

        Parameters
        ----------
        audio_features : NDArrayFloat
            he pre-computed audio features for the reference (e.g., score).
            Used in calculating the pitch observation probabilities.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self.audio_features = audio_features
        self.pitch_precision = pitch_precision
        self.pitch_norm_term = np.sqrt(0.5 * pitch_precision / np.pi)
        self.ioi_norm_term = np.sqrt(0.5 * ioi_precision / np.pi)
        self.ioi_precision = ioi_precision
        self.current_state = None
        self.states = np.arange(len(audio_features))

    def __call__(self, observation: NDArrayFloat) -> NDArrayFloat:
        pitch_obs, tempo_est = observation

        if self.current_state is None:
            estimated_state = 0
        else:
            estimated_state = self.current_state + tempo_est

        pitch_prob = compute_gaussian_audio_probabilities(
            audio_obs=pitch_obs,
            audio_features=self.audio_features,
            precision=self.pitch_precision,
            norm_term=self.pitch_norm_term,
        )

        exp_arg = -0.5 * ((self.states - estimated_state) ** 2) * self.ioi_precision

        tempo_prob = self.ioi_norm_term * np.exp(exp_arg)

        obs_prob = pitch_prob * tempo_prob

        return obs_prob


class CosineExpGaussianAudioPitchTempoObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed audio frame at the current moment/state.

    Parameters
    ----------
    audio_features : NDArrayFloat
        Audio features from the reference (score). Used in calculating
        the pitch observation probabilities.
    """

    def __init__(
        self,
        audio_features: NDArrayFloat,
        # ioi_matrix: Optional[NDArrayFloat] = None,
        pitch_rate: float = 1,
        ioi_precision: float = 1,
    ):
        """
        The initialization method.

        Parameters
        ----------
        audio_features : NDArrayFloat
            he pre-computed audio features for the reference (e.g., score).
            Used in calculating the pitch observation probabilities.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self.audio_features = audio_features
        self.pitch_rate = pitch_rate

        self.ioi_norm_term = np.sqrt(0.5 * ioi_precision / np.pi)
        self.ioi_precision = ioi_precision

        # if ioi_matrix is None:
        #     ioi_matrix = compute_ioi_matrix(
        #         unique_onsets=np.arange(len(audio_features))
        #     )
        # self.ioi_matrix = ioi_matrix
        self.current_state = None
        self.states = np.arange(len(audio_features))

    def __call__(self, observation: NDArrayFloat) -> NDArrayFloat:
        pitch_obs, tempo_est = observation

        # ioi_idx = self.current_state if self.current_state is not None else 0

        if self.current_state is None:
            estimated_state = 0
        else:
            estimated_state = self.current_state + tempo_est

        pitch_prob = compute_exponential_cosine_audio_probabilities(
            audio_obs=pitch_obs,
            audio_features=self.audio_features,
            rate=self.pitch_rate,
        )

        exp_arg = -0.5 * ((self.states - estimated_state) ** 2) * self.ioi_precision

        tempo_prob = self.ioi_norm_term * np.exp(exp_arg)

        obs_prob = pitch_prob * tempo_prob

        return obs_prob


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
            **self.pitch_prob_args,
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
            norm_term=np.sqrt(0.5 * ioi_precision / np.pi),
        )
        PitchIOIObservationModel.__init__(
            self,
            pitch_obs_prob_func=compute_bernoulli_pitch_probabilities,
            ioi_obs_prob_func=compute_gaussian_ioi_observation_probability,
            ioi_matrix=ioi_matrix,
            pitch_prob_args=pitch_prob_args,
            ioi_prob_args=ioi_prob_args,
        )


class ACCPitchIOIObservationModel(ObservationModel):
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

    def __init__(self, pitch_profiles, ioi_matrix, ioi_precision, piano_range=False):
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
        self.piano_range = piano_range

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
        if self.piano_range:
            pitch_prof_obs = np.zeros((1, 88))
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


class PitchIOIHMM(BaseHMM):
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
        reference_features: np.ndarray,  # snote_array
        queue: Optional[RECVQueue] = None,
        tempo_model: TempoModel = None,
        transition_model: Optional[TransitionModel] = None,
        observation_model: Optional[PitchIOIObservationModel] = None,
        transition_matrix: Optional[NDArrayFloat] = None,
        pitch_obs_prob_func: Optional[Callable[..., NDArrayFloat]] = None,
        ioi_obs_prob_func: Optional[Callable[..., NDArrayFloat]] = None,
        ioi_matrix: Optional[NDArrayFloat] = None,
        pitch_prob_args: Optional[Dict[str, Any]] = None,
        ioi_prob_args: Optional[Dict[str, Any]] = None,
        initial_probabilities: Optional[np.ndarray] = None,
        has_insertions: bool = False,
        piano_range: bool = False,
        **kwargs,
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
        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )
        self.reference_features = reference_features

        (
            observation_model,
            transition_matrix,
            initial_probabilities,
            tempo_model,
            unique_onsets,
        ) = self._build_hmm_modules(
            inserted_states=has_insertions,
            piano_range=piano_range,
            tempo_model=tempo_model,
        )

        if transition_model is not None and transition_matrix is not None:
            warnings.warn(
                "Both `transition_model` and `transition_matrix` were "
                "provided. Only `transition_model` will be used."
            )
        obs_model_params_given = [
            pitch_obs_prob_func is not None,
            ioi_obs_prob_func is not None,
            ioi_matrix is not None,
            pitch_prob_args is not None,
            ioi_prob_args is not None,
        ]
        if observation_model is not None and any(obs_model_params_given):
            warnings.warn(
                "`observation_model` and params were provided. "
                "Only `observation_model` will be used."
            )

        if observation_model is None and not all(obs_model_params_given):
            missing_params = [
                pn
                for pn, given in zip(
                    [
                        "pitch_obs_prob_func",
                        "ioi_obs_prob_func",
                        "ioi_matrix",
                        "pitch_prob_args",
                        "ioi_prob_args",
                    ],
                    obs_model_params_given,
                )
                if not given
            ]
            raise MatchmakerMissingParameterError(missing_params)

        if transition_model is None:
            transition_model = ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            )

        if observation_model is None:
            observation_model = PitchIOIObservationModel(
                pitch_obs_prob_func=pitch_obs_prob_func,
                ioi_obs_prob_func=ioi_obs_prob_func,
                ioi_matrix=ioi_matrix,
                pitch_prob_args=pitch_prob_args,
                ioi_prob_args=ioi_prob_args,
            )

        self.perf_onset = None
        self._prev_perf_time = None

        BaseHMM.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            score_positions=unique_onsets,
            tempo_model=tempo_model,
            has_insertions=has_insertions,
            queue=queue,
            **kwargs,
        )

    def step(self, features) -> None:
        perf_time = self.current_perf_time
        ioi_obs = (
            perf_time - self._prev_perf_time
            if self._prev_perf_time is not None
            else 0.0
        )
        self._prev_perf_time = perf_time

        current_state = self.forward_algorithm_step(
            observation=(features, ioi_obs, self.tempo_model.beat_period),
            log_probabilities=False,
        )
        self.input_index += 1

        prev_index = self.current_index
        if prev_index is None:
            prev_index = current_state

        if current_state > prev_index:
            update_tempo = (self.has_insertions and current_state % 2 == 0) or (
                not self.has_insertions
            )
            if update_tempo:
                self.tempo_model.update_beat_period(
                    performed_onset=perf_time,
                    score_onset=self.score_positions[current_state],
                )

        self.current_index = current_state
        self.observation_model.current_state = current_state

    def _build_hmm_modules(
        self,
        piano_range: bool = False,
        inserted_states: bool = True,
        observation_model=ACCPitchIOIObservationModel,
        tempo_model=KalmanTempoModel,
    ):
        snote_array = self.reference_features

        unique_sonsets = np.unique(snote_array["onset_beat"])
        unique_sonset_idxs = [
            np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
        ]
        chord_pitches = [snote_array["pitch"][uix] for uix in unique_sonset_idxs]

        pitch_profiles = compute_discrete_pitch_profiles(
            chord_pitches=chord_pitches,
            piano_range=piano_range,
            inserted_states=inserted_states,
        )
        ioi_matrix = compute_ioi_matrix(
            unique_onsets=unique_sonsets,
            inserted_states=inserted_states,
        )

        observation_model = observation_model(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=1,
            piano_range=piano_range,
        )

        if inserted_states:
            unique_onsets_s = np.insert(
                unique_sonsets,
                np.arange(1, len(unique_sonsets)),
                (unique_sonsets[:-1] + 0.5 * np.diff(unique_sonsets)),
            )
        else:
            unique_onsets_s = unique_sonsets

        if tempo_model == KalmanTempoModel:
            tempo_model = tempo_model(
                init_beat_period=60 / 100,
                init_score_onset=unique_sonsets.min(),
            )
        else:
            tempo_model = tempo_model(
                init_score_onset=unique_sonsets.min(),
                init_beat_period=60 / 100,
            )

        transition_matrix = stable_transition_matrix(
            n_states=len(ioi_matrix[0]),
            dist=gumbel_l,
            scale=1.0,  # 0.5,
            inserted_states=inserted_states,
        )
        initial_probabilities = init_dist(
            n_states=len(ioi_matrix[0]),
            dist=gumbel_l,
        )

        return (
            observation_model,
            transition_matrix,
            initial_probabilities,
            tempo_model,
            unique_onsets_s,
        )


def build_local_transition_matrix(n_states, window):
    matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        # Determine window bounds
        start = max(0, i - window)
        end = min(n_states, i + window + 1)
        width = end - start

        # Uniform probability across window
        matrix[i, start:end] = 1.0 / width

    return matrix


class GaussianAudioPitchHMM(BaseHMM):
    """
    Audio Gaussian HMM
    """

    def __init__(
        self,
        reference_features: np.ndarray,  # audio features
        queue: Optional[RECVQueue] = None,
        transition_model: Optional[TransitionModel] = None,
        observation_model: Optional[PitchIOIObservationModel] = None,
        transition_matrix: Optional[NDArrayFloat] = None,
        precision: Optional[float] = DEFAULT_GAUSSIAN_AUDIO_PRECISION,
        initial_probabilities: Optional[np.ndarray] = None,
        score_positions: Optional[NDArray] = None,
        patience: int = 50,
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
        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )

        if transition_model is not None and transition_matrix is not None:
            warnings.warn(
                "Both `transition_model` and `transition_matrix` were "
                "provided. Only `transition_model` will be used."
            )
        obs_model_params_given = [
            precision is not None,
        ]
        if observation_model is not None and any(obs_model_params_given):
            warnings.warn(
                "`observation_model` and params were provided. "
                "Only `observation_model` will be used."
            )

        if observation_model is None and not all(obs_model_params_given):
            missing_params = [
                pn
                for pn, given in zip(
                    [
                        "precision",
                    ],
                    obs_model_params_given,
                )
                if not given
            ]
            raise MatchmakerMissingParameterError(missing_params)

        if transition_model is None:
            if transition_matrix is None:
                transition_matrix = simple_transition_matrix(
                    n_states=len(reference_features),
                    trans_prob=0.5,
                    stay_prob=0.5,
                )
            if initial_probabilities is None:
                initial_probabilities = gumbel_init_dist(
                    n_states=len(reference_features)
                )
            transition_model = ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            )

        if observation_model is None:
            observation_model = GaussianAudioPitchObservationModel(
                audio_features=reference_features,
                precision=(
                    precision
                    if precision is not None
                    else DEFAULT_GAUSSIAN_AUDIO_PRECISION
                ),
            )
        self.perf_onset = None

        BaseHMM.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            score_positions=(
                score_positions
                if score_positions is not None
                else np.arange(len(reference_features))
            ),
            tempo_model=None,
            has_insertions=False,
            queue=queue,
            patience=patience,
        )

        self.input_features = None
        self.distance_func = "Euclidean"

    def __call__(self, input, *args, **kwargs):
        frame_index = args[0] if args else None
        frame, f_time = input
        current_state = self.forward_algorithm_step(
            observation=frame,
            log_probabilities=False,
        )
        self._alignment_path.append((self.input_index, current_state))
        self.input_index = self.input_index + 1 if frame_index is None else frame_index

        self.current_index = current_state
        self.observation_model.current_state = current_state

        return self.current_index

    def run(self, verbose: bool = True):
        same_state_counter = 0
        empty_counter = 0
        if verbose:
            pbar = progressbar.ProgressBar(
                max_val=self.n_states,  # redirect_stdout=True
            )
            pbar.start()

        while self.is_still_following():
            prev_state = self.current_index
            queue_input = self.queue.get(timeout=QUEUE_TIMEOUT)
            if queue_input is STREAM_END:
                break
            features, f_time = queue_input
            self.last_queue_update = time.time()
            self.input_features = (
                np.concatenate((self.input_features, features))
                if self.input_features is not None
                else features
            )
            if queue_input is not None:
                current_state = self(queue_input)
                empty_counter = 0
                if current_state == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0
                latency = time.time() - self.last_queue_update
                self.latency_stats = set_latency_stats(
                    latency, self.latency_stats, self.input_index
                )
                if verbose:
                    pbar.update(int(current_state))
                yield float(self.score_positions[current_state])

            if verbose:
                pbar.finish()
        return self.alignment_path


class GaussianAudioPitchTempoHMM(BaseHMM):
    """
    Audio Gaussian HMM
    """

    def __init__(
        self,
        reference_features: np.ndarray,  # audio features
        queue: Optional[RECVQueue] = None,
        transition_model: Optional[TransitionModel] = None,
        observation_model: Optional[PitchIOIObservationModel] = None,
        tempo_model: Optional[TempoModel] = None,
        transition_matrix: Optional[NDArrayFloat] = None,
        pitch_precision: Optional[float] = DEFAULT_GAUSSIAN_AUDIO_PRECISION,
        ioi_precision: Optional[float] = DEFAULT_GAUSSIAN_AUDIO_IOI_PRECISION,
        transition_scale: Optional[float] = DEFAULT_GUMBEL_AUDIO_SCALE,
        initial_probabilities: Optional[np.ndarray] = None,
        score_positions: Optional[NDArray] = None,
        patience: int = 200,
        **kwargs,
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
        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )

        if transition_model is not None and transition_matrix is not None:
            warnings.warn(
                "Both `transition_model` and `transition_matrix` were "
                "provided. Only `transition_model` will be used."
            )
        obs_model_params_given = [
            pitch_precision is not None,
        ]
        if observation_model is not None and any(obs_model_params_given):
            warnings.warn(
                "`observation_model` and params were provided. "
                "Only `observation_model` will be used."
            )

        if observation_model is None and not all(obs_model_params_given):
            missing_params = [
                pn
                for pn, given in zip(
                    [
                        "precision",
                    ],
                    obs_model_params_given,
                )
                if not given
            ]
            raise MatchmakerMissingParameterError(missing_params)

        if transition_model is None:
            if transition_matrix is None:
                transition_matrix = simple_transition_matrix(
                    n_states=len(reference_features),
                    trans_prob=0.475,
                    stay_prob=0.45,
                )

            if initial_probabilities is None:
                initial_probabilities = gumbel_init_dist(
                    n_states=len(reference_features)
                )
            transition_model = ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            )

        if observation_model is None:
            observation_model = GaussianAudioPitchTempoObservationModel(
                audio_features=reference_features,
                pitch_precision=(
                    pitch_precision
                    if pitch_precision is not None
                    else DEFAULT_GAUSSIAN_AUDIO_PRECISION
                ),
                ioi_precision=(
                    ioi_precision
                    if ioi_precision is not None
                    else DEFAULT_GAUSSIAN_AUDIO_IOI_PRECISION
                ),
            )

        if tempo_model is None:
            tempo_model = ReactiveTempoModel(
                init_beat_period=1.0,
                init_score_onset=0,
            )
        self.perf_onset = None
        self.input_features = None
        self.distance_func = "Euclidean"

        BaseHMM.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            score_positions=(
                score_positions
                if score_positions is not None
                else np.arange(len(reference_features))
            ),
            tempo_model=tempo_model,
            has_insertions=False,
            queue=queue,
            patience=patience,
        )

    def __call__(self, input, *args, **kwargs):
        frame_index = args[0] if args else None
        frame, f_time = input

        current_state = self.forward_algorithm_step(
            observation=(
                frame,
                self.tempo_model.beat_period,
            ),
            log_probabilities=False,
        )

        self._alignment_path.append((self.input_index, current_state))
        self.input_index = self.input_index + 1 if frame_index is None else frame_index

        if current_state >= self.current_index:
            # Only update tempo if jump is forward
            # current_so = self.score_positions[current_state]
            self.tempo_model.update_beat_period(
                performed_onset=self.input_index,
                score_onset=current_state,
            )
        self.perf_onset = f_time
        self.current_index = current_state
        self.observation_model.current_state = current_state

        return self.current_index

    def run(self, verbose: bool = True):
        same_state_counter = 0
        empty_counter = 0
        if verbose:
            pbar = progressbar.ProgressBar(
                max_val=self.n_states,  # redirect_stdout=True
            )
            pbar.start()

        while self.is_still_following():
            prev_state = self.current_index
            queue_input = self.queue.get(timeout=QUEUE_TIMEOUT)
            if queue_input is STREAM_END:
                break
            features, f_time = queue_input
            self.last_queue_update = time.time()
            self.input_features = (
                np.concatenate((self.input_features, features))
                if self.input_features is not None
                else features
            )
            if queue_input is not None:
                current_state = self(queue_input)
                empty_counter = 0
                if current_state == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0
                latency = time.time() - self.last_queue_update
                self.latency_stats = set_latency_stats(
                    latency, self.latency_stats, self.input_index
                )
                if verbose:
                    pbar.update(int(current_state))
                yield float(self.score_positions[current_state])

            if verbose:
                pbar.finish()
        return self.alignment_path
