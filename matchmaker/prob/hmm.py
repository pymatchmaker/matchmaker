#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements Hidden Markov Models for score following
"""
from typing import Optional, Union, Tuple, Dict, Any, Callable

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

# Alias for typing arrays
NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]


class BaseHMM(HiddenMarkovModel, OnlineAlignment):
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

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        state_space: Optional[Union[NDArrayFloat, NDArrayInt]] = None,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
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

    def __call__(self, input: NDArrayFloat) -> float:
        raise NotImplementedError(
            "This method needs to be implemented in the subclasses"
        )


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

    def __call__(self, input: NDArrayFloat) -> float:

        current_state = self.forward_algorithm_step(
            observation=input,
            log_probabilities=False,
        )

        return self.state_space[current_state]


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


class BernoulliGaussianPitchIOIObservationModel(ObservationModel):
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
        self.pitch_profiles = pitch_profiles
        self.ioi_matrix = ioi_matrix
        self.ioi_precision = ioi_precision
        # Compute the IOI normalization term:
        self.ioi_norm_term = np.sqrt(0.5 * self.ioi_precision / np.pi)
        self.current_state = None

    def __call__(self, observation: Tuple[NDArrayFloat, float, float]) -> NDArray:
        pitch_obs, ioi_obs, tempo_est = observation
        ioi_idx = self.current_state if self.current_state is not None else 0

        ioi_score = self.ioi_matrix[ioi_idx]
        observation_prob = compute_bernoulli_pitch_probabilities(
            pitch_obs=pitch_obs,
            pitch_profiles=self.pitch_profiles,
        ) * compute_gaussian_ioi_observation_probability(
            ioi_obs=ioi_obs,
            ioi_score=ioi_score,
            tempo_est=tempo_est,
            ioi_precision=self.ioi_precision,
            norm_term=self.ioi_norm_term,
        )

        return observation_prob


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
