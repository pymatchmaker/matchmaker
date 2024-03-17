#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements Hidden Markov Models for score following

TODO
----
* The definition of the HMMs need to be updated to work with audio
"""
from typing import Optional, Union

import numpy as np
import scipy.spatial.distance as sp_dist
from hiddenmarkov import (
    ConstantTransitionModel,
    HiddenMarkovModel,
    ObservationModel,
    TransitionModel,
)
from scipy.stats import gumbel_l

from matchmaker.base import OnlineAlignment
from matchmaker.utils.tempo_models import TempoModel
from numpy.typing import NDArray

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
        A tempo model

    
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
        raise NotImplementedError


class PitchHMM(BaseHMM):
    """ """

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
    pitch_prof_obs: NDArrayFloat,
    pitch_profiles: NDArrayFloat,
) -> NDArrayFloat:
    """
    Compute pitch observation probabilities
    """

    # Compute Bernoulli probability:
    pitch_prob = (pitch_profiles**pitch_prof_obs) * (
        (1 - pitch_profiles) ** (1 - pitch_prof_obs)
    )

    obs_prob = np.prod(pitch_prob, 1)

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
            pitch_prof_obs=observation,
            pitch_profiles=self.pitch_profiles,
        )
    

class BernoulliGaussianPitchIOIObservationModel(ObservationModel):
    pass




