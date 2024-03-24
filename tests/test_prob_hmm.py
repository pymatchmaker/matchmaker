#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the prob/hmm.py module.
"""
import unittest
import numpy as np
import partitura as pt

from matchmaker.features.midi import PitchProcessor
from matchmaker import EXAMPLE_MATCH
from matchmaker.prob.hmm import (
    gumbel_transition_matrix,
    gumbel_init_dist,
    compute_discrete_pitch_profiles,
    PitchHMM,
    BernoulliPitchObservationModel,
    BaseHMM
)

from hiddenmarkov import CategoricalObservationModel, ConstantTransitionModel

from tests.utils import process_midi_offline


class TestBaseHMM(unittest.TestCase):

    def test_init(self):
        # Non musical example, to test initialization

        obs = ("normal", "cold", "dizzy")
        observations = ("normal", "cold", "dizzy")
        states = np.array(["Healthy", "Fever"])
        observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
        transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
        expected_sequence = np.array(["Healthy", "Healthy", "Fever"])
        observation_model = CategoricalObservationModel(
            observation_probabilities, obs
        )

        init_probabilities = np.array([0.6, 0.4])

        transition_model = ConstantTransitionModel(
            transition_probabilities, init_probabilities
        )

        hmm = BaseHMM(
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=states,
            tempo_model=None,
            has_insertions=False
        )

        for ob, ex in zip(observations, expected_sequence):
            self.assertTrue(hmm(ob) == ex)



class TestPitchHMM(unittest.TestCase):

    def test_init(self):

        perf, _, score = pt.load_match(EXAMPLE_MATCH, create_score=True)

        snote_array = score.note_array()

        unique_sonsets = np.unique(snote_array["onset_beat"])

        unique_sonset_idxs = [
            np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
        ]

        chord_pitches = [snote_array["pitch"][uix] for uix in unique_sonset_idxs]

        pitch_profiles = compute_discrete_pitch_profiles(
            chord_pitches=chord_pitches,
            piano_range=True,
            inserted_states=False,
        )

        observation_model = BernoulliPitchObservationModel(
            pitch_profiles=pitch_profiles,
        )

        transition_matrix = gumbel_transition_matrix(
            n_states=len(chord_pitches),
            inserted_states=False,
        )

        hmm = PitchHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=unique_sonsets,
            initial_probabilities=None,
            has_insertions=False,
        )

        observations = process_midi_offline(
            perf_info=perf,
            features=[PitchProcessor(piano_range=True)],
        )
        
        for obs in observations:
            if obs[0] is not None:
                cp = hmm(obs[0])
                self.assertTrue(cp in unique_sonsets)
        
        self.assertTrue(isinstance(hmm.warping_path, np.ndarray))



