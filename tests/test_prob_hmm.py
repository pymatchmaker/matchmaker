#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the prob/hmm.py module.
"""
import unittest

import numpy as np
import partitura as pt
from hiddenmarkov import CategoricalObservationModel, ConstantTransitionModel
from matplotlib import pyplot as plt
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment

from matchmaker import EXAMPLE_AUDIO, EXAMPLE_MATCH, EXAMPLE_SCORE
from matchmaker.features.audio import (
    HOP_LENGTH,  # ChromagramIOIProcessor,
    SAMPLE_RATE,
    ChromagramProcessor,
)
from matchmaker.features.midi import PitchIOIProcessor, PitchProcessor
from matchmaker.prob.hmm import (
    BaseHMM,
    BernoulliGaussianPitchIOIObservationModel,
    BernoulliPitchObservationModel,
    PitchHMM,
    PitchIOIHMM,
    compute_continous_pitch_profiles,
    compute_discrete_pitch_profiles,
    compute_discrete_pitch_profiles_old,
    compute_ioi_matrix,
    gumbel_init_dist,
    gumbel_transition_matrix,
    simple_transition_matrix,
)
from matchmaker.utils.tempo_models import ReactiveTempoModel
from tests.utils import process_audio_offline, process_midi_offline


class TestBaseHMM(unittest.TestCase):
    def test_init(self):
        # Non musical example, to test initialization

        obs = ("normal", "cold", "dizzy")
        observations = ("normal", "cold", "dizzy")
        states = np.array(["Healthy", "Fever"])
        observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
        transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
        expected_sequence = np.array(["Healthy", "Healthy", "Fever"])
        observation_model = CategoricalObservationModel(observation_probabilities, obs)

        init_probabilities = np.array([0.6, 0.4])

        transition_model = ConstantTransitionModel(
            transition_probabilities, init_probabilities
        )

        hmm = BaseHMM(
            observation_model=observation_model,
            transition_model=transition_model,
            state_space=states,
            tempo_model=None,
            has_insertions=False,
        )

        for ob, ex in zip(observations, expected_sequence):
            self.assertTrue(hmm.state_space[hmm(ob)] == ex)

        self.assertIsInstance(hmm.warping_path, np.ndarray)


class TestPitchHMM(unittest.TestCase):
    def test_symbolic(self):

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

        pitch_profiles_old = compute_discrete_pitch_profiles_old(
            chord_pitches=chord_pitches,
            piano_range=True,
            inserted_states=False,
        )

        observation_model = BernoulliPitchObservationModel(
            pitch_profiles=pitch_profiles,
        )

        transition_matrix = simple_transition_matrix(
            n_states=len(chord_pitches),
            inserted_states=False,
        )

        initial_probabilities = np.zeros(len(chord_pitches)) + 1e-6
        initial_probabilities[0] = 1
        initial_probabilities /= initial_probabilities.sum()

        hmm = PitchHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=unique_sonsets,
            initial_probabilities=initial_probabilities,
            has_insertions=False,
        )

        observations = process_midi_offline(
            perf_info=perf,
            processor=PitchProcessor(piano_range=True),
        )

        for obs in observations:
            if obs is not None:
                cp = hmm(obs)
                # print(cp)
                self.assertTrue(hmm.state_space[cp] in unique_sonsets)

        self.assertTrue(isinstance(hmm.warping_path, np.ndarray))

class TestPitchIOIHMM(unittest.TestCase):
    def test_symbolic(self):

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

        ioi_matrix = compute_ioi_matrix(
            unique_onsets=unique_sonsets,
            inserted_states=False,
        )

        observation_model = BernoulliGaussianPitchIOIObservationModel(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=1,
        )

        transition_matrix = gumbel_transition_matrix(
            n_states=len(chord_pitches),
            inserted_states=False,
        )

        initial_probabilities = gumbel_init_dist(
            n_states=len(chord_pitches),
        )

        tempo_model = ReactiveTempoModel(init_score_onset=unique_sonsets.min())

        hmm = PitchIOIHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            reference_features=snote_array,
            initial_probabilities=initial_probabilities,
            has_insertions=False,
            tempo_model=tempo_model,
            queue=None,
        )

        observations = process_midi_offline(
            perf_info=perf,
            processor=PitchIOIProcessor(piano_range=True),
        )

        for obs in observations:
            if obs is not None:
                cp = hmm(obs)
                self.assertTrue(hmm.state_space[cp] in unique_sonsets)

        self.assertTrue(isinstance(hmm.warping_path, np.ndarray))
