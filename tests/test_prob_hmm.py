#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the prob/hmm.py module.
"""

import unittest

import numpy as np
import partitura as pt
from hiddenmarkov import CategoricalObservationModel, ConstantTransitionModel

from matchmaker import EXAMPLE_MATCH, EXAMPLE_PIECES
from matchmaker.features.midi import PitchProcessor
from matchmaker.prob.hmm import (
    BaseHMM,
    BernoulliGaussianPitchIOIObservationModel,
    BernoulliPitchObservationModel,
    GaussianAudioPitchHMM,
    PitchHMM,
    PitchIOIHMM,
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
        states = np.array([0.0, 1.0])  # numeric states for float-position contract
        observation_probabilities = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
        transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
        expected_sequence = np.array([0.0, 0.0, 1.0])
        observation_model = CategoricalObservationModel(observation_probabilities, obs)

        init_probabilities = np.array([0.6, 0.4])

        transition_model = ConstantTransitionModel(
            transition_probabilities, init_probabilities
        )

        hmm = BaseHMM(
            observation_model=observation_model,
            transition_model=transition_model,
            score_positions=states,
            tempo_model=None,
            has_insertions=False,
        )

        for ob, ex in zip(observations, expected_sequence):
            hmm(ob, 0.0)  # __call__(features, perf_time)
            self.assertEqual(hmm.score_positions[hmm.current_index], ex)

        self.assertIsInstance(hmm.alignment_path, np.ndarray)


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
            # inserted_states=False,
        )

        initial_probabilities = np.zeros(len(chord_pitches)) + 1e-6
        initial_probabilities[0] = 1
        initial_probabilities /= initial_probabilities.sum()

        hmm = PitchHMM(
            reference_features=snote_array,
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            initial_probabilities=initial_probabilities,
            has_insertions=False,
        )

        observations = process_midi_offline(
            perf_info=perf,
            processor=PitchProcessor(piano_range=True),
        )

        for obs in observations:
            if obs is not None:
                # MidiStream attaches perf_time, so obs is already
                # (pitch_array, perf_time)
                cp = hmm(*obs)
                # __call__ returns the float beat (current_position)
                self.assertTrue(cp in unique_sonsets)

        self.assertTrue(isinstance(hmm.alignment_path, np.ndarray))


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

        tempo_model = ReactiveTempoModel

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
            processor=PitchProcessor(piano_range=True),
        )

        for obs in observations:
            if obs is not None:
                # processor returns (pitch_array, chord_onset_time)
                cp = hmm(*obs)
                # __call__ returns the float beat (current_position)
                self.assertTrue(cp in unique_sonsets)

        self.assertTrue(isinstance(hmm.alignment_path, np.ndarray))


class TestBernoulliGaussianPitchIOIHMM(unittest.TestCase):
    def test_init(self):
        self.score_file = EXAMPLE_PIECES["bach_fugue"]["score"]
        self.performance_file_audio = EXAMPLE_PIECES["bach_fugue"]["audio"]
        self.performance_file_midi = EXAMPLE_PIECES["bach_fugue"]["midi"]
        self.performance_file_annotations = EXAMPLE_PIECES["bach_fugue"]["annotations"]

        self.performance = process_midi_offline(
            perf_info=self.performance_file_midi,
            processor=PitchProcessor(piano_range=True),
        )
