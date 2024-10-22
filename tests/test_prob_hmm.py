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
    HOP_LENGTH,
    SAMPLE_RATE,
    ChromagramIOIProcessor,
    ChromagramProcessor,
)
from matchmaker.features.midi import PitchIOIProcessor, PitchProcessor
from matchmaker.io.audio import CHUNK_SIZE
from matchmaker.prob.hmm import (
    BaseHMM,
    BernoulliGaussianPitchIOIObservationModel,
    BernoulliPitchObservationModel,
    PitchHMM,
    PitchIOIHMM,
    compute_continous_pitch_profiles,
    compute_discrete_pitch_profiles,
    compute_ioi_matrix,
    gumbel_init_dist,
    gumbel_transition_matrix,
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
            self.assertTrue(hmm(ob) == ex)


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

        observation_model = BernoulliPitchObservationModel(
            pitch_profiles=pitch_profiles,
        )

        transition_matrix = gumbel_transition_matrix(
            n_states=len(chord_pitches),
            inserted_states=False,
        )

        initial_probabilities = gumbel_init_dist(
            n_states=len(chord_pitches),
        )

        hmm = PitchHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=unique_sonsets,
            initial_probabilities=initial_probabilities,
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

    def test_audio(self):

        perf, alignment, score = pt.load_match(EXAMPLE_MATCH, create_score=True)

        # Get score features
        score_features = process_audio_offline(
            perf_info=score,
            processor=ChromagramProcessor(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            ),
            hop_length=HOP_LENGTH,
            sample_rate=SAMPLE_RATE,
            chunk_size=1,
        )

        score_features = np.vstack(score_features)
        score_feature_times = np.arange(len(score_features)) * HOP_LENGTH / SAMPLE_RATE

        snote_array = score.note_array()

        unique_sonsets = np.unique(snote_array["onset_beat"])

        pitch_profiles = compute_continous_pitch_profiles(  # TODO check this (with simple chromatic scale MIDI/audio features)
            spectral_features=score_features,
            spectral_feature_times=score_feature_times,
            onset_times=unique_sonsets,
            inserted_states=False,
        )

        observation_model = BernoulliPitchObservationModel(
            pitch_profiles=pitch_profiles,
        )

        transition_matrix = gumbel_transition_matrix(
            n_states=len(unique_sonsets),
            inserted_states=False,
        )

        initial_probabilities = gumbel_init_dist(
            n_states=len(unique_sonsets),
        )

        hmm = PitchHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=unique_sonsets,
            initial_probabilities=initial_probabilities,
            has_insertions=False,
        )

        # TODO: Check issue with librosa loading an empty file
        # https://stackoverflow.com/questions/74496808/mp3-loading-using-librosa-return-empty-data-when-start-time-metadata-is-0
        observations = process_audio_offline(
            perf_info=score,
            # perf_info=EXAMPLE_AUDIO,
            processor=ChromagramProcessor(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
            ),
            hop_length=HOP_LENGTH,
            sample_rate=SAMPLE_RATE,
            chunk_size=4,
        )
        observations = np.vstack(observations)
        for obs in observations:
            cp = hmm(obs)
            self.assertTrue(cp in unique_sonsets)

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
            score_onsets=unique_sonsets,
            initial_probabilities=initial_probabilities,
            has_insertions=False,
            tempo_model=tempo_model,
        )

        observations = process_midi_offline(
            perf_info=perf,
            features=[PitchIOIProcessor(piano_range=True)],
        )

        for obs in observations:
            if obs[0] is not None:
                cp = hmm(obs[0])
                self.assertTrue(cp in unique_sonsets)

        self.assertTrue(isinstance(hmm.warping_path, np.ndarray))

    def test_symbolic_insertions(self):

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
            inserted_states=True,
        )

        ioi_matrix = compute_ioi_matrix(
            unique_onsets=unique_sonsets,
            inserted_states=True,
        )

        state_space = ioi_matrix[0]
        n_states = len(state_space)

        observation_model = BernoulliGaussianPitchIOIObservationModel(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=1,
        )

        transition_matrix = gumbel_transition_matrix(
            n_states=n_states,
            inserted_states=True,
        )

        initial_probabilities = gumbel_init_dist(
            n_states=n_states,
        )

        tempo_model = ReactiveTempoModel(init_score_onset=unique_sonsets.min())

        hmm = PitchIOIHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=state_space,
            initial_probabilities=initial_probabilities,
            has_insertions=True,
            tempo_model=tempo_model,
        )

        observations = process_midi_offline(
            perf_info=perf,
            features=[PitchIOIProcessor(piano_range=True)],
        )

        for obs in observations:
            if obs[0] is not None:
                cp = hmm(obs[0])
                self.assertTrue(cp in unique_sonsets)

        self.assertTrue(isinstance(hmm.warping_path, np.ndarray))

    # def test_audio(self):

    #     perf, _, score = pt.load_match(EXAMPLE_MATCH, create_score=True)

    #     snote_array = score.note_array()

    #     unique_sonsets = np.unique(snote_array["onset_beat"])

    #     unique_sonset_idxs = [
    #         np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
    #     ]

    #     ioi_matrix = compute_ioi_matrix(
    #         unique_onsets=unique_sonsets,
    #         inserted_states=False,
    #     )

    #     state_space = ioi_matrix[0]
    #     n_states = len(state_space)

    #     # Get score features
    #     score_features = process_audio_offline(
    #         perf_info=score,
    #         features=[
    #             ChromagramIOIProcessor(
    #                 sample_rate=SAMPLE_RATE,
    #                 hop_length=HOP_LENGTH,
    #             )
    #         ],
    #         hop_length=HOP_LENGTH,
    #         sample_rate=SAMPLE_RATE,
    #         chunk_size=4,
    #         include_ftime=True,
    #     )

    #     # TODO: update this line when the api for score features is changed.
    #     score_features = np.vstack([sf[0][0] for sf in score_features])

    #     score_feature_times = np.arange(len(score_features)) * HOP_LENGTH / SAMPLE_RATE

    #     snote_array = score.note_array()

    #     unique_sonsets = np.unique(snote_array["onset_beat"])

    #     pitch_profiles = compute_continous_pitch_profiles(
    #         spectral_features=score_features,
    #         spectral_feature_times=score_feature_times,
    #         onset_times=unique_sonsets,
    #         inserted_states=False,
    #     )

    #     observation_model = BernoulliGaussianPitchIOIObservationModel(
    #         pitch_profiles=pitch_profiles,
    #         ioi_matrix=ioi_matrix,
    #         ioi_precision=1,
    #     )

    #     transition_matrix = gumbel_transition_matrix(
    #         n_states=n_states,
    #         inserted_states=False,
    #     )

    #     tempo_model = ReactiveTempoModel(init_score_onset=unique_sonsets.min())

    #     initial_probabilities = gumbel_init_dist(
    #         n_states=n_states,
    #     )

    #     hmm = PitchIOIHMM(
    #         observation_model=observation_model,
    #         transition_matrix=transition_matrix,
    #         score_onsets=state_space,
    #         initial_probabilities=initial_probabilities,
    #         has_insertions=False,
    #         tempo_model=tempo_model,
    #     )

    #     observations = process_audio_offline(
    #         perf_info=perf,
    #         features=[
    #             ChromagramIOIProcessor(
    #                 sample_rate=SAMPLE_RATE,
    #                 hop_length=HOP_LENGTH,
    #             )
    #         ],
    #         hop_length=HOP_LENGTH,
    #         sample_rate=SAMPLE_RATE,
    #         chunk_size=1,
    #         include_ftime=True,
    #     )

    #     # observations = np.vstack(observations)
    #     for obs in observations:
    #         cp = hmm(obs)
    #         self.assertTrue(cp in unique_sonsets)

    #     self.assertTrue(isinstance(hmm.warping_path, np.ndarray))
