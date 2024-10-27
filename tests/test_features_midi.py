#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the features/midi.py module
"""
import unittest

import numpy as np
import partitura as pt
from partitura.performance import PerformedPart

from matchmaker import EXAMPLE_MATCH, EXAMPLE_PERFORMANCE, EXAMPLE_SCORE
from matchmaker.features.midi import (
    PitchIOIProcessor,
    PitchProcessor,
    compute_features_from_symbolic,
)
from tests.utils import process_midi_offline


class TestPitchProcessor(unittest.TestCase):
    # @unittest.skipIf(True, "")
    def test_processor(self):

        note_array = np.empty(
            13,
            dtype=[
                ("pitch", int),
                ("onset_sec", float),
                ("duration_sec", float),
                ("velocity", int),
                ("id", str),
            ],
        )
        for i, pitch in enumerate(range(60, 73)):

            note_array[i] = (pitch, i, 0.5, 64, f"n{i}")

        perf = PerformedPart.from_note_array(note_array)

        feature_processor = PitchProcessor(
            piano_range=False,
            return_pitch_list=False,
        )
        feature_processor_pr = PitchProcessor(
            piano_range=True,
            return_pitch_list=False,
        )
        feature_processor_pl = PitchProcessor(
            piano_range=False,
            return_pitch_list=True,
        )

        feature_processor_pl_pr = PitchProcessor(
            piano_range=True,
            return_pitch_list=True,
        )
        # For coverage of the reset method, since it does not
        # do anything in this case.
        feature_processor.reset()
        polling_period = 0.01

        outputs = []
        for processor in [feature_processor, feature_processor_pr, feature_processor_pl, feature_processor_pl_pr,]:
            output = process_midi_offline(
                perf_info=perf,
                processor=feature_processor,
                polling_period=polling_period,
            )

            outputs.append(output)

        non_none_outputs = 0
        for output in outputs:

            if output[0] is not None:
                print("nn")
                pitch_obs = output[0]
                self.assertTrue(len(pitch_obs) == 128)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs)

            if output[1] is not None:
                print("nnn")
                pitch_obs = output[1]
                self.assertTrue(len(pitch_obs) == 88)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs - 21)

            if output[2] is not None:
                print("nnnn")
                pitch_obs = output[2]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs)

            if output[3] is not None:
                print("nnnnnn")
                pitch_obs = output[3]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs - 21)
                non_none_outputs += 1

        # self.assertTrue(non_none_outputs == len(note_array))


class TestPitchIOIProcessor(unittest.TestCase):
    @unittest.skipIf(True, "")
    def test_processor(self):

        note_array = np.empty(
            13,
            dtype=[
                ("pitch", int),
                ("onset_sec", float),
                ("duration_sec", float),
                ("velocity", int),
                ("id", str),
            ],
        )
        for i, pitch in enumerate(range(60, 73)):

            note_array[i] = (pitch, i, 0.5, 64, f"n{i}")

        perf = PerformedPart.from_note_array(note_array)

        feature_processor = PitchIOIProcessor(
            piano_range=False,
            return_pitch_list=False,
        )
        feature_processor_pr = PitchIOIProcessor(
            piano_range=True,
            return_pitch_list=False,
        )
        feature_processor_pl = PitchIOIProcessor(
            piano_range=False,
            return_pitch_list=True,
        )

        feature_processor_pl_pr = PitchIOIProcessor(
            piano_range=True,
            return_pitch_list=True,
        )
        # For coverage of the reset method, since it does not
        # do anything in this case.
        feature_processor.reset()
        polling_period = 0.01
        outputs = process_midi_offline(
            perf_info=perf,
            polling_period=polling_period,
            processor=[
                feature_processor,
                feature_processor_pr,
                feature_processor_pl,
                feature_processor_pl_pr,
            ],
        )

        non_none_outputs = 0
        for output in outputs:

            if output[0] is not None:
                pitch_obs, ioi_obs = output[0]
                self.assertTrue(len(pitch_obs) == 128)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(isinstance(ioi_obs, float))
                if non_none_outputs == 0:
                    self.assertTrue(
                        np.isclose(
                            ioi_obs,
                            0,
                            atol=polling_period,
                        )
                    )
                else:
                    self.assertTrue(
                        np.isclose(
                            ioi_obs,
                            1,
                            atol=polling_period,
                        )
                    )

                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs)

            if output[1] is not None:
                pitch_obs, ioi_obs = output[1]
                self.assertTrue(len(pitch_obs) == 88)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(isinstance(ioi_obs, float))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs - 21)

            if output[2] is not None:
                pitch_obs, ioi_obs = output[2]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(isinstance(ioi_obs, float))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs)

            if output[3] is not None:
                pitch_obs, ioi_obs = output[3]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(isinstance(ioi_obs, float))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs - 21)
                non_none_outputs += 1

        self.assertTrue(non_none_outputs == len(note_array))


class TestComputeFeaturesFromSymbolic(unittest.TestCase):
    @unittest.skipIf(True, "")
    def test_framed_features(self):

        score = pt.load_musicxml(EXAMPLE_SCORE)
        perf = pt.load_performance_midi(EXAMPLE_PERFORMANCE)
        input_types = [
            score,  # A Score object
            score[0],  # A Part object
            perf,  # A Performance object
            perf[0],  # A PerformedPart object
            perf.note_array(),  # A Performance note array
            EXAMPLE_MATCH,  # a path
        ]

        for ref_info in input_types:

            output_length = None
            features_list = [
                "pitch",
                "pitch_ioi",
                "pianoroll",
                "pitch_class_pianoroll",
                "cumsum_pianoroll",
            ]

            feature_kwargs = [
                dict(piano_range=True),  # PitchProcessor
                dict(piano_range=True),  # PitchIOIProcessor
                dict(piano_range=True),  # PianoRollProcessor
                dict(use_velocity=False),  # PitchClassPianoRollProcessor
                dict(piano_range=False),  # CumSumPianoRollProcessor
            ]

            for p_name, p_kwargs in zip(features_list, feature_kwargs):
                features = compute_features_from_symbolic(
                    ref_info=ref_info,
                    processor_name=p_name,
                    processor_kwargs=p_kwargs,
                )

                if output_length is None:
                    output_length = len(features)

                self.assertTrue(output_length == len(features))

    @unittest.skipIf(True, "")
    def test_nonframed_features(self):

        score = pt.load_musicxml(EXAMPLE_SCORE)
        perf = pt.load_performance_midi(EXAMPLE_PERFORMANCE)
        input_types = [
            score,  # A Score object
            score[0],  # A Part object
            perf,  # A Performance object
            perf[0],  # A PerformedPart object
            perf.note_array(),  # A Performance note array
            EXAMPLE_MATCH,  # a path
        ]

        for ref_info in input_types:

            output_length = None
            features_list = [
                "pitch",
                "pitch_ioi",
                "pianoroll",
                "pitch_class_pianoroll",
                "cumsum_pianoroll",
            ]

            for p_name in features_list:
                features = compute_features_from_symbolic(
                    ref_info=ref_info,
                    processor_name=p_name,
                    processor_kwargs=None,
                    polling_period=None,
                )

                if output_length is None:
                    output_length = len(features)

                self.assertTrue(output_length == len(features))
