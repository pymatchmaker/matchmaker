#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the features/midi.py module
"""

import unittest
from io import StringIO
from unittest.mock import patch

import numpy as np
import partitura as pt
from partitura.performance import PerformedPart

from matchmaker import EXAMPLE_MATCH, EXAMPLE_PERFORMANCE, EXAMPLE_SCORE
from matchmaker.features.midi import (
    PianoRollProcessor,
    PitchClassPianoRollProcessor,
    PitchProcessor,
    compute_features_from_symbolic,
)
from tests.utils import process_midi_offline


class TestPitchProcessor(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_processor(self, mock_io):
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

        # outputs = []
        for processor in [
            feature_processor,
            feature_processor_pr,
            feature_processor_pl,
            feature_processor_pl_pr,
        ]:
            output = process_midi_offline(
                perf_info=perf,
                processor=processor,
                polling_period=polling_period,
            )

            non_none_outputs = 0
            # MidiStream attaches perf_time, so each item is (features, perf_time)
            if processor.piano_range and processor.return_pitch_list:
                for out in output:
                    if out is not None:
                        pitch_obs, _ = out
                        self.assertEqual(len(pitch_obs), 1)
                        self.assertEqual(pitch_obs, non_none_outputs + 60 - 21)
                        non_none_outputs += 1

            elif not processor.piano_range and processor.return_pitch_list:
                for out in output:
                    if out is not None:
                        pitch_obs, _ = out
                        self.assertEqual(len(pitch_obs), 1)
                        self.assertEqual(pitch_obs, non_none_outputs + 60)
                        non_none_outputs += 1

            elif processor.piano_range and not processor.return_pitch_list:
                for out in output:
                    if out is not None:
                        pitch_obs, _ = out
                        self.assertEqual(len(pitch_obs), 88)
                        self.assertEqual(
                            np.argmax(pitch_obs), non_none_outputs + 60 - 21
                        )
                        non_none_outputs += 1

            elif not processor.piano_range and not processor.return_pitch_list:
                for out in output:
                    if out is not None:
                        pitch_obs, _ = out
                        self.assertEqual(len(pitch_obs), 128)
                        self.assertEqual(np.argmax(pitch_obs), non_none_outputs + 60)
                        non_none_outputs += 1

            self.assertEqual(non_none_outputs, len(note_array))


class TestPianoRollProcessor(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_processor(self, mock_io):
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

        feature_processor = PianoRollProcessor(
            piano_range=False,
        )
        feature_processor_pr = PianoRollProcessor(
            piano_range=True,
        )
        feature_processor_pr_vel = PianoRollProcessor(
            piano_range=True,
            use_velocity=True,
        )

        feature_processor.reset()
        polling_period = 0.01

        # outputs = []
        for processor in [
            feature_processor,
            feature_processor_pr,
            feature_processor_pr_vel,
        ]:
            output = process_midi_offline(
                perf_info=perf,
                processor=processor,
                polling_period=polling_period,
            )

            # MidiStream attaches perf_time, so each item is (pianoroll, perf_time)
            if processor.piano_range and processor.use_velocity:
                for out in output:
                    pr, _ = out
                    self.assertTrue(isinstance(pr, np.ndarray))
                    self.assertEqual(len(pr), 88)
                    self.assertTrue(np.sum(pr) == 64 or np.sum(pr) == 0)
                    if pr.sum() > 0:
                        self.assertTrue(np.argmax(pr) in note_array["pitch"] - 21)

            elif processor.piano_range and not processor.use_velocity:
                for out in output:
                    pr, _ = out
                    self.assertTrue(isinstance(pr, np.ndarray))
                    self.assertEqual(len(pr), 88)
                    self.assertTrue(np.sum(pr) == 1 or np.sum(pr) == 0)
                    if pr.sum() > 0:
                        self.assertTrue(np.argmax(pr) in note_array["pitch"] - 21)

            else:
                for out in output:
                    pr, _ = out
                    self.assertTrue(isinstance(pr, np.ndarray))
                    self.assertEqual(len(pr), 128)
                    self.assertTrue(np.sum(pr) == 1 or np.sum(pr) == 0)
                    if pr.sum() > 0:
                        self.assertTrue(np.argmax(pr) in note_array["pitch"])


class TestPitchClassPianoRollProcessor(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_processor(self, mock_io):
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

        feature_processor = PitchClassPianoRollProcessor()
        feature_processor_vel = PitchClassPianoRollProcessor(
            use_velocity=True,
        )

        feature_processor.reset()
        polling_period = 0.01

        for processor in [
            feature_processor,
            feature_processor_vel,
        ]:
            output = process_midi_offline(
                perf_info=perf,
                processor=processor,
                polling_period=polling_period,
            )

            # MidiStream attaches perf_time, so each item is (pitch_class, perf_time)
            if processor.use_velocity:
                for out in output:
                    pc, _ = out
                    self.assertTrue(isinstance(pc, np.ndarray))
                    self.assertEqual(len(pc), 12)
                    self.assertTrue(np.sum(pc) == 64 or np.sum(pc) == 0)
                    if pc.sum() > 0:
                        self.assertTrue(np.argmax(pc) in note_array["pitch"] % 12)

            else:
                for out in output:
                    pc, _ = out
                    self.assertTrue(isinstance(pc, np.ndarray))
                    self.assertEqual(len(pc), 12)
                    self.assertTrue(np.sum(pc) == 1 or np.sum(pc) == 0)
                    if pc.sum() > 0:
                        self.assertTrue(np.argmax(pc) in note_array["pitch"] % 12)

            self.assertTrue(len(processor.pitch_class_slices) > 0)
            processor.reset()
            self.assertEqual(len(processor.pitch_class_slices), 0)


class TestComputeFeaturesFromSymbolic(unittest.TestCase):
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
                "pianoroll",
                "pitch_class_pianoroll",
            ]

            feature_kwargs = [
                dict(piano_range=True),  # PitchProcessor
                dict(piano_range=True),  # PianoRollProcessor
                dict(use_velocity=False),  # PitchClassPianoRollProcessor
            ]

            for p_name, p_kwargs in zip(features_list, feature_kwargs):
                features = compute_features_from_symbolic(
                    ref_info=ref_info,
                    processor_name=p_name,
                    processor_kwargs=p_kwargs,
                )

                if output_length is None:
                    output_length = len(features)

                self.assertEqual(output_length, len(features))

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
                "pianoroll",
                "pitch_class_pianoroll",
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

                self.assertEqual(output_length, len(features))
