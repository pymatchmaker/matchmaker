#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the features/midi.py module
"""
import unittest
import numpy as np

from matchmaker import EXAMPLE_PERFORMANCE

import partitura as pt
from partitura.performance import PerformedPart

from matchmaker.features.midi import PitchIOIProcessor, PitchProcessor

from tests.utils import process_midi_offline

class TestPitchProcessor(unittest.TestCase):
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
        outputs = process_midi_offline(
            perf_info=perf,
            polling_period=polling_period,
            features=[
                feature_processor,
                feature_processor_pr,
                feature_processor_pl,
                feature_processor_pl_pr,
            ],
        )

        non_none_outputs = 0
        for output in outputs:

            if output[0] is not None:
                pitch_obs = output[0]
                self.assertTrue(len(pitch_obs) == 128)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs)

            if output[1] is not None:
                pitch_obs = output[1]
                self.assertTrue(len(pitch_obs) == 88)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(np.argmax(pitch_obs) == 60 + non_none_outputs - 21)

            if output[2] is not None:
                pitch_obs = output[2]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs)

            if output[3] is not None:
                pitch_obs = output[3]
                self.assertTrue(len(pitch_obs) == 1)
                self.assertTrue(isinstance(pitch_obs, np.ndarray))
                self.assertTrue(np.sum(pitch_obs) > 0)
                self.assertTrue(pitch_obs[0] == 60 + non_none_outputs - 21)
                non_none_outputs += 1

        self.assertTrue(non_none_outputs == len(note_array))

class TestPitchIOIProcessor(unittest.TestCase):

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
            features=[
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