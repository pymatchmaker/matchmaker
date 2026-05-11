import unittest

import matplotlib.pyplot as plt
import numpy as np
import partitura as pt
from partitura.utils.music import generate_random_performance_note_array

from matchmaker import EXAMPLE_MATCH
from matchmaker.features.midi import PitchProcessor
from matchmaker.prob.outer_product_hmm import (
    OuterProductHMM,
    compute_OuterProductHMM_pitch_probabilities,
    compute_transition_matrix,
    get_chords_from_score,
)
from tests.utils import process_midi_offline


class TestUtils(unittest.TestCase):
    def test_get_chords_from_score(self):
        score = pt.load_musicxml(pt.EXAMPLE_MUSICXML)

        # pass a score
        chords_from_score = get_chords_from_score(score)

        # pass a Part
        chords_from_Part = get_chords_from_score(score[0])

        # pass a note array
        chords_from_note_array = get_chords_from_score(score.note_array())

        expected_chords = [{69}, {72, 76}]

        self.assertTrue(len(chords_from_score) == 2)

        self.assertTrue(len(chords_from_Part) == 2)

        self.assertTrue(len(chords_from_note_array) == 2)

        self.assertTrue(chords_from_note_array == chords_from_Part)

        self.assertTrue(chords_from_score == chords_from_note_array)

        self.assertTrue(chords_from_score == expected_chords)

        invalid_note_array = generate_random_performance_note_array()

        with self.assertRaises(ValueError):
            invalid_result = get_chords_from_score(invalid_note_array)

    def test_compute_OuterProductHMM_pitch_probabilities(self):
        score = pt.load_musicxml(pt.EXAMPLE_MUSICXML)

        # pass a score
        chords = get_chords_from_score(score)

        b_table = compute_OuterProductHMM_pitch_probabilities(chords)

        self.assertTrue(b_table[0].argmax() == 69)
        self.assertTrue(b_table[0, 72] == b_table[0, 76])

    def test_compute_transition_matrix(self):
        score = pt.load_musicxml(pt.EXAMPLE_MUSICXML)

        # pass a score
        chords = get_chords_from_score(score)

        A, D1, D2 = compute_transition_matrix(len(chords))

        self.assertTrue(A.shape == (2, 2))
        self.assertTrue(D1 == 3)
        self.assertTrue(D2 == 3)
        self.assertTrue(np.allclose(A.sum(axis=1), 1.0))


class TestOuterProductHMM(unittest.TestCase):
    def test_outer_product_hmm(self):
        self.perf, _, self.score = pt.load_match(EXAMPLE_MATCH, create_score=True)

        snote_array = self.score.note_array()

        self.unique_sonsets = np.unique(snote_array["onset_beat"])

        observations = process_midi_offline(
            perf_info=self.perf,
            processor=PitchProcessor(piano_range=True),
        )

        self.outerhmm = OuterProductHMM(reference_features=self.score)

        for obs in observations:
            if obs is not None:
                beat = self.outerhmm(*obs)
                if beat is None:
                    continue  # chord continuation: no state advance
                self.assertTrue(beat in self.unique_sonsets)

        self.assertTrue(isinstance(self.outerhmm.alignment_path, np.ndarray))
