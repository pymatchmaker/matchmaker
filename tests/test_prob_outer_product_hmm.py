import unittest

import matplotlib.pyplot as plt
import numpy as np

import partitura as pt
from matchmaker.prob.outer_product_hmm import (
    compute_OuterProductHMM_pitch_probabilities,
    get_chords_from_score,
)
from partitura.utils.music import generate_random_performance_note_array


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
