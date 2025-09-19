#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains tests for the prob.particle_filter module.
"""

import unittest

import numpy as np

from matchmaker.prob.particle_filter_alt import (
    BasePF,
    InitialModel,
    MonophonicPitchInitialModel,
    ObservationModel,
    TransitionModel,
)
from tests.utils import generate_example_sequences

RNG = np.random.RandomState(1984)


class TestMonophonicPitchInitialModel(unittest.TestCase):
    def test_init(
        self,
    ) -> None:
        pitch = RNG.randint(0, 127, size=100)
        unique_onsets = np.arange(0, len(pitch))

        min_bpm = 10
        max_bpm = 100

        initial_position = 0

        initial_model = MonophonicPitchInitialModel(
            pitch=pitch,
            unique_onsets=unique_onsets,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            initial_position=initial_position,
            rng=RNG,
        )

        self.assertTrue(isinstance(initial_model.pitch, np.ndarray))
        self.assertTrue(isinstance(initial_model.n_states, int))
        self.assertTrue(isinstance(initial_model.min_tempo, float))
        self.assertTrue(isinstance(initial_model.max_tempo, float))
        self.assertTrue(isinstance(initial_model.initial_position, int))
        self.assertTrue(isinstance(initial_model.initial_position_choices, np.ndarray))

    def test_generate_given_initial_position(self) -> None:
        # Test giving an initial position
        pitch = RNG.randint(0, 127, size=100)
        unique_onsets = np.arange(0, len(pitch))
        min_bpm = 10
        max_bpm = 100

        initial_position = 0

        initial_model = MonophonicPitchInitialModel(
            pitch=pitch,
            unique_onsets=unique_onsets,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            initial_position=initial_position,
            rng=RNG,
        )

        n_particles = 100
        particles = initial_model.generate(n_particles)

        self.assertTrue(np.all(particles["state"] == initial_position))
        self.assertTrue(len(particles) == n_particles)

    def test_generate_not_given_initial_position(self) -> None:
        # Test giving an initial position
        pitch = RNG.randint(0, 127, size=100)
        unique_onsets = np.arange(0, len(pitch))
        min_bpm = 10
        max_bpm = 100

        initial_position = None

        initial_model = MonophonicPitchInitialModel(
            pitch=pitch,
            unique_onsets=unique_onsets,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            initial_position=initial_position,
            rng=RNG,
        )

        n_particles = 100
        particles = initial_model.generate(n_particles)

        self.assertTrue(np.any(particles["state"] != particles["state"][0]))
        self.assertTrue(len(particles) == n_particles)
