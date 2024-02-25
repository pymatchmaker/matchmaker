#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the io/midi.py module
"""
import unittest

import mido

import numpy as np
import time


from matchmaker.io.midi import MidiStream

from matchmaker.utils.misc import RECVQueue

from matchmaker.features.midi import (
    PitchIOIProcessor,
    PianoRollProcessor,
    CumSumPianoRollProcessor,
)

from typing import Optional

RNG = np.random.RandomState(1984)

from tests.utils import DummyMidiPlayer

from partitura import save_performance_midi
from partitura.performance import PerformedPart

from tempfile import NamedTemporaryFile


class TestMidiStream(unittest.TestCase):

    def set_up(self):

        # Open virtual MIDI port
        # the input uses the "created" virtual
        # port
        self.port = mido.open_input("port1", virtual=True)
        self.outport = mido.open_output("port1")
        self.queue = RECVQueue()

        # Generate a random MIDI file
        n_notes = 5
        iois = 2 * RNG.rand(n_notes - 1)
        note_array = np.empty(
            n_notes,
            dtype=[
                ("pitch", int),
                ("onset_sec", float),
                ("duration_sec", float),
                ("velocity", int),
            ],
        )

        note_array["pitch"] = RNG.randint(low=0, high=127, size=n_notes)
        note_array["onset_sec"] = np.r_[0, np.cumsum(iois)]
        note_array["duration_sec"] = 2 * RNG.rand(n_notes)
        note_array["velocity"] = RNG.randint(low=0, high=127, size=n_notes)

        tmp_file = NamedTemporaryFile(delete=True)
        save_performance_midi(
            performance_data=PerformedPart.from_note_array(note_array),
            out=tmp_file.name,
        )

        # self.mf = mido.MidiFile(tmp_file.name)
        self.midi_player = DummyMidiPlayer(
            port=self.outport,
            filename=tmp_file.name,
        )

        # close and delete tmp midi file
        tmp_file.close()

    def test_stream(self):
        self.set_up()
        features = [
            PitchIOIProcessor(),
            PianoRollProcessor(),
            CumSumPianoRollProcessor(),
        ]
        midi_stream = MidiStream(
            port=self.port,
            queue=self.queue,
            features=features,
        )
        midi_stream.start()
        # start_time = time.time()

        self.midi_player.start()

        while self.midi_player.is_playing:
            output = self.queue.recv()
            self.assertTrue(len(output)== len(features))

        midi_stream.stop_listening()
