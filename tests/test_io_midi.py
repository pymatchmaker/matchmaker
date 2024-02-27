#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the io/midi.py module
"""
import time
import unittest

import mido
import numpy as np
from matchmaker.features.midi import (
    CumSumPianoRollProcessor,
    PianoRollProcessor,
    PitchIOIProcessor,
)
from matchmaker.io.midi import FramedMidiStream, MidiStream
from matchmaker.utils.misc import RECVQueue

RNG = np.random.RandomState(1984)

from tempfile import NamedTemporaryFile

from partitura import save_performance_midi
from partitura.performance import PerformedPart

from tests.utils import DummyMidiPlayer


def setup_midi_player():
    """
    Setup dummy MIDI player for testing

    Returns
    -------
    port : mido.ports.BaseInput
        Virtual port for testing

    queue: RECVQueue
        Queue for getting the processed data

    midi_player : DummyMidiPlayer
        Midi player thread for testing

    note_array : np.ndarray
        Note array with performance information.
    """
    # Open virtual MIDI port
    # the input uses the "created" virtual
    # port
    port = mido.open_input("port1", virtual=True)
    outport = mido.open_output("port1")
    queue = RECVQueue()

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

    # Generate temporary midi file
    tmp_file = NamedTemporaryFile(delete=True)
    save_performance_midi(
        performance_data=PerformedPart.from_note_array(note_array),
        out=tmp_file.name,
    )
    # Create DummyMidiPlayer instance
    midi_player = DummyMidiPlayer(
        port=outport,
        filename=tmp_file.name,
    )
    # close and delete tmp midi file
    tmp_file.close()
    return port, queue, midi_player, note_array


class TestMidiStream(unittest.TestCase):
    """
    This class tests the MidiStream class

    TODO
    ----
    * Test mediator
    """

    def test_stream(self):
        """
        Test running an instance of a MidiStream class
        (i.e., getting features from a live input)
        """
        port, queue, midi_player, _ = setup_midi_player()
        features = [
            PitchIOIProcessor(),
            PianoRollProcessor(),
            CumSumPianoRollProcessor(),
        ]
        midi_stream = MidiStream(
            port=port,
            queue=queue,
            features=features,
        )
        midi_stream.start()

        midi_player.start()

        while midi_player.is_playing:
            output = queue.recv()
            self.assertTrue(len(output) == len(features))
        midi_stream.stop_listening()
        midi_player.join()
        port.close()

    def test_stream_with_midi_messages(self):
        """
        Test running an instance of a MidiStream class
        (i.e., getting features from a live input). This
        tests gets both computed features and input midi
        messages.
        """
        port, queue, midi_player, _ = setup_midi_player()
        features = [PitchIOIProcessor()]
        midi_stream = MidiStream(
            port=port,
            queue=queue,
            features=features,
            return_midi_messages=True,
        )
        midi_stream.start()

        midi_player.start()

        while midi_player.is_playing:
            (msg, msg_time), output = queue.recv()
            self.assertTrue(isinstance(msg, mido.Message))
            self.assertTrue(isinstance(msg_time, float))

            if msg.type == "note_on" and output[0] is not None:
                self.assertTrue(msg.note == int(output[0][0][0]))
            self.assertTrue(len(output) == len(features))
        midi_stream.stop_listening()
        midi_stream.join()
        midi_player.join()
        port.close()


class TestFramedMidiStream(unittest.TestCase):
    """
    This class tests the FramedMidiStream class

    TODO
    ----
    * Test getting midi messages
    * Test mediator
    """

    def test_stream(self):
        port, queue, midi_player, note_array = setup_midi_player()
        features = [
            PitchIOIProcessor(),
            PianoRollProcessor(),
            CumSumPianoRollProcessor(),
        ]
        polling_period = 0.05
        midi_stream = FramedMidiStream(
            port=port,
            queue=queue,
            features=features,
            polling_period=polling_period,
        )
        midi_stream.start()

        midi_player.start()

        perf_length = (
            note_array["onset_sec"] + note_array["duration_sec"]
        ).max() - note_array["onset_sec"].min()

        expected_frames = np.ceil(perf_length / polling_period)
        n_outputs = 0
        while midi_player.is_playing:
            output = queue.recv()
            self.assertTrue(len(output) == len(features))
            n_outputs += 1

        # Test whether the number of expected frames is within
        # 2 frames of the number of expected frames (due to rounding)
        # errors).
        self.assertTrue(abs(n_outputs - expected_frames) <= 2)
        midi_stream.stop_listening()
        midi_stream.join()
        midi_player.join()
        port.close()
