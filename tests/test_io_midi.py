#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the io/midi.py module
"""
from io import StringIO
import time
from typing import Optional
import unittest
from unittest.mock import patch

import mido
import numpy as np

from matchmaker import EXAMPLE_PERFORMANCE
from matchmaker.features.midi import (
    CumSumPianoRollProcessor,
    PianoRollProcessor,
    PitchIOIProcessor,
)
from matchmaker.io.mediator import CeusMediator
from matchmaker.io.midi import (
    MidiStream,
    Buffer,
)
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import DummyProcessor

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

    # normalize the random performance to last 1 second
    # (makes the tests a bit faster ;)
    max_duration = (note_array["onset_sec"] + note_array["duration_sec"]).max()
    note_array["onset_sec"] /= max_duration
    note_array["duration_sec"] /= max_duration

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

    mediator = CeusMediator()

    unique_pitches = np.unique(note_array["pitch"])
    mediator_pitches = RNG.choice(
        unique_pitches,
        size=int(np.round(0.3 * len(unique_pitches))),
        replace=False,
    )

    for mp in mediator_pitches:
        mediator.filter_append_pitch(midi_pitch=mp)
    return port, queue, midi_player, note_array, mediator


class TestMidiStream(unittest.TestCase):
    """
    This class tests the MidiStream class

    TODO
    ----
    * Test mediator
    """

    def setup(
        self,
        processor: str = "dummy",
        file_path: Optional[str] = None,
        polling_period: Optional[float] = None,
        port=None,
        mediator: Optional[CeusMediator] = None,
        queue: Optional[RECVQueue] = None,
        return_midi_messages: bool = False,
    ) -> None:
        """Setup a MidiStream for testing"""

        if processor == "dummy":
            processor = None
        elif processor == "pitchioi":
            processor = PitchIOIProcessor()
        elif processor == "pianoroll":
            processor = PianoRollProcessor()
        elif processor == "cumsumpianoroll":
            processor = CumSumPianoRollProcessor()

        self.stream = MidiStream(
            processor=processor,
            file_path=file_path,
            polling_period=polling_period,
            port=port,
            mediator=mediator,
            queue=queue,
            return_midi_messages=return_midi_messages,
        )

    def test_init(self):
        """Test that the MidiStream initializes correctly"""
        for processor in [
            "dummy",
            "pianoroll",
        ]:
            for file_path in [EXAMPLE_PERFORMANCE, None]:
                for polling_period in [None, 0.01]:
                    for port in [
                        mido.open_input(
                            "port1",
                            virtual=True,
                        ),
                        None,
                    ]:
                        for mediator in [None, CeusMediator()]:
                            self.setup(
                                processor=processor,
                                file_path=file_path,
                                polling_period=polling_period,
                                port=port,
                                mediator=mediator,
                            )

                            self.assertTrue(isinstance(self.stream, MidiStream))

                            if port is not None and file_path is not None:
                                self.assertTrue(self.stream.midi_in is None)

                            if polling_period is None:
                                self.assertFalse(self.stream.is_windowed)

                            else:
                                self.assertTrue(self.stream.is_windowed)

                            if port is not None:
                                port.close()

    @patch("sys.stdout", new_callable=StringIO)
    def test_run_online(self, mock_stdout):
        """
        Test running an instance of a MidiStream class
        (i.e., getting features from a live input)
        """

        for processor in ["dummy", "pianoroll"]:
            for return_midi_messages in [True, False]:
                for use_mediator in [True, False]:
                    for polling_period in [None, 0.01]:

                        port, queue, midi_player, _, mediator = setup_midi_player()

                        self.setup(
                            processor=processor,
                            file_path=None,
                            port=port,
                            queue=queue,
                            mediator=mediator if use_mediator else None,
                            return_midi_messages=return_midi_messages,
                            polling_period=polling_period,
                        )

                        if use_mediator:
                            self.assertTrue(
                                isinstance(self.stream.mediator, CeusMediator)
                            )
                        else:
                            self.assertIsNone(self.stream.mediator)

                        self.stream.start()

                        midi_player.start()

                        while midi_player.is_playing:
                            output = queue.recv()

                            if return_midi_messages and polling_period is None:
                                (msg, msg_time), output = output
                                self.assertTrue(isinstance(msg, mido.Message))
                                self.assertTrue(isinstance(msg_time, float))

                            elif return_midi_messages and polling_period is not None:
                                messages, output = output

                                for msg, msg_time in messages:
                                    self.assertTrue(isinstance(msg, mido.Message))
                                    self.assertTrue(isinstance(msg_time, float))

                            if processor == "pianoroll":
                                self.assertTrue(isinstance(output, np.ndarray))

                        self.stream.stop()
                        midi_player.join()
                        port.close()
    
    @patch("sys.stdout", new_callable=StringIO)
    def test_run_online_context_manager(self, mock_stdout):
        """
        Test running an instance of a MidiStream class
        (i.e., getting features from a live input) with the
        context manager interface.
        """

        polling_period = None
        processor = "pianoroll"
        return_midi_messages = True
        polling_period = 0.01

        port, queue, midi_player, _, _ = setup_midi_player()

        self.setup(
            processor=processor,
            file_path=None,
            port=port,
            queue=queue,
            mediator=None,
            return_midi_messages=return_midi_messages,
            polling_period=polling_period,
        )

        with self.stream as stream:

            midi_player.start()

            while midi_player.is_playing:
                output = stream.queue.recv()
                messages, output = output

                for msg, msg_time in messages:
                    self.assertTrue(isinstance(msg, mido.Message))
                    self.assertTrue(isinstance(msg_time, float))

                self.assertTrue(
                    isinstance(
                        output,
                        np.ndarray,
                    )
                )

            midi_player.join()
            port.close()


# class TestMidiStream(unittest.TestCase):
#     """
#     This class tests the MidiStream class

#     TODO
#     ----
#     * Test mediator
#     """

#     def test_stream(self):
#         """
#         Test running an instance of a MidiStream class
#         (i.e., getting features from a live input)
#         """
#         port, queue, midi_player, _ = setup_midi_player()
#         features = [
#             PitchIOIProcessor(),
#             PianoRollProcessor(),
#             CumSumPianoRollProcessor(),
#         ]
#         midi_stream = MidiStream(
#             port=port,
#             queue=queue,
#             processor=features,
#         )
#         midi_stream.start()

#         midi_player.start()

#         while midi_player.is_playing:
#             output = queue.recv()
#             self.assertTrue(len(output) == len(features))
#         midi_stream.stop_listening()
#         midi_player.join()
#         port.close()

#     def test_stream_with_midi_messages(self):
#         """
#         Test running an instance of a MidiStream class
#         (i.e., getting features from a live input). This
#         tests gets both computed features and input midi
#         messages.
#         """
#         port, queue, midi_player, _ = setup_midi_player()
#         features = [PitchIOIProcessor(return_pitch_list=True)]
#         midi_stream = MidiStream(
#             port=port,
#             queue=queue,
#             processor=features,
#             return_midi_messages=True,
#         )
#         midi_stream.start()

#         midi_player.start()

#         while midi_player.is_playing:
#             (msg, msg_time), output = queue.recv()
#             self.assertTrue(isinstance(msg, mido.Message))
#             self.assertTrue(isinstance(msg_time, float))

#             if msg.type == "note_on" and output[0] is not None:
#                 self.assertTrue(msg.note == int(output[0][0][0]))
#             self.assertTrue(len(output) == len(features))
#         midi_stream.stop_listening()
#         midi_stream.join()
#         midi_player.join()
#         port.close()


# class TestFramedMidiStream(unittest.TestCase):
#     """
#     This class tests the FramedMidiStream class

#     TODO
#     ----
#     * Test return_midi_messages=True
#     * Test mediator
#     * Test length and string of Buffer
#     """

#     def test_stream(self):
#         port, queue, midi_player, note_array = setup_midi_player()
#         features = [
#             PitchIOIProcessor(),
#             PianoRollProcessor(),
#             CumSumPianoRollProcessor(),
#         ]
#         polling_period = 0.05
#         midi_stream = FramedMidiStream(
#             port=port,
#             queue=queue,
#             features=features,
#             polling_period=polling_period,
#         )
#         midi_stream.start()

#         midi_player.start()

#         perf_length = (
#             note_array["onset_sec"] + note_array["duration_sec"]
#         ).max() - note_array["onset_sec"].min()

#         expected_frames = np.ceil(perf_length / polling_period)
#         n_outputs = 0
#         while midi_player.is_playing:
#             output = queue.recv()
#             self.assertTrue(len(output) == len(features))
#             n_outputs += 1

#         # Test whether the number of expected frames is within
#         # 2 frames of the number of expected frames (due to rounding)
#         # errors).
#         self.assertTrue(abs(n_outputs - expected_frames) <= 2)
#         midi_stream.stop_listening()
#         midi_stream.join()
#         midi_player.join()
#         port.close()


# class TestMockMidiStream(unittest.TestCase):
#     def test_stream(self):
#         """
#         Test running an instance of a MidiStream class
#         (i.e., getting features from a live input)
#         """

#         queue = RECVQueue()
#         features = [
#             PitchIOIProcessor(),
#             PianoRollProcessor(),
#             CumSumPianoRollProcessor(),
#         ]
#         midi_stream = MockMidiStream(
#             file_path=EXAMPLE_PERFORMANCE,
#             queue=queue,
#             features=features,
#         )

#         mf = mido.MidiFile(EXAMPLE_PERFORMANCE)

#         valid_messages = [msg for msg in mf if not isinstance(msg, mido.MetaMessage)]

#         midi_stream.start()
#         midi_stream.join()
#         # get all outputs of the queue at once
#         outputs = list(queue.queue)
#         self.assertTrue(len(outputs) == len(valid_messages))

#         for output in outputs:
#             self.assertTrue(len(output) == len(features))


# class TestMockFramedMidiStream(unittest.TestCase):
#     def test_stream(self):
#         """
#         Test running an instance of a MidiStream class
#         (i.e., getting features from a live input)
#         """

#         queue = RECVQueue()
#         features = [
#             PitchIOIProcessor(piano_range=True),
#             PianoRollProcessor(piano_range=True),
#             CumSumPianoRollProcessor(piano_range=True),
#         ]
#         midi_stream = MockFramedMidiStream(
#             file_path=EXAMPLE_PERFORMANCE,
#             queue=queue,
#             features=features,
#         )

#         midi_stream.start()
#         midi_stream.join()
#         # get all outputs of the queue at once
#         outputs = list(queue.queue)
#         self.assertTrue(len(outputs) >= 0)

#         for output in outputs:
#             self.assertTrue(len(output) == len(features))
