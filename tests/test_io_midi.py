#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the io/midi.py module
"""
import time
import unittest
from io import StringIO
from typing import Optional
from unittest.mock import patch

import mido
import numpy as np
import partitura as pt

from matchmaker import EXAMPLE_PERFORMANCE
from matchmaker.features.midi import PianoRollProcessor, PitchIOIProcessor
from matchmaker.io.mediator import CeusMediator
from matchmaker.io.midi import Buffer, MidiStream
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import DummyProcessor
from matchmaker.utils.symbolic import midi_messages_from_midi, panic_button

RNG = np.random.RandomState(1984)

from tempfile import NamedTemporaryFile

from partitura import save_performance_midi
from partitura.performance import PerformedPart

from tests.utils import DummyMidiPlayer


def setup_midi_player(use_example: bool = False):
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
    # panic_button()
    port = mido.open_input("port1", virtual=True)
    outport = mido.open_output("port1")
    queue = RECVQueue()

    if use_example:
        filename = EXAMPLE_PERFORMANCE
        perf = pt.load_performance_midi(filename)
        note_array = perf.note_array()
    else:
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
        note_array["onset_sec"] /= max_duration * 2
        note_array["duration_sec"] /= max_duration * 2

        # Generate temporary midi file
        tmp_file = NamedTemporaryFile(delete=True)
        save_performance_midi(
            performance_data=PerformedPart.from_note_array(note_array),
            out=tmp_file.name,
        )
        filename = tmp_file.name
    # Create DummyMidiPlayer instance
    midi_player = DummyMidiPlayer(
        port=outport,
        filename=filename,
    )

    if not use_example:
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
        virtual_port: bool = False,
    ) -> None:
        """Setup a MidiStream for testing"""

        if processor == "dummy":
            processor = None
        elif processor == "pitchioi":
            processor = PitchIOIProcessor()
        elif processor == "pianoroll":
            processor = PianoRollProcessor()

        self.stream = MidiStream(
            processor=processor,
            file_path=file_path,
            polling_period=polling_period,
            port=port,
            mediator=mediator,
            queue=queue,
            return_midi_messages=return_midi_messages,
            virtual_port=virtual_port,
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

    def test_init_port_selection(self):

        # Raise an error if port is incorrect
        with self.assertRaises(ValueError):
            self.setup(port="wrong_port")

        # test virtual
        self.setup(
            port="virtual",
            virtual_port=True,
        )

        self.assertIsInstance(self.stream, MidiStream)

        self.stream.midi_in.close()

        port = mido.open_input("virtual", virtual=True)
        self.setup(port=port)
        self.assertEqual(port, self.stream.midi_in)

        port.close()

        self.setup(file_path="test.mid")
        self.assertTrue(self.stream.midi_in is None)

    # @patch("sys.stdout", new_callable=StringIO)
    def test_run_online(self, mock_stdout=None):
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

    # @patch("sys.stdout", new_callable=StringIO)
    def test_run_online_context_manager(self, mock_stdout=None):
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

    # @patch("sys.stdout", new_callable=StringIO)
    def test_run_offline_single(self, mock_stdout=None):
        """
        Test run_offline_single method.
        """

        mf = mido.MidiFile(EXAMPLE_PERFORMANCE)

        valid_messages = [msg for msg in mf if not isinstance(msg, mido.MetaMessage)]
        for processor in [
            "pianoroll",
            "pitchioi",
            "dummy",
        ]:
            self.setup(
                processor=processor,
                polling_period=None,
                file_path=EXAMPLE_PERFORMANCE,
            )

            with self.stream as stream:
                pass

            outputs = list(self.stream.queue.queue)
            self.assertTrue(len(outputs) == len(valid_messages))

    # @patch("sys.stdout", new_callable=StringIO)
    def test_run_offline_windowed(self, mock_stdout=None):
        """
        Test run_offline_windowed method.
        """

        _, message_times = midi_messages_from_midi(
            filename=EXAMPLE_PERFORMANCE,
        )

        polling_period = 0.01
        expected_frames = int(np.ceil(message_times.max() / polling_period))

        for processor in [
            "pianoroll",
            "pitchioi",
            "dummy",
        ]:
            self.setup(
                processor=processor,
                polling_period=polling_period,
                file_path=EXAMPLE_PERFORMANCE,
            )

            with self.stream as stream:
                pass

            outputs = list(self.stream.queue.queue)

            self.assertTrue(len(outputs) == expected_frames)

    # @patch("sys.stdout", new_callable=StringIO)
    def test_clear_queue(self, mock_stdout=None):
        """
        Test clear_queue method
        """
        processor = "dummy"
        self.setup(
            processor=processor,
            polling_period=None,
            file_path=EXAMPLE_PERFORMANCE,
        )

        with self.stream as stream:
            pass

        self.stream.clear_queue()

        outputs = list(self.stream.queue.queue)
        self.assertTrue(len(outputs) == 0)

    # @patch("sys.stdout", new_callable=StringIO)
    def test_online_windowed_input(self, mock_stdout=None):
        port, queue, midi_player, note_array, _ = setup_midi_player()

        polling_period = 0.01

        self.setup(
            processor="dummy",
            port=port,
            queue=queue,
            polling_period=polling_period,
            return_midi_messages=True,
        )

        perf_length = (note_array["onset_sec"] + note_array["duration_sec"]).max()

        expected_frames = int(np.ceil(perf_length / polling_period))
        n_outputs = 0

        with self.stream as stream:
            midi_player.start()
            while midi_player.is_playing:
                output = queue.recv()

                if output is not None:
                    n_outputs += 1
                    # print(output, n_outputs)

        # Test whether the number of expected frames is within
        # 2 frames of the number of expected frames (due to rounding)
        # errors).
        self.assertTrue(abs(n_outputs - expected_frames) <= 2)

        midi_player.join()
        port.close()
