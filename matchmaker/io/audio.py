#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input audio stream
"""
import threading
import time
from typing import Callable, List, Optional

import librosa
import numpy as np
import pyaudio

from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import DummySequentialOutputProcessor, Stream

CHANNELS = 1


class AudioStream(threading.Thread, Stream):
    """
    A class to process audio stream in real-time

    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio stream
    hop_length : int
        Hop length of the audio stream
    queue : RECVQueue
        Queue to store the processed audio
    features : List[Callable]
        List of features to be processed (SequentialOutputProcessor)
    chunk_size : int
        Size of the audio chunk
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        queue: RECVQueue,
        features: List[Callable],
        chunk_size: int = None,
    ):
        if features is None:
            features = DummySequentialOutputProcessor()
        threading.Thread.__init__(self)
        Stream.__init__(self, features=features)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.queue = queue
        self.chunk_size = chunk_size or hop_length * 4
        self.format = pyaudio.paFloat32
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.last_chunk = None
        self.init_time = None
        self.listen = False

    def _process_frame(self, data, frame_count, time_info, status_flag):
        target_audio = np.frombuffer(data, dtype=np.float32)  # initial y
        self._process_feature(target_audio, time_info["input_buffer_adc_time"])

        return (data, pyaudio.paContinue)

    def _process_feature(self, target_audio, f_time):
        if self.last_chunk is None:  # add zero padding at the first block
            target_audio = np.concatenate((np.zeros(self.hop_length), target_audio))
        else:
            # add last chunk at the beginning of the block
            # making 5 block, 1 block overlap -> 4 frames each time
            target_audio = np.concatenate((self.last_chunk, target_audio))

        stacked_features = None
        for feature in self.features:
            feature_output, _ = feature((target_audio, f_time), f_time)
            stacked_features = (
                feature_output
                if stacked_features is None
                else np.vstack((stacked_features, feature_output))
            )

        self.queue.put(stacked_features)
        self.last_chunk = target_audio[-self.hop_length :]

    @property
    def current_time(self):
        """
        Get current time since starting to listen
        """
        return time.time() - self.init_time if self.init_time else None

    def start_listening(self):
        self.audio_stream.start_stream()
        print("* Start listening to audio stream....")
        self.listen = True
        self.init_time = self.audio_stream.get_time()

    def stop_listening(self):
        print("* Stop listening to audio stream....")
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio_interface.terminate()
        self.listen = False

    def run(self):
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._process_frame,
        )
        self.start_listening()

    def stop(self):
        self.stop_listening()


class MockAudioStream(AudioStream):
    """
    A class to process audio stream from a file

    Parameters
    ----------
    file_path : str
        Path to the audio file
    """

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        queue: RECVQueue,
        features: List[Callable],
        chunk_size: int = None,
        file_path: str = "",
    ):
        super().__init__(
            sample_rate=sample_rate,
            hop_length=hop_length,
            queue=queue,
            features=features,
            chunk_size=chunk_size,
        )
        self.file_path = file_path

    def start_listening(self):
        self.listen = True
        self.init_time = time.time()

    def stop_listening(self):
        self.listen = False

    def mock_stream(self):
        duration = int(librosa.get_duration(path=self.file_path))
        audio_y, _ = librosa.load(self.file_path, sr=self.sample_rate)
        padded_audio = np.concatenate(  # zero padding at the end
            (audio_y, np.zeros(duration * 2 * self.sample_rate))
        )
        trimmed_audio = padded_audio[  # trim to multiple of chunk_size
            : len(padded_audio) - (len(padded_audio) % self.chunk_size)
        ]
        self.start_listening()
        while self.listen and trimmed_audio.any():
            target_audio = trimmed_audio[: self.chunk_size]
            f_time = time.time() - self.init_time
            self._process_feature(target_audio, f_time)
            trimmed_audio = trimmed_audio[self.chunk_size :]

        # fill empty values with zeros after stream is finished
        additional_padding_size = duration * 2 * self.sample_rate
        while self.listen and additional_padding_size > 0:
            f_time = time.time() - self.init_time
            self._process_feature(target_audio, f_time)
            additional_padding_size -= self.chunk_size

    def run(self):
        print(f"* [Mocking] Loading existing audio file({self.file_path})....")
        self.mock_stream()
