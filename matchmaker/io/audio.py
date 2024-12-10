#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input audio stream
"""

import time
from types import TracebackType
from typing import Callable, Optional, Tuple, Type, Union

import librosa
import numpy as np
import pyaudio

from matchmaker.features.audio import HOP_LENGTH, SAMPLE_RATE, ChromagramProcessor
from matchmaker.utils.audio import (
    get_audio_devices,
    get_default_input_device_index,
    get_device_index_from_name,
)
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.stream import Stream

CHANNELS = 1


class AudioStream(Stream):
    """A class to process an audio stream in real-time

    Parameters
    ----------
    processor: Optional[Callable]
        The processor for the features
    file_path: Optional[str]
        If given, the audio stream will be simulated using the
        given file as an input instead.
    sample_rate : int
        Sample rate of the audio stream
    hop_length : int
        Hop length of the audio stream
    queue : RECVQueue
        Queue to store the processed audio
    device_name_or_index : Optional[Union[str, int]]
        Name or index of the audio device to be used. Ignored
        if `file_path` is given.
    """

    def __init__(
        self,
        processor: Optional[Callable] = None,
        file_path: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        queue: Optional[RECVQueue] = None,
        device_name_or_index: Optional[Union[str, int]] = None,
        wait: bool = True,
    ):
        if processor is None:
            processor = ChromagramProcessor(
                sample_rate=sample_rate, hop_length=hop_length
            )

        Stream.__init__(
            self,
            processor=processor,
            mock=file_path is not None,
        )

        if file_path is not None:
            # Do not activate audio device for running the
            # stream offline
            device_name_or_index = None

        self.file_path = file_path
        # Select device index, or raise an error if invalid device is selected.
        self.input_device_index = None

        # Name of the device is given
        if isinstance(device_name_or_index, str):
            self.input_device_index = get_device_index_from_name(
                device_name=device_name_or_index
            )

        # Index of the device is given
        elif isinstance(device_name_or_index, int):
            self.input_device_index = device_name_or_index
            audio_devices = get_audio_devices()

            if device_name_or_index > len(audio_devices):
                print(
                    f"`{device_name_or_index}` is an invalid device index!\n"
                    "The following audio devices are available:\n"
                )

                for ad in audio_devices:
                    print(ad)

                raise ValueError("Invalid index for audio device.")
        elif device_name_or_index is None and not file_path:
            default_index = get_default_input_device_index()
            if default_index is not None:
                self.input_device_index = default_index
            else:  # pragma: no cover
                raise ValueError("No audio devices found!")

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.queue = queue or RECVQueue()
        self.format = pyaudio.paFloat32
        self.audio_interface = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.last_chunk = None
        self.f_time = 0
        self.prev_time = None
        self.wait = wait  # only for offline mode making it same time as online

        if self.mock:
            self.run = self.run_offline
        else:
            self.run = self.run_online

    def __enter__(self) -> None:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.stop()
        if exc_type is not None:  # pragma: no cover
            # Returning True will suppress the exception
            # False means the exception will propagate
            return False
        return True

    def _process_frame(
        self,
        data: Union[bytes, np.ndarray],
        frame_count: int,
        time_info: dict,
        status_flag: int,
    ) -> Tuple[np.ndarray, int]:
        self.prev_time = time_info["input_buffer_adc_time"]
        # initial y
        target_audio = np.frombuffer(data, dtype=np.float32)
        self._process_feature(target_audio, time_info["input_buffer_adc_time"])

        return (data, pyaudio.paContinue)

    def _process_feature(
        self,
        target_audio: np.ndarray,
        f_time: float,
    ):
        if self.last_chunk is None:  # add zero padding at the first block
            target_audio = np.concatenate(
                (np.zeros(self.hop_length, dtype=np.float32), target_audio)
            )
        else:
            # add last chunk at the beginning of the block
            target_audio = np.concatenate((self.last_chunk, target_audio))

        features = self.processor(target_audio)
        self.queue.put((features, f_time))
        self.last_chunk = target_audio[-self.hop_length :]

    @property
    def current_time(self) -> Optional[float]:
        """
        Get current time since starting to listen.

        This property only makes sense in the context of live
        inputs.
        """
        return (
            self.audio_stream.get_time() - self.init_time
            if (self.init_time is not None and self.audio_stream is not None)
            else None
        )

    def start_listening(self) -> None:
        self.listen = True
        if self.mock:
            print("* Mock listening to stream....")
        else:
            self.audio_stream.start_stream()
            print("* Start listening to audio stream....")

    def stop_listening(self) -> None:
        """Stop listening to the audio stream.

        This method stops the audio stream and cleans up resources.
        For real-time mode, it stops and closes the audio stream,
        and terminates the audio interface.
        """
        print("* Stop listening to audio stream....")
        if not self.mock and self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_interface.terminate()
        self.listen = False

    def run_offline(self) -> None:
        """Process audio file in offline mode.

        This method simulates real-time processing by reading chunks from
        an audio file at regular intervals. The processing speed can be
        controlled using the `wait` parameter.

        Note
        ----
        The audio file is processed in chunks of size `hop_length`,
        and features are extracted for each chunk.
        """
        self.start_listening()
        self.init_time = time.time()
        duration = int(librosa.get_duration(path=self.file_path))
        time_interval = self.hop_length / self.sample_rate
        audio_y, _ = librosa.load(self.file_path, sr=self.sample_rate)
        padded_audio = np.concatenate(  # zero padding at the end
            (
                audio_y,
                np.zeros(int(duration * 0.1 * self.sample_rate), dtype=np.float32),
            )
        )
        trimmed_audio = padded_audio[  # trim to multiple of chunk_size
            : len(padded_audio) - (len(padded_audio) % self.hop_length)
        ]
        while trimmed_audio.any():
            start_time = time.time()
            target_audio = trimmed_audio[: self.hop_length]
            f_time = time.time()
            self.last_time = f_time
            self._process_feature(target_audio, f_time)
            trimmed_audio = trimmed_audio[self.hop_length :]
            elapsed_time = time.time() - start_time

            if self.wait:
                time.sleep(max(time_interval - elapsed_time, 0))

    def run_online(self) -> None:
        """Process audio in real-time from input device.

        This method sets up and starts real-time audio processing from
        the specified input device. It initializes the PyAudio interface
        and opens an audio stream with the configured parameters.

        Note
        ----
        The audio is processed in chunks of size `hop_length`,
        and features are extracted in real-time.
        """
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.hop_length,
            stream_callback=self._process_frame,
            input_device_index=self.input_device_index,
        )
        self.prev_time = self.audio_stream.get_time()
        self.init_time = self.audio_stream.get_time()
        self.start_listening()

    def stop(self):
        self.stop_listening()
        self.join()

    def clear_queue(self):
        if self.queue.not_empty:
            self.queue.queue.clear()
