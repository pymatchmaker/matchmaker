#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input audio stream
"""

import queue
import time
from types import TracebackType
from typing import Dict, Optional, Tuple, Type, Union

import librosa
import numpy as np

from matchmaker.features.audio import (
    HOP_LENGTH,
    SAMPLE_RATE,
    ChromagramProcessor,
)
from matchmaker.features.processor import Processor
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END, Stream
from matchmaker.utils.audio import (
    get_audio_devices,
    get_default_input_device_index,
    get_device_index_from_name,
)
from matchmaker.utils.misc import set_latency_stats

CHANNELS = 1
QUEUE_TIMEOUT = 10


class AudioStream(Stream):
    """A class to process an audio stream in real-time.

    Parameters
    ----------
    processor : Optional[Processor]
        The processor for the features. If ``None``, defaults to
        ``ChromagramProcessor``.
    file_path : Optional[str]
        If given, the audio stream will be simulated using the
        given file as an input instead.
    sample_rate : int
        Sample rate of the audio stream.
    hop_length : int
        Hop length of the audio stream.
    queue : RECVQueue
        Queue to store the processed audio.
    device_name_or_index : Optional[Union[str, int]]
        Name or index of the audio device to be used. Ignored
        if ``file_path`` is given.

    Notes
    -----
    Frame caching: each call to ``_process_feature`` prepends
    ``self.cache_size`` samples from the previous frame so that FFT
    windows wider than ``hop_length`` (e.g. SKF's
    ``RawSpectrumProcessor`` with ``n_fft=512``) have enough context.
    The cache size is auto-discovered at construction:

    - If ``processor`` exposes an ``n_fft`` attribute,
      ``cache_size = n_fft - hop_length``.
    - Otherwise ``cache_size = hop_length`` (one previous hop).

    The first frame is zero-padded by ``cache_size`` samples.
    """

    def __init__(
        self,
        processor: Processor = None,
        file_path: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        queue: Optional[RECVQueue] = None,
        device_name_or_index: Optional[Union[str, int]] = None,
        wait: bool = False,
        target_sr: int = SAMPLE_RATE,
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
        # See class docstring "Notes" for the cache_size convention.
        self.cache_size = getattr(processor, "n_fft", 2 * hop_length) - hop_length
        self.queue = queue or RECVQueue()
        self.format = None  # set to pyaudio.paFloat32 in run_online
        self.audio_interface = None
        self.audio_stream = None
        self.last_chunk = None
        self.f_time = 0
        self.prev_time = None
        self.wait = wait  # only for offline mode making it same time as online
        self.target_sr = target_sr
        self.last_data_received = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self.input_index = 0
        self._emit_count = 0
        self._preloaded_audio = None

        if self.mock:
            self.run = self.run_offline
            # Pre-load and resample audio so the stream thread can start
            # producing frames immediately (avoids queue-timeout race condition
            # when librosa.load takes longer than QUEUE_TIMEOUT).
            self._preload_audio()
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
        self.input_index += 1
        self.last_data_received = time.time()
        adc_time = time_info["input_buffer_adc_time"]
        self.prev_time = adc_time
        target_audio = np.frombuffer(data, dtype=np.float32)
        # perf_time is computed inside _process_feature based on emit_count
        self._process_feature(target_audio, adc_time)
        if not self.stream_start.is_set():
            self.stream_start.set()

        import pyaudio

        return (data, pyaudio.paContinue)

    def _process_feature(
        self,
        target_audio: np.ndarray,
        f_time: float,
    ):
        if self.last_chunk is None:  # add zero padding at the first block
            target_audio = np.concatenate(
                (np.zeros(self.cache_size, dtype=np.float32), target_audio)
            )
        else:
            # add last chunk at the beginning of the block
            target_audio = np.concatenate((self.last_chunk, target_audio))

        perf_time = self._emit_count * self.hop_length / float(self.sample_rate)
        output = self.processor((target_audio, perf_time))

        if self.last_chunk is not None:
            self._emit_count += 1
            self.queue.put(output)

        # update latency stats
        latency = time.time() - self.last_data_received
        self.latency_stats = set_latency_stats(
            latency, self.latency_stats, self.input_index
        )

        # cache last chunk (for the next frame window)
        self.last_chunk = target_audio[-self.cache_size :]

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

    def _preload_audio(self) -> None:
        """Pre-load and resample audio file so run_offline can start immediately."""
        audio_y, sr = librosa.load(self.file_path, sr=None)
        if sr != self.target_sr:
            audio_y = librosa.resample(y=audio_y, orig_sr=sr, target_sr=self.target_sr)
        self._preloaded_audio = audio_y

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

        if self._preloaded_audio is not None:
            audio_y = self._preloaded_audio
            self._preloaded_audio = None  # free memory
        else:
            audio_y, sr = librosa.load(self.file_path, sr=None)
            if sr != self.target_sr:
                audio_y = librosa.resample(
                    y=audio_y, orig_sr=sr, target_sr=self.target_sr
                )
        sr = self.target_sr

        # Pad to next hop_length boundary so no trailing samples are lost
        remainder = len(audio_y) % self.hop_length
        if remainder > 0:
            audio_y = np.concatenate(
                (audio_y, np.zeros(self.hop_length - remainder, dtype=np.float32))
            )
        trimmed_audio = audio_y
        time_interval = self.hop_length / float(sr)
        # Do not stop early on digital silence (all-zeros tails).
        while trimmed_audio.size > 0:
            self.input_index += 1
            self.last_data_received = time.time()
            target_audio = trimmed_audio[: self.hop_length]
            self._process_feature(target_audio, self.last_data_received)
            trimmed_audio = trimmed_audio[self.hop_length :]

            if not self.stream_start.is_set():
                self.stream_start.set()

            if self.wait:
                elapsed_time = time.time() - self.last_data_received
                time.sleep(max(time_interval - elapsed_time, 0))

        self.queue.put(STREAM_END)
        self.stop_listening()

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
        import pyaudio

        self.format = pyaudio.paFloat32
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


class BytesAudioStream(AudioStream):
    """An ``AudioStream`` variant that reads raw PCM bytes from an external queue.

    Designed for non-device audio sources (WebSocket handler, IPC pipe,
    subprocess, etc.). A producer pushes raw ``float32`` PCM ``bytes``
    into ``data_queue`` (one chunk of ``hop_length`` samples per item)
    and ``None`` to signal end of stream. This stream pulls from that
    queue and runs the regular Processor pipeline, putting
    ``(features, perf_time)`` tuples onto ``self.queue``.

    Inherits ``_process_feature`` (frame caching + Processor call) and
    the thread lifecycle from ``AudioStream``; overrides only the input
    source (``run``) and the start/stop hooks that touch PyAudio.

    Parameters
    ----------
    processor : Processor
        Feature processor (e.g. ``ChromagramProcessor``).
    sample_rate : int
        Sample rate of the incoming audio.
    hop_length : int
        Hop length used for feature extraction.
    data_queue : queue.Queue
        Source queue. Producer puts raw ``float32`` PCM ``bytes`` or
        ``None`` (disconnect sentinel).
    queue : RECVQueue, optional
        Output queue for ``(features, perf_time)``. Created if omitted.
    """

    def __init__(
        self,
        processor: Processor,
        sample_rate: int,
        hop_length: int,
        data_queue: queue.Queue,
        queue: Optional[RECVQueue] = None,
    ) -> None:
        # Bypass AudioStream's PyAudio device discovery; init via Stream.
        Stream.__init__(self, processor=processor, mock=False)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        # See AudioStream class docstring "Notes" for cache_size convention.
        self.cache_size = getattr(processor, "n_fft", 2 * hop_length) - hop_length
        self.queue = queue or RECVQueue()
        self.data_queue = data_queue
        self.last_chunk = None
        self._emit_count = 0
        self.input_index = 0
        self.last_data_received = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        # Sentinels: inherited stop_listening checks ``self.audio_stream``.
        self.audio_stream = None
        self.audio_interface = None

    def run(self) -> None:
        """Pull PCM chunks from ``data_queue`` and emit features."""
        self.start_listening()
        while self.listen:
            try:
                data = self.data_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                self.queue.put(STREAM_END)
                return
            if data is None:
                self.queue.put(STREAM_END)
                return
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            self.last_data_received = time.time()
            # _process_feature is inherited (cache_size prepend + processor call)
            self._process_feature(audio_chunk, self.last_data_received)
            if not self.stream_start.is_set():
                self.stream_start.set()

    def start_listening(self) -> None:
        """Override AudioStream.start_listening to skip device start."""
        self.listen = True
        print("* Start listening to bytes audio stream....")

    def stop(self) -> None:
        self.stop_listening()
        # Unblock the thread if it is waiting on data_queue.get()
        try:
            self.data_queue.put_nowait(None)
        except queue.Full:
            pass
        self.join(timeout=5)
