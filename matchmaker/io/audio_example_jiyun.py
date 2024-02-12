import queue
import time
from typing import Optional

import librosa
import numpy as np
import pyaudio
import threading

from .config import CHANNELS, CHUNK_SIZE
from .utils import process_chroma, process_mel, process_phonemes


class StreamProcessor:
    def __init__(
        self,
        sample_rate,
        chunk_size,
        hop_length,
        features=["chroma", "phoneme"],  # "chroma", "mel", "mfcc", "phoneme", etc.
    ):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.features = features
        self.format = pyaudio.paFloat32
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.buffer = queue.Queue()
        self.feature_buffer = queue.Queue()
        self.last_chunk = None
        self.index = 0
        self.mock = False

    def _process_feature(self, y, time_info=None):
        if self.last_chunk is None:  # add zero padding at the first block
            y = np.concatenate((np.zeros(self.hop_length), y))
        else:
            # add last chunk at the beginning of the block
            # making 5 block, 1 block overlap -> 4 frames each time
            y = np.concatenate((self.last_chunk, y))

        y_feature = None
        for feature in self.features:
            if feature == "chroma":
                y_chroma = process_chroma(y)
                y_feature = (
                    y_chroma if y_feature is None else np.vstack((y_feature, y_chroma))
                )
            elif feature == "mel":
                y_mel = process_mel(y)
                y_feature = (
                    y_mel if y_feature is None else np.vstack((y_feature, y_mel))
                )
            elif feature == "phoneme":
                y_phoneme = process_phonemes(y)
                y_feature = (
                    y_phoneme
                    if y_feature is None
                    else np.vstack((y_feature, y_phoneme))
                )

        current_chunk = {
            "timestamp": time_info if time_info else time.time(),
            "feature": y_feature[
                :, -int(self.chunk_size / self.hop_length) :
            ],  # trim to chunk_size
        }
        self.feature_buffer.put(current_chunk)
        self.last_chunk = y[-self.hop_length :]
        self.index += 1

    def _process_frame(self, data, frame_count, time_info, status_flag):
        target_audio = np.frombuffer(data, dtype=np.float32)  # initial y
        self.buffer.put(target_audio)
        self._process_feature(target_audio, time_info["input_buffer_adc_time"])

        return (data, pyaudio.paContinue)

    def mock_stream(self, file_path):
        duration = int(librosa.get_duration(path=file_path))
        audio_y, _ = librosa.load(file_path, sr=self.sample_rate)
        padded_audio = np.concatenate(  # zero padding at the end
            (audio_y, np.zeros(duration * 2 * self.sample_rate))
        )
        trimmed_audio = padded_audio[  # trim to multiple of chunk_size
            : len(padded_audio) - (len(padded_audio) % self.chunk_size)
        ]
        while trimmed_audio.any():
            audio_chunk = trimmed_audio[: self.chunk_size]
            time_info = {"input_buffer_adc_time": time.time()}
            self._process_feature(audio_chunk, time_info)
            trimmed_audio = trimmed_audio[self.chunk_size :]
            self.index += 1

        # fill empty values with zeros after stream is finished
        additional_padding_size = duration * 2 * self.sample_rate
        while additional_padding_size > 0:
            time_info = {"input_buffer_adc_time": time.time()}
            self._process_feature(
                np.zeros(self.chunk_size),
                time_info,
            )
            additional_padding_size -= self.chunk_size

    def run(self, mock=False, mock_file=""):
        if mock:  # mock processing
            print(f"* [Mocking] Loading existing audio file({mock_file})....")
            self.mock = True
            x = threading.Thread(target=self.mock_stream, args=(mock_file,))
            x.start()
            return

        # real-time processing
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=self.format,
            channels=CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._process_frame,
        )
        self.audio_stream.start_stream()
        self.start_time = self.audio_stream.get_time()

    def stop(self):
        if not self.mock:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.mock = False
            self.audio_interface.terminate()