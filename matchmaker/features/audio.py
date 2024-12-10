#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""

from typing import Dict, Optional, Tuple, Union

import librosa
import numpy as np

# from madmom.audio.chroma import DeepChromaProcessor
from matchmaker.utils.processor import Processor

SAMPLE_RATE = 44100
FRAME_RATE = 30
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
N_CHROMA = 12
N_MELS = 128
N_MFCC = 13
DCT_TYPE = 2
NORM = np.inf
FEATURES = "chroma"

# Type hint for Input Audio frame.
InputAudioSeries = np.ndarray

InputAudioFrame = Tuple[InputAudioSeries, float]


class ChromagramProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_chroma: int = N_CHROMA,
        norm: Optional[Union[float, str]] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_chroma = n_chroma
        self.norm = norm

    def __call__(
        self,
        data: InputAudioFrame,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        if isinstance(data, tuple):
            y, f_time = data
        else:
            y = data
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            norm=self.norm,
            center=False,
            dtype=np.float32,
        )
        return chroma.T


class ChromagramIOIProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_chroma: int = N_CHROMA,
        norm: Optional[Union[float, str]] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_chroma = n_chroma
        self.norm = norm
        self.prev_time = None

    def __call__(
        self,
        data: InputAudioFrame,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        y, f_time = data

        if self.prev_time is None:
            ioi_obs = 0
        else:
            ioi_obs = f_time - self.prev_time

        self.prev_time = f_time
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            norm=self.norm,
            center=False,
            dtype=np.float32,
        )
        return chroma.T, ioi_obs


class MFCCProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_mfcc: int = N_MFCC,
        norm: Optional[Union[float, str]] = "backward",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_mfcc = n_mfcc
        self.norm = norm

    def __call__(
        self,
        y: InputAudioSeries,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc=self.n_mfcc,
            center=False,
            norm=self.norm,
        )
        return mfcc.T


class MelSpectrogramProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        norm: Optional[Union[float, str]] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_mels = n_mels
        self.norm = norm

    def __call__(
        self,
        y: InputAudioSeries,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            norm=self.norm,
            center=False,
        )
        mel_spectrogram = np.log1p(mel_spectrogram * 5) / 4

        return mel_spectrogram.T


class LogSpectralEnergyProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length

    def __call__(
        self,
        y: InputAudioSeries,
    ):
        stft_result = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=False,
        )
        magnitude = np.abs(stft_result)

        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)

        linear_limit = 370
        log_limit = 12500
        linear_bins = magnitude[freqs <= linear_limit, :]
        log_bins = magnitude[(freqs > linear_limit) & (freqs <= log_limit), :]

        log_bin_edges = np.logspace(
            np.log10(linear_limit), np.log10(log_limit), num=84 - 34 - 1
        )
        log_mapped_bins = np.zeros((len(log_bin_edges), linear_bins.shape[1]))

        for i in range(log_mapped_bins.shape[1]):
            log_bin_idx = np.digitize(
                freqs[(freqs > linear_limit) & (freqs <= log_limit)], log_bin_edges
            )
            for j in range(1, len(log_bin_edges)):
                log_mapped_bins[j - 1, i] = np.sum(log_bins[log_bin_idx == j, i])

        high_freq_bin = np.sum(magnitude[freqs > log_limit, :], axis=0, keepdims=True)

        feature_vector = np.vstack(
            (linear_bins, log_mapped_bins, high_freq_bin), dtype=np.float32
        )

        diff_feature_vector = np.diff(
            feature_vector, axis=0, prepend=feature_vector[0:1, :]
        )
        half_wave_rectified_vector = np.maximum(diff_feature_vector, 0)

        return half_wave_rectified_vector.T


def compute_features_from_audio(
    ref_info: Union[np.ndarray, str],
    processor_name=FEATURES,
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
) -> np.ndarray:
    """
    Compute features from an audio file.
    """
    processor_mapping = {
        "chroma": ChromagramProcessor,
        "mel": MelSpectrogramProcessor,
        "mfcc": MFCCProcessor,
        "log_spectral": LogSpectralEnergyProcessor,
        # "deep_chroma": DeepChromaProcessor,
    }

    feature_processor = processor_mapping[processor_name](
        sample_rate=sample_rate,
        hop_length=hop_length,
    )

    if isinstance(ref_info, str):
        score_y, _ = librosa.load(ref_info, sr=sample_rate)
    elif isinstance(ref_info, np.ndarray):
        score_y = ref_info

    score_y = np.pad(score_y, (hop_length, 0), "constant")
    features = feature_processor(score_y)

    return features
