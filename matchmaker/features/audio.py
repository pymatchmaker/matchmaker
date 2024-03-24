#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""
from typing import Dict, Optional, Tuple, Callable, List

import librosa
import numpy as np

from matchmaker.utils.processor import Processor

SAMPLE_RATE = 16000
HOP_LENGTH = 640
N_CHROMA = 12
N_MELS = 128
N_MFCC = 13
DCT_TYPE = 2
NORM = np.inf
FEATURES = ["chroma"]

# Type hint for Input Audio frame.
InputAudioSeries = np.ndarray


class ChromagramProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_chroma: int = N_CHROMA,
        norm: Optional[float] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_chroma = n_chroma
        self.norm = norm

    def __call__(
        self,
        y: InputAudioSeries,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            norm=self.norm,
            center=False,
        )
        return chroma


class MFCCProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_mfcc: int = N_MFCC,
        dct_type: int = DCT_TYPE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type

    def __call__(
        self,
        y: InputAudioSeries,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            center=False,
        )
        return mfcc


class MelSpectrogramProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_mels = n_mels

    def __call__(
        self,
        y: InputAudioSeries,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            center=False,
        )
        return mel_spectrogram


def compute_features_from_audio(
    score_audio: str, features=FEATURES, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH
) -> Tuple[List[Callable], np.ndarray]:
    processor_mapping = {
        "chroma": ChromagramProcessor,
        "mel": MelSpectrogramProcessor,
        "mfcc": MFCCProcessor,
    }
    feature_processors = [
        processor_mapping[name](sample_rate=sample_rate, hop_length=hop_length)
        for name in features
    ]
    score_y, sr = librosa.load(score_audio, sr=sample_rate)
    score_y = np.pad(score_y, (hop_length, 0), "constant")
    stacked_features = None
    for feature_processor in feature_processors:
        feature = feature_processor(score_y)
        stacked_features = (
            feature
            if stacked_features is None
            else np.vstack((stacked_features, feature))
        )

    return feature_processors, stacked_features
