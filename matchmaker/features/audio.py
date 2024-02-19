#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
from matchmaker.utils.processor import Processor

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_CHROMA = 12
N_FFT = 512
N_MELS = 128
N_MFCC = 13
DCT_TYPE = 2
NORM = np.inf

# Type hint for Input Audio frame.
InputAudioFrame = Tuple[
    List[Tuple[np.ndarray, int]], float
]  # data, frame_count, time_info


class ChromagramProcessor(Processor):

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_chroma: int = N_CHROMA,
        norm: Optional[float] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.norm = norm

    def __call__(
        self,
        frame: InputAudioFrame,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        (y, f_time), f_time = frame
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            norm=self.norm,
            center=False,
        )
        return chroma, {}


class MFCCProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_mfcc: int = N_MFCC,
        dct_type: int = DCT_TYPE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type

    def __call__(
        self,
        frame: InputAudioFrame,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        (y, f_time), f_time = frame
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            center=False,
        )
        return mfcc, {}


class MelSpectrogramProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_mels: int = N_MELS,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels

    def __call__(
        self,
        frame: InputAudioFrame,
        kwargs: Dict = {},
    ) -> Tuple[Optional[np.ndarray], Dict]:
        (y, f_time), f_time = frame
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            center=False,
        )
        return mel_spectrogram, {}
