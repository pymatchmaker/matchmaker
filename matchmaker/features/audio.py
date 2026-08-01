#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""

from typing import Dict, Optional, Tuple, Union

import librosa
import numpy as np

from matchmaker.features.processor import Processor

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
        n_fft: int = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft if n_fft is not None else 2 * self.hop_length
        self.n_chroma = n_chroma
        self.norm = norm

    def __call__(
        self,
        data: InputAudioFrame,
    ) -> Optional[np.ndarray]:
        y, f_time = data
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
        return chroma.T, f_time


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
        data: InputAudioFrame,
    ) -> Optional[np.ndarray]:
        y, f_time = data
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mfcc=self.n_mfcc,
            center=False,
            norm=self.norm,
            dtype=np.float32,
        )
        return mfcc.T, f_time


class CQTProcessor(Processor):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        norm: Optional[Union[float, str]] = NORM,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.norm = norm

    def __call__(
        self,
        data: InputAudioFrame,
    ) -> Optional[np.ndarray]:
        y, f_time = data
        cqt = librosa.cqt(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            norm=self.norm,
            dtype=np.float32,
            fmin=librosa.note_to_hz("A0"),
            n_bins=88,
        )
        return np.abs(cqt).T[1:-1], f_time


class CQTSpectralFluxProcessor(Processor):
    """
    CQT spectrum (88 bins, A0-C8) with optional half-wave rectified spectral flux.
    Output shape: (n_frames, 88) or (n_frames, 89) if include_spectral_flux=True.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        norm: Optional[Union[float, str]] = NORM,
        fmin: Optional[float] = None,
        n_bins: int = 88,
        bins_per_octave: int = 12,
        include_spectral_flux: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.norm = norm
        self.fmin = fmin if fmin is not None else librosa.note_to_hz("A0")
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.include_spectral_flux = include_spectral_flux
        self.prev_magnitude = None

    def __call__(
        self,
        data: InputAudioFrame,
    ) -> Optional[np.ndarray]:
        y, f_time = data
        cqt = librosa.cqt(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            norm=self.norm,
            dtype=np.float32,
        )
        cqt_features = np.abs(cqt).T

        if self.include_spectral_flux:
            if self.prev_magnitude is None:
                spectral_flux = np.zeros((cqt_features.shape[0], 1), dtype=np.float32)
            else:
                diff = np.maximum(cqt_features - self.prev_magnitude, 0)
                spectral_flux = np.sum(diff, axis=1, keepdims=True)

            self.prev_magnitude = cqt_features.copy()
            features = np.hstack([cqt_features, spectral_flux])
        else:
            features = cqt_features

        return features[1:-1], f_time


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
        data: InputAudioFrame,
    ) -> Optional[np.ndarray]:
        y, f_time = data
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            norm=self.norm,
            center=False,
            dtype=np.float32,
        )

        return mel_spectrogram.T, f_time


class LogSpectralEnergyProcessor(Processor):
    """
    Log Spectral Energy feature processor from Dixon (2005).

    A ~46 ms Hamming-windowed STFT (2048 samples at 44.1 kHz) is summed as a
    power spectrum (the paper's "energy") into a linear-log frequency scale:
    identity bins up to the linear-to-log crossover (where the FFT bin
    spacing reaches one semitone), nearest-semitone bins above, capped at
    MIDI 127 (84 elements at 44.1 kHz). A half-wave rectified first-order
    difference emphasizes note onsets, and each difference vector is
    L2-normalized per frame.
    """

    WINDOW_DURATION = 2048 / 44100  # seconds (paper: "46ms (2048 points)")
    REF_FREQ = 440.0  # Hz

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = int(2 ** round(np.log2(self.WINDOW_DURATION * self.sample_rate)))
        self.window = np.hamming(self.n_fft)

        # Per-FFT-bin output index on the linear-log frequency scale.
        df = self.sample_rate / self.n_fft
        cross = int(2.0 / (2.0 ** (1.0 / 12.0) - 1.0))
        offset = int(round(np.log2(cross * df / self.REF_FREQ) * 12.0 + 69.0))
        n_bins = self.n_fft // 2 + 1
        self.freq_map = np.arange(n_bins)
        k = np.arange(cross + 1, n_bins)
        midi = np.minimum(np.log2(k * df / self.REF_FREQ) * 12.0 + 69.0, 127.0)
        self.freq_map[cross + 1 :] = cross + np.round(midi).astype(int) - offset
        self.dim = int(self.freq_map[-1] + 1)

        self.prev_spectrum = None

    def reset(self):
        self.prev_spectrum = None

    def __call__(
        self,
        data: InputAudioFrame,
    ):
        y, f_time = data
        stft_result = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
        )

        spectrum = stft_result.real**2 + stft_result.imag**2

        # Sum FFT bins into the linear-log frequency scale
        feature_vector = np.zeros((self.dim, spectrum.shape[1]), dtype=np.float32)
        np.add.at(feature_vector, self.freq_map, spectrum)

        # Half-wave rectified first-order difference (stateful for streaming)
        if self.prev_spectrum is not None:
            combined = np.hstack((self.prev_spectrum, feature_vector))
            diff = np.diff(combined, axis=1)
        else:
            diff = np.diff(
                feature_vector, axis=1, prepend=np.zeros_like(feature_vector[:, :1])
            )

        self.prev_spectrum = feature_vector[:, -1:]

        result = np.maximum(diff, 0).T
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-10)

        return result, f_time


class RawSpectrumProcessor(Processor):
    """Magnitude FFT spectrum, as used in Jiang & Raphael (ISMIR 2020).

    Returns (n_frames, n_fft//2 + 1) magnitude spectrum.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        hop_length: int = 128,
        n_fft: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __call__(self, data: InputAudioFrame):
        y, f_time = data
        stft = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            dtype=np.float32,
        )
        return np.abs(stft).T, f_time  # (n_frames, n_bins)


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
        "cqt": CQTProcessor,
        "cqt_spectral_flux": CQTSpectralFluxProcessor,
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
    features, _ = feature_processor((score_y, 0.0))

    return features
