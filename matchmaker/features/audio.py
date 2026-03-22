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
            dtype=np.float32,
        )
        return mfcc.T


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
        y: InputAudioSeries,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        cqt = librosa.cqt(
            y=y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            norm=self.norm,
            dtype=np.float32,
            fmin=librosa.note_to_hz("A0"),
            n_bins=88,
        )
        return np.abs(cqt).T[1:-1]


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
        y: InputAudioSeries,
    ) -> Tuple[Optional[np.ndarray], Dict]:
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

        return features[1:-1]


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
            dtype=np.float32,
        )

        return mel_spectrogram.T


class LogSpectralEnergyProcessor(Processor):
    """
    Log Spectral Energy feature processor based on Dixon (2005).

    Computes a spectral representation using a linear-log frequency scale,
    then applies half-wave rectified first-order difference to emphasize
    note onsets.

    The frequency axis is mapped to:
    - Linear below 370 Hz
    - Logarithmic spacing from 370 Hz to 12,500 Hz (49 bins)
    - One bin above 12,500 Hz
    """

    LINEAR_FREQ_LIMIT = 370  # Hz (F#4)
    LOG_FREQ_LIMIT = 12500  # Hz (G9)
    N_LOG_BINS = 49  # paper: 84 - 34 - 1

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.normalize = normalize
        self._prev_spectrum = None

        # Pre-compute frequency axis and masks
        self._freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        self._linear_mask = self._freqs <= self.LINEAR_FREQ_LIMIT
        self._log_mask = (self._freqs > self.LINEAR_FREQ_LIMIT) & (
            self._freqs <= self.LOG_FREQ_LIMIT
        )
        self._high_mask = self._freqs > self.LOG_FREQ_LIMIT

        # Create N+1 edges for N log-spaced bins
        self._log_bin_edges = np.logspace(
            np.log10(self.LINEAR_FREQ_LIMIT),
            np.log10(self.LOG_FREQ_LIMIT),
            num=self.N_LOG_BINS + 1,
        )

        # Pre-compute bin assignments for log frequencies
        log_freqs = self._freqs[self._log_mask]
        bin_idx = np.digitize(log_freqs, self._log_bin_edges) - 1
        self._log_bin_idx = np.clip(bin_idx, 0, self.N_LOG_BINS - 1)

    def reset(self):
        self._prev_spectrum = None

    def _map_frequencies(self, magnitude):
        """Map FFT magnitude spectrum to linear-log frequency scale."""
        linear_bins = magnitude[self._linear_mask]

        log_bins = magnitude[self._log_mask]
        n_frames = magnitude.shape[1]
        log_mapped = np.zeros((self.N_LOG_BINS, n_frames), dtype=np.float32)
        for b in range(self.N_LOG_BINS):
            mask = self._log_bin_idx == b
            if np.any(mask):
                log_mapped[b] = np.sum(log_bins[mask], axis=0)

        high_freq = np.sum(magnitude[self._high_mask], axis=0, keepdims=True)

        return np.vstack((linear_bins, log_mapped, high_freq)).astype(np.float32)

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
            dtype=np.float32,
        )
        magnitude = np.abs(stft_result)

        # Map to linear-log frequency scale
        feature_vector = self._map_frequencies(magnitude)

        # Half-wave rectified first-order difference (stateful for streaming)
        if self._prev_spectrum is not None:
            combined = np.hstack((self._prev_spectrum, feature_vector))
            diff = np.diff(combined, axis=1)
        else:
            diff = np.diff(
                feature_vector, axis=1, prepend=np.zeros_like(feature_vector[:, :1])
            )

        self._prev_spectrum = feature_vector[:, -1:]

        result = np.maximum(diff, 0).T

        if self.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            result = result / norms

        return result


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

    def __call__(self, y: InputAudioSeries):
        stft = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            dtype=np.float32,
        )
        return np.abs(stft).T  # (n_frames, n_bins)


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
    features = feature_processor(score_y)

    return features
