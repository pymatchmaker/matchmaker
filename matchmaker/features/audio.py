#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""

import pickle
import struct
from typing import Dict, Optional, Tuple, Union

import librosa
import numpy as np
from scipy.spatial.distance import mahalanobis

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
QUEUE_TIMEOUT = 1

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
    Processor for CQT spectrum with optional Spectral Flux features.

    Based on Nakamura et al. (2013) "Acoustic Score Following to Musical
    Performance with Errors and Arbitrary Repeats and Skips for Automatic Accompaniment".

    This processor extracts CQT (Constant-Q Transform) spectrum (88 dimensions
    matching piano keyboard range) and optionally combines it with spectral flux
    features for improved onset detection and score following.

    The CQT spectrum uses 88 bins corresponding to piano keys (MIDI note 21-108,
    A0 to C8), with one bin per semitone, matching the piano roll representation.
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
        # Default to A0 (MIDI note 21, 27.5 Hz) to match piano range
        # This ensures 88 bins correspond to piano keys (A0 to C8)
        self.fmin = fmin if fmin is not None else librosa.note_to_hz("A0")
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.include_spectral_flux = include_spectral_flux
        self.prev_magnitude = None

    def __call__(
        self,
        y: InputAudioSeries,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        # Compute CQT spectrum
        # CQT provides constant-Q (logarithmic) frequency resolution
        # with one bin per semitone, matching piano keyboard layout
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
        # Use magnitude spectrum (absolute value of complex CQT coefficients)
        cqt_magnitude = np.abs(cqt)

        # Transpose to get time frames as rows: (n_frames, n_bins)
        cqt_features = cqt_magnitude.T

        if self.include_spectral_flux:
            # Compute spectral flux (optional)
            # Spectral flux measures the rate of change of the spectrum
            if self.prev_magnitude is None:
                # For the first frame, use zero flux
                spectral_flux = np.zeros((cqt_features.shape[0], 1), dtype=np.float32)
            else:
                # Compute difference between current and previous magnitude
                diff = cqt_features - self.prev_magnitude
                # Half-wave rectification: only positive differences
                diff_rectified = np.maximum(diff, 0)
                # Sum across frequency bins to get spectral flux per frame
                spectral_flux = np.sum(diff_rectified, axis=1, keepdims=True)

            # Update previous magnitude for next call
            self.prev_magnitude = cqt_features.copy()

            # Combine CQT features with spectral flux
            features = np.hstack([cqt_features, spectral_flux])
        else:
            # Return only CQT features (88 dimensions)
            features = cqt_features

        # Return features: (n_frames, n_features)
        # For streaming, we want the last frame only
        # But for batch processing, return all frames
        return features[1:-1]  # Remove first and last frame (edge effects)


class GaussianToneModel:
    def __init__(
        self, means: np.ndarray, inv_covs: np.ndarray, const_terms: np.ndarray
    ):
        """
        ToneModel implementation based on C++ ToneModel.hpp

        Parameters
        ----------
        means : ndarray (N_models x n_features)
            Mean vectors for each tone model (usually 88 keys + noise)
        inv_covs : ndarray (N_models x n_features x n_features)
            Inverse covariance matrices (diagonal or full)
        const_terms : ndarray (N_models,)
            Precomputed log-constant terms for Gaussian PDF
        """
        self.means = means
        self.inv_covs = inv_covs
        self.const_terms = const_terms
        self.n_models = means.shape[0]
        self.n_features = means.shape[1]
        # True when generated by synthetic_cqt_templates fallback
        self.is_synthetic = False

    def compute_log_likelihood(self, feature: np.ndarray) -> np.ndarray:
        """
        Calculate Log-Observation Probability for a single feature vector.
        Corresponds to ToneModel::calc in C++

        Parameters
        ----------
        feature : ndarray (n_features,)
            Input CQT feature vector

        Returns
        -------
        log_probs : ndarray (N_models,)
            Log likelihood for each tone model
        """
        log_probs = np.zeros(self.n_models)

        # Vectorized implementation of Mahalanobis distance calculation
        # log_prob = const_term - 0.5 * (x - mu).T * inv_cov * (x - mu)
        # Note: C++ code uses full matrix multiplication

        diff = feature - self.means  # (N_models, n_features)

        # (x-mu)^T * inv_cov * (x-mu)
        # - Full covariance: inv_covs shape (N, F, F)
        # - Diagonal covariance: inv_covs shape (N, F) representing diag elements
        if self.inv_covs.ndim == 3:
            # 'ni,nij,nj->n' where n=models, i,j=features
            dist_sq = np.einsum("ni,nij,nj->n", diff, self.inv_covs, diff)
        elif self.inv_covs.ndim == 2:
            # diag quadratic form: sum_k (diff_k^2 * inv_var_k)
            dist_sq = np.sum((diff * diff) * self.inv_covs, axis=1)
        else:
            raise ValueError(
                f"Invalid inv_covs ndim={self.inv_covs.ndim}; expected 2 or 3."
            )

        log_probs = self.const_terms - 0.5 * dist_sq

        return log_probs

    @classmethod
    def load_binary_template(cls, filename: str) -> np.ndarray:
        """
        Port of C++ read_bintemplate function.
        Format: [FD (short)] [num_templates (short)] [data (double) * FD * num]
        """
        with open(filename, "rb") as f:
            header_bytes = f.read(4)
            if len(header_bytes) < 4:
                raise ValueError(f"File {filename} is too short.")

            # Little-endian short ('<h')
            fd, num_templates = struct.unpack("<hh", header_bytes)

            print(f"Loading {filename}: FD={fd}, Num={num_templates}")

            data_size = 8 * fd * num_templates
            data_bytes = f.read(data_size)

            if len(data_bytes) != data_size:
                raise ValueError("File size mismatch with header info.")

            data = np.frombuffer(data_bytes, dtype=np.float64)
            return data.reshape((num_templates, fd))

    @classmethod
    def from_templates(cls, mean_path: str, cov_path: str):
        """Load binary template files and create GaussianToneModel instance."""
        means = cls.load_binary_template(mean_path)
        raw_covs = cls.load_binary_template(cov_path)

        # Fallback to synthetic templates if binary files are corrupted
        maxabs = float(np.max(np.abs(means))) if means.size else float("inf")
        if not np.isfinite(maxabs) or maxabs > 1e6:
            import warnings

            warnings.warn(
                "GaussianToneModel.from_templates(): template means look corrupted "
                f"(max|mean|={maxabs:.3g}). Falling back to synthetic CQT templates.",
                RuntimeWarning,
            )
            return cls.synthetic_cqt_templates(
                n_bins=88,
                n_pitches=88,
                noise_template=True,
                pitch_variance=5e-2,
                noise_variance=5e-4,
                sigma=1.5,
                mix_uniform=0.8,
                noise_log_prior_penalty=50.0,
            )

        N, FD = means.shape

        inv_covs = np.zeros_like(raw_covs, dtype=np.float64)
        const_terms = np.zeros(N, dtype=np.float64)

        for i in range(N):
            safe_cov = np.maximum(np.asarray(raw_covs[i], dtype=np.float64), 1e-6)
            inv_covs[i] = 1.0 / safe_cov
            log_det_cov = np.sum(np.log(safe_cov))
            const_terms[i] = -0.5 * (FD * np.log(2.0 * np.pi) + log_det_cov)
            if not np.isfinite(const_terms[i]):
                const_terms[i] = -1e10

        return cls(np.asarray(means, dtype=np.float64), inv_covs, const_terms)

    @classmethod
    def synthetic_cqt_templates(
        cls,
        n_bins: int = 84,
        n_pitches: int = 88,
        noise_template: bool = True,
        pitch_variance: float = 5e-2,
        noise_variance: float = 5e-4,
        sigma: float = 1.5,
        mix_uniform: float = 0.8,
        noise_log_prior_penalty: float = 50.0,
    ) -> "GaussianToneModel":
        """
        Generate a simple, numerically-stable synthetic tone model.

        - pitch template k: Gaussian bump centered at a mapped bin index
        - noise template: uniform spectrum
        """
        n_models = int(n_pitches + (1 if noise_template else 0))
        means = np.zeros((n_models, n_bins), dtype=np.float64)

        # map 88 pitch indices to n_bins (84) linearly
        bin_pos = np.linspace(0, n_bins - 1, num=n_pitches)
        xs = np.arange(n_bins, dtype=np.float64)
        uni = np.ones(n_bins, dtype=np.float64) / n_bins
        for k in range(n_pitches):
            c = float(bin_pos[k])
            bump = np.exp(-0.5 * ((xs - c) / max(sigma, 1e-6)) ** 2)
            bump = bump / (np.sum(bump) + 1e-12)
            # Mix with uniform to account for CQT harmonic leakage
            a = float(np.clip(mix_uniform, 0.0, 1.0))
            means[k] = (1.0 - a) * bump + a * uni

        if noise_template:
            means[-1] = uni

        pitch_var = float(max(pitch_variance, 1e-8))
        noise_var = float(max(noise_variance, 1e-8))
        inv_covs = np.full((n_models, n_bins), 1.0 / pitch_var, dtype=np.float64)
        const_terms = np.full(
            (n_models,),
            -0.5 * (n_bins * np.log(2.0 * np.pi) + n_bins * np.log(pitch_var)),
            dtype=np.float64,
        )
        if noise_template:
            inv_covs[-1] = 1.0 / noise_var
            const_terms[-1] = -0.5 * (
                n_bins * np.log(2.0 * np.pi) + n_bins * np.log(noise_var)
            )
            # Prior penalty to prevent noise template from dominating
            const_terms[-1] -= float(max(noise_log_prior_penalty, 0.0))
        model = cls(means=means, inv_covs=inv_covs, const_terms=const_terms)
        model.is_synthetic = True
        return model


class HMMAudioProcessor:
    def __init__(
        self,
        tone_model: GaussianToneModel,
        sample_rate: int = 16000,
        hop_length: int = 320,  # 20ms at 16k
        n_bins: int = 88,
        accept_cqt_features: bool = False,
    ):
        self.tone_model = tone_model
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.accept_cqt_features = accept_cqt_features

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """
        Process audio and return observation probabilities for HMM.

        Parameters
        ----------
        y : ndarray
            If accept_cqt_features=False: Input audio signal (1D array, raw audio)
            If accept_cqt_features=True: CQT features (2D array, shape: (n_frames, n_bins))
                                         or single frame (1D array, shape: (n_bins,))

        Returns
        -------
        obs_probs : ndarray (N_models,)
            Observation likelihoods (Log scale) for HMM update
        """
        if self.accept_cqt_features:
            # Input is already CQT features
            if y.ndim == 2:
                # Multiple frames: take the last frame
                spec = y[-1]
            else:
                # Single frame
                spec = y

            # Flatten to 1D if needed (handle (n_bins, 1) shape)
            spec = spec.flatten()

            # Ensure it's magnitude (in case it's complex)
            spec = np.abs(spec)

            # Crop to match template feature dimension (84 bins)
            # Template expects FD=84, but CQTProcessor outputs 88 bins
            template_fd = self.tone_model.means.shape[
                1
            ]  # Get feature dimension from template
            if len(spec) > template_fd:
                # Crop to match template dimension (take first template_fd bins)
                spec = spec[:template_fd]
            elif len(spec) < template_fd:
                # Pad if needed (shouldn't happen, but handle gracefully)
                padding = np.zeros(template_fd - len(spec))
                spec = np.concatenate([spec, padding])

            # Final check: ensure 1D (handle case where spec might be (n_features, 1))
            spec = np.asarray(spec).flatten()
        else:
            # Input is raw audio: compute CQT
            # 1. CQT Extraction (Corresponds to Filtered_Spectrum in C++)
            # Note: C++ uses a custom pre-calculated filter matrix.
            # librosa.cqt is a good approximation.
            cqt = librosa.cqt(
                y=y,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=librosa.note_to_hz("A0"),  # MIDI 21
                n_bins=self.n_bins,
                bins_per_octave=12,
            )

            # Get magnitude spectrum for the center frame
            # (If processing a buffer, take the specific column)
            spec = np.abs(cqt).T[-1]  # Take the last frame if streaming

            # Crop to match template feature dimension (84 bins)
            # Template expects FD=84, but we might compute 88 bins
            template_fd = self.tone_model.means.shape[
                1
            ]  # Get feature dimension from template
            if len(spec) > template_fd:
                # Crop to match template dimension (take first template_fd bins)
                spec = spec[:template_fd]
            elif len(spec) < template_fd:
                # Pad if needed (shouldn't happen, but handle gracefully)
                padding = np.zeros(template_fd - len(spec))
                spec = np.concatenate([spec, padding])

            # Final check: ensure 1D
            spec = np.asarray(spec).flatten()

        # 2. Normalization (Matches C++ 'calc_log_obsprob')
        # double power=(*new_feature).sum();
        # (*new_feature).array()/=(*new_feature).sum();
        # Ensure spec is 1D before normalization
        spec = np.asarray(spec).flatten()
        power = np.sum(spec) + 1e-10
        normalized_spec = spec / power

        # Ensure normalized_spec is 1D (not (n_features, 1))
        normalized_spec = np.asarray(normalized_spec).flatten()

        # 3. Tone Model Calculation
        # C++: dists[i].calc(new_feature, power)
        log_obs_probs = self.tone_model.compute_log_likelihood(normalized_spec)

        return log_obs_probs


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
