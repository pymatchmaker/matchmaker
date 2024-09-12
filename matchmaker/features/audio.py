#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Features from audio files
"""
from typing import Callable, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from matchmaker.utils.processor import Processor
from transformers import AutoProcessor, EncodecModel
from madmom.audio.chroma import DeepChromaProcessor

SAMPLE_RATE = 44100
FRAME_RATE = 30
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
N_CHROMA = 12
N_MELS = 128
N_MFCC = 13
DCT_TYPE = 2
NORM = np.inf
FEATURES = ["chroma"]

# Type hint for Input Audio frame.
InputAudioSeries = np.ndarray

InputAudioFrame = Tuple[InputAudioSeries, float]


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
        data: InputAudioFrame,
        kwargs: Dict = {},
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
        norm: Optional[float] = NORM,
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
        kwargs: Dict = {},
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
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2 * self.hop_length
        self.n_mfcc = n_mfcc

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
            center=False,
            norm=np.inf,
        )
        return mfcc.T


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
            norm=np.inf,
            center=False,
        )
        mel_spectrogram = np.log1p(mel_spectrogram * 5) / 4

        return mel_spectrogram.T


class EnCodecProcessor(Processor):
    """
    EnCodec Process

    TODO:
    * Add reference.
    * Figure out 48kHz version?. The current pre-trained model for
      48kHz requires stereo input and is very slow.
    * Make offline version more efficient, using a single call with a
      larger batch size.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate in Hz, by default 24000.
    hop_length : int, optional
        Size of the hop, by default HOP_LENGTH
    return_discrete_codes : bool, optional
        If True, returns discrete codes, by default False
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = HOP_LENGTH,
        return_discrete_codes: bool = False,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate

        if self.sample_rate == 24000:
            pretrained_checkpoint = "facebook/encodec_24khz"
        # elif self.sample_rate == 48000:
        #     pretrained_checkpoint = "facebook/encodec_48khz"
        else:
            raise ValueError(
                "The pre-trained models require a sample rate of "
                f"either 24kHz, but is {sample_rate/1000:.2f}kHz"
            )

        self.pre_processor = AutoProcessor.from_pretrained(pretrained_checkpoint)

        self.model = EncodecModel.from_pretrained(pretrained_checkpoint)

        self.codebook_size = self.model.config.codebook_size

        self.return_discrete_codes = return_discrete_codes
        self.model.config.chunk_length_s = float(hop_length / sample_rate)

        self.hop_length = hop_length

        self.frame_length = 2 * hop_length

        self.process_offline = False

        # set model in evaluation mode
        self.model.eval()

    @property
    def process_offline(self) -> bool:
        return self._process_stepwise

    @process_offline.setter
    def process_offline(self, process_offline: bool) -> None:
        self._process_stepwise = process_offline

        if process_offline:
            self.call_func = self.batch_processor
        else:
            self.call_func = self.stepwise_processor

    def __call__(
        self,
        data: InputAudioSeries,
        kwargs: Optional[Dict] = None,
    ) -> np.ndarray:
        return self.call_func(data=data, kwargs=kwargs)

    def batch_processor(
        self,
        data: InputAudioSeries,
        kwargs: Optional[Dict] = None,
    ) -> np.ndarray:

        n_frames = int(np.ceil(len(data) / self.hop_length))

        outputs = []
        for i in range(n_frames):

            frame_start = i * self.hop_length
            y = data[frame_start : frame_start + self.frame_length]

            padding_needed = max(self.frame_length - len(y), 0)

            if padding_needed > 0:
                y = np.pad(
                    y,
                    (0, padding_needed),
                    "constant",
                    constant_values=(0),
                )
            feats = self.stepwise_processor(
                data=y,
            )
            outputs.append(feats)

        outputs = np.vstack(outputs).astype(np.float32)

        return outputs

    def stepwise_processor(
        self,
        data: InputAudioSeries,
        kwargs: Optional[Dict] = None,
    ) -> np.ndarray:

        with torch.no_grad():
            # the size of these outputs is(batch_size, channels, sequence_length)

            preprocessed_inputs = self.pre_processor(
                raw_audio=data,
                sampling_rate=self.pre_processor.sampling_rate,
                return_tensors="pt",
            )

            input_values = (
                preprocessed_inputs["input_values"]
                * preprocessed_inputs["padding_mask"]
            )
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale
            embeddings = self.model.encoder(input_values)

            if self.return_discrete_codes:
                embeddings = self.model.quantizer.encode(
                    embeddings=embeddings,
                    # TODO: experiment with this parameter
                    bandwidth=self.model.config.target_bandwidths[0],
                ).transpose(0, 1)

        # The output is a 1D vector
        output = embeddings.detach().numpy().astype(np.float32).squeeze().flatten()

        return output


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
        kwargs: Dict = {},
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
    audio_path: str,
    features=FEATURES,
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
) -> Tuple[List[Callable], np.ndarray]:
    processor_mapping = {
        "chroma": ChromagramProcessor,
        "mel": MelSpectrogramProcessor,
        "mfcc": MFCCProcessor,
        "encodec": EnCodecProcessor,
        "deep_chroma": DeepChromaProcessor,
        "log_spectral": LogSpectralEnergyProcessor,
    }
    feature_processors = [
        processor_mapping[name](sample_rate=sample_rate, hop_length=hop_length)
        for name in features
    ]
    score_y, sr = librosa.load(audio_path, sr=sample_rate)
    score_y = np.pad(score_y, (hop_length, 0), "constant")
    stacked_features = None
    for feature_processor in feature_processors:
        if hasattr(feature_processor, "process_offline"):
            feature_processor.process_offline = True
        feature = feature_processor(score_y)
        stacked_features = (
            feature
            if stacked_features is None
            else np.concatenate((stacked_features, feature), axis=1)
        )

    return feature_processors, stacked_features
