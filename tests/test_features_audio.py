#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Tests for the features/audio.py module
"""
import os
import tempfile
import unittest

import librosa
import numpy as np
from partitura.utils.music import midi_pitch_to_frequency, note_name_to_midi_pitch

from matchmaker.features.audio import (
    ChromagramIOIProcessor,
    ChromagramProcessor,
    LogSpectralEnergyProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    compute_features_from_audio,
)

SAMPLE_RATE = 44100
HOP_LENGTH = SAMPLE_RATE // 30
N_CHROMA = 12
N_MFCC = 13
N_MELS = 128

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def create_sample_audio_waveform(freq: float = 440.0) -> np.ndarray:
    """Prepare a sample audio waveform (sinusoidal wave) for testing"""
    # 1 second of audio
    duration = 1.0
    sample_audio = np.sin(
        2 * np.pi * freq * np.linspace(0, duration, int(SAMPLE_RATE * duration))
    )
    return sample_audio


class TestAudioProcessors(unittest.TestCase):

    def test_chromagram_processor(self):

        notes = [
            "C4",
            "C#4",
            "D4",
            "D#4",
            "E4",
            "F4",
            "F#4",
            "G4",
            "G#4",
            "A4",
            "A#4",
            "B4",
        ]

        for i, note in enumerate(notes):
            midi_pitch = note_name_to_midi_pitch(note)
            freq = midi_pitch_to_frequency(midi_pitch)
            sample_audio = create_sample_audio_waveform(freq)
            frame_time = 0
            processor = ChromagramProcessor()
            chroma_output = processor((sample_audio, frame_time))
            # check that the chroma output is an array
            self.assertIsInstance(chroma_output, np.ndarray)
            # check that the chroma output has the right number of chroma
            self.assertEqual(chroma_output.shape[1], N_CHROMA)
            # test that the output of the chroma corresponds to the right note
            self.assertEqual(chroma_output.sum(0).argmax(), i)
            # Expected shape for one second of audio
            self.assertEqual(chroma_output.shape[0], 30 - 1)

    def test_chromagram_ioi_processor(self):
        processor = ChromagramIOIProcessor()

        notes = [
            "C4",
            "C#4",
            "D4",
            "D#4",
            "E4",
            "F4",
            "F#4",
            "G4",
            "G#4",
            "A4",
            "A#4",
            "B4",
        ]

        for i, note in enumerate(notes):
            midi_pitch = note_name_to_midi_pitch(note)
            freq = midi_pitch_to_frequency(midi_pitch)
            sample_audio = create_sample_audio_waveform(freq)
            frame_time = i * 0.5
            chroma_output, ioi_obs = processor((sample_audio, frame_time))
            # check that the chroma output is an array
            self.assertIsInstance(chroma_output, np.ndarray)
            # check that the chroma output has the right number of chroma
            self.assertEqual(chroma_output.shape[1], N_CHROMA)
            # test that the output of the chroma corresponds to the right note
            self.assertEqual(chroma_output.sum(0).argmax(), i)
            # Expected shape for one second of audio
            self.assertEqual(chroma_output.shape[0], 30 - 1)
            # check ioi obs
            expected_ioi_obs = 0.5 if i > 0 else 0.0
            self.assertTrue(ioi_obs == expected_ioi_obs)

    def test_mfcc_processor(self):
        processor = MFCCProcessor()

        notes = [
            "C4",
            "C#4",
            "D4",
            "D#4",
            "E4",
            "F4",
            "F#4",
            "G4",
            "G#4",
            "A4",
            "A#4",
            "B4",
        ]

        for i, note in enumerate(notes):
            midi_pitch = note_name_to_midi_pitch(note)
            freq = midi_pitch_to_frequency(midi_pitch)
            sample_audio = create_sample_audio_waveform(freq)
            mfcc_output = processor(sample_audio)
            self.assertIsInstance(mfcc_output, np.ndarray)
            # check that we have the expected number of MFCCs
            self.assertEqual(mfcc_output.shape[1], N_MFCC)
            # Expected shape for one second of audio
            self.assertEqual(mfcc_output.shape[0], 30 - 1)
            # This represents the log energy of the signal. Since a pure sine wave
            # has a single frequency and no variation in energy across frequencies,
            # the first MFCC will capture most of the energy in the signal.
            self.assertTrue(mfcc_output.mean(0).argmax() == 1)

    def test_mel_spectrogram_processor(self):
        processor = MelSpectrogramProcessor()

        # Get the Mel frequencies corresponding to each Mel band
        mel_frequencies = librosa.mel_frequencies(
            n_mels=N_MELS,
            fmin=0,
            fmax=SAMPLE_RATE / 2,
        )

        for i, freq in enumerate(mel_frequencies):
            sample_audio = create_sample_audio_waveform(freq)
            mel_output = processor(sample_audio)
            self.assertIsInstance(mel_output, np.ndarray)
            # check expected number of mels
            self.assertEqual(mel_output.shape[1], N_MELS)
            # Expected shape for one second of audio
            self.assertEqual(mel_output.shape[0], 30 - 1)
            # Test that the frequencies are within one bin
            self.assertTrue(abs(mel_output.mean(0).argmax() - i) <= 1)

    def test_log_spectral_energy_processor(self):
        processor = LogSpectralEnergyProcessor()
        sample_audio = create_sample_audio_waveform(440)
        log_spectral_output = processor(sample_audio)
        self.assertIsInstance(log_spectral_output, np.ndarray)
        self.assertGreater(log_spectral_output.shape[1], 0)
        self.assertEqual(log_spectral_output.shape[0], 30 - 1)


class TestComputeFeaturesFromAudio(unittest.TestCase):

    def test_compute_features_from_audio_input_str(self):

        features = compute_features_from_audio(
            ref_info=os.path.join(CURRENT_PATH, "resources", "Bach-fugue_bwv_858.mp3")
        )

        self.assertIsInstance(features, np.ndarray)

    def test_compute_features_from_np_array(self):

        sample_audio = create_sample_audio_waveform(440)

        features = compute_features_from_audio(
            ref_info=sample_audio,
            sample_rate=SAMPLE_RATE,
        )

        self.assertIsInstance(features, np.ndarray)


if __name__ == "__main__":
    unittest.main()
