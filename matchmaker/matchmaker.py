import os
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportaudio import save_wav_fluidsynth
from partitura.io.exportmidi import get_ppq
from partitura.score import Part

from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.features.midi import PianoRollProcessor, PitchIOIProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchIOIHMM
from matchmaker.utils.misc import is_audio_file, is_midi_file

PathLike = Union[str, bytes, os.PathLike]
DEFAULT_TEMPO = 120


class Matchmaker(object):
    """
    A class to perform online score following with I/O support for audio and MIDI

    Parameters
    ----------
    score_file : Union[str, bytes, os.PathLike]
        Path to the score file
    performance_file : Union[str, bytes, os.PathLike, None]
        Path to the performance file. If None, live input is used.
    wait : bool (default: True)
        only for offline option. For debugging or fast testing, set to False
    input_type : str
        Type of input to use: audio or midi
    feature_type : str
        Type of feature to use
    method : str
        Score following method to use
    device_name_or_index : Union[str, int]
        Name or index of the audio device to be used.
        Ignored if `file_path` is given.

    """

    def __init__(
        self,
        score_file: PathLike,
        performance_file: Union[PathLike, None] = None,
        wait: bool = True,  # only for offline option. For debugging or fast testing, set to False
        input_type: str = "audio",  # 'audio' or 'midi'
        feature_type: str = None,
        method: str = None,
        device_name_or_index: Union[str, int] = None,
        sample_rate: int = SAMPLE_RATE,
        frame_rate: int = FRAME_RATE,
    ):
        self.score_file = score_file
        self.performance_file = performance_file
        self.input_type = input_type
        self.feature_type = feature_type
        self.frame_rate = frame_rate
        self.score_part: Optional[Part] = None
        self.device_name_or_index = device_name_or_index
        self.processor = None
        self.stream = None
        self.score_follower = None
        self.reference_features = None

        # setup score file
        if score_file is None:
            raise ValueError("Score file is required")

        try:
            self.score_part = partitura.load_score_as_part(score_file)
        except Exception as e:
            raise ValueError(f"Invalid score file: {e}")

        # setup feature processor
        if feature_type is None:
            feature_type = "chroma" if input_type == "audio" else "pitchclass"

        if feature_type == "chroma":
            self.processor = ChromagramProcessor(
                sample_rate=sample_rate,
            )
        elif feature_type == "mfcc":
            self.processor = MFCCProcessor(
                sample_rate=sample_rate,
            )
        elif feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=sample_rate,
            )
        elif feature_type == "pitchclass":
            self.processor = PitchIOIProcessor(piano_range=True)
        elif feature_type == "pianoroll":
            self.processor = PianoRollProcessor(piano_range=True)
        else:
            raise ValueError("Invalid feature type")

        # validate performance file and input_type
        if self.performance_file is not None:
            # check performance file type matches input type
            if self.input_type == "audio" and not is_audio_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected audio file, but got {self.performance_file}"
                )
            elif self.input_type == "midi" and not is_midi_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected MIDI file, but got {self.performance_file}"
                )

        # setup stream device
        if self.input_type == "audio":
            self.stream = AudioStream(
                processor=self.processor,
                device_name_or_index=self.device_name_or_index,
                file_path=self.performance_file,
                wait=wait,
            )
        elif self.input_type == "midi":
            self.stream = MidiStream(
                processor=self.processor,
                port=self.device_name_or_index,
                file_path=self.performance_file,
            )
        else:
            raise ValueError("Invalid input type")

        # preprocess score
        self.reference_features = self.preprocess_score()

        # setup score follower
        if method == "arzt" or (method is None and self.input_type == "audio"):
            self.score_follower = OnlineTimeWarpingArzt(
                reference_features=self.reference_features, queue=self.stream.queue
            )
        elif method == "dixon":
            self.score_follower = OnlineTimeWarpingDixon(
                reference_features=self.reference_features, queue=self.stream.queue
            )
        elif method == "hmm" or (method is None and self.input_type == "midi"):
            self.score_follower = PitchIOIHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )
        else:
            raise ValueError("Invalid method")

    def preprocess_score(self):
        if self.input_type == "audio":
            beat_type = self.score_part.time_sigs[0].beat_type
            musical_beats = self.score_part.time_sigs[0].musical_beats
            score_audio = save_wav_fluidsynth(
                self.score_part,
                bpm=DEFAULT_TEMPO * (beat_type / musical_beats),
            )
            reference_features = self.processor(score_audio.astype(np.float32))
            return reference_features
        else:
            return self.score_part.note_array()

    def convert_frame_to_beat(
        self, current_frame: int, frame_rate: int = FRAME_RATE
    ) -> float:
        """
        Convert frame number to relative beat position in the score.

        Parameters
        ----------
        frame_rate : int
            Frame rate of the audio stream
        current_frame : int
            Current frame number
        """
        tick = get_ppq(self.score_part)
        timeline_time = (current_frame / frame_rate) * tick * (DEFAULT_TEMPO / 60)
        beat_position = np.round(
            self.score_part.beat_map(timeline_time),
            decimals=2,
        )
        return beat_position

    def run(self, verbose: bool = True):
        """
        Run the score following process

        Yields
        ------
        float
            Beat position in the score (interpolated)

        Returns
        -------
        list
            Alignment results with warping path
        """
        with self.stream:
            for current_frame in self.score_follower.run(verbose=verbose):
                if self.input_type == "audio":
                    position_in_beat = self.convert_frame_to_beat(current_frame)
                    yield position_in_beat
                else:
                    yield float(self.score_follower.state_space[current_frame])

            return self.score_follower.warping_path
