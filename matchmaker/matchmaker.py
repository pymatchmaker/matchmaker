import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import partitura
import soundfile as sf
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
from matchmaker.utils.eval import TOLERANCES, get_evaluation_results
from matchmaker.utils.misc import (
    adjust_tempo_for_performance_audio,
    generate_score_audio,
    is_audio_file,
    is_midi_file,
)

PathLike = Union[str, bytes, os.PathLike]
DEFAULT_TEMPO = 120
DEFAULT_DISTANCE_FUNCS = {
    "arzt": OnlineTimeWarpingArzt.DEFAULT_DISTANCE_FUNC,
    "dixon": OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
    "hmm": None,
}

DEFAULT_METHODS = {
    "audio": "arzt",
    "midi": "hmm",
}

AVAILABLE_METHODS = ["arzt", "dixon", "hmm"]


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
        distance_func: Optional[str] = None,
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
        self.distance_func = distance_func
        self.device_name_or_index = device_name_or_index
        self.processor = None
        self.stream = None
        self.score_follower = None
        self.reference_features = None
        self.tempo = DEFAULT_TEMPO  # bpm for quarter note
        self._has_run = False

        # setup score file
        if score_file is None:
            raise ValueError("Score file is required")

        try:
            self.score_part = partitura.load_score_as_part(self.score_file)

        except Exception as e:
            raise ValueError(f"Invalid score file: {e}")

        # setup feature processor
        if self.feature_type is None:
            self.feature_type = "chroma" if input_type == "audio" else "pitchclass"

        if self.feature_type == "chroma":
            self.processor = ChromagramProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "mfcc":
            self.processor = MFCCProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "pitchclass":
            self.processor = PitchIOIProcessor(piano_range=True)
        elif self.feature_type == "pianoroll":
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
                target_sr=SAMPLE_RATE,
            )
        elif self.input_type == "midi":
            self.stream = MidiStream(
                processor=self.processor,
                port=self.device_name_or_index,
                file_path=self.performance_file,
            )
        else:
            raise ValueError("Invalid input type")

        # preprocess score (setting reference features, tempo)
        self.preprocess_score()

        # validate method first
        if method is None:
            method = DEFAULT_METHODS[self.input_type]
        elif method not in AVAILABLE_METHODS:
            raise ValueError(f"Invalid method. Available methods: {AVAILABLE_METHODS}")

        # setup distance function
        if distance_func is None:
            distance_func = DEFAULT_DISTANCE_FUNCS[method]

        # setup score follower
        if method == "arzt":
            self.score_follower = OnlineTimeWarpingArzt(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                distance_func=distance_func,
                frame_rate=self.frame_rate,
            )
        elif method == "dixon":
            self.score_follower = OnlineTimeWarpingDixon(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                distance_func=distance_func,
                frame_rate=self.frame_rate,
            )
        elif method == "hmm":
            self.score_follower = PitchIOIHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )

    def preprocess_score(self):
        if self.input_type == "audio":
            if self.performance_file is not None:
                # tempo is slightly adjusted to reflect the tempo of the performance audio
                self.tempo = adjust_tempo_for_performance_audio(
                    self.score_part, self.performance_file
                )

            # generate score audio
            score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)

            # save score audio (for debugging)
            score_annots = self.build_score_annotations()
            score_annots_audio = librosa.clicks(
                times=score_annots,
                sr=SAMPLE_RATE,
                click_freq=1000,
                length=len(score_audio),
            )
            score_audio_mixed = score_audio + score_annots_audio
            sf.write(
                f"score_audio_{Path(self.score_file).stem}.wav",
                score_audio_mixed,
                SAMPLE_RATE,
                subtype="PCM_24",
            )

            reference_features = self.processor(score_audio.astype(np.float32))
            self.reference_features = reference_features
        else:
            self.reference_features = self.score_part.note_array()

    def _convert_frame_to_beat(self, current_frame: int) -> float:
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
        timeline_time = (current_frame / self.frame_rate) * tick * (self.tempo / 60)
        beat_position = np.round(
            self.score_part.beat_map(timeline_time),
            decimals=2,
        )
        return beat_position

    def run(self, verbose: bool = True, wait: bool = True):
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
                    position_in_beat = self._convert_frame_to_beat(current_frame)
                    yield position_in_beat
                else:
                    yield float(self.score_follower.state_space[current_frame])

        self._has_run = True
        return self.score_follower.warping_path

    def build_score_annotations(self, level="beat"):
        score_annots = []
        if level == "beat":  # TODO: add bar-level, note-level
            note_array = np.unique(self.score_part.note_array()["onset_beat"])
            start_beat = np.ceil(note_array.min())
            end_beat = np.floor(note_array.max())
            beats = np.arange(start_beat, end_beat + 1)

            beat_timestamp = [
                self.score_part.inv_beat_map(beat)
                / get_ppq(self.score_part)
                * (60 / self.tempo)
                for beat in beats
            ]

            score_annots = np.array(beat_timestamp)
        return score_annots

    def run_evaluation(
        self,
        perf_annotations: PathLike,
        level: str = "beat",
        tolerance: list = TOLERANCES,
    ) -> dict:
        """
        Evaluate the score following process

        Parameters
        ----------
        perf_annotations : PathLike
            Path to the performance annotations file (tab-separated)
        level : str
            Level of annotations to use: bar, beat or note
        tolerance : list
            Tolerances to use for evaluation (in milliseconds)

        Returns
        -------
        dict
            Evaluation results with mean, median, std, skewness, kurtosis, and
            accuracy for each tolerance
        """
        if not self._has_run:
            raise ValueError("Must call run() before evaluation")

        score_annots = self.build_score_annotations(level)
        perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)

        # print(f"score annotations: {score_annots}, len: {len(score_annots)}")
        # print(f"performance annotations: {perf_annots}, len: {len(perf_annots)}")

        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]

        return get_evaluation_results(
            score_annots, perf_annots, self.score_follower.warping_path, self.frame_rate
        )
