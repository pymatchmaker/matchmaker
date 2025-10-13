import csv
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.score import Part

from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    ChromagramProcessor,
    CQTProcessor,
    LogSpectralEnergyProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.features.midi import PianoRollProcessor, PitchIOIProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import (
    CosineExpGaussianAudioPitchTempoObservationModel,
    GaussianAudioPitchHMM,
    GaussianAudioPitchTempoHMM,
    PitchIOIHMM,
)
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MILLISECONDS,
    get_evaluation_results,
    transfer_from_perf_to_predicted_score,
    transfer_from_score_to_predicted_perf,
)
from matchmaker.utils.misc import (
    adjust_tempo_for_performance_audio,
    generate_score_audio,
    is_audio_file,
    is_midi_file,
    save_debug_results,
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

AVAILABLE_METHODS = ["arzt", "dixon", "hmm", "pthmm"]


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
        self.score_file = str(score_file)
        self.performance_file = (
            str(performance_file) if performance_file is not None else None
        )
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
        self.method = method

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
        elif self.feature_type == "cqt":
            self.processor = CQTProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "lse":
            self.processor = LogSpectralEnergyProcessor(
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
        elif method == "hmm" and self.input_type == "midi":
            self.score_follower = PitchIOIHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )
        elif method == "hmm" and self.input_type == "audio":
            # state_space = self._convert_frame_to_beat(np.arange(len(self.reference_features)))
            self.score_follower = GaussianAudioPitchHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                # state_space=state_space,
                # patience=50,
            )

        elif method == "pthmm" and self.input_type == "audio":
            self.score_follower = GaussianAudioPitchTempoHMM(
                reference_features=self.reference_features,
                # observation_model=obs_model,
                queue=self.stream.queue,
                # pitch_precision=0.5,
                # ioi_precision=2,
                transition_scale=0.05,
            )

    def preprocess_score(self):
        if self.input_type == "audio":
            if self.performance_file is not None:
                # tempo is slightly adjusted to reflect the tempo of the performance audio
                self.tempo = adjust_tempo_for_performance_audio(
                    self.score_part, self.performance_file
                )

            # generate score audio
            self.score_audio = generate_score_audio(
                self.score_part, self.tempo, SAMPLE_RATE
            ).astype(np.float32)

            reference_features = self.processor(self.score_audio)
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

    def build_score_annotations(self, level="beat", musical_beat: bool = False):
        score_annots = []
        if level == "beat":  # TODO: add bar-level, note-level
            if musical_beat:
                self.score_part.use_musical_beat()  # for asap dataset
            note_array = np.unique(self.score_part.note_array()["onset_beat"])
            start_beat = np.ceil(note_array.min())
            end_beat = np.floor(note_array.max())
            self.beats = np.arange(start_beat, end_beat + 1)

            beat_timestamp = [
                self.score_part.inv_beat_map(beat)
                / self.score_part.quarter_duration_map(
                    self.score_part.inv_beat_map(beat)
                )
                * (60 / self.tempo)
                for beat in self.beats
            ]

            score_annots = np.array(beat_timestamp)
        return score_annots

    def convert_timestamps_to_beats(self, timestamps):
        """
        Convert an array of timestamps (in seconds) to beat positions.

        Parameters
        ----------
        timestamps : array-like
            Array of timestamps in seconds

        Returns
        -------
        beats : np.ndarray
            Array of beat positions corresponding to the input timestamps
        """
        beats = []
        tick = get_ppq(self.score_part)

        for timestamp in timestamps:
            timeline_time = timestamp * tick * (self.tempo / 60)

            beat_position = np.round(
                self.score_part.beat_map(timeline_time),
                decimals=2,
            )
            beats.append(beat_position)

        return np.array(beats)

    def get_latency_stats(self):
        feature_stats = self.stream.latency_stats
        inference_stats = self.score_follower.latency_stats

        return {
            "f_avg_latency": round(
                feature_stats["total_latency"] / feature_stats["total_frames"] * 1000,
                3,
            ),
            "i_avg_latency": round(
                inference_stats["total_latency"]
                / inference_stats["total_frames"]
                * 1000,
                3,
            ),
        }

    def run_evaluation(
        self,
        perf_annotations: PathLike,
        level: str = "beat",
        tolerances: list = TOLERANCES_IN_MILLISECONDS,
        musical_beat: bool = False,  # beat annots are difference in some dataset
        debug: bool = False,
        save_dir: PathLike = None,
        run_name: str = None,
        in_seconds: bool = True,  # 'True' for performance-based, 'False' for score-based
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
        debug : bool
            Whether to save the score and performance audio with beat annotations
        axis : str
            Evaluation axis, either 'score' or 'performance'

        Returns
        -------
        dict
            Evaluation results with mean, median, std, skewness, kurtosis, and
            accuracy for each tolerance
        """
        if not self._has_run:
            raise ValueError("Must call run() before evaluation")

        score_annots = self.build_score_annotations(level, musical_beat)
        perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)
        original_perf_annots_length = len(perf_annots)

        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]

        perf_annots_predicted = transfer_from_score_to_predicted_perf(
            self.score_follower.warping_path, score_annots, frame_rate=self.frame_rate
        )

        score_annots_predicted = transfer_from_perf_to_predicted_score(
            self.score_follower.warping_path, perf_annots, frame_rate=self.frame_rate
        )
        score_annots = score_annots[: len(score_annots_predicted)]

        if original_perf_annots_length != len(perf_annots_predicted):
            print(
                f"Length of the annotation changed: {original_perf_annots_length} -> {len(perf_annots_predicted)}"
            )

        if debug:
            save_debug_results(
                self.score_file,
                self.score_audio,
                score_annots,
                score_annots_predicted,
                self.performance_file,
                perf_annots,
                perf_annots_predicted,
                self.score_follower,
                self.frame_rate,
                save_dir,
                run_name,
            )

        if in_seconds:
            eval_results = get_evaluation_results(
                perf_annots,
                perf_annots_predicted,
                total_length=original_perf_annots_length,
                tolerances=tolerances,
            )
        else:
            score_annots = self.beats
            score_annots_predicted = self.convert_timestamps_to_beats(
                score_annots_predicted
            )
            if tolerances == TOLERANCES_IN_MILLISECONDS:
                tolerances = TOLERANCES_IN_BEATS  # switch to beats
            eval_results = get_evaluation_results(
                score_annots,
                score_annots_predicted,
                total_length=original_perf_annots_length,
                tolerances=tolerances,
                in_seconds=False,
            )

        latency_results = self.get_latency_stats()
        eval_results.update(latency_results)
        return eval_results

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
