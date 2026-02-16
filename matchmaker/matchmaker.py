import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.score import Part, merge_parts

from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    ChromagramProcessor,
    CQTProcessor,
    CQTSpectralFluxProcessor,
    LogSpectralEnergyProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
)
from matchmaker.features.midi import (
    PianoRollProcessor,
    PitchClassPianoRollProcessor,
    PitchIOIProcessor,
)
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import (
    GaussianAudioPitchHMM,
    GaussianAudioPitchTempoHMM,
    PitchHMM,
    PitchIOIHMM,
    PitchHMM,
)
from matchmaker.prob.outer_product_hmm import OuterProductHMM
from matchmaker.prob.outer_product_hmm_audio import AudioOuterProductHMM
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
    get_tempo_from_score,
    is_audio_file,
    is_midi_file,
    plot_and_save_gt_vs_pred_points,
    save_debug_results,
)
from matchmaker.utils.tempo_models import KalmanTempoModel

sys.setrecursionlimit(10_000)

PathLike = Union[str, bytes, os.PathLike]
DEFAULT_TEMPO = 120


class _LastFrameProcessor:
    """Wrapper that returns only the last frame from a processor's output."""

    def __init__(self, base_processor):
        self.base_processor = base_processor

    def __call__(self, y):
        feats = np.asarray(self.base_processor(y))
        return feats[-1] if feats.ndim == 2 else feats


DEFAULT_DISTANCE_FUNCS = {
    "arzt": OnlineTimeWarpingArzt.DEFAULT_DISTANCE_FUNC,
    "dixon": OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
    "hmm": None,
    "outerhmm": None,
    "audio_outerhmm": None,
    "pthmm": None,
}

DEFAULT_METHODS = {
    "audio": "arzt",
    "midi": "outerhmm",
}

AVAILABLE_METHODS = ["arzt", "dixon", "hmm", "pthmm", "outerhmm", "audio_outerhmm"]
KWARGS = {
    "audio": {
        "dixon": {
            "window_size": 10,
        },
        "arzt": {},
        "audio_outerhmm": {
            "sample_rate": 16000,
            "frame_rate": 50,
        },
    },
    "midi": {
        "arzt": {
            "processor": "pianoroll",
            "piano_range": True,
        },
        "dixon": {
            "processor": "pianoroll",
            "piano_range": True,
            "window_size": 30,
        },
        "hmm": {
            "processor": "pitch_ioi",
            "tempo_model": KalmanTempoModel,
            "piano_range": True,
        },
        "pthmm": {
            "processor": "pitch_ioi",
            "piano_range": True,
        },
        "outerhmm": {
            "processor": "pitch_ioi",
            "piano_range": True,
        },
    },
}


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
    tempo : float, optional
        Tempo in BPM. If None, reads from score; if score has no tempo marking,
        defaults to 120 BPM.
    adjust_tempo : bool (default: False)
        If True and performance_file is provided, adjusts tempo based on
        performance audio analysis. Applies to all methods.

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
        tempo: Optional[float] = None,
        kwargs=KWARGS,
        unfold_score=True,
        auto_adjust_tempo: bool = False,
    ):
        self.score_file = str(score_file)
        self.performance_file = (
            str(performance_file) if performance_file is not None else None
        )

        # if input_type not in ("audio", "midi"):
        #     raise ValueError(f"Invalid input_type {input_type}")
        self.input_type = input_type
        self.feature_type = feature_type
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.hop_length = sample_rate // self.frame_rate
        self.score_part: Optional[Part] = None
        self.distance_func = distance_func
        self.device_name_or_index = device_name_or_index
        self.processor = None
        self.stream = None
        self.score_follower = None
        self.reference_features = None
        self._has_run = False

        # validate method first
        if method is None:
            method = DEFAULT_METHODS[self.input_type]
        elif method not in AVAILABLE_METHODS:
            raise ValueError(f"Invalid method. Available methods: {AVAILABLE_METHODS}")

        self.method = method
        self.config = kwargs[input_type][self.method]
        self.auto_adjust_tempo = auto_adjust_tempo

        # Apply method-specific defaults from config (only if not explicitly provided by caller)
        if sample_rate == SAMPLE_RATE and "sample_rate" in self.config:
            self.sample_rate = self.config["sample_rate"]
        if frame_rate == FRAME_RATE and "frame_rate" in self.config:
            self.frame_rate = self.config["frame_rate"]
        self.hop_length = self.sample_rate // self.frame_rate

        # setup score file
        try:
            ext = Path(self.score_file).suffix.lower()
            if ext in (".musicxml", ".xml", ".mxl"):
                score = partitura.load_musicxml(
                    self.score_file, ignore_invisible_objects=True
                )
            else:
                score = partitura.load_score(self.score_file)

            if unfold_score:
                score = partitura.score.unfold_part_maximal(score, ignore_leaps=False)
            self.score_part = merge_parts(score.parts)
        except Exception as e:
            raise ValueError(f"Invalid score file: {e}")

        # Set tempo: user-provided > adjust_tempo (always 120) > score marking > default (120 BPM)
        if tempo is not None:
            self.tempo = float(tempo)
        elif auto_adjust_tempo:
            self.tempo = DEFAULT_TEMPO
        else:
            score_tempo = get_tempo_from_score(self.score_part, self.score_file)
            self.tempo = score_tempo if score_tempo is not None else DEFAULT_TEMPO

        # setup feature processor
        if self.feature_type is None:
            if input_type == "audio":
                if method == "audio_outerhmm":
                    # For audio_outerhmm with tone model, can use either raw_audio or cqt
                    # Using cqt is more efficient (CQT computed once, not twice)
                    self.feature_type = "cqt_spectral_flux"
                else:
                    self.feature_type = "chroma"
            else:
                self.feature_type = "pitch_ioi"

        if self.feature_type == "chroma":
            self.processor = ChromagramProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_type == "mfcc":
            self.processor = MFCCProcessor(
                sample_rate=self.sample_rate,
            )
        elif self.feature_type == "cqt":
            self.processor = CQTProcessor(
                sample_rate=self.sample_rate,
            )
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=self.sample_rate,
            )
        elif self.feature_type == "lse":
            self.processor = LogSpectralEnergyProcessor(
                sample_rate=self.sample_rate,
            )
        elif self.feature_type == "pitch_ioi":
            self.processor = PitchIOIProcessor(piano_range=self.config["piano_range"])
        elif self.feature_type == "pitchclass":
            self.processor = PitchClassPianoRollProcessor()
        elif self.feature_type == "pianoroll":
            self.processor = PianoRollProcessor(piano_range=self.config["piano_range"])
        elif self.feature_type == "cqt_spectral_flux":
            self.processor = CQTSpectralFluxProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
            if self.input_type == "audio" and method == "audio_outerhmm":
                self.processor = _LastFrameProcessor(self.processor)

        else:
            raise ValueError(f"Invalid feature type `{self.feature_type}`")

        if self.performance_file is not None:
            if self.input_type == "audio" and not is_audio_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected audio file, but got {self.performance_file}"
                )
            elif self.input_type == "midi" and not is_midi_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected MIDI file, but got {self.performance_file}"
                )

        # setup distance function
        if distance_func is None:
            distance_func = DEFAULT_DISTANCE_FUNCS[method]
        # setup stream device

        if self.input_type == "audio":
            self.stream = AudioStream(
                processor=self.processor,
                device_name_or_index=self.device_name_or_index,
                file_path=self.performance_file,
                wait=wait,
                target_sr=self.sample_rate,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.input_type == "midi":
            self.stream = MidiStream(
                processor=self.processor,
                port=self.device_name_or_index,
                file_path=self.performance_file,
                **({"polling_period": None} if method == "outerhmm" else {}),
            )
        else:
            raise ValueError(f"Invalid input type {self.input_type}")

        use_score_audio = self.input_type == "audio" and method in {"dixon", "arzt"}
        self.reference_features = self.preprocess_score(use_score_audio)

        if distance_func is None:
            distance_func = DEFAULT_DISTANCE_FUNCS[method]

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
                tempo_model=self.config["tempo_model"],
                has_insertions=True,
                piano_range=self.config["piano_range"],
            )
        elif method == "outerhmm" and self.input_type == "midi":
            self.score_follower = OuterProductHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )
        elif method == "pthmm" and self.input_type == "audio":
            self.score_follower = GaussianAudioPitchTempoHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )
        elif method == "audio_outerhmm" and self.input_type == "audio":
            self.score_follower = AudioOuterProductHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                tempo=self.tempo,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif method == "pthmm" and self.input_type == "midi":
            self.score_follower = PitchHMM(
                reference_features=self.reference_features,
                # observation_model=obs_model,
                queue=self.stream.queue,
                has_insertions=True,
                piano_range=self.config["piano_range"],
            )
        elif method == "outerhmm" and self.input_type == "midi":
            self.score_follower = OuterProductHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )
        else:
            raise ValueError("Invalid method")

    def preprocess_score(self, use_score_audio: bool = False):
        """Preprocess score to extract reference features."""
        if self.auto_adjust_tempo and self.performance_file is not None:
            self.tempo = adjust_tempo_for_performance_audio(
                self.score_part, self.performance_file, self.tempo
            )

        if use_score_audio:
            self.score_audio = generate_score_audio(
                self.score_part, self.tempo, self.sample_rate
            ).astype(np.float32)
            reference_features = self.processor(self.score_audio)
            self.reference_features = reference_features
            self.processor.reset()
        else:
            return self.score_part.note_array()

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

    def build_score_annotations(
        self,
        level="beat",
        musical_beat: bool = False,
        return_type: str = "beats",  # "beat" or "seconds"
    ):
        """
        Build score annotations in beat or second unit.

        Parameters
        ----------
        level : str
            Level of annotations to use: beat or note (chord onset level)
        musical_beat : bool
            Whether to use musical beat
        return_type : {"beat", "seconds"}
            Type of annotations to return: beat or seconds (time unit)

        Returns
        -------
        score_annots : np.ndarray
            Array of score annotations in beat or second unit
        """
        score_annots = []
        if level == "beat":
            if musical_beat:
                self.score_part.use_musical_beat()  # for asap dataset
            note_array = np.unique(self.score_part.note_array()["onset_beat"])
            start_beat = np.ceil(note_array.min())
            end_beat = np.floor(note_array.max())
            score_annots_in_beat = np.arange(start_beat, end_beat + 1)
        elif level == "note":
            snote_array = self.score_part.note_array()
            score_annots_in_beat = np.unique(snote_array["onset_beat"])
        else:
            raise ValueError(f"Invalid score annotation level: {level}")

        if return_type == "beats":
            return score_annots_in_beat
        elif return_type == "seconds":
            score_annots_in_seconds = [
                self.score_part.inv_beat_map(beat)
                / self.score_part.quarter_duration_map(
                    self.score_part.inv_beat_map(beat)
                )
                * (60 / self.tempo)
                for beat in score_annots_in_beat
            ]
            return np.array(score_annots_in_seconds)
        else:
            raise ValueError(f"Invalid return type: {return_type}")

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
        perf_annotations: Union[PathLike, np.ndarray],
        level: str = "note",
        tolerances: list = TOLERANCES_IN_MILLISECONDS,
        musical_beat: bool = False,  # beat annots are difference in some dataset
        debug: bool = False,
        save_dir: PathLike = None,
        run_name: str = None,
        domain: str = "performance",  # "score" or "performance"
    ) -> dict:
        """
        Evaluate the score following process

        Parameters
        ----------
        perf_annotations : PathLike or np.ndarray
            Path to the performance annotations file (tab-separated),
            or numpy array of annotation times in seconds.
        level : str
            Level of annotations to use: bar, beat or note
        tolerance : list
            Tolerances to use for evaluation (in milliseconds)
        debug : bool
            Whether to save the score and performance audio with beat annotations
        domain : str
            Evaluation domain, either "score" or "performance".
            "score" domain evaluates in beat unit, "performance" domain evaluates in second unit. (Default: "performance")

        Returns
        -------
        dict
            Evaluation results with mean, median, std, skewness, kurtosis, and
            accuracy for each tolerance
        """
        if not self._has_run:
            raise ValueError("Must call run() before evaluation")

        if isinstance(perf_annotations, np.ndarray):
            perf_annots = perf_annotations
        else:
            perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)

        return_type = "seconds" if domain == "performance" else "beats"
        score_annots = self.build_score_annotations(level, musical_beat, return_type)

        original_perf_annots_counts = len(perf_annots)

        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]

        mode = (
            "state"
            if (self.input_type == "midi" or self.method == "audio_outerhmm")
            else "frame"
        )
        perf_annots_predicted = transfer_from_score_to_predicted_perf(
            self.score_follower.warping_path,
            score_annots,
            frame_rate=self.frame_rate,
            mode=mode,
        )

        score_annots_predicted = transfer_from_perf_to_predicted_score(
            self.score_follower.warping_path,
            perf_annots,
            frame_rate=self.frame_rate,
            mode=mode,
        )
        score_annots = score_annots[: len(score_annots_predicted)]

        if original_perf_annots_counts != len(perf_annots_predicted):
            print(
                f"Length of the annotation changed: {original_perf_annots_counts} -> {len(perf_annots_predicted)}"
            )

        if self.input_type == "audio":
            if debug:
                save_debug_results(
                    self.score_file,
                    getattr(self, "score_audio", None),
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
            # plot_and_save_gt_vs_pred_points(
            #     perf_annots,
            #     perf_annots_predicted,
            #     save_dir,
            #     run_name,
            #     score_y=score_annots,
            #     frame_rate=self.frame_rate,
            #     x_unit="frames",
            # )

        if domain == "performance":
            eval_results = get_evaluation_results(
                perf_annots,
                perf_annots_predicted,
                total_counts=original_perf_annots_counts,
                tolerances=tolerances,
            )
        else:
            score_annots_predicted = self.convert_timestamps_to_beats(
                score_annots_predicted
            )
            if tolerances == TOLERANCES_IN_MILLISECONDS:
                tolerances = TOLERANCES_IN_BEATS  # switch to beats
            eval_results = get_evaluation_results(
                score_annots,
                score_annots_predicted,
                total_counts=original_perf_annots_counts,
                tolerances=tolerances,
                in_seconds=False,
            )
        if self.input_type == "audio":
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
            for current_position in self.score_follower.run(verbose=verbose):
                if self.input_type == "audio" and self.method != "audio_outerhmm":
                    position_in_beat = self._convert_frame_to_beat(current_position)
                    yield position_in_beat
                else:
                    yield float(self.score_follower.state_space[current_position])

        self._has_run = True
        return self.score_follower.warping_path
