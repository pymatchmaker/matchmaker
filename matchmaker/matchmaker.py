import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment
from partitura.score import Part, merge_parts

from matchmaker.dp import (
    OnlineTimeWarpingArztEvent,
    OnlineTimeWarpingArztFrame,
    OnlineTimeWarpingDixonEvent,
    OnlineTimeWarpingDixonFrame,
)
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
    OnsetPianoRollProcessor,
    PianoRollProcessor,
    PitchClassPianoRollProcessor,
    PitchIOIProcessor,
    onset_pianoroll,
)
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob import AudioOuterProductHMM, OuterProductHMM, PitchHMM, PitchIOIHMM
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MILLISECONDS,
    evaluate_alignment,
    transfer_positions,
)
from matchmaker.utils.misc import (
    adjust_tempo_for_performance_file,
    generate_score_audio,
    get_tempo_from_score,
    is_audio_file,
    is_midi_file,
    save_debug_results,
)
from matchmaker.utils.tempo_models import KalmanTempoModel

PathLike = Union[str, bytes, os.PathLike]
sys.setrecursionlimit(10_000)

DEFAULT_TEMPO = 120
MIDI_FRAME_RATE = 1  # dummy value for MIDI input
DEFAULT_PROCESSOR = {
    "audio": "chroma",
    "midi": "pitch_ioi",
}
DEFAULT_METHODS = {
    "audio": "arzt",
    "midi": "outerhmm",
}
AVAILABLE_METHODS = ["arzt", "dixon", "hmm", "pthmm", "outerhmm"]
OLTW_METHODS = {"arzt", "dixon"}
KWARGS = {
    "audio": {
        "dixon": {
            "processor": "lse",
            "window_size": 10,
        },
        "arzt": {
            "window_size": 10,
            "start_window_size": 0.1,
            "step_size": 3,
        },
        "outerhmm": {
            "processor": "cqt_spectral_flux",
            "sample_rate": 16000,
            "frame_rate": 25,
            "s_j": 0.0,
        },
    },
    "midi": {
        "arzt": {
            "processor": "pianoroll",
            "piano_range": True,
            "frame_rate": 100,
            "window_size": 2,
            "start_window_size": 2,
            "step_size": 5,
        },
        "dixon": {
            "processor": "pianoroll",
            "piano_range": True,
            "frame_rate": 100,
            "window_size": 0.3,
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
    processor : str
        Type of feature processor to use
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
        input_type: str = "audio",
        method: str = None,
        *,
        processor: str = None,
        device_name_or_index: Union[str, int] = None,
        tempo: Optional[float] = None,
        sample_rate: int = SAMPLE_RATE,
        frame_rate: int = FRAME_RATE,
        auto_adjust_tempo: bool = False,
        wait: bool = False,
        unfold_score=True,
        kwargs=KWARGS,
    ):
        self.score_file = str(score_file)
        self.performance_file = (
            str(performance_file) if performance_file is not None else None
        )

        self.input_type = input_type
        self.score_part: Optional[Part] = None
        self.device_name_or_index = device_name_or_index
        self.processor = None
        self.stream = None
        self.score_follower = None
        self.reference_features = None
        self._has_run = False
        self.alignment_duration = None

        # validate method first
        if method is None:
            method = DEFAULT_METHODS[self.input_type]
        elif method not in AVAILABLE_METHODS:
            raise ValueError(f"Invalid method. Available methods: {AVAILABLE_METHODS}")

        self.method = method
        self.config = dict(kwargs.get(self.input_type, {}).get(self.method, {}))
        self.auto_adjust_tempo = auto_adjust_tempo

        # Resolve sample_rate and frame_rate: config overrides defaults
        self.sample_rate = self.config.pop("sample_rate", sample_rate)
        self.frame_rate = self.config.pop(
            "frame_rate", frame_rate if input_type == "audio" else MIDI_FRAME_RATE
        )
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
                try:
                    # Ensure recursion limit is high enough for deepcopy of
                    # complex scores. External libraries (e.g. madmom) may
                    # lower it during processing.
                    _prev_limit = sys.getrecursionlimit()
                    sys.setrecursionlimit(max(_prev_limit, 10_000))
                    unfolded = partitura.score.unfold_part_maximal(
                        score, ignore_leaps=False
                    )
                    self.score_part = merge_parts(unfolded.parts)
                    sys.setrecursionlimit(_prev_limit)
                except Exception:
                    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
                    self.score_part = merge_parts(score.parts)
            else:
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

        processor_type = processor or self.config.pop(
            "processor", DEFAULT_PROCESSOR[input_type]
        )
        self.processor = self._build_processor(method, processor_type)

        if self.performance_file is not None:
            if self.input_type == "audio" and not is_audio_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected audio file, but got {self.performance_file}"
                )
            elif self.input_type == "midi" and not is_midi_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected MIDI file, but got {self.performance_file}"
                )

        self.stream = self._build_stream(method, wait)
        self.reference_features = self.preprocess_score()
        self.score_follower = self._build_score_follower(method)

    def _build_processor(self, method, processor_type):
        audio_kw = dict(sample_rate=self.sample_rate, hop_length=self.hop_length)

        AUDIO_PROCESSORS = {
            "chroma": lambda: ChromagramProcessor(**audio_kw),
            "mfcc": lambda: MFCCProcessor(**audio_kw),
            "cqt": lambda: CQTProcessor(**audio_kw),
            "mel": lambda: MelSpectrogramProcessor(**audio_kw),
            "lse": lambda: LogSpectralEnergyProcessor(**audio_kw),
            "cqt_spectral_flux": lambda: CQTSpectralFluxProcessor(**audio_kw),
        }
        MIDI_PROCESSORS = {
            "pitch_ioi": lambda: PitchIOIProcessor(
                piano_range=self.config["piano_range"],
                return_pitch_list=(method == "hmm"),
            ),
            "pitchclass": lambda: PitchClassPianoRollProcessor(),
            "pianoroll": lambda: PianoRollProcessor(
                piano_range=self.config["piano_range"],
            ),
        }

        if processor_type in AUDIO_PROCESSORS:
            return AUDIO_PROCESSORS[processor_type]()
        elif processor_type in MIDI_PROCESSORS:
            return MIDI_PROCESSORS[processor_type]()
        raise ValueError(f"Invalid feature type '{processor_type}'")

    def _build_stream(self, method, wait):
        if self.input_type == "audio":
            return AudioStream(
                processor=self.processor,
                device_name_or_index=self.device_name_or_index,
                file_path=self.performance_file,
                wait=wait,
                target_sr=self.sample_rate,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.input_type == "midi":
            if method in OLTW_METHODS:
                processor = OnsetPianoRollProcessor(
                    piano_range=self.config.get("piano_range", True),
                )
            else:
                processor = self.processor
            return MidiStream(
                processor=processor,
                port=self.device_name_or_index,
                file_path=self.performance_file,
                polling_period=None,
            )
        raise ValueError(f"Invalid input type '{self.input_type}'")

    def _build_score_follower(self, method):
        if self.input_type == "audio":
            return self._build_audio_follower(method)
        elif self.input_type == "midi":
            return self._build_symbolic_follower(method)
        raise ValueError(f"Invalid input_type '{self.input_type}'")

    def _build_audio_follower(self, method):
        ref = self.reference_features
        queue = self.stream.queue
        state_space = np.unique(self.score_part.note_array()["onset_beat"])

        if method in OLTW_METHODS:
            self.ppart = partitura.utils.music.performance_from_part(
                self.score_part, bpm=self.tempo
            )
            self.ppart.sustain_pedal_threshold = 127
            try:
                stm, rtm = self.get_time_maps()
            except Exception:
                stm, rtm = None, None
            cls = (
                OnlineTimeWarpingArztFrame
                if method == "arzt"
                else OnlineTimeWarpingDixonFrame
            )
            return cls(
                reference_features=ref,
                queue=queue,
                state_space=state_space,
                frame_rate=self.frame_rate,
                state_to_ref_time_map=stm,
                ref_to_state_time_map=rtm,
                ref_frame_to_beat=self._build_ref_frame_to_beat(),
                **self.config,
            )
        elif method == "outerhmm":
            return AudioOuterProductHMM(
                reference_features=ref,
                queue=queue,
                tempo=self.tempo,
                hop_length=self.hop_length,
                **self.config,
            )
        raise ValueError(f"No audio follower for method '{method}'")

    def _build_symbolic_follower(self, method):
        ref = self.reference_features
        queue = self.stream.queue

        if method in OLTW_METHODS:
            # Convert note_array to onset pianoroll for event-level OLTW
            onset_ref, state_space = onset_pianoroll(
                ref,
                onset_key="onset_beat",
                piano_range=self.config.get("piano_range", True),
            )
            # Filter out frame-level config keys
            skip = {
                "window_size",
                "start_window_size",
                "frame_rate",
                "processor",
                "piano_range",
            }
            config = {k: v for k, v in self.config.items() if k not in skip}
            cls = (
                OnlineTimeWarpingArztEvent
                if method == "arzt"
                else OnlineTimeWarpingDixonEvent
            )
            return cls(
                reference_features=onset_ref,
                queue=queue,
                state_space=state_space,
                **config,
            )
        elif method == "hmm":
            return PitchIOIHMM(
                reference_features=ref,
                queue=queue,
                has_insertions=True,
                **self.config,
            )
        elif method == "pthmm":
            return PitchHMM(
                reference_features=ref,
                queue=queue,
                has_insertions=True,
                **self.config,
            )
        elif method == "outerhmm":
            return OuterProductHMM(
                reference_features=ref,
                queue=queue,
                **self.config,
            )
        raise ValueError(f"No MIDI follower for method '{method}'")

    def _wp_perf_to_seconds(self, wp_perf):
        """Convert warping path performance axis to absolute seconds."""
        if self.input_type == "audio":
            return wp_perf / self.frame_rate
        elif self.method in OLTW_METHODS:
            return wp_perf  # OnsetPianoRollProcessor provides absolute timestamps
        else:
            # HMM: IOI-accumulated from 0; shift by first note onset
            _perf = partitura.load_performance_midi(self.performance_file)
            return wp_perf + float(_perf.note_array()["onset_sec"].min())

    def preprocess_score(self):
        """Extract reference features from the score."""
        if self.auto_adjust_tempo and self.performance_file is not None:
            self.tempo = adjust_tempo_for_performance_file(
                self.score_part, self.performance_file, self.tempo
            )

        if self.input_type == "audio" and self.method in OLTW_METHODS:
            score_audio = generate_score_audio(
                self.score_part, self.tempo, self.sample_rate
            ).astype(np.float32)
            features = self.processor(score_audio)
            self.processor.reset()
            return features

        return self.score_part.note_array()

    def get_time_maps(self):
        sna = self.score_part.note_array()
        pna = self.ppart.note_array()
        note_ids = sna["id"]
        # If note IDs are missing, use index-based IDs
        if len(set(note_ids)) <= 1:
            synth_ids = [f"n{i}" for i in range(len(sna))]
            sna = sna.copy()
            sna["id"] = synth_ids
            pna = pna.copy()
            pna["id"] = synth_ids[: len(pna)]
            note_ids = synth_ids
        alignment = [
            {"label": "match", "score_id": nid, "performance_id": nid}
            for nid in note_ids
        ]
        return get_time_maps_from_alignment(pna, sna, alignment)

    def _convert_frame_to_beat(self, current_frame: int) -> float:
        """Convert frame number to beat position in the score."""
        tick = get_ppq(self.score_part)
        timeline_time = (current_frame / self.frame_rate) * tick * (self.tempo / 60)
        beat_position = np.round(
            self.score_part.beat_map(timeline_time),
            decimals=2,
        )
        return beat_position

    def _build_ref_frame_to_beat(self) -> np.ndarray:
        """Precompute beat position for each reference feature frame."""
        n_ref = self.reference_features.shape[0]
        return np.array(
            [self._convert_frame_to_beat(i) for i in range(n_ref)],
        )

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
        tolerances: list = None,
        musical_beat: bool = False,
        debug: bool = False,
        save_dir: PathLike = None,
        run_name: str = None,
        domain: str = "score",
        plot_dist_matrix: bool = True,
    ) -> dict:
        """
        Evaluate the score following process.

        When domain="score" (default), returns beat-based metrics as primary
        and ms-based metrics under "ms" key. When domain="performance",
        returns ms-based metrics only (legacy behavior).

        Parameters
        ----------
        perf_annotations : PathLike or np.ndarray
            Path to the performance annotations file or numpy array of onset times (seconds).
        level : str
            Annotation level: "beat" or "note"
        tolerances : list or None
            Tolerances for evaluation. If None, uses default for the domain.
        musical_beat : bool
            Whether to use musical beat
        debug : bool
            Whether to save debug outputs
        domain : str
            "score" (default, beat-based primary) or "performance" (ms-based, legacy)

        Returns
        -------
        dict
            Evaluation results. If domain="score", includes both beat and ms metrics.
        """
        if tolerances is None:
            tolerances = (
                TOLERANCES_IN_BEATS if domain == "score" else TOLERANCES_IN_MILLISECONDS
            )
        if not self._has_run:
            raise ValueError("Must call run() before evaluation")

        if isinstance(perf_annotations, np.ndarray):
            perf_annots = perf_annotations
        else:
            perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)

        wp = self.score_follower.warping_path
        wp_score = wp[0].astype(float)
        wp_perf_sec = self._wp_perf_to_seconds(wp[1].astype(float))

        score_annots_beats = self.build_score_annotations(
            level, musical_beat, return_type="beats"
        )
        min_length = min(len(score_annots_beats), len(perf_annots))
        score_annots_beats = score_annots_beats[:min_length]
        perf_annots = perf_annots[:min_length]

        eval_results = evaluate_alignment(
            wp_score,
            wp_perf_sec,
            score_annots_beats,
            perf_annots,
            beat_tolerances=tolerances if domain == "score" else TOLERANCES_IN_BEATS,
            ms_tolerances=TOLERANCES_IN_MILLISECONDS,
        )

        # Real-Time Factor (domain-independent)
        if self.alignment_duration is not None:
            finite_perf = perf_annots[np.isfinite(perf_annots)]
            if len(finite_perf) > 0:
                perf_duration = float(np.max(finite_perf) - np.min(finite_perf))
                if perf_duration > 0:
                    eval_results["rtf"] = float(
                        f"{self.alignment_duration / perf_duration:.4f}"
                    )

        if self.input_type == "audio":
            latency_results = self.get_latency_stats()
            eval_results.update(latency_results)

        if debug and save_dir is not None:
            wp_sec = np.array([wp_score, wp_perf_sec])
            sf = self.score_follower
            save_debug_results(
                warping_path=wp_sec,
                score_annots=score_annots_beats,
                perf_annots=perf_annots,
                perf_annots_predicted=transfer_positions(
                    wp_sec,
                    score_annots_beats,
                    frame_rate=1,
                    domain="performance",
                ),
                eval_results=eval_results,
                frame_rate=self.frame_rate,
                save_dir=save_dir,
                run_name=run_name or "results",
                state_space=getattr(sf, "state_space", None),
                ref_features=(
                    getattr(sf, "reference_features", None)
                    if plot_dist_matrix
                    else None
                ),
                input_features=(
                    getattr(sf, "input_features", None) if plot_dist_matrix else None
                ),
                distance_func=(
                    getattr(sf, "distance_func", None) if plot_dist_matrix else None
                ),
                ref_frame_to_beat=getattr(sf, "_ref_frame_to_beat", None),
            )

        return eval_results

    def run(self, verbose: bool = True):
        """
        Run the score following process.

        Yields
        ------
        float
            Beat position in the score (interpolated)

        Returns
        -------
        np.ndarray
            Warping path (2, T)
        """
        with self.stream:
            self.stream.stream_start.wait()
            t0 = time.time()
            for current_position in self.score_follower.run(verbose=verbose):
                yield current_position
        self.alignment_duration = time.time() - t0

        self._has_run = True
        return self.score_follower.warping_path
