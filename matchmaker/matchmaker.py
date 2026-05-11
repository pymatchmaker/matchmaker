import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.score import Part, merge_parts
from partitura.utils.music import performance_notearray_from_score_notearray

from matchmaker.dp import (
    OnlineTimeWarpingArztEvent,
    OnlineTimeWarpingArztFrame,
    OnlineTimeWarpingDixonEvent,
    OnlineTimeWarpingDixonFrame,
)
from matchmaker.external import OnlineParangonarAlignment
from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    ChromagramProcessor,
    CQTProcessor,
    CQTSpectralFluxProcessor,
    LogSpectralEnergyProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    RawSpectrumProcessor,
)
from matchmaker.features.midi import (
    OnsetOnlyPianoRollProcessor,
    PianoRollProcessor,
    PitchClassPianoRollProcessor,
    PitchProcessor,
    onset_pianoroll,
)
from matchmaker.io.midi import POLLING_PERIOD
from matchmaker.prob import AudioOuterProductHMM, OuterProductHMM, PitchHMM, PitchIOIHMM
from matchmaker.prob.particle_filter import ParticleFilter
from matchmaker.prob.skf import SwitchingKalmanFilterFollower
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MILLISECONDS,
    evaluate_alignment,
    transfer_positions,
)
from matchmaker.utils.misc import (
    generate_score_audio,
    get_tempo_from_score,
    is_audio_file,
    is_midi_file,
    save_debug_results,
)
from matchmaker.utils.symbolic import framed_midi_messages_from_performance
from matchmaker.utils.tempo_models import KalmanTempoModel

PathLike = Union[str, bytes, os.PathLike]
sys.setrecursionlimit(10_000)

DEFAULT_TEMPO = 120
MIDI_FRAME_RATE = 1  # dummy value for MIDI input
OLTW_METHODS = {"arzt", "dixon"}
PARANGONAR_METHODS = {"SLT_OLTW", "SL_OLTW", "OTM", "OPTM"}
AVAILABLE_METHODS = {
    "audio": sorted(OLTW_METHODS) + ["outerhmm", "skf", "pf"],
    "midi": sorted(OLTW_METHODS)
    + ["hmm", "pthmm", "outerhmm", "pf"]
    + sorted(PARANGONAR_METHODS),
}
DEFAULT_METHOD = {"audio": "arzt", "midi": "pthmm"}
DEFAULT_PROCESSOR = {"audio": "chroma", "midi": "pitch"}
DEFAULT_KWARGS = {
    "audio": {
        "arzt": {"window_size": 10, "start_window_size": 0.1, "step_size": 3},
        "dixon": {"processor": "lse", "window_size": 10},
        "outerhmm": {
            "processor": "cqt_spectral_flux",
            "sample_rate": 16000,
            "frame_rate": 25,
            "s_j": 0.0,
        },
        "skf": {
            "processor": "raw_spectrum",
            "sample_rate": 8000,
            "hop_length": 128,
            "n_fft": 512,
        },
        "pf": {
            "processor": "chroma",
            "sample_rate": int(SAMPLE_RATE / 4),
            "hop_length": int(SAMPLE_RATE / 100),
            "n_fft": int(
                int(SAMPLE_RATE / 4) / 21.533203125
            ),  # converts to closest power of 2 for a 46ms window (as proposed by Duan et al.) for default sample rates.
            "frame_rate": 100,
            "num_particles": 1000,
        },
    },
    "midi": {
        "arzt": {
            "processor": "onset_only_pianoroll",
            "piano_range": True,
            "window_size": 2,
            "start_window_size": 2,
            "step_size": 5,
        },
        "dixon": {
            "processor": "onset_only_pianoroll",
            "piano_range": True,
            "window_size": 0.3,
        },
        "hmm": {
            "processor": "pitch",
            "tempo_model": KalmanTempoModel,
            "piano_range": True,
        },
        "pf": {
            "processor": "pitchclass",
            "piano_range": True,
            "num_particles": 1000,
        },
        "pthmm": {"processor": "pitch", "piano_range": True},
        "outerhmm": {"processor": "pitch", "piano_range": True},
        "SLT_OLTW": {"processor": "pitch", "piano_range": True},
        "SL_OLTW": {"processor": "pitch", "piano_range": True},
        "OTM": {"processor": "pitch", "piano_range": True},
        "OPTM": {"processor": "pitch", "piano_range": True},
    },
}


class Matchmaker(object):
    """
    A class to perform online score following with I/O support for audio and MIDI

    Parameters
    ----------
    score_file : Union[str, bytes, os.PathLike]
        Path to the score file.
    performance_file : Union[str, bytes, os.PathLike, None]
        Path to the performance file. If None, live input is used.
    input_type : str
        Type of input to use: ``"audio"`` or ``"midi"``.
    method : str
        Score following method to use.
    stream : Stream, optional
        Custom input stream (e.g. ``AudioStream`` / ``MidiStream`` or a
        user subclass). If None, one is built from ``method`` defaults.
    processor : str, optional
        Registered processor name (looked up in the built-in processor
        registry, overrides ``kwargs["processor"]``). If None, defaults
        are used based on ``method``.
    device_name_or_index : Union[str, int]
        Name or index of the audio/MIDI device. Ignored if ``performance_file``
        is given.
    tempo : float, optional
        Tempo in BPM. If None, reads from score; if score has no tempo marking,
        defaults to 120 BPM.
    wait : bool (default: False)
        Offline mode only. If True, simulates real-time playback speed.
    unfold_score : bool (default: True)
        If True, unfolds score repeats maximally before processing.
    kwargs : dict, optional
        Method-specific configuration dict. If None, uses built-in defaults
        for the given ``input_type`` and ``method``.

        **audio keys**

        - ``processor`` (str): Feature type. Default: ``"chroma"``.
          Choices: ``"chroma"``, ``"mfcc"``, ``"cqt"``, ``"mel"``,
          ``"lse"``, ``"cqt_spectral_flux"``, ``"raw_spectrum"``.
        - ``sample_rate`` (int): Sample rate in Hz. Default: 22050.
        - ``frame_rate`` (int): Frames per second. Default: 50.
          Ignored if ``hop_length`` is set.
        - ``hop_length`` (int): Hop length in samples. Overrides ``frame_rate``.

        **midi keys**

        - ``processor`` (str): Feature type. Default: ``"pitch"``.
          Choices: ``"pitch"``, ``"pianoroll"``, ``"onset_only_pianoroll"``,
          ``"pitchclass"``.
        - ``piano_range`` (bool): Restrict pitch to 88-key piano range
          (MIDI 21-108). Default: True.
        - ``polling_period`` (float or None): Window size in seconds for
          frame-based MIDI accumulation. ``None`` = event-based (one note
          per frame). When set, all note-ons within each window are emitted
          as one chord observation.

    Notes
    -----
    ``Matchmaker`` is a convenience class for the common case of running a
    registered method (one of ``AVAILABLE_METHODS``). For full control —
    e.g. a novel score follower, a custom stream, or audio-to-audio
    alignment without a score — compose ``Stream`` + ``Processor`` +
    ``OnlineAlignment`` directly. See ``HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md``.
    """

    def __init__(
        self,
        score_file: PathLike,
        performance_file: Union[PathLike, None] = None,
        input_type: str = "audio",
        method: str = None,
        *,
        stream=None,
        processor: str = None,
        device_name_or_index: Union[str, int] = None,
        tempo: Optional[float] = None,
        wait: bool = False,
        unfold_score=True,
        kwargs=None,
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

        # validate method
        if method is None:
            method = DEFAULT_METHOD[self.input_type]
        elif method not in AVAILABLE_METHODS.get(self.input_type, []):
            raise ValueError(
                f"Invalid method '{method}' for {input_type}. "
                f"Available: {AVAILABLE_METHODS.get(self.input_type, [])}"
            )

        self.method = method
        self.config = dict(
            kwargs
            if kwargs is not None
            else DEFAULT_KWARGS[self.input_type].get(self.method, {})
        )

        if input_type == "midi":
            # outerhmm uses event-based (single-message) mode; everything else
            # defaults to MidiStream's POLLING_PERIOD (0.01s windowed).
            default_polling = None if method == "outerhmm" else POLLING_PERIOD
            self.polling_period = self.config.pop("polling_period", default_polling)
            self.frame_rate = MIDI_FRAME_RATE
        else:
            # Audio: hop_length (if given) is primary; else derive from frame_rate.
            self.sample_rate = self.config.pop("sample_rate", SAMPLE_RATE)
            hop_length_cfg = self.config.pop("hop_length", None)
            if hop_length_cfg is not None:
                self.hop_length = int(hop_length_cfg)
                self.frame_rate = self.sample_rate / self.hop_length
                self.config.pop("frame_rate", None)
            else:
                self.frame_rate = self.config.pop("frame_rate", FRAME_RATE)
                self.hop_length = int(self.sample_rate // self.frame_rate)

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

        # Set tempo: user-provided > score marking > default (120 BPM)
        if tempo is not None:
            self.tempo = float(tempo)
        else:
            score_tempo = get_tempo_from_score(self.score_part, self.score_file)
            self.tempo = score_tempo if score_tempo is not None else DEFAULT_TEMPO

        processor_type = processor or self.config.pop(
            "processor", DEFAULT_PROCESSOR[self.input_type]
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

        self.stream = stream if stream is not None else self._build_stream(method, wait)
        self.reference_features = self.preprocess_score()
        self.score_follower = self._build_score_follower(method)

    def _build_processor(self, method, processor_type):
        if self.input_type == "audio":
            audio_kw = dict(sample_rate=self.sample_rate, hop_length=self.hop_length)
            if method == "pf":
                audio_kw["n_fft"] = self.config.get("n_fft", 1024)
            AUDIO_PROCESSORS = {
                "chroma": lambda: ChromagramProcessor(**audio_kw),
                "mfcc": lambda: MFCCProcessor(**audio_kw),
                "cqt": lambda: CQTProcessor(**audio_kw),
                "mel": lambda: MelSpectrogramProcessor(**audio_kw),
                "lse": lambda: LogSpectralEnergyProcessor(**audio_kw),
                "cqt_spectral_flux": lambda: CQTSpectralFluxProcessor(**audio_kw),
                "raw_spectrum": lambda: RawSpectrumProcessor(
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    n_fft=self.config.get("n_fft", 512),
                ),
            }
            if processor_type in AUDIO_PROCESSORS:
                return AUDIO_PROCESSORS[processor_type]()
            raise ValueError(f"Invalid feature type '{processor_type}'")

        # All MIDI processors are stateless aggregators over their input frame.
        # Time-based grouping (e.g., chords) is the stream's job: set
        # ``polling_period`` on ``MidiStream`` to bin events. Cross-frame
        # chord-merging, if needed, should be inside the tracker class.
        MIDI_PROCESSORS = {
            "pitch": lambda: PitchProcessor(
                piano_range=self.config["piano_range"],
                return_pitch_list=(method == "hmm"),
            ),
            "pitchclass": lambda: PitchClassPianoRollProcessor(),
            "pianoroll": lambda: PianoRollProcessor(
                piano_range=self.config["piano_range"],
            ),
            "onset_only_pianoroll": lambda: OnsetOnlyPianoRollProcessor(
                piano_range=self.config.get("piano_range", True),
            ),
        }
        if processor_type in MIDI_PROCESSORS:
            return MIDI_PROCESSORS[processor_type]()
        raise ValueError(f"Invalid feature type '{processor_type}'")

    def _build_stream(self, method, wait):
        try:
            if self.input_type == "audio":
                from matchmaker.io.audio import AudioStream

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
                from matchmaker.io.midi import MidiStream

                return MidiStream(
                    processor=self.processor,
                    port=self.device_name_or_index,
                    file_path=self.performance_file,
                    polling_period=self.polling_period,
                )
        except ImportError as e:
            raise ImportError(
                f"{e}. To use local audio/MIDI devices, "
                "install with: pip install pymatchmaker[devices]"
            ) from e
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
        score_positions = np.unique(self.score_part.note_array()["onset_beat"])

        if method in OLTW_METHODS:
            cls = (
                OnlineTimeWarpingArztFrame
                if method == "arzt"
                else OnlineTimeWarpingDixonFrame
            )
            return cls(
                reference_features=ref,
                score_positions=score_positions,
                queue=queue,
                frame_rate=self.frame_rate,
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
        elif method == "skf":
            return SwitchingKalmanFilterFollower(
                reference_features=self.score_part.note_array(),
                queue=queue,
                tempo=self.tempo,
                sample_rate=self.sample_rate,
                n_fft=self.config.get("n_fft", 512),
                hop_length=self.hop_length,
            )
        elif method == "pf":
            return ParticleFilter(
                reference_features=ref,
                score_positions=score_positions,
                score_boundaries=self._get_score_onsets_and_offsets_in_beats(
                    self.score_part, score_positions
                ),
                notated_tempo=self.tempo,
                hop_size=self.hop_length / self.sample_rate,
                queue=queue,
                num_particles=self.config.get("num_particles", 1000),
            )
        raise ValueError(f"No audio follower for method '{method}'")

    def _build_symbolic_follower(self, method):
        ref = self.reference_features
        queue = self.stream.queue

        if method in OLTW_METHODS:
            # Convert note_array to onset pianoroll for event-level OLTW
            onset_ref, score_positions = onset_pianoroll(
                ref,
                onset_key="onset_beat",
                piano_range=self.config.get("piano_range", True),
            )
            # Filter out frame-level config keys
            skip = {
                "window_size",
                "start_window_size",
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
                score_positions=score_positions,
                queue=queue,
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
        elif method == "pf":
            return ParticleFilter(
                reference_features=ref,
                score_positions=np.unique(self.score_part.note_array()["onset_beat"]),
                score_boundaries=self._get_score_onsets_and_offsets_in_beats(
                    self.score_part,
                    np.unique(self.score_part.note_array()["onset_beat"]),
                ),
                notated_tempo=self.tempo,
                hop_size=POLLING_PERIOD,
                queue=queue,
                num_particles=self.config.get("num_particles", 1000),
            )
        elif method in PARANGONAR_METHODS:
            sna = self.score_part.note_array(include_grace_notes=True)
            return OnlineParangonarAlignment(
                reference_features=sna,
                performance_file=self.performance_file,
                method=method,
                queue=queue,
            )
        raise ValueError(f"No MIDI follower for method '{method}'")

    def _wp_perf_to_seconds(self, wp_perf):
        """Convert alignment path performance axis to absolute seconds.

        All trackers now store absolute perf time in alignment_path[1].
        """
        return wp_perf

    def preprocess_score(self):
        """Extract reference features from the score."""
        if self.input_type == "audio" and self.method in sorted(OLTW_METHODS) + ["pf"]:
            score_audio = generate_score_audio(
                self.score_part, self.tempo, self.sample_rate
            ).astype(np.float32)
            features, _ = self.processor((score_audio, 0.0))
            self.processor.reset()
            return features

        if self.input_type == "midi" and self.method == "pf":
            performed_notearray = performance_notearray_from_score_notearray(
                self.score_part.note_array(),
                bpm=self.tempo,
            )

            frames_array, frame_times = framed_midi_messages_from_performance(
                performed_notearray,
                polling_period=self.polling_period,
            )
            score_pitchclass_pianoroll_processor = PitchClassPianoRollProcessor()
            features = []
            for frame, frame_time in zip(frames_array, frame_times):
                feat, _ = score_pitchclass_pianoroll_processor((frame, frame_time))
                features.append(feat)

            features = np.array(features)
            return features

        return self.score_part.note_array()

    def _convert_frame_to_beat(self, current_frame: int) -> float:
        """Convert frame number to beat position in the score."""
        tick = get_ppq(self.score_part)
        timeline_time = (current_frame / self.frame_rate) * tick * (self.tempo / 60)
        return float(self.score_part.beat_map(timeline_time))

    def _build_ref_frame_to_beat(self) -> np.ndarray:
        """Precompute beat position for each reference feature frame."""
        n_ref = self.reference_features.shape[0]
        return np.array(
            [self._convert_frame_to_beat(i) for i in range(n_ref)],
        )

    def _get_score_onsets_and_offsets_in_beats(
        self, score_part: Part, score_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the beat positions of note onsets and offsets in the score.

        Parameters
        ----------
        score_part : Part
            Partitura Part object representing the score

        Returns
        -------
        np.ndarray
            Array of beat positions corresponding to note onsets and offsets
        """
        note_array = score_part.note_array()
        # add a column for offsets, which is onset_beat + duration_beat
        note_array = np.lib.recfunctions.append_fields(
            note_array,
            "offset_beat",
            note_array["onset_beat"] + note_array["duration_beat"],
            usemask=False,
        )
        score_boundaries = np.unique(
            np.concatenate((note_array["onset_beat"], note_array["offset_beat"]))
        )

        # for every entry in score_boundaries, find the highest beat in state_space that is smaller than or equal to it,
        # and replace the entry with that beat (to ensure boundaries are aligned with score frames)
        score_boundaries = np.array(
            [
                score_positions[
                    np.searchsorted(score_positions, boundary, side="right") - 1
                ]
                for boundary in score_boundaries
            ]
        )
        return score_boundaries

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
        make_plot: bool = True,
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

        wp = self.score_follower.alignment_path
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
                alignment_path=wp_sec,
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
                score_positions=sf.score_positions,
                ref_features=sf.reference_features if plot_dist_matrix else None,
                input_features=(
                    getattr(sf, "input_features", None) if plot_dist_matrix else None
                ),
                distance_func=getattr(sf, "distance_func", None),
                ref_frame_to_beat=getattr(sf, "_ref_frame_to_beat", None),
                make_plot=make_plot,
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
            Alignment path (2, T): row 0 score beat, row 1 perf time (sec).
        """
        with self.stream:
            self.stream.stream_start.wait()
            t0 = time.time()
            for current_position in self.score_follower.run(verbose=verbose):
                yield current_position
        self.alignment_duration = time.time() - t0

        self._has_run = True
        return self.score_follower._alignment_path
