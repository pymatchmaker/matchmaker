import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
import scipy.interpolate
from partitura.io.exportmidi import get_ppq
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment
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
)
from matchmaker.prob.outer_product_hmm import OuterProductHMM
from matchmaker.prob.outer_product_hmm_audio import AudioOuterProductHMM
from matchmaker.utils.eval import (
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MILLISECONDS,
    get_evaluation_results,
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
DEFAULT_METHODS = {
    "audio": "arzt",
    "midi": "outerhmm",
}
AVAILABLE_METHODS = ["arzt", "dixon", "hmm", "pthmm", "outerhmm", "audio_outerhmm"]
KWARGS = {
    "audio": {
        "dixon": {
            "feature_type": "lse",
            "window_size": 10,
        },
        "arzt": {
            "window_size": 10,
            "start_window_size": 0.1,
            "step_size": 3,
        },
        "audio_outerhmm": {
            "feature_type": "cqt_spectral_flux",
            "sample_rate": 16000,
            "frame_rate": 25,
            "s_j": 0.0,
        },
    },
    "midi": {
        "arzt": {
            "processor": "pianoroll",
            "piano_range": True,
            "window_size": 200,
            "start_window_size": 200,
            "step_size": 5,
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
        input_type: str = "audio",
        method: str = None,
        *,
        feature_type: str = None,
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
        self.feature_type = feature_type
        self.frame_rate = frame_rate if input_type == "audio" else 1
        self.sample_rate = sample_rate
        self.hop_length = sample_rate // self.frame_rate
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
        self.config = dict(kwargs[self.input_type][self.method])
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

        # setup feature processor
        if self.feature_type is None:
            default = "chroma" if input_type == "audio" else "pitch_ioi"
            self.feature_type = self.config.get("feature_type", default)

        if self.feature_type == "chroma":
            self.processor = ChromagramProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_type == "mfcc":
            self.processor = MFCCProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_type == "cqt":
            self.processor = CQTProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
            )
        elif self.feature_type == "lse":
            self.processor = LogSpectralEnergyProcessor(
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
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

        self.reference_features = self.preprocess_score()

        if method == "arzt":
            try:
                state_to_ref_time_map, ref_to_state_time_map = self.get_time_maps()
            except Exception:
                state_to_ref_time_map, ref_to_state_time_map = None, None
            self.score_follower = OnlineTimeWarpingArzt(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                frame_rate=self.frame_rate,
                state_to_ref_time_map=state_to_ref_time_map,
                ref_to_state_time_map=ref_to_state_time_map,
                state_space=np.unique(self.score_part.note_array()["onset_beat"]),
                ref_frame_to_beat=self._build_ref_frame_to_beat(),
                **self.config,
            )
        elif method == "dixon":
            try:
                state_to_ref_time_map, ref_to_state_time_map = self.get_time_maps()
            except Exception:
                state_to_ref_time_map, ref_to_state_time_map = None, None
            self.score_follower = OnlineTimeWarpingDixon(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                frame_rate=self.frame_rate,
                state_to_ref_time_map=state_to_ref_time_map,
                ref_to_state_time_map=ref_to_state_time_map,
                state_space=np.unique(self.score_part.note_array()["onset_beat"]),
                ref_frame_to_beat=self._build_ref_frame_to_beat(),
                **self.config,
            )
        elif method == "hmm" and self.input_type == "midi":
            self.score_follower = PitchIOIHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                has_insertions=True,
                **self.config,
            )
        elif method == "pthmm" and self.input_type == "audio":
            self.score_follower = GaussianAudioPitchTempoHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                **self.config,
            )
        elif method == "audio_outerhmm" and self.input_type == "audio":
            self.score_follower = AudioOuterProductHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                tempo=self.tempo,
                hop_length=self.hop_length,
                **self.config,
            )
        elif method == "pthmm" and self.input_type == "midi":
            self.score_follower = PitchHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                has_insertions=True,
                **self.config,
            )
        elif method == "outerhmm" and self.input_type == "midi":
            self.score_follower = OuterProductHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                **self.config,
            )
        else:
            raise ValueError("Invalid method")

    def preprocess_score(self):
        """Preprocess score to extract reference features."""
        if self.auto_adjust_tempo and self.performance_file is not None:
            self.tempo = adjust_tempo_for_performance_file(
                self.score_part, self.performance_file, self.tempo
            )

        if self.method in {"arzt", "dixon"}:
            self.ppart = partitura.utils.music.performance_from_part(
                self.score_part, bpm=self.tempo
            )
            self.ppart.sustain_pedal_threshold = 127
            if self.input_type == "audio":
                self.score_audio = generate_score_audio(
                    self.score_part, self.tempo, self.sample_rate
                ).astype(np.float32)
                reference_features = self.processor(self.score_audio)
                self.processor.reset()
                return reference_features
            else:
                polling_period = 0.01
                reference_features = (
                    partitura.utils.music.compute_pianoroll(
                        note_info=self.ppart,
                        time_unit="sec",
                        time_div=int(np.round(1 / polling_period)),
                        binary=True,
                        piano_range=self.config["piano_range"],
                    )
                    .toarray()
                    .T
                ).astype(np.float32)
                return reference_features
        else:
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

        return_type = "seconds" if domain == "performance" else "beats"
        score_annots = self.build_score_annotations(level, musical_beat, return_type)

        original_perf_annots_counts = len(perf_annots)

        # min_length = min(len(score_annots), len(perf_annots))
        # score_annots = score_annots[:min_length]
        # perf_annots = perf_annots[:min_length]

        wp = self.score_follower.warping_path
        score_annots_beats = self.build_score_annotations(
            level, musical_beat, return_type="beats"
        )

        # --- Per-frame evaluation ---
        # Build GT interpolator: score beat → perf time (seconds)
        valid_gt = np.isfinite(perf_annots)
        gt_interp = scipy.interpolate.interp1d(
            score_annots_beats[valid_gt],
            perf_annots[valid_gt],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        wp_score = wp[0].astype(float)
        wp_perf = wp[1].astype(float)

        # Convert wp perf axis to seconds
        if self.input_type == "midi":
            # MIDI: wp_perf is IOI-accumulated from 0; shift by first note onset
            _perf = partitura.load_performance_midi(self.performance_file)
            midi_offset = float(_perf.note_array()["onset_sec"].min())
            wp_perf_sec = wp_perf + midi_offset
        else:
            # Audio: wp_perf is frame index
            wp_perf_sec = wp_perf / self.frame_rate

        # For each wp entry: GT perf time for predicted beat vs actual perf time
        gt_perf_times = gt_interp(wp_score)
        perf_annots_predicted = transfer_positions(
            wp,
            score_annots_beats,
            frame_rate=self.frame_rate,
            domain="performance",
        )

        if domain == "performance":
            eval_results = get_evaluation_results(
                gt_perf_times,
                wp_perf_sec,
                total_counts=len(wp_score),
                tolerances=tolerances,
            )
        else:
            # Score domain: beat-based (primary) + ms-based (secondary)
            score_annots_predicted = transfer_positions(
                wp, perf_annots, frame_rate=self.frame_rate, domain="score"
            )
            score_annots = score_annots[: len(score_annots_predicted)]
            beat_tolerances = (
                tolerances
                if tolerances != TOLERANCES_IN_MILLISECONDS
                else TOLERANCES_IN_BEATS
            )
            beat_results = get_evaluation_results(
                score_annots,
                score_annots_predicted,
                total_counts=original_perf_annots_counts,
                tolerances=beat_tolerances,
                in_seconds=False,
            )
            ms_results = get_evaluation_results(
                gt_perf_times,
                wp_perf_sec,
                total_counts=len(wp_score),
                tolerances=TOLERANCES_IN_MILLISECONDS,
            )
            eval_results = {"beat": beat_results, "ms": ms_results}

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

        # Debug: save warping path TSV, results JSON, and plots
        if debug and save_dir is not None:
            # For plot y-axis: use beats when wp[0] is in beats
            debug_score_annots = score_annots_beats
            save_debug_results(
                warping_path=self.score_follower.warping_path,
                score_annots=debug_score_annots,
                perf_annots=perf_annots,
                perf_annots_predicted=perf_annots_predicted,
                eval_results=eval_results,
                frame_rate=self.frame_rate,
                save_dir=save_dir,
                run_name=run_name or "results",
                state_space=getattr(self.score_follower, "state_space", None),
                ref_features=(
                    getattr(self.score_follower, "reference_features", None)
                    if plot_dist_matrix
                    else None
                ),
                input_features=(
                    getattr(self.score_follower, "input_features", None)
                    if plot_dist_matrix
                    else None
                ),
                distance_func=(
                    getattr(self.score_follower, "distance_func", None)
                    if plot_dist_matrix
                    else None
                ),
                ref_frame_to_beat=getattr(
                    self.score_follower, "_ref_frame_to_beat", None
                ),
            )

        return eval_results

    def run(self, verbose: bool = True, wait: bool = True):
        """
        Run the score following process.

        Measures wall-clock time as ``alignment_duration`` (seconds),
        which covers both feature extraction (producer thread) and
        score following inference (main thread) running concurrently.
        RTF is computed as ``alignment_duration / performance_duration``.

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
            self.stream.stream_start.wait()
            t0 = time.time()
            for current_position in self.score_follower.run(verbose=verbose):
                yield current_position
        self.alignment_duration = time.time() - t0

        self._has_run = True
        return self.score_follower.warping_path
