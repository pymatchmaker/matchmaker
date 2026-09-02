import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import partitura
from partitura.io.exportmidi import get_ppq
from partitura.score import Part, merge_parts

from matchmaker.features.audio import FRAME_RATE, SAMPLE_RATE
from matchmaker.io.midi import POLLING_PERIOD
from matchmaker.registry import REGISTRY
from matchmaker.utils.misc import (
    get_tempo_from_score,
    is_audio_file,
    is_midi_file,
)

PathLike = Union[str, bytes, os.PathLike]
sys.setrecursionlimit(10_000)

DEFAULT_TEMPO = 120
MIDI_FRAME_RATE = 1  # dummy value for MIDI input

#: Which methods and processors exist, and how each is constructed, is declared
#: in ``matchmaker/methods.yaml`` and interpreted by :mod:`matchmaker.registry`.
#: The tables below are views onto that spec, kept for backwards compatibility;
#: ``AVAILABLE_METHODS`` and ``DEFAULT_KWARGS`` are the live dicts that
#: :func:`register_method` extends.
AVAILABLE_METHODS = REGISTRY.available_methods
DEFAULT_KWARGS = REGISTRY.default_kwargs
DEFAULT_METHOD = REGISTRY.default_method
DEFAULT_PROCESSOR = REGISTRY.default_processor
OLTW_METHODS = REGISTRY.family("oltw")
PARANGONAR_METHODS = REGISTRY.family("parangonar")

#: Score followers registered at runtime by :func:`register_method`, keyed by
#: ``(input_type, name)``. These are built by the same ``Matchmaker`` pipeline
#: as the methods above; only the construction step differs.
CUSTOM_METHODS = {}


def register_method(
    name: str,
    *,
    input_type: str,
    build_follower,
    build_processor=None,
    build_reference=None,
    default_kwargs: Optional[dict] = None,
    overwrite: bool = False,
) -> None:
    """Register a score follower so ``Matchmaker(method=name)`` can build it.

    This is the supported way to plug a follower that lives outside this
    package into the Matchmaker pipeline. A registered method is built by the
    same code as a built-in one — same score loading, same stream, same
    ``alignment_path`` — so it also works with anything downstream that takes a
    ``Matchmaker``, such as the benchmark's evaluation.

    A follower that lives *inside* the package is better declared in
    ``matchmaker/methods.yaml`` instead: it needs no Python builder at all
    unless its constructor arguments fall outside the spec's vocabulary
    (see :mod:`matchmaker.registry`).

    Parameters
    ----------
    name : str
        Method name, as passed to ``Matchmaker(method=...)``. Must not collide
        with an existing method for the same ``input_type``.
    input_type : {"audio", "midi"}
        Which stream the follower consumes.
    build_follower : callable
        ``build_follower(mm) -> OnlineAlignment``. Called once per
        ``Matchmaker``, after the stream and reference features exist. Read
        what you need off ``mm``: ``mm.score_part``, ``mm.tempo``,
        ``mm.reference_features``, ``mm.frame_rate``, ``mm.config``, and
        ``mm.stream.queue`` (pass that as the follower's ``queue``).
    build_processor : callable, optional
        ``build_processor(mm) -> Processor``. Omit to use the standard
        processor named by ``default_kwargs["processor"]`` (or the default for
        this input type), which is usually what you want.
    build_reference : callable, optional
        ``build_reference(mm) -> Any``, the score-side features. Omit for the
        score note array. Audio followers that align against a synthesised
        score rendering override this.
    default_kwargs : dict, optional
        Defaults for ``Matchmaker(kwargs=...)``, exactly like a method's
        ``default_kwargs`` in ``matchmaker/methods.yaml``: ``processor``,
        ``sample_rate``, ``frame_rate`` / ``hop_length`` for audio,
        ``polling_period`` for MIDI, plus anything your follower reads from
        ``mm.config``.
    overwrite : bool, optional
        Allow replacing an already-registered method of the same name.

    Examples
    --------
    >>> from matchmaker import Matchmaker, register_method
    >>> from matchmaker.base import OnlineAlignment
    >>> class MarchForward(OnlineAlignment):
    ...     def step(self, features):
    ...         self.current_index += 1
    >>> register_method(
    ...     "march-forward",
    ...     input_type="midi",
    ...     build_follower=lambda mm: MarchForward(
    ...         reference_features=mm.reference_features,
    ...         score_positions=np.unique(
    ...             mm.score_part.note_array()["onset_beat"]
    ...         ),
    ...         queue=mm.stream.queue,
    ...     ),
    ... )
    """
    if input_type not in AVAILABLE_METHODS:
        raise ValueError(
            f"Invalid input_type '{input_type}'. Available: {sorted(AVAILABLE_METHODS)}"
        )
    if not callable(build_follower):
        raise TypeError("build_follower must be callable.")
    for label, hook in (
        ("build_processor", build_processor),
        ("build_reference", build_reference),
    ):
        if hook is not None and not callable(hook):
            raise TypeError(f"{label} must be callable or None.")

    key = (input_type, name)
    if not overwrite:
        if key in CUSTOM_METHODS:
            raise ValueError(
                f"Method '{name}' is already registered for {input_type}. "
                "Pass overwrite=True to replace it."
            )
        if name in AVAILABLE_METHODS[input_type]:
            raise ValueError(
                f"'{name}' is a built-in {input_type} method and cannot be "
                "replaced by registration."
            )

    CUSTOM_METHODS[key] = {
        "build_follower": build_follower,
        "build_processor": build_processor,
        "build_reference": build_reference,
    }
    if name not in AVAILABLE_METHODS[input_type]:
        AVAILABLE_METHODS[input_type].append(name)
    if default_kwargs:
        DEFAULT_KWARGS[input_type][name] = dict(default_kwargs)


def unregister_method(name: str, input_type: str) -> None:
    """Undo a :func:`register_method`. Mainly for tests."""
    CUSTOM_METHODS.pop((input_type, name), None)
    if name in AVAILABLE_METHODS.get(input_type, []):
        AVAILABLE_METHODS[input_type].remove(name)
    DEFAULT_KWARGS.get(input_type, {}).pop(name, None)


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
        Method-specific configuration dict. If None, uses the method's
        ``default_kwargs`` from the spec (``DEFAULT_KWARGS[input_type][method]``).
        Anything the keys below do not claim is passed on to the follower's
        constructor, so the accepted keys are ultimately the follower's own —
        see ``matchmaker/methods.yaml`` for what each method declares.

        **audio keys**

        - ``processor`` (str): Feature type. Default: ``"chroma"``.
          Choices: the entries under ``processors.audio`` in the spec.
        - ``sample_rate`` (int): Sample rate in Hz. Default: 44100.
        - ``frame_rate`` (int): Frames per second. Default: 30.
          Ignored if ``hop_length`` is set.
        - ``hop_length`` (int): Hop length in samples. Overrides ``frame_rate``.
        - ``norm`` (float or None): LSE per-frame norm. Default: 2.

        **midi keys**

        - ``processor`` (str): Feature type. Default: ``"pitch"``.
          Choices: the entries under ``processors.midi`` in the spec.
        - ``piano_range`` (bool): Restrict pitch to 88-key piano range
          (MIDI 21-108). Default: True.
        - ``polling_period`` (float or None): Window size in seconds for
          frame-based MIDI accumulation. ``None`` = event-based (one note
          per frame). When set, all note-ons within each window are emitted
          as one chord observation.

    Notes
    -----
    ``Matchmaker`` is a convenience class for the common case of running a
    registered method (one of ``AVAILABLE_METHODS``). Built-in methods are
    declared in ``matchmaker/methods.yaml``; followers living outside this
    package are added with :func:`register_method`. For full control —
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
            # Methods flagged ``event_based`` in the spec consume one MIDI
            # message per frame; everything else defaults to MidiStream's
            # POLLING_PERIOD (0.01s windowed).
            spec = REGISTRY.methods["midi"].get(method)
            default_polling = (
                None if spec is not None and spec.event_based else POLLING_PERIOD
            )
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

        # ``processor`` always leaves the config: it configures Matchmaker and
        # must not leak into a follower that takes ``**config``.
        configured_processor = self.config.pop(
            "processor", DEFAULT_PROCESSOR[self.input_type]
        )
        processor_type = processor or configured_processor
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

    def _custom_spec(self, method):
        """The registration for ``method``, or None if it is a built-in."""
        return CUSTOM_METHODS.get((self.input_type, method))

    def _build_processor(self, method, processor_type):
        """The feature processor for ``processor_type``, per the spec."""
        spec = self._custom_spec(method)
        if spec is not None and spec["build_processor"] is not None:
            return spec["build_processor"](self)
        return REGISTRY.build_processor(self, processor_type)

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
        """The score follower for ``method``, per the spec."""
        spec = self._custom_spec(method)
        if spec is not None:
            return spec["build_follower"](self)
        return REGISTRY.build_follower(self, method)

    def _wp_perf_to_seconds(self, wp_perf):
        """Convert alignment path performance axis to absolute seconds.

        alignment_path[0] already holds absolute perf seconds, so this is a pass-through.
        """
        return wp_perf

    def preprocess_score(self):
        """Extract reference features from the score.

        Which strategy is used comes from the method's ``reference`` key in the
        spec — the score note array unless the method says otherwise.
        """
        spec = self._custom_spec(self.method)
        if spec is not None:
            if spec["build_reference"] is not None:
                return spec["build_reference"](self)
            return self.score_part.note_array()
        return REGISTRY.build_reference(self, self.method)

    def _convert_frame_to_beat(self, current_frame: int) -> float:
        """Convert frame number to beat position in the score."""
        tick = get_ppq(self.score_part)
        timeline_time = (current_frame / self.frame_rate) * tick * (self.tempo / 60)
        return float(self.score_part.beat_map(timeline_time))

    @property
    def score_positions(self) -> np.ndarray:
        """Ascending score beat of every note onset — the follower's states."""
        return np.unique(self.score_part.note_array()["onset_beat"])

    def ref_frame_to_beat(self) -> np.ndarray:
        """Score beat position of each reference *frame*.

        Only meaningful when ``reference_features`` is a frame array, i.e. for
        audio followers aligning against a synthesised score rendering.
        """
        return self._build_ref_frame_to_beat()

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
            Alignment path (2, T): row 0 perf time (sec), row 1 score beat.
        """
        with self.stream:
            self.stream.stream_start.wait()
            t0 = time.time()
            for current_position in self.score_follower.run(verbose=verbose):
                yield current_position
        self.alignment_duration = time.time() - t0

        self._has_run = True
        return self.score_follower._alignment_path
