# -*- coding: utf-8 -*-
"""Construction of the ``"ensemble"`` method.

The ensemble is a *composite* method: it builds one follower per member and
drives them all from a single merged input stream. That does not fit the
one-processor-one-follower shape the rest of :mod:`matchmaker.registry`
declares, so the two steps that need Python live here and are reached from the
spec through a provider (``{from: ensemble_members}``) and a stream builder
(``stream: merged``).

Members are built by a full sub-``Matchmaker`` each, so every member goes
through exactly the same processor / reference / follower construction as it
would as a top-level method -- including methods added via ``methods.yaml`` or
``register_method``.
"""

from typing import TYPE_CHECKING, Any, List, Tuple

from matchmaker.features.audio import FRAME_RATE, SAMPLE_RATE
from matchmaker.io.queue import RECVQueue

from .follower import EnsembleMember
from .merged_stream import MergedStream, RawProcessor

if TYPE_CHECKING:  # pragma: no cover
    from matchmaker.matchmaker import Matchmaker

__all__ = ["build_members", "build_merged_stream", "midi_polling_period"]

MIDI_FRAME_RATE = 1  # dummy value for MIDI input


class _PlaceholderStream:
    """Minimal stand-in stream used while building ensemble members.

    Member followers are constructed with ``queue=stream.queue``; the ensemble
    drives them directly via a :class:`MergedStream` instead of their own
    stream, so this placeholder only needs to expose a queue (no device or file
    is opened).
    """

    def __init__(self) -> None:
        self.queue = RECVQueue()


def _audio_framing(mm: "Matchmaker") -> Tuple[int, int, float]:
    """``(sample_rate, hop_length, frame_rate)`` shared by all audio members.

    Every audio member reads the same capture, so they must agree on framing.
    """
    audio_cfg = dict(mm.config.get("audio", {}))
    sample_rate = int(audio_cfg.get("sample_rate", SAMPLE_RATE))
    if audio_cfg.get("hop_length") is not None:
        hop_length = int(audio_cfg["hop_length"])
        frame_rate = sample_rate / hop_length
    else:
        frame_rate = audio_cfg.get("frame_rate", FRAME_RATE)
        hop_length = int(sample_rate // frame_rate)
    return sample_rate, hop_length, frame_rate


def _performance_files(mm: "Matchmaker") -> Tuple[Any, Any]:
    """``(audio_file, midi_file)`` to simulate from, per modality."""
    audio = mm.config.get("audio_performance_file") or (
        mm.performance_file if mm.input_type == "audio" else None
    )
    midi = mm.config.get("midi_performance_file") or (
        mm.performance_file if mm.input_type == "midi" else None
    )
    return audio, midi


def build_members(mm: "Matchmaker") -> List[EnsembleMember]:
    """Build the ensemble's members from ``mm.config['members']``.

    Each entry is ``{"method": ..., "input_type": ..., "processor": ...,
    "name": ..., "kwargs": {...}}``; only ``method`` is required. A member's
    ``kwargs`` are merged over that method's ``default_kwargs`` rather than
    replacing them (the top-level ``Matchmaker`` semantics), so listing a
    method several times with different ``kwargs`` gives several differently
    configured members. Repeated names are suffixed ``_2``, ``_3``, ...
    """
    from matchmaker.matchmaker import DEFAULT_KWARGS, Matchmaker

    members_spec = mm.config.get("members")
    if not members_spec:
        raise ValueError(
            "method='ensemble' requires kwargs['members'] (a non-empty list "
            "of {'method': ..., 'input_type': ...} dicts)."
        )

    sample_rate, hop_length, _ = _audio_framing(mm)
    audio_perf, midi_perf = _performance_files(mm)

    members: List[EnsembleMember] = []
    used_names = set()
    for spec in members_spec:
        method = spec["method"]
        modality = spec.get("input_type", mm.input_type)
        if modality not in ("audio", "midi"):
            raise ValueError(
                f"ensemble member modality must be 'audio' or 'midi', "
                f"got '{modality}'"
            )

        member_kwargs = dict(DEFAULT_KWARGS[modality].get(method, {}))
        member_kwargs.update(spec.get("kwargs", {}))
        if modality == "audio":
            member_kwargs["sample_rate"] = sample_rate
            member_kwargs["hop_length"] = hop_length

        sub = Matchmaker(
            score_file=mm.score_file,
            performance_file=audio_perf if modality == "audio" else midi_perf,
            input_type=modality,
            method=method,
            processor=spec.get("processor"),
            tempo=mm.tempo,
            unfold_score=mm.unfold_score,
            kwargs=member_kwargs,
            stream=_PlaceholderStream(),
        )

        name = spec.get("name", method)
        duplicate = 1
        while name in used_names:
            duplicate += 1
            name = f"{method}_{duplicate}"
        used_names.add(name)

        members.append(
            EnsembleMember(
                name=name,
                follower=sub.score_follower,
                processor=sub.processor,
                modality=modality,
                polling_period=(
                    sub.polling_period if modality == "midi" else None
                ),
            )
        )
    return members


def midi_polling_period(mm: "Matchmaker", members: List[EnsembleMember]):
    """The frame window the merged MIDI stream runs at.

    Every MIDI member reads the same capture, so the stream runs at the finest
    period any of them asked for (``None`` -- one message per frame -- being
    the finest of all). Finer than requested costs a member nothing: extra
    frames carry no note-ons and its processor returns ``None`` for them.
    Coarser merges messages a member needed apart, and
    ``ParangonarProcessor`` refuses such a frame outright.

    ``kwargs['midi_polling_period']`` overrides the derivation, including with
    ``None`` to force event-based framing. An override that would starve a
    member is refused here rather than at the first chord of the performance.
    """
    periods = [m.polling_period for m in members if m.modality == "midi"]
    event_based = [
        m.name for m in members if m.modality == "midi" and m.polling_period is None
    ]
    if "midi_polling_period" in mm.config:
        override = mm.config["midi_polling_period"]
        if override is not None and event_based:
            raise ValueError(
                f"midi_polling_period={override!r} frames several MIDI messages "
                f"together, but member(s) {event_based} consume one message at a "
                "time. Leave it unset (the members decide) or pass None."
            )
        return override
    if not periods or event_based:
        return None
    return min(periods)


def build_merged_stream(mm: "Matchmaker", wait: bool) -> MergedStream:
    """One raw stream per modality the members use, merged into one queue.

    Also fixes ``mm.frame_rate``, which depends on which modalities are
    actually present rather than on the ensemble's own ``input_type``.
    """
    members = ensemble_members(mm)
    modalities = {member.modality for member in members}

    sample_rate, hop_length, audio_frame_rate = _audio_framing(mm)
    devices = mm.config.get("device", {})
    audio_perf, midi_perf = _performance_files(mm)

    children: List[Tuple[str, Any]] = []
    if "audio" in modalities:
        from matchmaker.io.audio import AudioStream

        # The raw capture must be wide enough for the hungriest member.
        n_fft = max(
            [2 * hop_length]
            + [
                int(getattr(member.processor, "n_fft", 2 * hop_length))
                for member in members
                if member.modality == "audio"
            ]
        )
        children.append(
            (
                "audio",
                AudioStream(
                    processor=RawProcessor(n_fft=n_fft),
                    device_name_or_index=devices.get("audio", mm.device_name_or_index),
                    file_path=audio_perf,
                    wait=wait,
                    target_sr=sample_rate,
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                ),
            )
        )
    if "midi" in modalities:
        from matchmaker.io.midi import MidiStream

        children.append(
            (
                "midi",
                MidiStream(
                    processor=RawProcessor(),
                    port=devices.get("midi", mm.device_name_or_index),
                    file_path=midi_perf,
                    polling_period=midi_polling_period(mm, members),
                ),
            )
        )

    mm.frame_rate = audio_frame_rate if "audio" in modalities else MIDI_FRAME_RATE
    return MergedStream(children)


def ensemble_members(mm: "Matchmaker") -> List[EnsembleMember]:
    """``build_members`` memoised on the Matchmaker.

    The stream builder needs the members to know which modalities to capture,
    and the follower needs the same objects afterwards; both go through here.
    """
    cached = getattr(mm, "_ensemble_members_cache", None)
    if cached is None:
        cached = build_members(mm)
        mm._ensemble_members_cache = cached
    return cached
