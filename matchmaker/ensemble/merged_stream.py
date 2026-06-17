# -*- coding: utf-8 -*-
"""Raw fan-in plumbing for :class:`~matchmaker.ensemble.follower.EnsembleFollower`.

The ensemble feeds the *same* input to every member, but members use different
feature processors (chroma vs. pitch vs. chord-onset ...), so a single shared
feature stream is not possible. Instead each underlying ``Stream`` runs a
:class:`RawProcessor` (a passthrough) and the ensemble applies each member's own
processor internally.

:class:`MergedStream` wraps one or two such raw streams (at most one audio + one
MIDI), tags each frame with its modality, and merges them into a single queue.
This both supplies the raw frames and naturally reconciles the different stepping
rates of the members: each member advances only when its modality produces a
frame.
"""
from __future__ import annotations

import threading
from queue import Empty
from threading import Thread
from typing import Any, Dict, List, Tuple

from matchmaker.features.processor import Processor
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END


class RawProcessor(Processor):
    """Passthrough processor: forwards the stream frame unchanged.

    The underlying ``Stream`` calls ``processor((frame, frame_time))`` and
    forwards the return value to its queue, so returning the tuple as-is makes
    the queue carry raw ``(frame, frame_time)`` items for the ensemble to
    process per member.

    Parameters
    ----------
    n_fft : int or None
        Advertised FFT size. ``AudioStream`` sizes its overlap cache as
        ``n_fft - hop_length``; set this to the widest ``n_fft`` across the
        audio members so every member's processor gets enough context.
    """

    def __init__(self, n_fft: int = None) -> None:
        if n_fft is not None:
            self.n_fft = n_fft

    def __call__(self, data: Any, **kwargs) -> Any:
        return data

    def reset(self) -> None:
        pass


class MergedStream:
    """Merge one or more raw streams into a single modality-tagged queue.

    Each queue item is ``((modality, raw_frame), perf_time)`` so it unpacks
    directly through ``OnlineAlignment.__call__(observation, perf_time)`` with
    ``observation = (modality, raw_frame)``. A single ``STREAM_END`` is emitted
    only after *all* child streams have ended.

    Parameters
    ----------
    children : list of (str, Stream)
        ``(modality, stream)`` pairs. ``modality`` is ``"audio"`` or ``"midi"``;
        each ``stream`` must already be built with a :class:`RawProcessor`.
    """

    def __init__(self, children: List[Tuple[str, Any]]) -> None:
        if not children:
            raise ValueError("MergedStream requires at least one child stream.")
        self.children = children
        self.queue = RECVQueue()
        self.stream_start = threading.Event()
        self._forwarders: List[Thread] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._remaining = len(children)

    @property
    def latency_stats(self) -> Dict[str, float]:
        """Aggregate child stream latency stats (summed)."""
        agg = {
            "total_latency": 0.0,
            "total_frames": 0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
        }
        for _, st in self.children:
            stats = getattr(st, "latency_stats", None)
            if not stats:
                continue
            agg["total_latency"] += stats.get("total_latency", 0.0)
            agg["total_frames"] += stats.get("total_frames", 0)
            agg["max_latency"] = max(agg["max_latency"], stats.get("max_latency", 0.0))
            agg["min_latency"] = min(agg["min_latency"], stats.get("min_latency", float("inf")))
        return agg

    def _forward(self, modality: str, stream: Any) -> None:
        while not self._stop.is_set():
            try:
                item = stream.queue.get(timeout=0.5)
            except Empty:
                continue
            if item is STREAM_END:
                with self._lock:
                    self._remaining -= 1
                    if self._remaining == 0:
                        self.queue.put(STREAM_END)
                return
            if item is None:
                continue
            raw, perf_time = item
            self.queue.put(((modality, raw), perf_time))
            if not self.stream_start.is_set():
                self.stream_start.set()

    def __enter__(self) -> "MergedStream":
        for _, stream in self.children:
            stream.__enter__()
        for modality, stream in self.children:
            t = Thread(target=self._forward, args=(modality, stream), daemon=True)
            t.start()
            self._forwarders.append(t)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        for _, stream in self.children:
            try:
                stream.__exit__(exc_type, exc_value, traceback)
            except Exception:
                pass
        for t in self._forwarders:
            t.join(timeout=1.0)
        return False
