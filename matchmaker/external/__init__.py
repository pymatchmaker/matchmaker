#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Wrapper around parangonar (https://github.com/sildater/parangonar) online
note matchers so they can be plugged into the Matchmaker pipeline.

Supported method keys:
  - "SL_OLTW"  : parangonar.OLTWMatcher  (symbolic-level OLTW)
  - "SLT_OLTW" : parangonar.TOLTWMatcher (symbolic-level tempo OLTW)
  - "OTM"      : parangonar.OnlineTransformerMatcher
  - "OPTM"     : parangonar.OnlinePureTransformerMatcher
"""

from typing import Generator

import numpy as np
import parangonar as pa
import partitura as pt

from matchmaker.base import OnlineAlignment

_OLTW_MATCHERS = {"SL_OLTW", "SLT_OLTW"}
_TRANSFORMER_MATCHERS = {"OTM", "OPTM"}


def _ensure_unique_ids(note_array: np.ndarray, prefix: str) -> np.ndarray:
    """Guarantee unique string ids in the note array, copying if needed."""
    ids = note_array["id"]
    if len(set(ids)) == len(ids) and all(bool(i) for i in ids):
        return note_array
    out = note_array.copy()
    out["id"] = np.array([f"{prefix}{i}" for i in range(len(out))])
    return out


class OnlineParangonarAlignment(OnlineAlignment):
    """
    Adapter that exposes a parangonar online matcher through the
    `OnlineAlignment` interface Matchmaker expects.

    Parameters
    ----------
    reference_features : np.ndarray
        Score note array (structured, with `onset_beat`, `pitch`, `id`,
        `is_grace`).
    performance_file : str
        Path to the performance MIDI file. The adapter reads it directly
        rather than consuming the MidiStream queue, because parangonar
        matchers operate on full note rows.
    method : str
        One of {"SL_OLTW", "SLT_OLTW", "OTM", "OPTM"}.
    queue : RECVQueue or None
        The MidiStream queue. Drained but not used; kept so Matchmaker's
        stream lifecycle stays intact.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        performance_file: str,
        method: str,
        queue=None,
    ):
        if method not in _OLTW_MATCHERS | _TRANSFORMER_MATCHERS:
            raise ValueError(f"Unknown parangonar method: {method}")
        score_note_array = _ensure_unique_ids(reference_features, prefix="s")
        score_positions = np.unique(score_note_array["onset_beat"]).astype(np.float32)
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.method = method
        self.performance_file = performance_file
        self.score_note_array = score_note_array
        self.matcher = self._build_matcher(method, self.score_note_array)

    @staticmethod
    def _build_matcher(method: str, sna: np.ndarray):
        if method == "SLT_OLTW":
            return pa.TOLTWMatcher(sna)
        if method == "SL_OLTW":
            return pa.OLTWMatcher(sna)
        if method == "OTM":
            return pa.OnlineTransformerMatcher(sna)
        if method == "OPTM":
            return pa.OnlinePureTransformerMatcher(sna)
        raise ValueError(method)

    def step(self, performance_note) -> None:
        self.matcher.online(performance_note)
        s_onset = float(self.matcher._prev_score_onset)
        idx = int(np.searchsorted(self.score_positions, s_onset))
        self.current_index = max(0, min(idx, len(self.score_positions) - 1))

    def _load_performance_note_array(self) -> np.ndarray:
        perf = pt.load_performance_midi(self.performance_file)
        pna = perf.note_array()
        return _ensure_unique_ids(pna, prefix="p")

    def run(self, verbose: bool = True) -> Generator[float, None, np.ndarray]:
        pna = self._load_performance_note_array()

        # OLTW-based matchers
        if self.method in _OLTW_MATCHERS:
            tracking_path = self.matcher(pna)
            score_idx = np.asarray(tracking_path[0], dtype=int)
            perf_idx = np.asarray(tracking_path[1], dtype=int)
            score_beats = self.score_positions[score_idx]
            perf_secs = pna["onset_sec"][perf_idx]
            for beat, perf_t in zip(score_beats, perf_secs):
                self.current_index = int(
                    np.searchsorted(self.score_positions, float(beat))
                )
                self.current_index = max(
                    0, min(self.current_index, len(self.score_positions) - 1)
                )
                self.current_position = float(beat)
                self.current_perf_time = float(perf_t)
                self._alignment_path.append(
                    (self.current_perf_time, self.current_position)
                )
                yield self.current_position
            return self.alignment_path

        # Transformer-based matchers
        self.matcher.prepare_performance(float(pna[0]["onset_sec"]))
        for note in pna:
            yield self(note, float(note["onset_sec"]))
        return self.alignment_path
