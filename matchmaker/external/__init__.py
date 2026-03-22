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

from typing import Generator, Optional

import numpy as np
import parangonar as pa
import partitura as pt

from matchmaker.base import OnlineAlignment

_BATCH_MATCHERS = {"SL_OLTW", "SLT_OLTW"}
_INCREMENTAL_MATCHERS = {"OTM", "OPTM"}


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
        super().__init__(reference_features=reference_features)
        if method not in _BATCH_MATCHERS | _INCREMENTAL_MATCHERS:
            raise ValueError(f"Unknown parangonar method: {method}")
        self.method = method
        self.performance_file = performance_file
        self.queue = queue

        self.score_note_array = _ensure_unique_ids(reference_features, prefix="s")
        self.matcher = self._build_matcher(method, self.score_note_array)

        self.warping_path: Optional[np.ndarray] = None
        self.input_features = None
        self.state_space = np.unique(self.score_note_array["onset_beat"])

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

    def __call__(self, performance_note):
        return float(performance_note["onset_sec"])

    def _load_performance_note_array(self) -> np.ndarray:
        perf = pt.load_performance_midi(self.performance_file)
        pna = perf.note_array()
        return _ensure_unique_ids(pna, prefix="p")

    def run(self, verbose: bool = True) -> Generator[float, None, None]:
        pna = self._load_performance_note_array()
        unique_onsets = np.unique(self.score_note_array["onset_beat"])

        if self.method in _BATCH_MATCHERS:
            tracking_path = self.matcher(pna)
            score_idx = np.asarray(tracking_path[0], dtype=int)
            perf_idx = np.asarray(tracking_path[1], dtype=int)
            score_beats = unique_onsets[score_idx]
            perf_secs = pna["onset_sec"][perf_idx]
            for beat in score_beats:
                yield float(beat)
            self.warping_path = np.vstack(
                [score_beats.astype(float), perf_secs.astype(float)]
            )
            return

        self.matcher.prepare_performance(float(pna[0]["onset_sec"]))
        score_beats_list = []
        perf_secs_list = []
        for note in pna:
            self.matcher.online(note)
            s_onset = float(self.matcher._prev_score_onset)
            p_onset = float(note["onset_sec"])
            score_beats_list.append(s_onset)
            perf_secs_list.append(p_onset)
            yield s_onset
        self.warping_path = np.vstack(
            [
                np.asarray(score_beats_list, dtype=float),
                np.asarray(perf_secs_list, dtype=float),
            ]
        )
