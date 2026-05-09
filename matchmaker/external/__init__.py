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
        **kwargs
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
        self.matcher = self._build_matcher(method, self.score_note_array, **kwargs)
        self._private_score_position = self.score_positions[0]

    @staticmethod
    def _build_matcher(method: str, sna: np.ndarray, **kwargs):
        if method == "SLT_OLTW":
            return pa.TOLTWMatcher(sna, tracker_type=method, **kwargs) 
        if method == "SL_OLTW":
            return pa.OLTWMatcher(sna,tracker_type=method, **kwargs)
        if method == "OTM":
            return pa.OnlineTransformerMatcher(sna, **kwargs)
        if method == "OPTM":
            return pa.OnlinePureTransformerMatcher(sna, **kwargs)
        raise ValueError(method)

    def step(self, performance_note) -> None:
        self._private_score_position = float(self.matcher(performance_note))

    def get_current_position(self):
	    return self._private_score_position