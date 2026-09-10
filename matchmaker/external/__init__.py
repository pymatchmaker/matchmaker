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

from collections import deque
from typing import Generator

import numpy as np
import parangonar as pa
import partitura as pt

from matchmaker.base import OnlineAlignment
from matchmaker.features.processor import Processor
from matchmaker.utils.typing import InputMIDIFrame
from typing import Optional, Tuple


_OLTW_MATCHERS = {"SL_OLTW", "SLT_OLTW"}
_TRANSFORMER_MATCHERS = {"OTM", "OPTM"}

#: How many recent notes ``confidence()`` averages its per-note score over.
CONFIDENCE_WINDOW = 8
#: Width of the "tied with the winner" band in an OLTW cost window, as a
#: fraction of the gap between the best and the average cost in that window.
#: Relative rather than absolute because the two OLTW cost metrics live on
#: different scales: plain OLTW's is a pitch-set distance taking two values,
#: the tempo-augmented one adds a continuous time term that no two cells ever
#: hit exactly alike.
COST_TIE_TOL = 0.02


def _pitch_sets_by_onset(note_array: np.ndarray, onsets: np.ndarray) -> list:
    """Set of MIDI pitches sounding at each unique score onset.

    ``onsets`` must be the sorted unique ``onset_beat`` values of
    ``note_array`` (i.e. the follower's score positions), so that entry ``i``
    describes state ``i``.
    """
    sets = [set() for _ in range(len(onsets))]
    index_of = np.searchsorted(onsets, note_array["onset_beat"])
    for i, pitch in zip(index_of, note_array["pitch"]):
        sets[int(i)].add(int(pitch))
    return sets


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
    confidence_window : int
        Number of recent notes ``confidence()`` averages its score over.

    Notes
    -----
    Every parangonar matcher indexes the same states as this follower: the
    sorted unique score onsets. So its return value is ``current_index``
    directly, and ``set_position`` can write a state index straight back into
    the matcher.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        method: str,
        queue=None,
        confidence_window: int = CONFIDENCE_WINDOW,
        **kwargs
    ):
        if method not in _OLTW_MATCHERS | _TRANSFORMER_MATCHERS:
            raise ValueError(f"Unknown parangonar method: {method}")
        score_note_array = _ensure_unique_ids(reference_features, prefix="s")
        unique_onsets = np.unique(score_note_array["onset_beat"])
        super().__init__(
            reference_features=reference_features,
            score_positions=unique_onsets.astype(np.float32),
            queue=queue,
        )
        self.method = method
        self.score_note_array = score_note_array
        self.matcher = self._build_matcher(method, self.score_note_array, **kwargs)
        self._pitches_at_onset = _pitch_sets_by_onset(score_note_array, unique_onsets)
        self._recent_scores = deque(maxlen=max(1, int(confidence_window)))

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
        # The matchers predict a state index within a window around their own
        # position, which can reach past the last state near the end of the
        # score; clamp so ``score_positions[current_index]`` stays valid.
        index = int(self.matcher(performance_note))
        self.current_index = max(0, min(index, len(self.score_positions) - 1))
        self._recent_scores.append(self._score_step(performance_note))

    def _score_step(self, performance_note) -> float:
        """How well-determined this note left the tracker's position, in [0, 1].

        Two different signals, because the two matcher families make their
        decision differently and neither signal is meaningful for the other:

        * OLTW: how unique the winning cell of the DP window was. Asking
          instead whether the played pitch is written at the position would be
          circular -- the position *is* where that pitch matched, so it scores
          1.0 even when the tracker is lost (measured: identical means for
          tracking and lost frames).
        * transformer: whether the played pitch is written where the network
          put the position. Here the position comes from the network rather
          than from pitch matching, so the check carries information.
        """
        if self.method in _OLTW_MATCHERS:
            return self._cost_uniqueness()
        return float(self._is_hit(performance_note))

    def _cost_uniqueness(self) -> float:
        """``1 / (1 + ties)`` over the DP cells that share the winning cost.

        After a step, the finite cells of the tracker's rolling cost column are
        exactly the window it just searched. Many cells tied with the best one
        means many score positions explain the input equally well, which is
        what being lost looks like from inside the DP.
        """
        tracker = getattr(self.matcher, "tracker", None)
        column = getattr(tracker, "global_cost_matrix", None)
        if column is None:
            return 1.0
        costs = np.asarray(column[:, 0], dtype=float)
        costs = costs[np.isfinite(costs)]
        if len(costs) == 0:
            return 1.0
        best = costs.min()
        spread = float(costs.mean() - best)
        if spread <= 0:  # every cell equally good: no information at all
            return 1.0 / len(costs)
        ties = int(np.count_nonzero(costs <= best + COST_TIE_TOL * spread)) - 1
        return 1.0 / (1.0 + ties)

    def _is_hit(self, performance_note) -> bool:
        """Is the played pitch written at the tracker's current position?"""
        pitch = int(performance_note["pitch"])
        return pitch in self._pitches_at_onset[self.current_index]

    def confidence(self) -> Optional[float]:
        """Mean of :meth:`_score_step` over the recent notes.

        Not calibrated -- like the other followers' confidences it is a
        monotone score, meant for ranking members against each other within one
        ensemble step. ``None`` until the first note has been consumed.
        """
        if not self._recent_scores:
            return None
        return float(np.mean(self._recent_scores))

    def set_position(self, beat: float, strength: float = 1.0) -> bool:
        """Re-anchor the matcher at ``beat`` (hard snap).

        None of the matchers carries a belief to blend, so ``strength`` only
        gates whether the snap happens. This is the only way back for the OLTW
        matchers in particular: their position is monotonic, so a tracker that
        has run ahead cannot recover on its own.
        """
        if strength <= 0 or self.score_positions is None:
            return False
        index = self._snap_index(beat)
        moved = (
            self._reanchor_oltw(index)
            if self.method in _OLTW_MATCHERS
            else self._reanchor_transformer(index)
        )
        if not moved:
            return False
        self.current_index = index
        # The recent scores describe the position we just left.
        self._recent_scores.clear()
        return True

    def _reanchor_oltw(self, index: int) -> bool:
        """Move the (T)OLTW playhead and reseed its rolling cost column.

        The trackers keep two columns of accumulated cost and search a window
        centred on ``current_position``. Clearing the previous column and
        seeding it at ``index`` leaves the next step exactly one finite path to
        continue from, which recentres the window on the corrected state.
        """
        tracker = getattr(self.matcher, "tracker", None)
        cost = getattr(tracker, "global_cost_matrix", None)
        lengths = getattr(tracker, "global_path_length_matrix", None)
        if cost is None or lengths is None:
            return False
        tracker.current_position = index
        cost[:] = np.inf
        cost[index + 1, 0] = 0.0
        lengths[:] = 0.0
        lengths[index + 1, 0] = 1.0
        return True

    def _reanchor_transformer(self, index: int) -> bool:
        """Move the transformer matcher's playhead to ``index``.

        Its prediction window is taken around ``current_position``, so writing
        that (plus the onset OTM greedily matches against) is the whole
        correction. The counters that track how long it has been stuck are
        reset with it: they describe the position being left behind.
        """
        matcher = self.matcher
        if not hasattr(matcher, "current_position"):
            return False
        matcher.current_position = index
        if hasattr(matcher, "_prev_score_onset"):
            matcher._prev_score_onset = float(self.score_positions[index])
        for counter in ("stuck_with_no_options", "time_since_nn_update"):
            if hasattr(matcher, counter):
                setattr(matcher, counter, 0)
        if hasattr(matcher, "reset_buffers"):
            # OPTM's "lostness" buffers, which trigger its own jump-back.
            matcher.reset_buffers()
        return True


class ParangonarProcessor(Processor):
    """Aggregate ``note_on`` events in a MIDI frame into a minimal 
    partitura note array row.

    All concurrent ``note_on`` events present in the input frame are merged
    into a single observation (i.e. a frame containing a chord emits one
    observation covering all chord pitches). Frame-level time grouping is
    the ``MidiStream``'s responsibility via ``polling_period``; this
    processor itself is stateless.


    Returns
    -------
    None if the frame has no note_on events. Otherwise a tuple
    ``(pitch_obs, f_time)``
    """

    def __init__(
        self
        ) -> None:
        super().__init__()
        self.id = 0

    def __call__(
        self,
        frame: InputMIDIFrame,
    ) -> Optional[Tuple[np.ndarray, float]]:
        data, f_time = frame
        pitch_obs = []
        
        pitches = []
        note_times = []

        for msg, m_time in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitches.append(msg.note)
                note_times.append(m_time)
        

        if len(pitches) > 1:
               raise ValueError("ParangonarProcessor requires event-based MIDI input with polling_period=None")
                
        if len(pitches) > 0:
            
            fields = [
                        ("onset_sec", "f4"),
                        ("pitch", "i4"),
                        ("id", "U256"),
                    ]
            note_array = []
            for i in range(len(pitches)):
                note_array.append((note_times[i], pitches[i], "n"+str(self.id)))
                self.id += 1

            arr_slice = np.array(note_array, dtype=fields)

            obs_time = min(note_times)

            return arr_slice[0], obs_time
        else:
            return None

    def reset(self) -> None:
        self.id = 0
        