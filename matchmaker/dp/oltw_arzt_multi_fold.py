#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Parallel Multi-Fold Online Time Warping (OLTWArztMultiFold)

A baseline for folded scores with repeats. The score is unfolded into a few versions
(every repeat taken, the folded score itself, and the first four versions partitura's
``iter_unfolded_parts`` yields, duplicates removed); one frame-level Arzt OLTW matcher
(``OnlineTimeWarpingArztFrame``) runs on each version from the start, and the output, in
notated (folded) beats, is the matcher whose local matching cost averaged over the last
``selection_window`` frames is lowest, switching only when another is lower by
``switch_margin``.

The idea of running several OLTW matchers in parallel and comparing their alignment costs
is adapted from
  A. Arzt, G. Widmer and S. Dixon, "Automatic page turning for musicians via real-time
      machine listening", Proc. ECAI, 2008 (Strategy 3)
  G. Widmer and A. Arzt, "Robust real-time music tracking", Proc. Vienna Talk, 2010.
The unfolded versions, the cost window and the switching margin are this implementation's
own choices, not those papers'.
"""

from __future__ import annotations

import collections
import copy
import re
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import partitura as pt
from numpy.typing import NDArray
from partitura.score import Part, merge_parts

from matchmaker.base import OnlineAlignment
from matchmaker.dp.oltw_arzt import (
    OnlineTimeWarpingArztFrame,
    START_WINDOW_SIZE,
    STEP_SIZE,
    WINDOW_SIZE,
)
from matchmaker.features.audio import FRAME_RATE
from matchmaker.features.processor import Processor
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import generate_score_audio, set_latency_stats
from partitura.io.exportmidi import get_ppq


class OnlineTimeWarpingArztMultiFold(OnlineTimeWarpingArztFrame):
    """Multi-fold parallel OLTW tracker for folded scores with repeats.

    Parameters
    ----------
    score_part : Part
        The original (folded) partitura Part.
    tempo : float
        Nominal score tempo in BPM.
    sample_rate : int
        Audio sample rate.
    processor : Processor
        Audio feature processor (e.g. ChromaProcessor).
    frame_rate : int
        Audio frame rate (default: FRAME_RATE, 30).
    window_size : int
        Search window size in seconds (default: 10).
    step_size : int
        Maximum forward advance per input frame (default: 3).
    start_window_size : float or int
        Window size during warmup (default: 0.1).
    selection_window : int
        Number of trailing frames over which running matching cost is averaged
        for hypothesis selection (default: 50 frames, about 1.7 s at 30 fps).
    switch_margin : float
        Hysteresis factor requiring an alternative hypothesis to have cost
        strictly lower by this fraction before switching (default: 0.05).
    queue : Optional[RECVQueue]
        Input queue for streaming observations.
    """

    def __init__(
        self,
        reference_features: Optional[NDArray[np.float32]] = None,
        score_positions: Optional[NDArray[np.float32]] = None,
        score_part: Optional[Part] = None,
        tempo: Optional[float] = None,
        sample_rate: int = 44100,
        processor: Optional[Processor] = None,
        frame_rate: int = FRAME_RATE,
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        distance_func: str = "Manhattan",
        selection_window: int = 50,
        switch_margin: float = 0.05,
        queue: Optional[RECVQueue] = None,
        ref_frame_to_beat: Optional[NDArray] = None,
        **kwargs,
    ) -> None:
        if score_part is None:
            raise ValueError("OnlineTimeWarpingArztMultiFold requires `score_part`.")
        if tempo is None:
            tempo = 120.0

        self.score_part = score_part
        self.tempo = float(tempo)
        self.sample_rate = sample_rate
        self.processor = processor
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.step_size = step_size
        self.start_window_size = start_window_size
        self.distance_func_name = distance_func
        self.selection_window = selection_window
        self.switch_margin = switch_margin
        self.queue_timeout = QUEUE_TIMEOUT

        # Folded note lookup: original note ID -> notated beat
        folded_notes = self.score_part.note_array()
        self.folded_note_dict = {
            str(n["id"]): float(n["onset_beat"]) for n in folded_notes
        }
        notated_positions = np.unique(folded_notes["onset_beat"]).astype(np.float32)

        ref_feat = (
            reference_features
            if reference_features is not None
            else np.empty((0, 12), dtype=np.float32)
        )
        dummy_ref_beat = (
            ref_frame_to_beat
            if ref_frame_to_beat is not None
            else np.zeros(max(1, len(ref_feat)), dtype=np.float32)
        )

        # Build parallel track hypotheses from the score repeat structure
        # before the parent __init__ so that reset() always finds them.
        self._build_hypotheses()

        super().__init__(
            reference_features=ref_feat,
            score_positions=notated_positions,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            distance_func=distance_func,
            frame_rate=frame_rate,
            ref_frame_to_beat=dummy_ref_beat,
            queue=queue,
            **kwargs,
        )
        self.reset()

    def _build_hypotheses(self) -> None:
        """Enumerate distinct structural fold clones safely and initialize trackers."""
        import itertools
        import sys

        _prev_limit = sys.getrecursionlimit()
        clones = []
        try:
            sys.setrecursionlimit(max(_prev_limit, 25_000))
            # 1. Maximal unfold (canonical repeat-performed performance)
            try:
                max_unfolded = pt.score.unfold_part_maximal(
                    self.score_part, ignore_leaps=False
                )
                clones.append(
                    merge_parts(max_unfolded.parts)
                    if hasattr(max_unfolded, "parts")
                    else max_unfolded
                )
            except Exception:
                pass

            # 2. Minimal / folded score (repeat-skipped performance)
            clones.append(self.score_part)

            # 3. Intermediate unfold variants (bounded to prevent combinatorial explosion)
            try:
                gen = pt.score.iter_unfolded_parts(self.score_part, update_ids=True)
                for c in itertools.islice(gen, 4):
                    clones.append(c)
            except Exception:
                pass
        finally:
            sys.setrecursionlimit(_prev_limit)

        # Deduplicate clones by note count and duration
        unique_clones = []
        seen = set()
        for c in clones:
            na = c.note_array()
            key = (len(na), round(float(na["onset_beat"].max()), 2))
            if key not in seen:
                seen.add(key)
                unique_clones.append(c)

        if not unique_clones:
            unique_clones = [self.score_part]

        self.hypotheses: List[Dict[str, Any]] = []
        tick = get_ppq(self.score_part)

        for i, clone in enumerate(unique_clones):
            # 1. Synthesize audio and extract reference features
            clone_audio = generate_score_audio(clone, self.tempo, self.sample_rate)
            if self.processor is not None:
                features, _ = self.processor((clone_audio.astype(np.float32), 0.0))
                self.processor.reset()
            else:
                raise ValueError(
                    "OnlineTimeWarpingArztMultiFold requires an audio processor."
                )

            n_ref_frames = features.shape[0]

            # 2. Frame to unfolded beat
            times = (
                (np.arange(n_ref_frames) / self.frame_rate)
                * tick
                * (self.tempo / 60.0)
            )
            ref_frame_to_unfolded_beat = np.array(
                [float(clone.beat_map(t)) for t in times], dtype=np.float32
            )
            score_positions_k = np.unique(clone.note_array()["onset_beat"]).astype(
                np.float32
            )

            # 3. Unfolded beat -> Notated beat mapping via note ID correspondence
            clone_notes = clone.note_array()
            u_beats = []
            n_beats = []
            for n in clone_notes:
                orig_id = re.sub(r"-\d+$", "", str(n["id"]))
                if orig_id in self.folded_note_dict:
                    u_beats.append(float(n["onset_beat"]))
                    n_beats.append(self.folded_note_dict[orig_id])

            u_beats = np.asarray(u_beats, dtype=float)
            n_beats = np.asarray(n_beats, dtype=float)
            sort_idx = np.argsort(u_beats)
            u_sorted = u_beats[sort_idx]
            n_sorted = n_beats[sort_idx]
            u_uniq, u_first = np.unique(u_sorted, return_index=True)
            n_uniq = n_sorted[u_first]

            # Precompute frame to notated beat directly
            if len(n_uniq) > 0:
                ref_frame_to_notated_beat = np.interp(
                    ref_frame_to_unfolded_beat,
                    u_uniq,
                    n_uniq,
                    left=n_uniq[0],
                    right=n_uniq[-1],
                ).astype(np.float32)
            else:
                ref_frame_to_notated_beat = ref_frame_to_unfolded_beat.astype(np.float32)

            # 4. Instantiate follower
            tracker = OnlineTimeWarpingArztFrame(
                reference_features=features,
                score_positions=score_positions_k,
                window_size=self.window_size,
                step_size=self.step_size,
                start_window_size=self.start_window_size,
                distance_func=self.distance_func_name,
                frame_rate=self.frame_rate,
                ref_frame_to_beat=ref_frame_to_unfolded_beat,
                queue=None,
            )

            self.hypotheses.append(
                {
                    "index": i,
                    "clone": clone,
                    "tracker": tracker,
                    "ref_frame_to_notated_beat": ref_frame_to_notated_beat,
                    "recent_costs": collections.deque(
                        maxlen=self.selection_window
                    ),
                    "mean_recent_cost": float("inf"),
                }
            )

    def reset(self) -> None:
        super().reset()
        self.active_hypothesis_idx = 0
        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        for hyp in self.hypotheses:
            hyp["tracker"].reset()
            hyp["recent_costs"].clear()
            hyp["mean_recent_cost"] = float("inf")

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        self.step(observation)
        current_beat = self.get_current_position()
        self._alignment_path.append((float(perf_time), float(current_beat)))
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return current_beat

    def step(self, input_features: NDArray[np.float32]) -> None:
        obs_squeezed = input_features.squeeze()

        # 1. Advance every hypothesis tracker
        for i, hyp in enumerate(self.hypotheses):
            tracker: OnlineTimeWarpingArztFrame = hyp["tracker"]
            tracker.step(input_features)

            # Measure local matching error at current hypothesis frame
            curr_frame = tracker._current_frame
            ref_feat = tracker.reference_features[curr_frame]
            loc_cost = float(tracker.distance_func(ref_feat, obs_squeezed))

            hyp["recent_costs"].append(loc_cost)
            mean_cost = float(np.mean(hyp["recent_costs"]))
            hyp["mean_recent_cost"] = mean_cost

        # 2. Decision Maker: select winning hypothesis with hysteresis
        current_hyp_cost = self.hypotheses[self.active_hypothesis_idx][
            "mean_recent_cost"
        ]
        min_hyp_idx = min(
            range(len(self.hypotheses)),
            key=lambda i: self.hypotheses[i]["mean_recent_cost"],
        )
        min_hyp_cost = self.hypotheses[min_hyp_idx]["mean_recent_cost"]

        if min_hyp_cost < current_hyp_cost * (1.0 - self.switch_margin):
            self.active_hypothesis_idx = min_hyp_idx

        self.input_index += 1

    def get_current_position(self) -> float:
        """Return the current position projected into notated (folded) score beats."""
        winning_hyp = self.hypotheses[self.active_hypothesis_idx]
        curr_frame = winning_hyp["tracker"]._current_frame
        notated_beat = winning_hyp["ref_frame_to_notated_beat"][
            min(curr_frame, len(winning_hyp["ref_frame_to_notated_beat"]) - 1)
        ]
        return float(notated_beat)

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))
