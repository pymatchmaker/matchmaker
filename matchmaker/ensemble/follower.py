# -*- coding: utf-8 -*-
"""Ensemble score follower.

Runs several :class:`~matchmaker.base.OnlineAlignment` members on the same
input and uses a swappable :class:`~matchmaker.ensemble.policy.MetaPolicy` to
decide, at each step, which member's score-position estimate to trust. The
chosen position is fed back into the members (via their ``set_position``) so a
member that has drifted gets another chance to recover.

The ensemble is itself an ``OnlineAlignment`` subclass, so it plugs into the
existing ``run()`` / ``alignment_path`` / evaluation machinery and into the
``Matchmaker`` top-level object exactly like any other follower.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.features.processor import Processor
from matchmaker.io.queue import RECVQueue

from .policy import MemberState, MetaPolicy, build_policy


@dataclass
class EnsembleMember:
    """One member of the ensemble.

    Parameters
    ----------
    name : str
        Unique member name (also its action label for learned policies).
    follower : OnlineAlignment
        The wrapped score follower. It is driven directly by the ensemble
        (its own ``run()`` is never called), so it does not need a queue.
    processor : Processor
        The member's feature processor, applied by the ensemble to the raw
        frame before stepping the follower.
    modality : str
        ``"audio"`` or ``"midi"`` — selects which raw frames feed this member.
    """

    name: str
    follower: OnlineAlignment
    processor: Processor
    modality: str
    _stepped: bool = field(default=False, repr=False)
    _last_perf_time: Optional[float] = field(default=None, repr=False)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class EnsembleFollower(OnlineAlignment):
    """Meta follower combining several members under a meta-policy.

    Parameters
    ----------
    members : list of EnsembleMember
        The followers to combine (must be non-empty).
    score_positions : np.ndarray
        Unique score onset beats; used for progress/termination and to snap the
        chosen position to a state index.
    reference_features : Any, optional
        Unused by the ensemble itself (each member holds its own reference);
        kept for interface compatibility.
    queue : RECVQueue, optional
        Tagged input queue (see :class:`MergedStream`). Each item is
        ``((modality, raw_frame), perf_time)``.
    policy : MetaPolicy or str or None
        Meta-policy instance, registry name (``"agreement"``,
        ``"confidence_median"``, ``"rl"``), or ``None`` for the default
        agreement-gated policy.
    feedback : bool
        If True, push the chosen position back into drifted members.
    feedback_strength : float
        Correction strength in ``[0, 1]`` passed to ``set_position``.
    feedback_threshold : float
        Only members whose estimate is farther than this many beats from the
        chosen position are corrected, so members tracking well keep their
        independence (preserving ensemble diversity).
    feedback_exclude_selected : bool
        If True, never correct the member the policy currently trusts.
    """

    def __init__(
        self,
        members: List[EnsembleMember],
        score_positions: NDArray[np.float32],
        reference_features: Any = None,
        queue: Optional[RECVQueue] = None,
        policy=None,
        feedback: bool = True,
        feedback_strength: float = 0.5,
        feedback_threshold: float = 2.0,
        feedback_exclude_selected: bool = False,
    ) -> None:
        if not members:
            raise ValueError("EnsembleFollower requires at least one member.")
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.members = members
        self.policy: MetaPolicy = build_policy(policy, [m.name for m in members])
        self.feedback = feedback
        self.feedback_strength = feedback_strength
        self.feedback_threshold = feedback_threshold
        self.feedback_exclude_selected = feedback_exclude_selected
        self._chosen_position: float = 0.0
        # Block on the queue; termination is via STREAM_END / is_still_following.
        self.queue_timeout = None
        self.latency_stats = {
            "total_latency": 0.0,
            "total_frames": 0,
            "max_latency": 0.0,
            "min_latency": float("inf"),
        }

    # -- member introspection ------------------------------------------------

    def _member_states(self, modality: str) -> List[MemberState]:
        states = []
        for m in self.members:
            f = m.follower
            states.append(
                MemberState(
                    name=m.name,
                    position=_safe(lambda f=f: float(f.get_current_position())),
                    confidence=_safe(lambda f=f: f.confidence()),
                    is_following=bool(_safe(lambda f=f: f.is_still_following(), True)),
                    modality=m.modality,
                    stepped=m._stepped,
                )
            )
        return states

    # -- core step -----------------------------------------------------------

    def __call__(self, observation: Any, perf_time: float) -> float:
        """Process one tagged raw frame.

        Unlike the base class, a path point is recorded only on frames where at
        least one member actually advanced (a real observation arrived). Empty
        frames (e.g. silent MIDI polling windows) would otherwise pollute the
        alignment path with flat segments and bias evaluation.
        """
        self.current_perf_time = perf_time
        self.latency_stats["total_frames"] += 1
        stepped_any = self._ingest(observation)
        if not stepped_any:
            return self.current_position
        record_time = self._decide(observation[0], perf_time)
        self.current_position = self.get_current_position()
        self._alignment_path.append((self.current_position, record_time))
        return self.current_position

    def step(self, observation: Any) -> None:
        """Route + decide without recording (direct/offline use)."""
        if self._ingest(observation):
            self._decide(observation[0], self.current_perf_time)

    def _ingest(self, observation: Any) -> bool:
        """Run the matching members on the raw frame; return whether any stepped."""
        modality, raw_frame = observation
        perf_time = self.current_perf_time
        stepped_any = False
        for m in self.members:
            m._stepped = False
            if m.modality != modality:
                continue
            out = m.processor((raw_frame, perf_time))
            if out is None:
                continue
            features, member_perf_time = out
            m.follower(features, member_perf_time)
            m._stepped = True
            m._last_perf_time = member_perf_time
            stepped_any = True
        return stepped_any

    def _decide(self, modality: str, perf_time: float) -> float:
        """Run the meta-policy + feedback; return the perf time to record."""
        states = self._member_states(modality)
        context = {
            "last_position": self._chosen_position,
            "perf_time": perf_time,
            "modality": modality,
        }
        chosen = float(self.policy.select(states, context))
        self._chosen_position = chosen
        if self.score_positions is not None:
            self.current_index = self._snap_index(chosen)

        selected = self._apply_feedback(chosen)
        # Record the chosen member's observation time so the perf-time axis
        # matches what a standalone member would produce.
        if selected is not None and selected._last_perf_time is not None:
            return selected._last_perf_time
        return perf_time

    def _apply_feedback(self, chosen: float) -> Optional[EnsembleMember]:
        """Correct drifted members toward ``chosen``; return the trusted member.

        The member whose estimate is closest to the chosen position is treated
        as the one currently trusted (returned for perf-time recording). When
        ``feedback`` is enabled, every other member that has drifted beyond
        ``feedback_threshold`` beats is nudged toward ``chosen``.
        """
        positioned = [
            m for m in self.members
            if _safe(lambda m=m: float(m.follower.get_current_position())) is not None
        ]
        selected = (
            min(
                positioned,
                key=lambda m: abs(float(m.follower.get_current_position()) - chosen),
            )
            if positioned
            else None
        )
        if self.feedback:
            for m in self.members:
                if self.feedback_exclude_selected and m is selected:
                    continue
                pos = _safe(lambda m=m: float(m.follower.get_current_position()))
                if pos is None:
                    continue
                if abs(pos - chosen) > self.feedback_threshold:
                    _safe(
                        lambda m=m: m.follower.set_position(
                            chosen, self.feedback_strength
                        )
                    )
        return selected

    def get_current_position(self) -> float:
        return self._chosen_position

    # -- convenience direct-call API (offline / testing) ---------------------

    def observe(self, modality: str, raw_frame: Any, perf_time: float) -> float:
        """Feed one raw frame for ``modality`` directly (bypassing the queue)."""
        return self((modality, raw_frame), perf_time)

    def run(self, verbose: bool = True):
        self.policy.reset()
        return (yield from super().run(verbose=verbose))
