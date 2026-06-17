# -*- coding: utf-8 -*-
"""Meta-policies for :class:`~matchmaker.ensemble.follower.EnsembleFollower`.

A meta-policy looks at the per-member score-position estimates (and their
self-reported confidences) at each step and decides on a single position to
trust. Policies are swappable so a simple, hand-written gating rule can be
replaced by a learned (e.g. RL) policy without touching the ensemble plumbing.

All policies implement :meth:`MetaPolicy.select`, which receives the current
list of :class:`MemberState` plus a free-form ``context`` dict and returns the
chosen beat position (float).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# Default confidence assigned to a member that reports ``None`` (no usable
# confidence signal). Kept below 1.0 so that members which *do* report
# confidence can outweigh those that don't.
DEFAULT_CONFIDENCE = 0.5


@dataclass
class MemberState:
    """Snapshot of one ensemble member at a single ensemble step.

    Parameters
    ----------
    name : str
        Registered member name (e.g. ``"pthmm"``).
    position : float or None
        Latest ``current_position`` (beat). ``None`` if the member has not
        produced an estimate yet.
    confidence : float or None
        Member's self-reported confidence in ``[0, 1]`` or ``None``.
    is_following : bool
        Whether the member still considers itself active
        (``is_still_following()``).
    modality : str
        ``"audio"`` or ``"midi"``.
    stepped : bool
        Whether the member advanced on the current ensemble step (its modality
        fired and its processor emitted a feature).
    """

    name: str
    position: Optional[float]
    confidence: Optional[float]
    is_following: bool
    modality: str
    stepped: bool

    @property
    def weight(self) -> float:
        """Confidence clamped to ``[0, 1]`` with the ``None`` fallback applied."""
        c = DEFAULT_CONFIDENCE if self.confidence is None else self.confidence
        return float(np.clip(c, 0.0, 1.0))


class MetaPolicy(ABC):
    """Base class for ensemble meta-policies."""

    @abstractmethod
    def select(self, members: List[MemberState], context: Dict) -> float:
        """Return the trusted beat position given the current member states."""

    def reset(self) -> None:
        """Reset any per-run internal state (no-op by default)."""

    def record_reward(self, reward: float) -> None:
        """Hook for online/learning policies to receive a reward (no-op)."""


def _valid(members: List[MemberState]) -> List[MemberState]:
    """Members that have produced a position and are still following."""
    return [m for m in members if m.position is not None and m.is_following]


def _weighted_median(positions: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``positions`` (robust to outlier members)."""
    order = np.argsort(positions)
    positions = positions[order]
    weights = weights[order]
    cum = np.cumsum(weights)
    cutoff = cum[-1] / 2.0
    return float(positions[np.searchsorted(cum, cutoff)])


class ConfidenceWeightedMedianPolicy(MetaPolicy):
    """Robust blend: confidence-weighted median of the member positions.

    Less prone to a single confidently-wrong member than an argmax, while still
    biasing toward members that report high confidence.
    """

    def select(self, members: List[MemberState], context: Dict) -> float:
        valid = _valid(members)
        if not valid:
            return float(context.get("last_position", 0.0))
        positions = np.array([m.position for m in valid], dtype=float)
        weights = np.array([m.weight for m in valid], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        return _weighted_median(positions, weights)


class AgreementGatedPolicy(MetaPolicy):
    """Vote-then-gate selection — the default policy.

    1. Cluster the member positions: members within ``tolerance`` beats of each
       other form an agreement group.
    2. Pick the group with the highest total confidence (ties broken by size),
       i.e. the position the ensemble collectively agrees on.
    3. Within that group, trust the single most-confident member's position.

    This both *votes* (robust to a lone outlier) and *jumps* (snaps onto
    whichever follower is currently most trustworthy), matching the intent of a
    meta-follower that hops between members.

    Parameters
    ----------
    tolerance : float
        Maximum beat distance for two members to be considered in agreement.
    stickiness : float
        Small bonus added to the group containing the previously chosen
        position, to avoid chattering between near-equal groups.
    """

    def __init__(self, tolerance: float = 1.0, stickiness: float = 0.15) -> None:
        self.tolerance = tolerance
        self.stickiness = stickiness

    def select(self, members: List[MemberState], context: Dict) -> float:
        valid = _valid(members)
        if not valid:
            return float(context.get("last_position", 0.0))
        if len(valid) == 1:
            return float(valid[0].position)

        last = context.get("last_position", None)
        positions = np.array([m.position for m in valid], dtype=float)

        # Greedy single-link clustering on sorted positions.
        order = np.argsort(positions)
        groups: List[List[MemberState]] = []
        current: List[MemberState] = [valid[order[0]]]
        for prev, cur in zip(order[:-1], order[1:]):
            if positions[cur] - positions[prev] <= self.tolerance:
                current.append(valid[cur])
            else:
                groups.append(current)
                current = [valid[cur]]
        groups.append(current)

        def group_score(group: List[MemberState]) -> float:
            score = sum(m.weight for m in group)
            if last is not None and any(
                abs(m.position - last) <= self.tolerance for m in group
            ):
                score += self.stickiness
            return score

        best_group = max(groups, key=group_score)
        # Within the winning group, trust the most confident member.
        leader = max(best_group, key=lambda m: m.weight)
        return float(leader.position)


class RLMetaPolicy(MetaPolicy):
    """Reinforcement-learning meta-policy (selection over members).

    The RL agent observes a fixed-length feature vector built from the member
    states and outputs the index of the member to trust this step. Training is
    necessarily *offline* (reward = alignment error against ground-truth
    annotations, which only exist in a labelled corpus); at inference the
    trained policy is frozen.

    This class deliberately carries no RL framework dependency. Supply a trained
    ``policy_fn`` mapping a feature vector to a member index; until one is
    provided it delegates to ``fallback`` (the hand-written gating policy), so
    the ensemble is fully usable before any training has happened.

    Parameters
    ----------
    member_names : list of str
        Fixed member ordering, defining the action space and feature layout.
    policy_fn : callable, optional
        ``f(features: np.ndarray) -> int`` returning a member index.
    fallback : MetaPolicy, optional
        Policy used when ``policy_fn`` is ``None`` or returns an invalid index.
    """

    #: per-member features emitted by :meth:`featurize`
    FEATURES_PER_MEMBER = 4

    def __init__(
        self,
        member_names: List[str],
        policy_fn: Optional[Callable[[np.ndarray], int]] = None,
        fallback: Optional[MetaPolicy] = None,
    ) -> None:
        self.member_names = list(member_names)
        self.policy_fn = policy_fn
        self.fallback = fallback or AgreementGatedPolicy()
        self._last_features: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    def featurize(self, members: List[MemberState], context: Dict) -> np.ndarray:
        """Build the RL observation vector (fixed length, member order stable).

        Layout per member: ``[position, confidence, is_following, stepped]``,
        with ``position`` expressed relative to the ensemble's last chosen
        position so the policy generalizes across pieces.
        """
        by_name = {m.name: m for m in members}
        last = float(context.get("last_position", 0.0))
        feats: List[float] = []
        for name in self.member_names:
            m = by_name.get(name)
            if m is None or m.position is None:
                feats.extend([0.0, 0.0, 0.0, 0.0])
            else:
                feats.extend(
                    [
                        m.position - last,
                        m.weight,
                        1.0 if m.is_following else 0.0,
                        1.0 if m.stepped else 0.0,
                    ]
                )
        return np.asarray(feats, dtype=np.float32)

    def select(self, members: List[MemberState], context: Dict) -> float:
        features = self.featurize(members, context)
        self._last_features = features
        if self.policy_fn is not None:
            try:
                action = int(self.policy_fn(features))
            except Exception:
                action = -1
            if 0 <= action < len(self.member_names):
                self._last_action = action
                chosen = next(
                    (
                        m
                        for m in members
                        if m.name == self.member_names[action]
                        and m.position is not None
                    ),
                    None,
                )
                if chosen is not None:
                    return float(chosen.position)
        return self.fallback.select(members, context)

    def reset(self) -> None:
        self._last_features = None
        self._last_action = None
        self.fallback.reset()


#: Registry for ``policy=`` string lookup in the ensemble / Matchmaker config.
POLICIES: Dict[str, Callable[..., MetaPolicy]] = {
    "agreement": AgreementGatedPolicy,
    "confidence_median": ConfidenceWeightedMedianPolicy,
    "rl": RLMetaPolicy,
}


def build_policy(spec, member_names: List[str]) -> MetaPolicy:
    """Resolve a policy spec (name, instance, or ``None``) to a ``MetaPolicy``.

    ``None`` -> default :class:`AgreementGatedPolicy`. A string is looked up in
    :data:`POLICIES` (``"rl"`` is given ``member_names``). A ``MetaPolicy``
    instance is returned unchanged.
    """
    if spec is None:
        return AgreementGatedPolicy()
    if isinstance(spec, MetaPolicy):
        return spec
    if isinstance(spec, str):
        if spec not in POLICIES:
            raise ValueError(
                f"Unknown policy '{spec}'. Available: {sorted(POLICIES)}"
            )
        if spec == "rl":
            return RLMetaPolicy(member_names=member_names)
        return POLICIES[spec]()
    raise TypeError(f"Invalid policy spec of type {type(spec)}")
