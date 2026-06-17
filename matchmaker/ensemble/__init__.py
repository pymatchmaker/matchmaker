# -*- coding: utf-8 -*-
"""Ensemble score following: run several followers and arbitrate between them.

See :class:`~matchmaker.ensemble.follower.EnsembleFollower` and the meta-policies
in :mod:`~matchmaker.ensemble.policy`.
"""
from .follower import EnsembleFollower, EnsembleMember
from .merged_stream import MergedStream, RawProcessor
from .policy import (
    AgreementGatedPolicy,
    ConfidenceWeightedMedianPolicy,
    MemberState,
    MetaPolicy,
    RLMetaPolicy,
    build_policy,
)

__all__ = [
    "EnsembleFollower",
    "EnsembleMember",
    "MergedStream",
    "RawProcessor",
    "MetaPolicy",
    "MemberState",
    "AgreementGatedPolicy",
    "ConfidenceWeightedMedianPolicy",
    "RLMetaPolicy",
    "build_policy",
]
