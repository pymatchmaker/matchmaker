# -*- coding: utf-8 -*-
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import progressbar
from numpy.typing import NDArray

from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END


class OnlineAlignment(object):
    """Base class for online score followers.

    Sits at the end of the pipeline: receives ``(features, perf_time)``
    observations (from a queue or directly) and emits the current score
    position per step plus a final ``alignment_path``.

    Input
    -----
    Each ``__call__(observation)`` takes a ``(features, perf_time)`` tuple,
    where ``features`` is the per-frame feature vector from the Processor
    and ``perf_time`` is the timestamp (chord onset for chord-buffering
    MIDI processors, frame time otherwise). The stream-driven ``run()``
    pulls these tuples from ``self.queue`` until a ``STREAM_END`` sentinel.

    Output
    ------
    Per step: ``current_position`` (float beat).
    On completion: ``alignment_path`` — a ``(2, T)`` ``np.ndarray`` whose
    rows are score beats and performance times in seconds.

    Subclass requirements
    ---------------------
    Subclasses must implement ``step(features)`` to update
    ``self.current_index`` (an integer index into ``self.score_positions``).
    The base class derives ``self.current_position`` from that index in
    ``__call__``, and provides defaults for ``run``, ``alignment_path``,
    ``is_still_following``, and ``get_current_position``.

    Parameters
    ----------
    reference_features : Any
        Reference (score) features.
    score_positions : np.ndarray, optional
        Score beat position (float) for each alignment state.
    queue : RECVQueue, optional
        Input queue for stream-driven ``run()``. Not needed when calling
        ``__call__`` directly per observation.
    """

    def __init__(
        self,
        reference_features: Any = None,
        score_positions: Optional[NDArray[np.float32]] = None,
        queue: Optional[RECVQueue] = None,
    ) -> None:
        self.reference_features = reference_features
        self.score_positions = score_positions
        self.queue = queue
        self.queue_timeout: Optional[float] = None
        self.current_index: int = 0
        self.current_position: float = 0.0
        self.current_perf_time: float = 0.0
        self._alignment_path: List[Tuple[float, float]] = []

    def __call__(self, observation: Any, perf_time: float) -> float:
        self.current_perf_time = perf_time
        self.step(observation)
        self.current_position = self.get_current_position()
        self._alignment_path.append((self.current_position, perf_time))
        return self.current_position

    @property
    def alignment_path(self) -> NDArray[np.float64]:
        if not self._alignment_path:
            return np.empty((2, 0))
        return np.array(self._alignment_path).T

    def is_still_following(self) -> bool:
        return self.current_index < len(self.score_positions) - 1

    def get_current_position(self) -> float:
        """Compute current_position (float beat) from internal state.

        Default: ``score_positions[current_index]``. Override in subclasses
        that need higher precision than score-state granularity (e.g.
        frame-level OLTW with a per-frame beat mapping).
        """
        return float(self.score_positions[self.current_index])

    def step(self, features: Any) -> None:
        """Update self.current_index based on the observation features."""
        raise NotImplementedError

    def set_position(self, beat: float, strength: float = 1.0) -> bool:
        """Nudge this follower's internal state toward an external estimate.

        Used by meta-followers (e.g. ``EnsembleFollower``) to feed a trusted
        position back into every member so they get another chance to recover.

        Parameters
        ----------
        beat : float
            Target score position in beat units (matched against
            ``self.score_positions``).
        strength : float, optional
            Correction strength in ``[0, 1]``. ``1.0`` snaps as hard as the
            follower allows; smaller values blend the correction with the
            follower's own belief (meaningful for probabilistic followers).

        Returns
        -------
        bool
            ``True`` if the follower applied the correction, ``False`` if it
            does not support correction (default).
        """
        return False

    def confidence(self) -> Optional[float]:
        """Self-reported confidence in the latest ``current_position``.

        Returns a value in ``[0, 1]`` (higher = more certain) or ``None`` when
        the follower exposes no usable confidence signal. Meta-policies fall
        back to uniform weighting for ``None``.
        """
        return None

    @staticmethod
    def _blend_belief(
        belief: NDArray[np.float64], target_index: int, strength: float
    ) -> NDArray[np.float64]:
        """Blend a probability vector toward a Gaussian bump at ``target_index``.

        Helper for probabilistic followers' ``set_position``: builds a peaked
        correction distribution over state indices and mixes it with the
        current ``belief`` according to ``strength`` (1.0 = fully replace).
        Returns a normalized vector of the same length.
        """
        n = len(belief)
        p = np.asarray(belief, dtype=np.float64)
        s = p.sum()
        p = np.full(n, 1.0 / n) if s <= 0 else p / s

        sigma = max(1.0, 0.01 * n)
        states = np.arange(n)
        bump = np.exp(-0.5 * ((states - target_index) / sigma) ** 2)
        bump /= bump.sum()

        strength = float(np.clip(strength, 0.0, 1.0))
        blended = strength * bump + (1.0 - strength) * p
        total = blended.sum()
        return blended / total if total > 0 else blended

    @staticmethod
    def _belief_confidence(belief: Optional[NDArray[np.float64]]) -> Optional[float]:
        """Confidence in ``[0, 1]`` from a belief vector's peakedness
        (``1 - normalized entropy``). ``None`` if the vector is empty/degenerate."""
        if belief is None:
            return None
        p = np.asarray(belief, dtype=np.float64)
        s = p.sum()
        if s <= 0:
            return None
        p = p / s
        nz = p[p > 0]
        if len(nz) <= 1:
            return 1.0
        entropy = -np.sum(nz * np.log(nz))
        return float(1.0 - entropy / np.log(len(p)))

    def _snap_index(self, beat: float) -> int:
        """Index of the score position nearest to ``beat`` (helper for
        ``set_position`` in index-based followers)."""
        if self.score_positions is None:
            return self.current_index
        idx = int(np.searchsorted(self.score_positions, beat))
        idx = max(0, min(idx, len(self.score_positions) - 1))
        if idx > 0 and abs(self.score_positions[idx - 1] - beat) <= abs(
            self.score_positions[idx] - beat
        ):
            idx -= 1
        return idx

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        """Drive the score follower from `self.queue`.

        Pulls `(features, perf_time)` items from `self.queue`, calls `self(item)` per step,
        and yields the current beat each step. The final return value is the alignment path.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print a progress bar.

        Returns
        -------
        (yield) beat : float
            The current beat.
        (return) alignment_path : NDArray
            The alignment path.
        """
        if verbose:
            pbar = progressbar.ProgressBar(
                max_value=len(self.score_positions),
                redirect_stdout=True,
                redirect_stderr=True,
            )
            pbar.start()

        while self.is_still_following():
            item = self.queue.get(timeout=self.queue_timeout)
            if item is STREAM_END:
                break
            if item is None:
                continue
            beat = self(*item)
            if verbose:
                pbar.update(int(np.searchsorted(self.score_positions, beat)))
            yield beat

        if verbose:
            pbar.finish()
        return self.alignment_path


if __name__ == "__main__":  # pragma: no cover
    pass
