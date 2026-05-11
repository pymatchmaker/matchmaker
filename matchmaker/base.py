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
