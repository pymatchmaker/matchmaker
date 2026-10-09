#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Time Warping (OLTWArzt)

OLTW with step-size constraint and adaptive window, based on:
  Arzt & Widmer (2010) "Simple Tempo Models for Real-Time Music Tracking"
  Arzt & Widmer (2015) "Real-Time Music Tracking Using Multiple Performances
                        as a Reference"

Classes:
  OnlineTimeWarpingArzt      — base class (common properties, step-size clamp, run loop)
  OnlineTimeWarpingArztFrame — frame-level variant for audio (Cython-accelerated)
  OnlineTimeWarpingArztTempoFrame — Arzt, Widmer & Dixon (2008) OLTW with the
                                    Arzt & Widmer (2010) tempo model (audio)
  OnlineTimeWarpingArztEvent — event-level variant for MIDI (onset-by-onset)
"""

import math
import time
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.features.audio import FRAME_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
    distances,
)
from matchmaker.utils.distances import Metric, vdist
from matchmaker.utils.errors import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
)
from matchmaker.utils.misc import set_latency_stats

STEP_SIZE: int = 3
START_WINDOW_SIZE: Union[float, int] = 0.1
WINDOW_SIZE: int = 10


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArzt(OnlineAlignment):
    """Base class for Arzt-style OLTW with step-size constraint.

    Parameters
    ----------
    reference_features : np.ndarray
        Feature matrix for the reference (score) sequence.
    score_positions : np.ndarray
        Score beat positions for unique onsets.
    window_size : int
        Search window size (interpretation depends on subclass).
    step_size : int
        Maximum position advance per input step.
    start_window_size : int
        Window size during warmup.
    queue : RECVQueue or None
        Input queue for streaming.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size=WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        start_window_size=START_WINDOW_SIZE,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        self.N_ref: int = self.reference_features.shape[0]
        self.step_size: int = step_size

    def reset(self) -> None:
        self.current_index = 0
        self._alignment_path: list = []
        self.global_cost_matrix: NDArray[np.float32] = np.full(
            (self.N_ref + 1, 2), np.inf, dtype=np.float32
        )
        self.input_index: int = 0

    @property
    def window_index(self) -> int:
        return self.current_index

    def get_window(self) -> Tuple[int, int]:
        """Return (start, end) of the search window."""
        raise NotImplementedError

    def step(self, input_features: NDArray[np.float32]) -> None:
        """Process one input element and update self.current_index."""
        raise NotImplementedError

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))


# ---------------------------------------------------------------------------
# Frame-level subclass (audio)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArztFrame(OnlineTimeWarpingArzt):
    """Frame-level OLTW for audio input.

    Uses Cython-accelerated ``oltw_arzt_loop`` and Matchmaker distance
    functions. Window size is in seconds (converted to frames via frame_rate).
    """

    DEFAULT_DISTANCE_FUNC: str = "Manhattan"

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        distance_func: Union[
            str, Callable, Tuple[str, Dict[str, Any]]
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: Union[float, int] = START_WINDOW_SIZE,
        frame_rate: int = FRAME_RATE,
        ref_frame_to_beat: NDArray = None,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            **kwargs,
        )
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Arzt requires `ref_frame_to_beat` (per-frame beat mapping)."
            )
        self.frame_rate = frame_rate
        self._ref_frame_to_beat = ref_frame_to_beat
        self._window_size = int(np.round(window_size * self.frame_rate))
        self._start_window_size = int(np.round(start_window_size * frame_rate))
        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self._init_distance_func(distance_func)
        self.reset()

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        self.input_features.append(observation)
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return beat

    def reset(self) -> None:
        super().reset()
        self.input_features: list = []
        self._current_frame = 0

    @property
    def window_index(self) -> int:
        return self._current_frame

    def _frame_to_beat(self, frame: int) -> float:
        """Per-frame beat value (frame-precision)."""
        return float(
            self._ref_frame_to_beat[min(frame, len(self._ref_frame_to_beat) - 1)]
        )

    def _frame_to_score_idx(self, frame: int) -> int:
        """Project a reference-frame index to the score_positions index."""
        beat = self._frame_to_beat(frame)
        idx = int(np.searchsorted(self.score_positions, beat, side="right") - 1)
        return max(0, min(idx, len(self.score_positions) - 1))

    def get_current_position(self) -> float:
        return self._frame_to_beat(self._current_frame)

    def _init_distance_func(self, distance_func):
        if not (isinstance(distance_func, (str, tuple)) or callable(distance_func)):
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="distance_func",
                required_parameter_type=(str, tuple, Callable),
                actual_parameter_type=type(distance_func),
            )

        if isinstance(distance_func, str):
            if distance_func not in CYTHONIZED_METRICS_WO_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_WO_ARGUMENTS,
                    value=distance_func,
                )
            self.distance_func = getattr(distances, distance_func)()
        elif isinstance(distance_func, tuple):
            if distance_func[0] not in CYTHONIZED_METRICS_W_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_W_ARGUMENTS,
                    value=distance_func[0],
                )
            self.distance_func = getattr(distances, distance_func[0])(
                **distance_func[1]
            )
        elif callable(distance_func):
            self.distance_func = distance_func

        if isinstance(self.distance_func, Metric):
            self.vdist = vdist
        else:
            self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(
                np.float32
            )

    def get_window(self) -> Tuple[int, int]:
        w = self._window_size
        if self.window_index < self._start_window_size:
            w = self._start_window_size
        start = max(self.window_index - w, 0)
        end = min(self.window_index + w, self.N_ref)
        return start, end

    def step(self, input_features: NDArray[np.float32]) -> None:
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            input_features.squeeze(),
            self.distance_func,
        )

        self.global_cost_matrix, min_index, min_costs = oltw_arzt_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
        )

        if self.input_index > 0:
            self._current_frame = int(
                np.clip(
                    min_index,
                    self._current_frame,
                    self._current_frame + self.step_size,
                )
            )
        else:
            self._current_frame = min_index
        self.current_index = self._frame_to_score_idx(self._current_frame)
        self.input_index += 1


_ADV_IN, _ADV_SC, _BOTH = 0, 1, 2  # Dixon (2005) GetInc: Row, Column, Both
_DIAG, _HORIZ, _VERT = 0, 1, 2  # predecessor: (x-1, y-1), (x-1, y), (x, y-1)
_TILE = 256


def _scan(
    diag, straight, d, wa, wb, first_along, straight_code, along_code, along_first
):
    """DTW recursion along one new line (a column or a row) of the cost matrix.

    ``diag[k]`` and ``straight[k]`` are the accumulated costs of the diagonal
    predecessor and of the predecessor on the previous line; the predecessor
    along this line is cell ``k - 1`` (``first_along`` for ``k = 0``). Ties go
    to the diagonal, then to the horizontal, then to the vertical predecessor.
    """
    n = d.shape[0]
    dg, st, dd = diag.tolist(), straight.tolist(), d.tolist()
    out = [0.0] * n
    codes = [0] * n
    prev = first_along
    for k in range(n):
        dk = dd[k]
        cd = dg[k] + wb * dk
        cs = st[k] + wa * dk
        ca = prev + wa * dk
        best, bc = cd, _DIAG
        if along_first:
            if ca < best:
                best, bc = ca, along_code
            if cs < best:
                best, bc = cs, straight_code
        else:
            if cs < best:
                best, bc = cs, straight_code
            if ca < best:
                best, bc = ca, along_code
        if not math.isfinite(best):
            bc = -1
        out[k] = best
        codes[k] = bc
        prev = best
    return np.array(out, dtype=np.float64), np.array(codes, dtype=np.int8)


class _CostGrid:
    """Sparse cost matrix in tiles: accumulated cost D (inf = not computed) and backpointers."""

    def __init__(self):
        self.D = {}
        self.B = {}

    def _tile(self, tx, ty):
        key = (tx, ty)
        if key not in self.D:
            self.D[key] = np.full((_TILE, _TILE), np.inf)
            self.B[key] = np.full((_TILE, _TILE), -1, dtype=np.int8)

    def get(self, x, y):
        if x < 0 or y < 0:
            return np.inf
        t = self.D.get((x // _TILE, y // _TILE))
        return np.inf if t is None else t[x % _TILE, y % _TILE]

    def get_bp(self, x, y):
        b = self.B.get((x // _TILE, y // _TILE))
        return -1 if b is None else b[x % _TILE, y % _TILE]

    def set(self, x, y, v, c):
        self._tile(x // _TILE, y // _TILE)
        self.D[(x // _TILE, y // _TILE)][x % _TILE, y % _TILE] = v
        self.B[(x // _TILE, y // _TILE)][x % _TILE, y % _TILE] = c

    def col(self, x, y0, y1):
        """D(x, y) for y in [y0, y1]."""
        out = np.full(y1 - y0 + 1, np.inf)
        if x < 0 or y1 < 0:
            return out
        y = max(y0, 0)
        while y <= y1:
            ty = y // _TILE
            hi = min(y1, ty * _TILE + _TILE - 1)
            t = self.D.get((x // _TILE, ty))
            if t is not None:
                out[y - y0 : hi - y0 + 1] = t[x % _TILE, y % _TILE : hi % _TILE + 1]
            y = hi + 1
        return out

    def row(self, y, x0, x1):
        """D(x, y) for x in [x0, x1]."""
        out = np.full(x1 - x0 + 1, np.inf)
        if y < 0 or x1 < 0:
            return out
        x = max(x0, 0)
        while x <= x1:
            tx = x // _TILE
            hi = min(x1, tx * _TILE + _TILE - 1)
            t = self.D.get((tx, y // _TILE))
            if t is not None:
                out[x - x0 : hi - x0 + 1] = t[x % _TILE : hi % _TILE + 1, y % _TILE]
            x = hi + 1
        return out

    def set_col(self, x, y0, vals, codes):
        y, y1 = y0, y0 + len(vals) - 1
        while y <= y1:
            ty = y // _TILE
            hi = min(y1, ty * _TILE + _TILE - 1)
            self._tile(x // _TILE, ty)
            self.D[(x // _TILE, ty)][x % _TILE, y % _TILE : hi % _TILE + 1] = vals[
                y - y0 : hi - y0 + 1
            ]
            self.B[(x // _TILE, ty)][x % _TILE, y % _TILE : hi % _TILE + 1] = codes[
                y - y0 : hi - y0 + 1
            ]
            y = hi + 1

    def set_row(self, y, x0, vals, codes):
        x, x1 = x0, x0 + len(vals) - 1
        while x <= x1:
            tx = x // _TILE
            hi = min(x1, tx * _TILE + _TILE - 1)
            self._tile(tx, y // _TILE)
            self.D[(tx, y // _TILE)][x % _TILE : hi % _TILE + 1, y % _TILE] = vals[
                x - x0 : hi - x0 + 1
            ]
            self.B[(tx, y // _TILE)][x % _TILE : hi % _TILE + 1, y % _TILE] = codes[
                x - x0 : hi - x0 + 1
            ]
            x = hi + 1

    def prune_below(self, y_min):
        for key in [k for k in self.D if (k[1] + 1) * _TILE <= y_min]:
            del self.D[key]
            del self.B[key]


class OnlineTimeWarpingArztTempoFrame(OnlineAlignment):
    """Arzt, Widmer & Dixon (2008) on-line time warping with the simple tempo
    model of Arzt & Widmer (2010).

    References
    ----------
    - S. Dixon (2005), "An On-Line Time Warping Algorithm for Tracking
      Musical Performances", IJCAI -- the OLTW algorithm (Fig. 1: GetInc,
      MaxRunCount, search width c).
    - A. Arzt, G. Widmer & S. Dixon (2008), "Automatic Page Turning for
      Musicians via Real-Time Machine Listening", ECAI, Sec. 3 -- the tracker:
      straight steps weighted 1.3 and diagonal steps 2 (Eq. 1), MaxRunCount 6,
      c = 500 frames (10 s at 50 fps), a 1 s initial diagonal phase, and
      Strategy 1 (backward-forward: every 2 input frames go back b = 10 score
      frames on the backward path, every fifth time b = 50, and compute a new
      forward path from there).
    - A. Arzt & G. Widmer (2010), "Simple Tempo Models for Real-Time Music
      Tracking", SMC, Sec. 4 -- the tempo model: after every input frame, the
      relative tempo t is the Eq. 1-weighted mean of the local tempi at the 20
      most recent score onsets at least 1 s in the past, each over a 3 s
      window of the rectified backward path (as in Mueller et al. (2009), ISMIR,
      Sec. 3.1 / 3.3: path rectified at onsets, extended with slope 1 past its
      ends). The score representation is then altered at the last processed
      score frame ps: with probability 1 - 1/t (t > 1) frames ps+1 and ps+2
      are merged into their mean, with probability 1 - t (t < 1) the mean of
      ps and ps+1 is inserted; onset frames are not duplicated (the insertion
      is postponed), and at most 3 alterations are made in a row.

    Use the ``onset2008`` features (see ``Onset2008Processor``).

    Differences from the papers
    ---------------------------
    - Features: the 2008 paper normalises each frame to sum 1 and then keeps
      the positive difference; on synthesised score audio against recordings
      this loses the onset energy and the tracker fails (eval146: 6/146 pieces
      tracked). Here the positive difference is L2-normalised instead
      (crossover at 370 Hz kept).
    - Strategy 2 (onset-based) needs details only given in a thesis, and
      Strategy 3 (multiple instances for structural changes) does not apply
      to unfolded scores; neither is implemented.
    - Where the papers are silent: the weights multiply the local cost d and
      GetInc compares D / (i + j); the tempo window is taken on the (altered)
      score representation; the reported position is the end of the forward
      path; the backward path used for the tempo is not limited in length.

    Parameters
    ----------
    reference_features : np.ndarray
        Score features, one row per frame.
    score_positions : np.ndarray
        Score beat positions of the onsets.
    ref_frame_to_beat : np.ndarray
        Beat position of each reference frame.
    frame_rate : int
        Frames per second (50 in the papers).
    window_size : float
        Search width c in seconds.
    max_run_count : int
        MaxRunCount.
    init_seconds : float
        Length of the initial diagonal phase in seconds.
    use_tempo_model : bool
        Apply the 2010 tempo model; False gives the 2008 tracker alone.
    tempo_window_size : float
        Window around an onset for its local tempo, in seconds.
    tempo_min_past : float
        Onsets used must be at least this many seconds in the past.
    tempo_n_onsets : int
        Number of recent onsets averaged.
    random_seed : int or None
        Seed for the random score alterations.
    queue : RECVQueue or None
        Input queue for streaming.
    """

    WA = 1.3  # straight steps (2008, Sec. 3.2)
    WB = 2.0  # diagonal steps (2008, Eq. 1)

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        ref_frame_to_beat: NDArray = None,
        frame_rate: int = 50,
        window_size: float = 10.0,
        max_run_count: int = 6,
        init_seconds: float = 1.0,
        use_tempo_model: bool = True,
        tempo_window_size: float = 3.0,
        tempo_min_past: float = 1.0,
        tempo_n_onsets: int = 20,
        random_seed: Optional[int] = 1984,
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        if ref_frame_to_beat is None:
            raise ValueError(
                "Frame-level Arzt requires `ref_frame_to_beat` (per-frame beat mapping)."
            )
        self.frame_rate = frame_rate
        self.c = int(round(window_size * frame_rate))
        self.max_run_count = max_run_count
        self.init_frames = int(round(init_seconds * frame_rate))
        self.use_tempo_model = use_tempo_model
        self.tempo_w = int(round(tempo_window_size * frame_rate))
        self.tempo_min_past = int(round(tempo_min_past * frame_rate))
        self.tempo_n_onsets = tempo_n_onsets
        self.random_seed = random_seed
        self._orig_V = np.asarray(reference_features, dtype=np.float64).copy()
        self._orig_f2b = np.asarray(ref_frame_to_beat, dtype=np.float64).copy()
        self.queue_timeout = 1
        self.reset()

    def reset(self) -> None:
        # The tempo model alters the score representation on the fly; each run
        # starts again from the notated score.
        self.V = self._orig_V.copy()
        self.f2b = self._orig_f2b.copy()
        self.N = len(self.V)
        on = np.searchsorted(self.f2b, self.score_positions)
        self.is_onset = np.zeros(self.N, dtype=bool)
        self.is_onset[on[on < self.N]] = True
        self.grid = _CostGrid()
        self.U = []
        self.T = self.J = self.x = self.y = -1
        self.run_count = 0
        self.previous = None
        self.pending = []
        self.n_received = 0
        self.n_bf = 0
        self.finished = False
        self._rng = np.random.RandomState(self.random_seed)
        self._consecutive = 0
        self._insert_pending = False
        self.current_relative_tempo = 1.0
        self.n_inserted = self.n_deleted = self.n_postponed = 0
        self._min_y_needed = 0
        self._alignment_path = []
        self.current_index = 0

    def is_still_following(self) -> bool:
        return not self.finished

    def get_current_position(self) -> float:
        return float(self.f2b[max(self.y, 0)])

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        self.reset()
        return (yield from super().run(verbose=verbose))

    # Axes: x = live input frame (columns), y = score representation frame (rows).

    def _dist(self, A, b):
        return np.sqrt(((A - b) ** 2).sum(axis=-1))

    def _new_column(self):
        """INPUT u(t): t := t+1, EvaluatePathCost(t, k) for k = j-c+1 .. j."""
        self.U.append(np.asarray(self.pending.pop(0), dtype=np.float64).reshape(-1))
        self.T += 1
        x = self.T
        y0, y1 = max(0, self.J - self.c + 1), self.J
        d = self._dist(self.V[y0 : y1 + 1], self.U[x])
        straight = self.grid.col(x - 1, y0, y1)
        diag = self.grid.col(x - 1, y0 - 1, y1 - 1)
        first = self.grid.get(x, y0 - 1)
        vals, codes = _scan(
            diag, straight, d, self.WA, self.WB, first, _HORIZ, _VERT, False
        )
        self.grid.set_col(x, y0, vals, codes)

    def _new_row(self):
        """j := j+1, EvaluatePathCost(k, j) for k = t-c+1 .. t."""
        self.J += 1
        y = self.J
        x0, x1 = max(0, self.T - self.c + 1), self.T
        d = self._dist(np.asarray(self.U[x0 : x1 + 1]), self.V[y])
        straight = self.grid.row(y - 1, x0, x1)
        diag = self.grid.row(y - 1, x0 - 1, x1 - 1)
        first = self.grid.get(x0 - 1, y)
        vals, codes = _scan(
            diag, straight, d, self.WA, self.WB, first, _VERT, _HORIZ, True
        )
        self.grid.set_row(y, x0, vals, codes)

    def _norm(self, xs, ys, D):
        # path cost normalised by the path length i + j (1-based indices)
        return D / (xs + ys + 2.0)

    def _get_inc(self):
        x, y = self.x, self.y
        if x + 1 < self.init_frames:
            return _BOTH
        if self.run_count > self.max_run_count:
            return _ADV_SC if self.previous == _ADV_IN else _ADV_IN
        best = self._norm(x, y, self.grid.get(x, y))
        bx, by = x, y
        x0, y0 = max(0, x - self.c + 1), max(0, y - self.c + 1)
        r = self._norm(np.arange(x0, x + 1), y, self.grid.row(y, x0, x))
        k = int(np.argmin(r))
        if r[k] < best:
            best, bx, by = r[k], x0 + k, y
        cl = self._norm(x, np.arange(y0, y + 1), self.grid.col(x, y0, y))
        k = int(np.argmin(cl))
        if cl[k] < best:
            best, bx, by = cl[k], x, y0 + k
        # 2008, Sec. 3.1: "If this occurs elsewhere in row j a new row is
        # calculated and if this occurs elsewhere in column i a new column is
        # calculated" (rows = score frames, columns = input frames)
        if bx < x:
            return _ADV_SC
        if by < y:
            return _ADV_IN
        return _BOTH

    def _update_run_count(self):
        inc = self._get_inc()
        if inc == self.previous:
            self.run_count += 1
        else:
            self.run_count = 1
        if inc != _BOTH:
            self.previous = inc

    def _loop(self):
        """The LOOP of Dixon (2005), Fig. 1, run until it needs the next input frame."""
        while not self.finished:
            if self._get_inc() != _ADV_SC:
                if self.x == self.T:
                    if not self.pending:
                        return
                    self._new_column()
                self.x += 1
            if self._get_inc() != _ADV_IN:
                if self.y == self.J:
                    if self.J + 1 >= self.N:
                        self.finished = True
                        return
                    self._new_row()
                self.y += 1
            self._update_run_count()

    def _backward_forward(self):
        """2008, Sec. 3.2, Strategy 1."""
        b = 50 if self.n_bf % 5 == 4 else 10
        self.n_bf += 1
        x, y = self.x, self.y
        target = y - b
        while y > target:
            code = self.grid.get_bp(x, y)
            if code < 0:
                break
            if code == _DIAG:
                x, y = x - 1, y - 1
            elif code == _HORIZ:
                x -= 1
            else:
                y -= 1
        self._min_y_needed = min(self._min_y_needed, y) if self._min_y_needed else y
        # a new forward path from (x, y) until column T or row J is reached
        self.x, self.y = x, y
        self.run_count, self.previous = 0, None
        while self.x < self.T and self.y < self.J:
            if self._get_inc() != _ADV_SC:
                if not np.isfinite(
                    self.grid.get(self.x + 1, self.y)
                ) and not np.isfinite(self.grid.get(self.x + 1, self.y + 1)):
                    break
                self.x += 1
            if self.x <= self.T and self.y < self.J and self._get_inc() != _ADV_IN:
                self.y += 1
            if not np.isfinite(self.grid.get(self.x, self.y)):
                break
            self._update_run_count()

    def _relative_tempo(self):
        """2010, Sec. 4.1: relative tempo from the rectified backward path."""
        half = (self.tempo_w - 1) // 2
        limit = self.T - self.tempo_min_past
        x, y = self.x, self.y
        phi = {}  # row -> min input index on the backward path
        onsets = []  # onset rows on the path, most recent first
        last_row = y
        phi[y] = x
        need_below = None
        while True:
            code = self.grid.get_bp(x, y)
            if code < 0:
                break
            if code == _DIAG:
                x, y = x - 1, y - 1
            elif code == _HORIZ:
                x -= 1
            else:
                y -= 1
            phi[y] = x
            if y != last_row:
                if self.is_onset[last_row] and phi[last_row] <= limit:
                    onsets.append(last_row)
                last_row = y
                if need_below is None and len(onsets) >= self.tempo_n_onsets:
                    need_below = float(onsets[self.tempo_n_onsets - 1]) - half
                if (
                    need_below is not None
                    and float(y) < need_below
                    and self.is_onset[y]
                ):
                    break
        if (
            self.is_onset[last_row]
            and phi[last_row] <= limit
            and last_row not in onsets
        ):
            onsets.append(last_row)
        self._min_y_needed = y if not self._min_y_needed else min(self._min_y_needed, y)
        if not onsets:
            return None
        # anchors: onsets on the path plus both ends (Mueller et al. 2009, Sec. 3.3)
        rows = sorted(set([y, self.y] + [r for r in phi if self.is_onset[r]]))
        g = np.array(rows, dtype=np.float64)
        p = np.array([phi[r] for r in rows], dtype=np.float64)

        def phi_r(n):
            if n <= g[0]:
                return p[0] - (g[0] - n)  # extended with slope 1
            if n >= g[-1]:
                return p[-1] + (n - g[-1])
            return float(np.interp(n, g, p))

        sel = onsets[: self.tempo_n_onsets][::-1]  # oldest first (Eq. 1 weights 1..n)
        tempi = []
        for o in sel:
            go = float(o)
            n1, n2 = go - half, go + half
            tempi.append((n2 - n1 + 1) / (phi_r(n2) - phi_r(n1) + 1))
        w = np.arange(1, len(tempi) + 1)
        return float(np.sum(np.array(tempi) * w) / np.sum(w))

    def _alter(self, t):
        """2010, Sec. 4.2: alter the score representation at the last processed frame."""
        ps = self.J
        if t is None:
            self._consecutive = 0
            return
        if t >= 1.0:
            self._insert_pending = False
        if ps < 0 or ps + 2 >= self.N:
            return
        if self._consecutive >= 3:  # at most 3 alterations in a row
            self._consecutive = 0
            return
        r = self._rng.uniform(0.0, 1.0)
        if t > 1.0 and r > 1.0 / t:
            V, f2b, on = self.V, self.f2b, self.is_onset
            V[ps + 1] = 0.5 * (V[ps + 1] + V[ps + 2])
            if on[ps + 2] and not on[ps + 1]:
                f2b[ps + 1] = f2b[ps + 2]
            elif not on[ps + 1]:
                f2b[ps + 1] = 0.5 * (f2b[ps + 1] + f2b[ps + 2])
            on[ps + 1] |= on[ps + 2]
            self.V = np.delete(V, ps + 2, axis=0)
            self.f2b = np.delete(f2b, ps + 2)
            self.is_onset = np.delete(on, ps + 2)
            self.N -= 1
            self.n_deleted += 1
            self._consecutive += 1
        elif t < 1.0 and (self._insert_pending or r > t):
            if (
                self.is_onset[ps] or self.is_onset[ps + 1]
            ):  # onset vectors are not duplicated
                self._insert_pending = True
                self.n_postponed += 1
                self._consecutive = 0
                return
            self._insert_pending = False
            self.V = np.insert(
                self.V, ps + 1, 0.5 * (self.V[ps] + self.V[ps + 1]), axis=0
            )
            self.f2b = np.insert(
                self.f2b, ps + 1, 0.5 * (self.f2b[ps] + self.f2b[ps + 1])
            )
            self.is_onset = np.insert(self.is_onset, ps + 1, False)
            self.N += 1
            self.n_inserted += 1
            self._consecutive += 1
        else:
            self._consecutive = 0

    def step(self, input_features: NDArray[np.float32]) -> None:
        self.pending.append(input_features)
        self.n_received += 1
        if self.T < 0:  # t := 1; j := 1; INPUT u(1); EvaluatePathCost(1, 1)
            self.J = 0
            self.U.append(np.asarray(self.pending.pop(0), dtype=np.float64).reshape(-1))
            self.T = 0
            self.grid.set(0, 0, float(self._dist(self.V[0], self.U[0])), -1)
            self.x = self.y = 0
            return
        if self.use_tempo_model:  # after every input frame, before the path computation
            t = self._relative_tempo()
            if t is not None:
                self.current_relative_tempo = t
            self._alter(t)
        self._loop()
        if self.n_received % 2 == 0:  # Strategy 1, after every 2 input frames
            self._backward_forward()
            self._loop()
        if self.n_received % self.c == 0:
            keep = min(self.J - 2 * self.c, self._min_y_needed) - 1
            if keep > 0:
                self.grid.prune_below(keep)
            self._min_y_needed = 0
        self.current_index = 0


# ---------------------------------------------------------------------------
# Event-level subclass (MIDI)
# ---------------------------------------------------------------------------


class OnlineTimeWarpingArztEvent(OnlineTimeWarpingArzt):
    """Event-level OLTW for MIDI input.

    Each step processes one onset event. Uses scipy for distance computation
    and a pure-Python rolling DP loop. Window size is in number of events.
    """

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        score_positions: NDArray[np.float32],
        window_size: int = 30,
        step_size: int = 5,
        start_window_size: int = 5,
        distance_func: str = "cosine",
        queue: Optional[RECVQueue] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            window_size=window_size,
            step_size=step_size,
            start_window_size=start_window_size,
            queue=queue,
            **kwargs,
        )
        self._window_size = window_size
        self._start_window_size = start_window_size
        self._scipy_metric = distance_func
        self.reset()

    def get_window(self) -> Tuple[int, int]:
        w = self._window_size
        if self.input_index < self._start_window_size:
            w = self._start_window_size
        start = max(self.window_index - self.step_size, 0)
        end = min(start + w, self.N_ref)
        return start, end

    def step(self, input_features: NDArray[np.float32]) -> None:
        feat = np.asarray(input_features, dtype=np.float32).squeeze()

        window_start, window_end = self.get_window()
        window_cost = scipy.spatial.distance.cdist(
            self.reference_features[window_start:window_end],
            feat.reshape(1, -1),
            metric=self._scipy_metric,
        ).flatten()

        # Rolling 2-column DP (same logic as oltw_arzt_loop)
        min_costs = np.inf
        min_index = max(self.window_index - self.step_size, 0)

        for idx_w, score_index in enumerate(range(window_start, window_end)):
            si = score_index + 1  # 1-indexed in cost matrix

            if score_index == 0 and self.input_index == 0:
                self.global_cost_matrix[1, 1] = window_cost[idx_w]
                min_costs = window_cost[idx_w]
                min_index = 0
                continue

            local_dist = window_cost[idx_w]
            d1 = self.global_cost_matrix[si - 1, 1] + local_dist
            d2 = self.global_cost_matrix[si, 0] + local_dist
            d3 = self.global_cost_matrix[si - 1, 0] + local_dist
            best = min(d1, d2, d3)
            self.global_cost_matrix[si, 1] = best

            norm_cost = best / (self.input_index + score_index + 1.0)
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        # Shift columns
        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        self.global_cost_matrix[:, 1] = np.inf

        if self.input_index > 0:
            self.current_index = int(
                np.clip(
                    min_index,
                    self.current_index,
                    self.current_index + self.step_size,
                )
            )
        else:
            self.current_index = min_index
        self.input_index += 1


if __name__ == "__main__":
    pass  # pragma: no cover
