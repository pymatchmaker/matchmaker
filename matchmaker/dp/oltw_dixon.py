#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping (Dixon 2005)
"""

import time
from enum import IntEnum
from typing import Dict

import numpy as np
import progressbar
import scipy
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE, QUEUE_TIMEOUT
from matchmaker.utils.misc import set_latency_stats


class Direction(IntEnum):
    REF = 0
    TARGET = 1
    BOTH = 2


MAX_RUN_COUNT: int = 3
FRAME_PER_SEG = 1
WINDOW_SIZE = 10  # seconds


class OnlineTimeWarpingDixon(OnlineAlignment):
    """
    On-line Dynamic Time Warping (Dixon)

    Parameters
    ----------
    reference_features : np.ndarray
        A 2D array with dimensions (`F(n_features)`, `T(n_timesteps)`) containing the
        features of the reference the input is going to be aligned to.

    distance_func : str, tuple (str, dict) or callable
        Distance function for computing pairwise distances.

    window_size : int
        Size of the window for searching the optimal path in the cumulative cost matrix.

    max_run_count : int
        Maximum consecutive advances in one direction before the other
        direction is forced. Constrains the slope to [1/max_run_count, max_run_count].

    frame_per_seg : int
        Number of frames per segment (audio chunk).

    frame_rate : int
        Frame rate of the audio stream.

    Attributes
    ----------
    warping_path : np.ndarray [shape=(2, T)]
        Warping path. ``warping_path[0]`` = reference index,
        ``warping_path[1]`` = input index.
    """

    DEFAULT_DISTANCE_FUNC: str = "euclidean"
    distance_func: str

    def __init__(
        self,
        reference_features,
        queue,
        window_size=WINDOW_SIZE,
        distance_func=DEFAULT_DISTANCE_FUNC,
        max_run_count=MAX_RUN_COUNT,
        frame_per_seg=FRAME_PER_SEG,
        frame_rate=FRAME_RATE,
        state_to_ref_time_map = None,
        ref_to_state_time_map = None,
        state_space = None,
        **kwargs,
    ):
        super().__init__(reference_features=reference_features)
        self.queue = queue
        self.N_ref = self.reference_features.shape[0]
        self.frame_rate = frame_rate
        self.w = int(window_size * self.frame_rate)
        self.distance_func = distance_func.lower()
        self.max_run_count = max_run_count
        self.frame_per_seg = frame_per_seg
        self.state_to_ref_time_map = state_to_ref_time_map
        self.ref_to_state_time_map = ref_to_state_time_map
        self.state_space = state_space
        self.reset()

    def reset(self):
        """Reset mutable state for a new alignment run."""
        self.input_features = np.empty((0, self.reference_features.shape[1]))
        self.acc_dist_matrix = np.full((self.w, self.w), np.inf, dtype=np.float32)
        self.wp = np.array([[0, 0]]).T  # [shape=(2, T)]
        self.ref_pointer = 0
        self.input_pointer = 0
        self.run_count_ref = 0
        self.run_count_input = 0
        self.best_ref = 0
        self.best_input = 0
        self.current_position = 0
        self.input_index = 0
        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        self._initialized = False

    @property
    def warping_path(self) -> NDArray[np.float32]:  # [shape=(2, T)]
        return self.wp

    def _ref_offset(self):
        return max(0, self.ref_pointer - self.w)

    def _input_offset(self):
        return max(0, self.input_pointer - self.w)

    def _ref_win_size(self):
        return min(self.w, self.ref_pointer)

    def _input_win_size(self):
        return min(self.w, self.input_pointer)

    def offset(self):
        """Calculate the offset for the current window position."""
        return np.array([self._ref_offset(), self._input_offset()])

    def _compute_cell(self, r_win, i_win):
        """
        Compute DTW accumulated cost for cell (r_win, i_win) in window coordinates.

        Uses Sakoe-Chiba symmetric weighting (DAFx 2005, Eq. 3):
            D[i,j] = min(D[i-1,j] + d, D[i,j-1] + d, D[i-1,j-1] + 2d)

        Matrix convention: acc_dist_matrix[ref_win, input_win]
        """
        abs_ref = self._ref_offset() + r_win
        abs_input = self._input_offset() + i_win

        ref_feature = self.reference_features[abs_ref]
        input_feature = self.input_features[abs_input]

        local_dist = scipy.spatial.distance.cdist(
            ref_feature.reshape(1, -1),
            input_feature.reshape(1, -1),
            metric=self.distance_func,
        )[0, 0]

        if r_win == 0 and i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = local_dist
        elif r_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win, i_win - 1] + local_dist
            )
        elif i_win == 0:
            self.acc_dist_matrix[r_win, i_win] = (
                self.acc_dist_matrix[r_win - 1, i_win] + local_dist
            )
        else:
            candidates = [
                self.acc_dist_matrix[r_win - 1, i_win] + local_dist,
                self.acc_dist_matrix[r_win, i_win - 1] + local_dist,
                self.acc_dist_matrix[r_win - 1, i_win - 1] + 2 * local_dist,
            ]
            self.acc_dist_matrix[r_win, i_win] = min(candidates)

    def _accept_input(self, input_features):
        """Store input features and advance input pointer."""
        self.input_features = np.vstack([self.input_features, input_features])
        self.input_pointer += self.frame_per_seg

    def _compute_input_column(self):
        """Shift the window left if full, then fill the new rightmost column."""
        if self.input_pointer > self.w:
            self.acc_dist_matrix[:, :-1] = self.acc_dist_matrix[:, 1:]
            self.acc_dist_matrix[:, -1] = np.inf

        input_col = self._input_win_size() - 1
        for r in range(self._ref_win_size()):
            self._compute_cell(r, input_col)

    def _advance_ref(self):
        """Advance ref pointer, shift the window up if full, then fill the new bottom row."""
        self.ref_pointer += self.frame_per_seg

        if self.ref_pointer > self.w:
            self.acc_dist_matrix[:-1, :] = self.acc_dist_matrix[1:, :]
            self.acc_dist_matrix[-1, :] = np.inf

        ref_row = self._ref_win_size() - 1
        for c in range(self._input_win_size()):
            self._compute_cell(ref_row, c)

    def _determine_direction(self):
        """Apply MaxRunCount slope constraint to the suggested direction"""
        suggested_direction = self.get_expand_direction()

        if self.run_count_ref >= self.max_run_count:
            return Direction.TARGET
        elif self.run_count_input >= self.max_run_count:
            return Direction.REF
        else:
            return suggested_direction

    def get_expand_direction(self):
        """
        Determine expansion direction (GetInc from Dixon 2005, IJCAI Fig. 1).

        Searches the L-shaped boundary of the sliding window — the last ref row
        and the last input column — for the cell with minimum normalised
        accumulated cost.  The direction is chosen based on where that minimum
        falls:
        - Corner (last row AND last col) → BOTH
        - In the column (not at last row) → TARGET (advance input to catch up)
        - In the row (not at last col)    → REF    (advance ref to catch up)

        Normalization: cost / (1 + i + j), an O(1) approximation of the true
        path-length divisor (DAFx 2005, Sec. 2).
        """
        ref_ws = self._ref_win_size()
        input_ws = self._input_win_size()
        ref_off = self._ref_offset()
        input_off = self._input_offset()

        # Current row and column in window coordinates
        row = ref_ws - 1  # last ref row
        col = input_ws - 1  # last input column

        best_cost = np.inf
        best_row = row
        best_col = col

        # Search current column (last input column) across all ref rows
        for r in range(ref_ws):
            abs_r = ref_off + r
            abs_i = input_off + col
            cost = self.acc_dist_matrix[r, col]
            norm_cost = cost / (1 + abs_r + abs_i)
            if norm_cost < best_cost:
                best_cost = norm_cost
                best_row = r
                best_col = col

        # Search current row (last ref row) across all input columns
        for c in range(input_ws):
            abs_r = ref_off + row
            abs_i = input_off + c
            cost = self.acc_dist_matrix[row, c]
            norm_cost = cost / (1 + abs_r + abs_i)
            if norm_cost < best_cost:
                best_cost = norm_cost
                best_row = row
                best_col = c

        # Convert best window coords to absolute
        self.best_ref = ref_off + best_row
        self.best_input = input_off + best_col

        # Determine direction based on where the best point is
        if best_row == row and best_col == col:
            return Direction.BOTH
        elif best_row < row:
            # Best is in the column (not at current row) → input needs to catch up
            return Direction.TARGET
        else:
            # Best is in the row (not at current col) → ref needs to catch up
            return Direction.REF

    def save_history(self):
        """Append current best alignment point to warping path."""
        new_point = np.array([[self.best_ref], [self.best_input]])
        self.wp = np.concatenate((self.wp, new_point), axis=1)

    def __call__(self, input_features: NDArray[np.float32]) -> int:
        self.step(input_features)
        return self.current_position

    def step(self, input_features):
        """Process one input frame and update alignment state.

        Handles REF-only catch-up internally so that each call
        consumes exactly one input frame.

        Parameters
        ----------
        input_features : NDArray[np.float32]
            A single input feature frame.
        """
        # First call: initialization (matches run()'s pre-loop logic)
        if not self._initialized:
            self._accept_input(input_features)
            self.ref_pointer += self.frame_per_seg
            self._compute_cell(0, 0)
            self._initialized = True
            self.input_index += 1
            return

        # Phase 1: REF-only catch-up (0 or more iterations)
        direction = None
        while self.is_still_following():
            direction = self._determine_direction()
            if direction != Direction.REF:
                break
            self._advance_ref()
            self.run_count_ref += 1
            self.run_count_input = 0
            self.save_history()
            self.current_position = self.best_ref

        if not self.is_still_following():
            return

        # Phase 2: consume input (direction is TARGET or BOTH)
        self._accept_input(input_features)
        self._compute_input_column()

        if direction is not Direction.TARGET:  # BOTH
            self._advance_ref()

        # Update run counts
        if direction == Direction.TARGET:
            self.run_count_input += 1
            self.run_count_ref = 0
        else:  # BOTH
            self.run_count_ref = 1
            self.run_count_input = 0

        self.save_history()
        self.current_position = self.best_ref
        self.input_index += 1

    def is_still_following(self):
        return self.ref_pointer <= (self.N_ref - self.frame_per_seg)

    def run(self, verbose=True):
        """Run the online alignment process.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show progress bar, by default True

        Yields
        ------
        int
            Current position in the reference sequence

        Returns
        -------
        NDArray[np.float32]
            The warping path as a 2D array where each column contains
            (reference_position, input_position)
        """
        self.reset()

        if verbose:
            pbar = progressbar.ProgressBar(max_value=self.N_ref, redirect_stdout=True)

        while self.is_still_following():
            input_feature, f_time = self.queue.get(timeout=QUEUE_TIMEOUT)
            self.last_queue_update = time.time()
            self.step(input_feature)

            if verbose:
                pbar.update(int(self.current_position))

            latency = time.time() - self.last_queue_update
            self.latency_stats = set_latency_stats(
                latency, self.latency_stats, self.input_index
            )
            yield self.current_position

        if verbose:
            pbar.finish()

        return self.wp
