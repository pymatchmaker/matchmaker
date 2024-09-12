#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping
"""
import time
from enum import IntEnum
from typing import Callable

import numpy as np
import scipy
from numpy.typing import NDArray
from tqdm import tqdm

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import SAMPLE_RATE
from matchmaker.utils.misc import RECVQueue


class Direction(IntEnum):
    REF = 0
    TARGET = 1

    def toggle(self):
        return Direction(self ^ 1)


DEFAULT_LOCAL_COST: str = "euclidean"
FRAME_RATE = 25
MAX_RUN_COUNT: int = 30
WINDOW_SIZE = 5
HOP_LENGTH = SAMPLE_RATE // FRAME_RATE
FRAME_PER_SEG = 1
QUEUE_TIMEOUT = 10


class OnlineTimeWarpingDixon(OnlineAlignment):
    """
    On-line Dynamic Time Warping (Dixon)

    Parameters
    ----------
    reference_features : np.ndarray
        A 2D array with dimensions (`F(n_features)`, `T(n_timesteps)`) containing the
        features of the reference the input is going to be aligned to.

    local_cost_fun : str
        Local metric for computing pairwise distances.

    window_size : int
        Size of the window for searching the optimal path in the cumulative cost matrix.

    max_run_count : int
        Maximum number of times the class can run in the same direction.

    frame_per_seg : int
        Number of frames per segment (audio chunk).

    frame_rate : int
        Frame rate of the audio stream.

    Attributes
    ----------
    warping_path : np.ndarray [shape=(2, T)]
        Warping path with pairs of indices of the reference and target features.
        where warping_path[0] is the index of the reference feature and warping_path[1] is the index of the target(input) feature.
    """

    local_cost_fun: str
    vdist: Callable[[NDArray[np.float32]], NDArray[np.float32]]

    def __init__(
        self,
        reference_features,
        window_size=WINDOW_SIZE,
        local_cost_fun=DEFAULT_LOCAL_COST,
        max_run_count=MAX_RUN_COUNT,
        frame_per_seg=FRAME_PER_SEG,
        frame_rate=FRAME_RATE,
        queue=None,
        perf_stream=None,
        **kwargs,
    ):
        super().__init__(reference_features=reference_features)
        self.N_ref = self.reference_features.shape[0]
        self.input_features = np.zeros(
            (self.N_ref * 3, self.reference_features.shape[1])  # [3N, F]
        )
        self.w = window_size * frame_rate
        self.local_cost_fun = local_cost_fun
        self.max_run_count = max_run_count
        self.frame_per_seg = frame_per_seg
        self.current_position = 0
        self.wp = np.array([[0, 0]]).T  # [shape=(2, T)]
        self.queue = queue or RECVQueue()
        self.ref_pointer = 0
        self.target_pointer = 0
        self.input_index: int = 0
        self.previous_direction = None
        self.last_queue_update = time.time()
        self.perf_stream = perf_stream

    @property
    def warping_path(self) -> NDArray[np.float32]:  # [shape=(2, T)]
        return self.wp

    def offset(self):
        offset_x = max(self.ref_pointer - self.w, 0)
        offset_y = max(self.target_pointer - self.w, 0)
        return np.array([offset_x, offset_y])

    def init_matrix(self):
        x = self.ref_pointer
        y = self.target_pointer
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))
        x_seg = self.reference_features[x - wx : x]  # [wx, 12]
        y_seg = self.input_features[min(y - d, 0) : y]  # [d, 12]
        dist = scipy.spatial.distance.cdist(
            x_seg, y_seg, metric=self.local_cost_fun
        )  # [wx, d]

        for i in range(wx):
            for j in range(d):
                local_dist = dist[i, j]
                update_x0 = 0
                update_y0 = wy - d
                if i == 0 and j == 0:
                    new_acc[i, j] = local_dist
                elif i == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i, update_y0 - 1]
                    new_len_acc[i, update_y0 + j] = 1 + new_len_acc[i, update_y0 - 1]
                elif j == 0:
                    new_acc[i, update_y0 + j] = local_dist + new_acc[i - 1, update_y0]
                    new_len_acc[i, update_y0 + j] = (
                        local_dist + new_len_acc[i - 1, update_y0]
                    )
                else:
                    compares = [
                        new_acc[i - 1, update_y0 + j],
                        new_acc[i, update_y0 + j - 1],
                        new_acc[i - 1, update_y0 + j - 1] * 0.98,
                    ]
                    len_compares = [
                        new_len_acc[i - 1, update_y0 + j],
                        new_len_acc[i, update_y0 + j - 1],
                        new_len_acc[i - 1, update_y0 + j - 1],
                    ]
                    local_direction = np.argmin(compares)
                    new_acc[i, update_y0 + j] = local_dist + compares[local_direction]
                    new_len_acc[i, update_y0 + j] = 1 + len_compares[local_direction]
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc
        self.select_candidate()

    def update_accumulate_matrix(self, direction):
        # local cost matrix
        x = self.ref_pointer
        y = self.target_pointer
        d = self.frame_per_seg
        wx = min(self.w, x)
        wy = min(self.w, y)
        new_acc = np.zeros((wx, wy))
        new_len_acc = np.zeros((wx, wy))

        if direction is Direction.REF:
            new_acc[:-d, :] = self.acc_dist_matrix[d:]
            new_len_acc[:-d, :] = self.acc_len_matrix[d:]
            x_seg = self.reference_features[x - d : x]  # [d, 12]
            y_seg = self.input_features[y - wy : y]  # [wy, 12]
            dist = scipy.spatial.distance.cdist(
                x_seg, y_seg, metric=self.local_cost_fun
            )  # [d, wy]

            for i in range(d):
                for j in range(wy):
                    local_dist = dist[i, j]
                    update_x0 = wx - d
                    update_y0 = 0
                    if j == 0:
                        new_acc[update_x0 + i, j] = (
                            local_dist + new_acc[update_x0 + i - 1, j]
                        )
                        new_len_acc[update_x0 + i, j] = (
                            new_len_acc[update_x0 + i - 1, j] + 1
                        )
                    else:
                        compares = [
                            new_acc[update_x0 + i - 1, j],
                            new_acc[update_x0 + i, j - 1],
                            new_acc[update_x0 + i - 1, j - 1] * 0.98,
                        ]
                        len_compares = [
                            new_len_acc[update_x0 + i - 1, j],
                            new_len_acc[update_x0 + i, j - 1],
                            new_len_acc[update_x0 + i - 1, j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[update_x0 + i, j] = (
                            local_dist + compares[local_direction]
                        )
                        new_len_acc[update_x0 + i, j] = (
                            1 + len_compares[local_direction]
                        )

        elif direction is Direction.TARGET:
            overlap_y = wy - d
            new_acc[:, :-d] = self.acc_dist_matrix[:, -overlap_y:]
            new_len_acc[:, :-d] = self.acc_len_matrix[:, -overlap_y:]
            x_seg = self.reference_features[x - wx : x]  # [wx, 12]
            y_seg = self.input_features[y - d : y]  # [d, 12]
            dist = scipy.spatial.distance.cdist(
                x_seg, y_seg, metric=self.local_cost_fun
            )  # [wx, d]

            for i in range(wx):
                for j in range(d):
                    local_dist = dist[i, j]
                    update_x0 = 0
                    update_y0 = wy - d
                    if i == 0:
                        new_acc[i, update_y0 + j] = (
                            local_dist + new_acc[i, update_y0 - 1]
                        )
                        new_len_acc[i, update_y0 + j] = (
                            1 + new_len_acc[i, update_y0 - 1]
                        )
                    else:
                        compares = [
                            new_acc[i - 1, update_y0 + j],
                            new_acc[i, update_y0 + j - 1],
                            new_acc[i - 1, update_y0 + j - 1] * 0.98,
                        ]
                        len_compares = [
                            new_len_acc[i - 1, update_y0 + j],
                            new_len_acc[i, update_y0 + j - 1],
                            new_len_acc[i - 1, update_y0 + j - 1],
                        ]
                        local_direction = np.argmin(compares)
                        new_acc[i, update_y0 + j] = (
                            local_dist + compares[local_direction]
                        )
                        new_len_acc[i, update_y0 + j] = (
                            1 + len_compares[local_direction]
                        )
        self.acc_dist_matrix = new_acc
        self.acc_len_matrix = new_len_acc

    def update_path_cost(self, direction):
        self.update_accumulate_matrix(direction)
        self.select_candidate()

    def select_candidate(self):
        norm_x_edge = self.acc_dist_matrix[-1, :] / self.acc_len_matrix[-1, :]
        norm_y_edge = self.acc_dist_matrix[:, -1] / self.acc_len_matrix[:, -1]
        cat = np.concatenate((norm_x_edge, norm_y_edge))
        min_idx = np.argmin(cat)
        offset = self.offset()
        if min_idx <= len(norm_x_edge):
            self.candidate = np.array([self.ref_pointer - offset[0], min_idx])
        else:
            self.candidate = np.array(
                [min_idx - len(norm_x_edge), self.target_pointer - offset[1]]
            )

    def save_history(self):
        new_coordinate = np.expand_dims(
            self.offset() + self.candidate, axis=1
        )  # [2, 1]
        self.wp = np.concatenate((self.wp, new_coordinate), axis=1)

    def select_next_direction(self):
        if self.target_pointer <= self.w:
            next_direction = Direction.TARGET
        elif self.run_count > self.max_run_count:
            next_direction = self.previous_direction.toggle()
        else:
            offset = self.offset()
            x0, y0 = offset[0], offset[1]
            if self.candidate[0] == self.ref_pointer - x0:
                next_direction = Direction.REF
            else:
                assert self.candidate[1] == self.target_pointer - y0
                next_direction = Direction.TARGET
        return next_direction

    def get_new_input(self):
        target_feature, f_time = self.queue.get()
        self.before_algorithm = time.time()
        q_length = self.frame_per_seg
        self.input_features[self.target_pointer : self.target_pointer + q_length] = (
            target_feature
        )
        self.target_pointer += q_length
        self.last_queue_update = time.time()

    def is_still_following(self):
        no_update = time.time() - self.last_queue_update > QUEUE_TIMEOUT
        return self.ref_pointer <= (self.N_ref - self.frame_per_seg) and not no_update

    def run(self):
        self.ref_pointer += self.w
        self.get_new_input()
        self.init_matrix()

        pbar = tqdm(total=self.N_ref)
        while self.is_still_following():
            updated_position = self.wp[0][-1]
            pbar.update(updated_position - self.current_position)
            pbar.set_description(
                f"[{self.ref_pointer}/{self.N_ref}] ref: {self.ref_pointer}, target: {self.target_pointer}"
            )
            self.current_position = updated_position
            self.save_history()
            direction = self.select_next_direction()

            if direction is Direction.TARGET:
                self.get_new_input()
            elif direction is Direction.REF:
                self.ref_pointer += self.frame_per_seg

            self.update_path_cost(direction)

            if direction == self.previous_direction:
                self.run_count += 1
            else:
                self.run_count = 1

            self.previous_direction = direction
            self.input_index += 1

        pbar.close()

        return self.wp
