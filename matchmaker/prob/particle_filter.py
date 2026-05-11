import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import progressbar
from numpy.typing import NDArray
from partitura.io.exportmidi import get_ppq
from partitura.utils.generic import interp1d

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE, SAMPLE_RATE
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]
SEED = 1984
RNG = np.random.RandomState(SEED)

HOP_SIZE = 1.0 / FRAME_RATE


class ParticleFilter(OnlineAlignment):
    def __init__(
        self,
        reference_features,  # shape: (num_score_frames, 12)
        score_positions,  # shape: (num_score_frames,)
        score_boundaries,  # shape: (num_score_boundaries,)
        notated_tempo: float = 120.0,  # BPM from score
        hop_size: float = HOP_SIZE,  # hop size in seconds
        queue: Optional[RECVQueue] = None,
        num_particles=1000,
    ):
        self.reference_features = reference_features
        self.notated_tempo = notated_tempo
        self.hop_size = hop_size
        self.num_particles = num_particles

        self.score_positions = score_positions
        self.score_boundaries = score_boundaries

        self.current_state_in_frame_index = 0
        self.current_position = 0
        self.previous_state = None
        self.queue = queue
        self.queue_timeout: Optional[float] = QUEUE_TIMEOUT
        self.N_ref = len(score_positions)
        self.input_index = 0

        self.input_features: List = []
        self.rng = RNG

        self.beat_std = 0.25

        self.p_ioi = None
        self.f_time_prev = None

        # Tempo limits
        self.v_min = 0.5 * notated_tempo
        self.v_max = 2.0 * notated_tempo

        init_tempo_std = 0.05 * notated_tempo

        # Tempo noise (paper: quarter of notated tempo)
        self.sigma_v = 0.25 * notated_tempo

        # Particle state arrays
        self.x = np.zeros(self.num_particles)  # Beat position of each particle
        self.prev_x = self.x.copy()
        self.v = self.rng.uniform(
            self.v_min, self.v_max, self.num_particles
        )  # Tempo of each particle

        self.tempo_mean = np.mean(self.v)

        self.weights = np.ones(num_particles) / num_particles

        self._alignment_path = [(self.current_position, self.input_index)]

        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

    @property
    def warping_path(self) -> np.ndarray:
        return np.array(self._alignment_path).T

    def is_still_following(self) -> bool:
        if self.current_position is not None:
            return self.current_position < self.score_positions[-1]
        return False

    def predict(self):
        self.prev_x = self.x.copy()
        # Update score position - each particle advances based on its tempo
        self.x += (self.v / 60.0) * self.hop_size  # Convert BPM to beats per second

        # Keep within bounds
        self.x = np.clip(self.x, self.score_positions[0], self.score_positions[-1])

    def compute_likelihood(self, feature):
        likelihoods = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            score_feature = self._get_score_feature(self.x[i])
            alpha = self._cosine_angle(feature, score_feature)
            likelihoods[i] = np.exp(-(alpha**2) / 0.2**2)

        return likelihoods

    def _get_score_feature(self, beat_position):
        # Find interval
        idx = np.searchsorted(self.score_positions, beat_position)

        if idx <= 0:
            return self.reference_features[0]
        if idx >= len(self.score_positions):
            return self.reference_features[-1]

        left = idx - 1
        right = idx

        beat_left = self.score_positions[left]
        beat_right = self.score_positions[right]

        frac = (beat_position - beat_left) / (beat_right - beat_left + 1e-12)

        return (1 - frac) * self.reference_features[
            left
        ] + frac * self.reference_features[right]

    @staticmethod
    def _cosine_angle(ca, cm):
        norm_a = np.linalg.norm(ca)
        norm_m = np.linalg.norm(cm)

        if norm_a == 0 and norm_m == 0:
            return 0.0
        if norm_a == 0 or norm_m == 0:
            return np.pi / 2

        cos_angle = np.dot(ca, cm) / (norm_a * norm_m)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle)

    def check_crossing(self):
        if self.previous_state is not None and self.current_position is not None:
            # check if there is a score boundary between previous and current state
            if self.previous_state < self.current_position:
                crossed_boundaries = self.score_boundaries[
                    (self.score_boundaries > self.previous_state)
                    & (self.score_boundaries <= self.current_position)
                ]
                if len(crossed_boundaries) > 0:
                    # for each crossed boundary, check if it falls in between self.prev_x and self.x for each particle, and store the indexes of the particles that crossed the boundary
                    indices_crossed = []
                    for boundary in crossed_boundaries:
                        particles_crossed_indices = np.where(
                            (self.prev_x < boundary) & (self.x >= boundary)
                        )[0]
                        indices_crossed.append(particles_crossed_indices)

                    unique_indices_crossed = np.unique(np.concatenate(indices_crossed))
                    if len(unique_indices_crossed) > 0:
                        # For particles that crossed the boundary, reset their tempo to a random value around the mean tempo
                        self.v[unique_indices_crossed] = self.rng.normal(
                            self.tempo_mean, self.sigma_v, len(unique_indices_crossed)
                        )
                        self.v = np.clip(self.v, self.v_min, self.v_max)

    def resample(self):
        indices = self.rng.choice(
            self.num_particles, size=self.num_particles, p=self.weights
        )
        self.x = self.x[indices]
        self.x += self.rng.normal(
            0, 0.01, self.num_particles
        )  # Add noise after resampling

        self.v = self.v[indices]
        self.v += self.rng.normal(
            0, 1, self.num_particles
        )  # Add tempo noise after resampling
        self.v = np.clip(self.v, self.v_min, self.v_max)
        # Reset weights only after resampling
        self.weights.fill(1.0 / self.num_particles)

    def step(self, feature):
        self.predict()

        current_position = round(np.mean(self.x), 2)

        self.previous_state = self.current_position
        self.current_position = current_position
        self.check_crossing()

        likelihoods = self.compute_likelihood(feature)

        self.weights *= likelihoods
        self.weights += 1e-12  # avoid zero
        self.weights /= np.sum(self.weights)

        self.resample()

        self.tempo_mean = np.mean(self.v)
        self.sigma_v = 0.25 * self.tempo_mean

        return self.current_position

    def __call__(self, observation: Any, perf_time: float) -> float:
        t0 = time.time()
        self.input_features.append(observation)
        beat = super().__call__(observation, perf_time)
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        self.input_index += 1
        return beat

    def get_current_position(self) -> float:
        return self.current_position

    def run(self, verbose: bool = True) -> Generator[float, None, NDArray]:
        return (yield from super().run(verbose=verbose))
