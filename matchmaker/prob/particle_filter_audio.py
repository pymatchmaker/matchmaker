from typing import Generator, List, Optional

import numpy as np
import progressbar
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.utils.misc import RECVQueue
from partitura.utils.generic import interp1d

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]
SEED = 1984
RNG = np.random.RandomState(SEED)

QUEUE_TIMEOUT = 10


class ParticleFilterAudio(OnlineAlignment):
    def __init__(
        self,
        reference_features,  # shape: (num_score_frames, 12)
        state_space,  # beat position of each score frame
        score_boundaries,  # beat positions of note onsets/offsets
        notated_tempo,  # BPM from score
        hop_size,  # hop size in seconds
        queue: Optional[RECVQueue] = None,
        num_particles=1000,
    ):
        self.reference_features = reference_features
        self.state_space = state_space
        self.score_boundaries = np.array(score_boundaries)
        self.notated_tempo = notated_tempo
        self.hop_size = hop_size
        self.num_particles = num_particles
        self.current_state_in_frame_index = 0
        self.current_state = 0
        self.queue = queue
        self.N_ref = len(state_space)
        self.beat_to_frame_map = interp1d(
            x=self.state_space,
            y=np.arange(len(state_space)),
            dtype=int,
        )
        self.input_index = 0

        self.input_features: List[NDArray[np.float32]] = None
        self.rng = RNG

        self.beat_std = 0.25

        self.p_ioi = None
        self.f_time_prev = None

        # Tempo limits
        self.v_min = 0.5 * notated_tempo
        self.v_max = 2.0 * notated_tempo

        # Tempo noise (paper: quarter of notated tempo)
        self.sigma_v = 0.25 * notated_tempo
        self.tempo_noise = self.rng.normal(0, self.sigma_v, self.num_particles)

        # Particle state arrays
        self.x = np.zeros(self.num_particles)  # Beat position of each particle
        self.v = self.rng.normal(
            notated_tempo, self.sigma_v, num_particles
        )  # Normal distribution around notated tempo
        self.v = np.clip(self.v, self.v_min, self.v_max)  # Clip to valid range
        self.weights = np.ones(num_particles) / num_particles

        self.warping_path = [(self.current_state_in_frame_index, self.input_index)]

    def is_still_following(self) -> bool:
        if self.current_state is not None:
            return self.current_state < self.state_space[-1]

        return False

    def predict(self):
        # Update score position - each particle advances based on its tempo
        self.x += (self.v / 60.0) * self.hop_size  # Convert BPM to beats per second

        # Keep within bounds
        self.x = np.clip(self.x, self.state_space[0], self.state_space[-1])

    def compute_likelihood(self, feature):
        likelihoods = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            score_feature = self._get_score_feature(self.x[i])
            alpha = self._cosine_angle(feature, score_feature)
            likelihoods[i] = np.exp(-(alpha**2))

        # print(f"likelihoods: {likelihoods.argmax()}")
        return likelihoods

    def _get_score_feature(self, beat_position):
        # Find interval
        idx = np.searchsorted(self.state_space, beat_position)

        if idx <= 0:
            return self.reference_features[0]
        if idx >= len(self.state_space):
            return self.reference_features[-1]

        left = idx - 1
        right = idx

        beat_left = self.state_space[left]
        beat_right = self.state_space[right]

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
        cos_angle = np.clip(cos_angle, 0, 1)
        return np.arccos(cos_angle)

    def compute_likelihood_timing(self, ioi, idx):
        # Bounds checking: if at the start, use next beat instead
        if idx <= 0:
            return np.ones(self.num_particles)  # No timing info at start

        beat_diff = self.state_space[idx] - self.state_space[idx - 1]
        expected_ioi = (60.0 * beat_diff) / self.notated_tempo

        # Estimate tempo from observed IOI
        tempo_estimate = (60.0 * beat_diff) / max(ioi, 1e-6)

        # tempo update
        alpha = 0.1
        self.v += alpha * (tempo_estimate - self.v)
        self.v = np.clip(self.v, self.v_min, self.v_max)  # Ensure valid range

        # Likelihood based on how well IOI matches expected
        return np.exp(-0.5 * ((ioi - expected_ioi) / max(self.sigma_v, 1e-6)) ** 2)

    def step(self, feature, f_time):
        self.predict()

        likelihoods = self.compute_likelihood(feature)

        self.weights *= likelihoods
        self.weights += 1e-12  # avoid zero
        self.weights /= np.sum(self.weights)

        indices = self.rng.choice(
            self.num_particles, size=self.num_particles, p=self.weights
        )
        self.x = self.x[indices]
        self.v = self.v[indices]
        # Reset weights only after resampling
        self.weights.fill(1.0 / self.num_particles)

        current_state = np.clip(
            round(np.mean(self.x), 2),
            a_min=self.state_space.min(),
            a_max=self.state_space.max(),
        )
        self.current_state = current_state
        self.check_crossing(self.current_state)

        return self.current_state

    def check_crossing(self, previous_state):
        if previous_state is not None and self.current_state is not None:
            crossed = np.where(
                (self.score_boundaries > previous_state)
                & (self.score_boundaries <= self.current_state)
            )[0]
            if len(crossed) > 0:
                for idx in crossed:
                    self.v[idx] += self.rng.normal(
                        0, self.sigma_v
                    )  # Add tempo noise on crossing

    def __call__(self, feature, f_time):
        return self.step(feature, f_time)

    def run(self, verbose: bool = True) -> Generator[int, None, NDArray[np.float32]]:
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
        if verbose:
            pbar = progressbar.ProgressBar(max_value=self.N_ref, redirect_stdout=True)

        while self.is_still_following():
            features, f_time = self.queue.get(timeout=QUEUE_TIMEOUT)
            self.input_features = (
                np.concatenate((self.input_features, features))
                if self.input_features is not None
                else features
            )

            self.current_state = self(features, f_time)
            self.current_state_in_frame_index = int(
                self.beat_to_frame_map(self.current_state)
            )
            self.warping_path.append(
                (self.current_state_in_frame_index, self.input_index)
            )
            if verbose:
                pbar.update(self.current_state_in_frame_index)

            yield self.current_state

            self.input_index += 1

        if verbose:
            pbar.finish()

        self.warping_path = np.array(self.warping_path).T
        return self.warping_path
