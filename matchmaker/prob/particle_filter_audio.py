from typing import List, Optional, Generator

import progressbar

from matchmaker.utils.misc import RECVQueue
from matchmaker.base import OnlineAlignment

import numpy as np
from numpy.typing import NDArray

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]
SEED = 1984
RNG = np.random.RandomState(SEED)

QUEUE_TIMEOUT = 10

class ParticleFilterAudio(OnlineAlignment):
    def __init__(
        self,
        reference_features,          # shape: (num_score_frames, 12)
        score_beats,            # beat position of each score frame
        score_boundaries,       # beat positions of note onsets/offsets
        notated_tempo,          # BPM from score
        hop_size,               # seconds
        queue: Optional[RECVQueue] = None,
        num_particles=1000
    ):
        self.reference_features = reference_features
        self.score_beats = score_beats
        self.score_boundaries = np.array(score_boundaries)
        self.notated_tempo = notated_tempo
        self.hop_size = hop_size
        self.num_particles = num_particles
        self.current_state = 0
        self.queue = queue
        self.N_ref: int = self.reference_features.shape[0]
        self.input_features: List[NDArray[np.float32]] = None
        self.rng = RNG

        # Tempo limits
        self.v_min = 0.5 * notated_tempo
        self.v_max = 2.0 * notated_tempo

        # Tempo noise (paper: quarter of notated tempo)
        self.sigma_v = 0.25 * notated_tempo

        # Particle state arrays
        self.x = self.rng.uniform(score_beats[0], score_beats[-1], num_particles)  # beat positions
        self.v = self.rng.uniform(self.v_min, self.v_max, num_particles)
        self.weights = np.ones(num_particles) / num_particles

        # Initialize at first beat
        self.x[:] = score_beats[0]

    def is_still_following(self) -> bool:
        if self.current_state is not None:
            return self.current_state <= self.N_ref - 1

        return False

    def predict(self):
        avg_tempo = np.mean(self.v)
        # Update score position
        self.x += (self.v / 60.0) * self.hop_size

        # Check boundary crossing
        crossed = self._crossed_boundary(self.x)

        # Update tempo only if crossed
        noise = self.rng.normal(0, self.sigma_v, self.num_particles)
        self.v = np.where(
            crossed,
            self.v + noise,
            self.v
        )

        # Clip tempo
        self.v = np.clip(self.v, self.v_min, self.v_max)

    def _crossed_boundary(self, new_positions):
        """
        Check if particle crossed any score boundary.
        """
        crossed = np.zeros(self.num_particles, dtype=bool)

        for b in self.score_boundaries:
            crossed |= (new_positions >= b)

        return crossed

    def compute_likelihood(self, feature):
        likelihoods = np.zeros(self.num_particles)

        for i in range(self.num_particles):
            score_feature = self._get_score_feature(self.x[i])
            alpha = self._cosine_angle(feature, score_feature)
            likelihoods[i] = np.exp(-alpha ** 2)

        return likelihoods

    def _get_score_feature(self, beat_position):
        idx = np.argmin(np.abs(self.score_beats - beat_position))
        return self.reference_features[idx]

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

    def step(self, feature):
        self.predict()

        likelihoods = self.compute_likelihood(feature)
        self.weights *= likelihoods
        self.weights += 1e-12  # avoid zero
        self.weights /= np.sum(self.weights)

        indices = self.rng.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )

        self.x = self.x[indices]
        self.v = self.v[indices]

        # Reset weights
        self.weights.fill(1.0 / self.num_particles)

        return int(round(np.mean(self.x)))
    
    
    def __call__(self, feature):
        return self.step(feature)
    
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
            self.current_state = self(features)

            if verbose:
                pbar.update(int(self.current_state))

            yield self.current_state

        if verbose:
            pbar.finish()

        return self.warping_path
