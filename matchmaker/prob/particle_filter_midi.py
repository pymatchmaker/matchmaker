from typing import List, Optional

import progressbar

from matchmaker.utils.misc import RECVQueue
from matchmaker.base import OnlineAlignment
from matchmaker.utils.symbolic import get_chords_from_score
from matchmaker.prob.hmm import compute_discrete_pitch_profiles, compute_bernoulli_pitch_probabilities

import numpy as np
from numpy.typing import NDArray


IOI_THRESHOLD = 0.035

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]
SEED = 1984
RNG = np.random.RandomState(SEED)


class ParticleFilterMIDI(OnlineAlignment):
    """Particle filter for online alignment of MIDI performances to a reference score."""
    def __init__(
            self, 
            num_particles: int,
            reference_features: np.ndarray,
            tempo: float = 120.0,
            tempo_std: float = 1.0,
            beat_std: float = 0.01,
            queue: Optional[RECVQueue] = None,
            other_prob: float = 1e-6,
            patience: int = 10,
            score_boundaries_in_seconds: Optional[List[float]] = None
            ):
        """
        Args:
            num_particles: Number of particles in the filter.
            reference_features: Note array or score like object.
            tempo: Tempo in BPM for the particles.
            tempo_std: Standard deviation for tempo noise.
            beat_std: Standard deviation for beat noise.
            queue: Optional queue for receiving real-time performance features.
            other_prob: Small probability for unseen events in likelihood calculation.
        """
        self.num_particles = num_particles
        self.reference_features = reference_features
        self.init_tempo = tempo
        self.current_tempo = tempo
        self.tempo_std = tempo_std
        self.beat_std = beat_std
        self.other_prob = other_prob
        self.patience = patience
        self.score_boundaries = score_boundaries_in_seconds


        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )

        self.queue = queue

        chords, unique_onsets = get_chords_from_score(self.reference_features, return_unique_onsets=True, return_one_hot=True)
        self.n_states = len(chords)
        self.state_space = unique_onsets
        self.state_space_min = min(self.state_space)
        self.state_space_max = max(self.state_space)
        self.chords = chords
        self.pitch_profiles = compute_discrete_pitch_profiles(self.chords)
        self._warping_path = []

        # Tempo limits
        self.tempo_min = 0.5 * tempo
        self.tempo_max = 2.0 * tempo

        self.patience = patience

        self.current_state = 0

        self._current_chord = np.zeros(128, dtype=int)

        self.particles = np.zeros((self.num_particles, 2))
        self.particles[:, 0] = np.zeros(self.num_particles)  # Initialize positions uniformly across states
        self.particles[:, 1] = RNG.uniform(self.tempo_min, self.tempo_max, self.num_particles)
        self.prev_x = self.particles[:, 0].copy()
        self.tempo_mean = np.mean(self.particles[:, 1])

        self.weights = np.ones(self.num_particles) / self.num_particles
        self._current_position = 0

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)
    
    @staticmethod
    def beats_to_seconds(beats, tempo_bpm):
        return 60.0 * beats / tempo_bpm
    
    @staticmethod
    def seconds_to_beats(seconds, tempo_bpm):
        return tempo_bpm * seconds / 60.0
    
    def is_still_following(self) -> bool:
        if self.current_state is not None:
            return self.current_state <= self.n_states - 1

        return False
        
    def predict(self, ioi: float):
        self.prev_x = self.particles[:, 0].copy()  # Store previous positions for warping path
        # Update particle states based on their tempo
        beats_step = self.seconds_to_beats(ioi, self.particles[:, 1])
        #step_noise = RNG.normal(0, self.beat_std, self.num_particles)
        self.particles[:, 0] += beats_step #+ step_noise
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.state_space_min, self.state_space_max)

    def check_crossing(self):
        if self.previous_state is not None and self.current_state is not None:
            # check if there is a score boundary between previous and current state
            if self.previous_state < self.current_state:
                crossed_boundaries = self.score_boundaries[
                    (self.score_boundaries > self.previous_state)
                    & (self.score_boundaries <= self.current_state)
                ]
                if len(crossed_boundaries) > 0:
                    # for each crossed boundary, check if it falls in between self.prev_x and self.x for each particle, and store the indexes of the particles that crossed the boundary
                    indices_crossed = []
                    for boundary in crossed_boundaries:
                        particles_crossed_indices = np.where(
                            (self.prev_x < boundary) & (self.particles[:, 0] >= boundary)
                        )[0]
                        indices_crossed.append(particles_crossed_indices)

                    unique_indices_crossed = np.unique(np.concatenate(indices_crossed))
                    if len(unique_indices_crossed) > 0:
                        self.particles[unique_indices_crossed, 1] = RNG.normal(self.tempo_mean, self.tempo_std, len(unique_indices_crossed))
                        self.particles[:, 1] = np.clip(self.particles[:, 1], self.tempo_min, self.tempo_max)

    
    def pitch_likelihood(
        self, 
        pitch: np.ndarray, 
        pitch_profile: np.ndarray
        ) -> NDArrayFloat:

        pitch_prob = (pitch_profile**pitch) * ((1 - pitch_profile) ** (1 - pitch))
        pitch_prob = np.prod(pitch_prob)
        pitch_prob = max(pitch_prob, self.other_prob)  # Avoid zero probability
        return pitch_prob
        
    # def timing_likelihood(self, ioi: float, idx: int):
    #     beat_diff = abs(self.state_space[idx] - self.state_space[idx - 1])
    #     expected_ioi = self.beats_to_seconds(beat_diff, self.current_tempo)
    #     tempo_estimate = 60.0 * beat_diff / max(ioi, 1e-6)
    #     alpha = 0.2
    #     self.particles[:, 1] += alpha * (tempo_estimate - self.particles[:, 1])  # Update tempo estimates
    #     self.particles[:, 1] = np.clip(self.particles[:, 1], self.tempo_min, self.tempo_max)
    #     return np.exp(-0.5 * ((ioi - expected_ioi) / self.tempo_std) ** 2)
    
    def update(self, pitch: np.ndarray, ioi: float):
        # Calculate likelihoods for each particle
        likelihoods = np.zeros(self.num_particles)
        for i in range(self.num_particles):
            # Find the closest state index for the particle's position
            idx = np.argmin(np.abs(self.state_space - self.particles[i, 0]))
            pitch_likelihood = self.pitch_likelihood(pitch, self.pitch_profiles[idx])
            # if idx == 0:
            #     # If we're at the first state, we can't compute timing likelihood based on previous state
            #     timing_likelihood = 1.0
            # else:
            #     timing_likelihood = self.timing_likelihood(ioi, idx)
            
            # likelihoods[i] = pitch_likelihood * timing_likelihood
            likelihoods[i] = pitch_likelihood
        
        # Update weights
        self.weights *= likelihoods
        self.weights += 1e-10  # Avoid zero weights
        self.weights /= np.sum(self.weights)  # Normalize

    def resample(self):
        # Resample particles based on weights
        indices = RNG.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.particles[:, 0] += RNG.normal(0, self.beat_std, self.num_particles)  # Add noise after resampling
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.state_space_min, self.state_space_max)
        self.particles[:, 1] += RNG.normal(0, self.tempo_std, self.num_particles)  # Add tempo noise after resampling
        self.particles[:, 1] = np.clip(self.particles[:, 1], self.tempo_min, self.tempo_max)
        self.weights.fill(1.0 / self.num_particles)  # Reset weights after resampling
    
    def step(self, pitch: np.ndarray, ioi: float):
        self.predict(ioi)
        self._current_position_in_beats = min((np.mean(self.particles[:, 0])), self.state_space_max)  # Ensure we don't go beyond the last state
        self.previous_state = self.current_state
        # Map the current position in beats to the corresponding state index
        self.current_state = np.argmin(np.abs(self.state_space - self._current_position_in_beats))
        self.check_crossing()
        self.update(pitch, ioi)
        self.resample()
        self.tempo_mean = np.mean(self.particles[:, 1])

        return self.current_state
    
    def __call__(
            self, 
            input: tuple[np.ndarray, float],
            *args, **kwargs
            ) -> int:
        pitch, ioi = input
        
        if ioi < IOI_THRESHOLD: 
            # If IOI is very small, we assume it's a continuation of the current chord, irrespective of the following notes.
            # Maybe change this in the future?
            self._current_chord = np.maximum(self._current_chord, pitch)
            return self.current_state  # Return current alignment estimate without updating
        else:
            self._current_chord = pitch
            return self.step(pitch, ioi)
        
    def run(
            self, 
            verbose: bool = True,
            ) -> NDArrayInt:
        same_state_counter = 0
        if verbose:
            pbar = progressbar.ProgressBar(
                maxval=self.n_states,  # redirect_stdout=True
            )
            pbar.start()

        while self.is_still_following():
            prev_state = self.current_state

            queue_input = self.queue.get()
            if queue_input is not None:
                self(queue_input)
                if self.current_state == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0
                    self._warping_path.append(self.current_state)

                if verbose:
                    # current_state may be None (no state yet); guard the update
                    if self.current_state is not None:
                        pbar.update(round(self.current_state))
                yield self.current_state

            if verbose:
                pbar.finish()
        return self._warping_path
