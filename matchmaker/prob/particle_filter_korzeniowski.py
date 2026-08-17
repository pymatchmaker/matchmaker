import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import FRAME_RATE
from matchmaker.features.processor import KorzeniowskiScoreModel
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats

NDArrayFloat = NDArray[np.float32]

SEED = 1984
RNG = np.random.RandomState(SEED)

HOP_SIZE = 1.0 / FRAME_RATE


class KorzeniowskiParticleFilter(OnlineAlignment):
    """
    Rao-Blackwellised Particle Filter described in

    Korzeniowski et al.
    "Tracking Rests and Tempo Changes:
    Improved Score Following with Particle Filters"

    ISMIR 2014.

    Hidden sampled variables
    ------------------------
    x : beat position
    m : note tempo (log2 BPM)
    l : local tempo (log2 BPM)

    Analytically inferred
    ---------------------
    o : onset state
    s : sounding/rest state
    """

    def __init__(
        self,
        score_model,
        observation_type,
        notated_tempo=120.0,
        hop_size=HOP_SIZE,
        queue: Optional[RECVQueue] = None,
        num_particles=500,
    ):
        super().__init__()

        self.score_model = score_model
        self.score_positions = score_model.onset_positions
        self.observation_type = observation_type.lower()

        if self.observation_type not in ("audio", "midi"):
            raise ValueError(
                "observation_type must be "
                "'audio' or 'midi'."
            )

        self.notated_tempo = notated_tempo
        self.hop_size = hop_size

        self.queue = queue
        self.queue_timeout = QUEUE_TIMEOUT

        self.num_particles = num_particles
        self.rng = RNG

        self.initial_logtempo = np.log2(self.notated_tempo)


        # Particle positions (in beats) are initialized to the first score onset.
        self.x = np.full(
            num_particles,
            self.score_positions[0],
            dtype=np.float64,
        )

        # Particle note log-tempi are initialized to the notated tempo with some Gaussian noise.
        self.m = self.initial_logtempo + self.rng.normal(
            0.0,
            0.1,
            self.num_particles,
        )

        # Particle local (slow-moving/averaged) log-tempi are initialized to the notated tempo with some Gaussian noise.
        self.l = self.m.copy()

        # to adjust tempo according to crossed onsets and expected tempo
        # midi has a lower phase gain because the note onsets are more reliable than audio onsets
        # additionally, sustained note frames in midi are identical to each other in midi, 
        # so the phase gain should be lower to avoid overreacting to these frames
        self.phase_gain = 0.45 if self.observation_type == "audio" else 0.05

        # initial particle weights are uniform
        self.weights = np.ones(num_particles) / num_particles


        # strictness of matching expected and observed notes, used in the midi feature likelihood computation
        self.note_sigma = 0.2


        # sigma value while sampling note tempo from a normal distribution around local tempo
        self.sigma_ms = 0.1

        # a certain proportion of particles will sample their note tempo with a wider sigma, to allow for larger tempo deviations.
        # weight_fast determines the proportion, sigma_mf is the wider sigma value.
        self.weight_fast = 0.2
        self.sigma_mf = 0.4

        self.sigma_onset = 0.25

        # effective sample threshold
        self.resample_threshold = num_particles / 2

        self.current_position = self.score_positions[0]
        self.previous_position = self.score_positions[0]

        self.first_score_onset = self.score_positions[0]
        self.input_features = []

        self.first_input_found = False
        self.input_index = 0

        self.previous_time = 0.0

        self._alignment_path = []

        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

        self.sound_prob = np.ones(
            self.num_particles,
        )

        self.rest_prob = np.zeros(
            self.num_particles,
        )

        self.sounding_prob_mu = -30.0 if self.observation_type == "audio" else -15.0
        self.resting_prob_mu = -70.0

        self.onset_match_probability = 0.95

        # Loudness calculations from MIDI are not very reliable, so we do not consider them in the likelihood computation.
        self.feature_weight = 0.70 if self.observation_type == "audio" else 0.80
        self.onset_weight = 0.20
        self.loudness_weight = 0.10 if self.observation_type == "audio" else 0.0

        self.last_onset_time = np.full(
                        self.num_particles,
                        self.previous_time,
                        dtype=np.float64,
                    )
        
        self.small_mvt = (notated_tempo/3) * hop_size / 60.0
        self.large_mvt = (notated_tempo*3) * hop_size / 60.0
        self.notated_mvt = notated_tempo * hop_size / 60.0

    @property
    def warping_path(self):
        return np.array(self._alignment_path).T


    def is_still_following(self):
        return True


    @staticmethod
    def gaussian(x, mu, sigma):
        return (
            np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )


    @staticmethod
    def log2tempo_to_bpm(m):
        return 2.0 ** m


    @staticmethod
    def bpm_to_log2(bpm):
        return np.log2(bpm)
    
    def beat_index(
        self,
        beat: float,
    ) -> int:
        """
        Return the index of the closest beat-grid position.

        Uses binary search (O(log N)) since beat_grid is sorted.
        """

        beat_grid = self.score_model.beat_grid

        idx = np.searchsorted(
            beat_grid,
            beat,
        )

        # Clamp to valid range.
        if idx == 0:
            return 0

        if idx == len(beat_grid):
            return len(beat_grid) - 1

        # Choose the closer of the neighbouring grid points.
        before = idx - 1
        after = idx

        if beat - beat_grid[before] <= beat_grid[after] - beat:
            return before

        return after
    
    def moving_average_tempo(
        self,
        local_tempo,
        note_tempo,
        n=3,
    ):
        """

        Moving average of the tempo in BPM.

        Parameters
        ----------
        n : averaging window.
        """

        bpm_local = self.log2tempo_to_bpm(local_tempo)
        bpm_note = self.log2tempo_to_bpm(note_tempo)

        # For the particles in bpm_note that are 3 times faster than the local tempo, divide by 2, 
        # and for the particles that are 3 times slower than the local tempo, multiply by 2. 
        # This is to prevent the moving average from being skewed by extreme values from the phase gain.
        bpm_note = np.where(
            bpm_note > 3 * bpm_local,
            bpm_note / 2,
            np.where(
                bpm_note < bpm_local / 3,
                bpm_note * 2,
                bpm_note,
            ),
        )

        bpm_new = ((n - 1) * bpm_local + bpm_note) / n

        return self.bpm_to_log2(bpm_new)

    def phase_error(
        self,
        crossed: np.ndarray,
        curr_idx: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the normalized timing error for particles
        that crossed a score onset, using the expected interval 
        from their tempo and the actual time since the last onset.

        Returns
        -------
        ndarray
            Timing error in units of beats, with one value
            for every particle in ``crossed``.
        """

        current_time = self.current_perf_time

        onset_idx = curr_idx[crossed]

        phase = np.zeros(
            len(onset_idx),
            dtype=np.float64,
        )

        valid = (
            (onset_idx > 0)
            & (onset_idx < len(self.score_positions))
        )

        if not np.any(valid):
            return phase

        valid_onset_idx = onset_idx[valid]

        crossed_indices = np.flatnonzero(crossed)
        valid_particle_indices = crossed_indices[valid]

        delta_beats = (
            self.score_positions[valid_onset_idx]
            -
            self.score_positions[valid_onset_idx - 1]
        )

        bps = (
            2.0 ** self.m[valid_particle_indices]
        ) / 60.0

        expected_interval = (
            delta_beats
            / bps
        )

        actual_interval = (
            current_time
            -
            self.last_onset_time[valid_particle_indices]
        )

        phase[valid] = (
            actual_interval
            -
            expected_interval
        ) / (expected_interval * 60)


        self.last_onset_time[valid_particle_indices] = current_time

        return phase
    
    
    def is_rest_position(self, beat):
        idx = self.beat_index(beat)
        return self.score_model.rest_mask[idx]


    def spectral_template(self, idx):
        return self.score_model.templates[idx]


    def nearest_onset(self, idx):
        return self.score_model.nearest_onsets[idx]
    
    def sample_note_tempo(
        self,
        local_tempo,
    ):
        """
        Sample Eq. (11).

        Parameters
        ----------
        local_tempo : ndarray
            Local tempos of particles that crossed
            a score onset.

        Returns
        -------
        ndarray
            Sampled note tempos.
        """

        n = len(local_tempo)

        choose_fast = (
            self.rng.rand(n)
            < self.weight_fast
        )

        sigma = np.where(
            choose_fast,
            self.sigma_mf,
            self.sigma_ms,
        )

        return local_tempo + self.rng.normal(
            loc=0.0,
            scale=sigma,
            size=n,
        )
    
    def predict_position(self):
        """
        Equation (5).

        x_t = x_{t-1}
              + tempo * tau / 60
        """

        bpm = self.log2tempo_to_bpm(self.m)

        delta_beats = (
            bpm
            * (self.current_perf_time - self.previous_time)
            / 60.0
        )

        delta_beats = np.where(
            delta_beats > self.large_mvt,
            self.notated_mvt,
            np.where(
                delta_beats < self.small_mvt,
                self.small_mvt,
                delta_beats,
            )
        )

        previous = self.x.copy()

        self.x += delta_beats

        self.x = np.clip(
            self.x,
            self.score_positions[0],
            self.score_positions[-1],
        )

        return previous
    
    def update_tempi(self, previous_positions):
        """
        Vectorized implementation of Eqs. (8), (10), (12).

        Tempo updates occur only if a score onset was crossed.
        """

        prev_idx = np.searchsorted(
            self.score_positions,
            previous_positions,
            side="right",
        )

        curr_idx = np.searchsorted(
            self.score_positions,
            self.x,
            side="right",
        )

        crossed = curr_idx > prev_idx

        if not np.any(crossed):
            return

        phase = self.phase_error(
            crossed,
            curr_idx,
        )

        self.m[crossed] += self.phase_gain * phase

        self.l[crossed] = self.moving_average_tempo(
            self.l[crossed],
            self.m[crossed],
        )

        self.m[crossed] = self.sample_note_tempo(
            self.l[crossed]
        )

    
    def predict(self):
        """
        State transition.

        1. propagate score position

        2. update local tempo

        3. sample note tempo
        """

        previous_positions = self.predict_position()

        self.update_tempi(previous_positions)

    def propagate_sound_state(self):
        """
        Rao-Blackwell update of Table 2.

        Computes

            P(s_t)

        recursively.
        """

        sounding = np.empty_like(
            self.sound_prob
        )

        resting = np.empty_like(
            self.rest_prob
        )

        rest_mask = np.array(
            [
                self.is_rest_position(x)
                for x in self.x
            ]
        )

        # sounding score positions

        sounding[~rest_mask] = 1.0

        resting[~rest_mask] = 0.0

        # rests
        sounding[rest_mask] = (
            0.8
            * self.sound_prob[rest_mask]
        )

        resting[rest_mask] = (
            0.2
            * self.sound_prob[rest_mask]
            +
            self.rest_prob[rest_mask]
        )

        total = sounding + resting

        self.sound_prob = sounding / total

        self.rest_prob = resting / total
    
    def onset_probability(
        self,
        beat,
        idx,
    ):
        """
        Equation (15).
        """

        onset = self.nearest_onset(idx)

        d = beat - onset

        return np.exp(
            -(d ** 2)
            /
            (2 * self.sigma_onset ** 2)
        )
    
    def estimate_position(self):
        """
        MAP estimate.

        Mean of best 20% particles.
        """

        n = max(
            1,
            int(0.20 * self.num_particles),
        )

        idx = np.argsort(
            self.weights
        )[-n:]

        return float(
            np.mean(self.x[idx])
        )
    
    def spectral_probability(
        self,
        spectrum,
        beat,
        idx,
    ):
        """
        Eq. (14).

        Assumes both vectors are normalized.
        """

        if self.is_rest_position(beat):
            return 1.0

        template = self.spectral_template(idx)

        corr = np.dot(
            spectrum,
            template,
        )

        return max(0.0, corr)
    
    def note_probability(
        self,
        observed_notes,
        beat,
        idx,
    ):
        """
        Compute the symbolic note likelihood.
        """

        expected_notes = self.score_model.active_notes[idx]

        expected = set(expected_notes.tolist())
        observed = set(observed_notes.tolist())

        # Silence.
        if not expected and not observed:
            return 1.0

        if not expected or not observed:
            return 0.0

        tp = len(expected & observed)

        precision = tp / len(observed)

        recall = tp / len(expected)

        if precision + recall == 0:
            return 0.0

        similarity = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

        return np.exp(
            -0.5
            * (
                (1.0 - similarity)
                / self.note_sigma
            ) ** 2
        )
    
    def onset_observation_probability(
        self,
        onset_feature,
        onset_present,
    ):
        """
        Implements Eq. (17).

        onset_feature in [0,1]
        """

        if self.observation_type == "audio":
            k = 10.0

            if onset_present:

                return 0.5 * (
                    1.0 +
                    np.tanh(
                        k * (onset_feature - 0.5)
                    )
                )

            return 0.5 * (
                1.0 +
                np.tanh(
                    k * (0.5 - onset_feature)
                )
            )
        
        else:
            observed_onset = len(onset_feature) > 0

            if onset_present:

                return (
                    self.onset_match_probability
                    if observed_onset
                    else 1.0 - self.onset_match_probability
                )

            return (
                1.0 - self.onset_match_probability
                if observed_onset
                else self.onset_match_probability
            )
    
    def feature_probability(
        self,
        feature,
        beat,
        idx,
    ):
        """
        Compute the observation feature likelihood.

        For audio observations this is the spectral likelihood
        (Eq. 14).

        For MIDI observations this is the note-set likelihood.
        """

        if self.observation_type == "audio":

            return self.spectral_probability(
                feature,
                beat,
                idx,
            )

        elif self.observation_type == "midi":

            return self.note_probability(
                feature,
                beat,
                idx,
            )

        raise ValueError(
            f"Unknown observation type "
            f"{self.observation_type}"
        )
    
    
    def onset_probability_feature(
        self,
        onset_feature,
        beat,
        idx,
    ):
        """
        Equation (18).
        """

        p_onset = self.onset_probability(beat=beat, idx=idx)

        p_no = 1.0 - p_onset

        return (
            self.onset_observation_probability(
                onset_feature,
                True,
            ) * p_onset
            +
            self.onset_observation_probability(
                onset_feature,
                False,
            ) * p_no
        )
    
    def loudness_probability(
        self,
        loudness,
        particle,
    ):
        """
        Marginalized loudness likelihood.

        Implements Eqs. (19)-(20).
        """

        ps = self.gaussian(
            loudness,
            self.sounding_prob_mu,
            8,
        )

        pr = self.gaussian(
            loudness,
            self.resting_prob_mu,
            8,
        )

        return (
            self.sound_prob[particle] * ps
            +
            self.rest_prob[particle] * pr
        )
    
    def compute_weights(
        self,
        observation,
    ):
        """
        Equation (13).

        Computes the particle weights from the observation
        likelihood.

        Audio:
            spectrum × onset × loudness

        MIDI:
            note-set × onset × loudness
        """

        likelihood = np.empty(
            self.num_particles,
            dtype=np.float64,
        )

        loudness = observation.loudness

        if self.observation_type == "audio":

            feature = observation.spectrum
            onset = observation.onset

        else:

            feature = observation.active_notes
            onset = observation.onset_notes

        for i in range(self.num_particles):

            beat = self.x[i]
            idx = self.beat_index(beat)


            # Feature likelihood
            pf = self.feature_probability(
                feature,
                beat,
                idx,
            )

            # Onset likelihood
            po = self.onset_probability_feature(
                onset,
                beat,
                idx,
            )

            # Loudness likelihood
            pl = self.loudness_probability(
                loudness,
                i,
            )


            likelihood[i] = np.exp(
                self.feature_weight * np.log(max(pf, 1e-300))
                + self.onset_weight * np.log(max(po, 1e-300))
                + self.loudness_weight * np.log(max(pl, 1e-300))
            )

        self.weights *= likelihood

        self.weights += 1e-300

        self.weights /= np.sum(
            self.weights
        )

    def effective_sample_size(self):

        return 1.0 / np.sum(
            self.weights ** 2
        )
    
    def systematic_resample(self):
        """
        Low-variance systematic resampling.
        """

        positions = (
            self.rng.rand()
            +
            np.arange(self.num_particles)
        ) / self.num_particles

        cumulative = np.cumsum(
            self.weights
        )

        indices = np.searchsorted(
            cumulative,
            positions,
        )

        self.x = self.x[indices]
        self.m = self.m[indices]
        self.l = self.l[indices]

        self.sound_prob = self.sound_prob[indices]
        self.rest_prob = self.rest_prob[indices]

        self.weights.fill(
            1.0 / self.num_particles
        )

    def step(
        self,
        observation,
    ):
        """
        One iteration of the Rao-Blackwellised
        particle filter.
        """

        if self.first_input_found is False:
            if self.observation_type == "midi":
                if len(observation.active_notes) == 0:
                    # no active notes in this frame, skip update
                    self._alignment_path.append(
                        (self.current_perf_time, self.current_position)
                    )
                    return self.current_position
                else:
                    self.first_input_found = True
            else:
                if observation.loudness < -60.0:
                    # no sound in this frame, skip update
                    self._alignment_path.append(
                        (self.current_perf_time, self.current_position)
                    )
                    return self.current_position
                else:
                    self.first_input_found = True


        self.predict()

        self.propagate_sound_state()

        self.compute_weights(observation)

        estimate = self.estimate_position()

        if (
            self.effective_sample_size()
            <
            self.resample_threshold
        ):

            self.systematic_resample()

        self.previous_position = self.current_position
        self.current_position = estimate

        return estimate
    
    
    def get_current_position(self):
        return self.current_position


    def __call__(
        self,
        observation,
        perf_time,
    ):
        t0 = time.time()

        beat = super().__call__(
            observation,
            perf_time,
        )

        self.latency_stats = set_latency_stats(
            time.time() - t0,
            self.latency_stats,
            self.input_index,
        )

        self.input_index += 1
        self.previous_time = perf_time

        return beat


    def run(
        self,
        verbose=True,
    ):

        return (
            yield from
            super().run(
                verbose=verbose
            )
        )