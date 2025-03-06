#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module implements Particle Filter models for score following
"""
from typing import Any, Generator, List, Optional, Tuple, Union
import numpy as np

from matchmaker.base import OnlineAlignment
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.tempo_models import TempoModel
from matchmaker.utils.typing import NDArrayFloat, NDArrayInt
from matchmaker.utils.math import normal_density, discrete_normal_density

RNG = np.random.RandomState(1984)


# reconstructions from estimated score positions
def reconstruct_onset_times(score, est_info):
    """This function is to match onsets from the performance to score positions
    It deals with insertions and skips to be able to plot simultaneously the estimated performance and the score
    """
    N = score.shape[0]
    est_onsets = np.NAN * np.ones(
        (N, 2)
    )  # there will be NAN if the note is not detected
    for i in range(N):
        est_onsets[i, 1] = i
        est_p = np.where(est_info[:, 1] == i)[0]
        if len(est_p):
            est_onsets[i, 0] = est_info[est_p[0], 0]
    return est_onsets


def reconstruct_tempi(score, est_info):
    N = score.shape[0]
    est_tempi = np.NAN * np.ones(
        (N, 2)
    )  # there will be NAN if the note is not detected
    for i in range(N):
        est_tempi[i, 1] = i
        est_p = np.where(est_info[:, 1] == i)[0]
        if len(est_p):
            est_tempi[i, 0] = est_info[est_p[0], 2]
    return est_tempi


def reconstruct_pitches(score, est_idx):
    Nperf = est_idx.shape[0]
    est_pitches = np.NAN * np.ones(Nperf)  # there will be NAN at each insertion
    for i in range(Nperf):
        if est_idx[i] % 1 == 0:
            est_pitches[i] = score[int(est_idx[i]), 2]
    return est_pitches


def reconstruct_onsets_beats(score, est_idx):
    Nperf = est_idx.shape[0]
    est_onsets = np.NAN * np.ones(Nperf)  # there will be NAN at each insertion
    for i in range(Nperf):
        if est_idx[i] % 1 == 0:
            est_onsets[i] = score[int(est_idx[i]), 0]
    return est_onsets


def get_inserts_and_skips(est_idx):
    Nperf = est_idx.shape[0]
    # save inserts and skips
    out = np.ones(Nperf)
    for i in range(Nperf):
        if est_idx[i] % 1:
            out[i] = 0
        else:
            if i > 0 and est_idx[i] - est_idx[i - 1] > 1:
                out[i] = 2
    return out


def get_real_performance_positions(perf):
    Nperf = perf.shape[0]
    real_perf_idx = np.zeros(Nperf)
    count = 0
    for i in range(Nperf):
        if perf[i, 3] == 1:
            real_perf_idx[i] = count
            count += 1
        if perf[i, 3] == 0:
            real_perf_idx[i] = count - 0.5
        if perf[i, 3] == 2:
            real_perf_idx[i] = count + 1
            count += 2

    return real_perf_idx


def get_real_onset_times(score, perf):
    N = score.shape[0]
    real_perf_idx = get_real_performance_positions(perf)

    obs_score_onsets = np.NAN * np.ones(
        (N, 2)
    )  # there will be NAN if the note is not played
    for i in range(N):
        obs_score_onsets[i, 1] = i
        obs_p = np.where(real_perf_idx == i)[0]
        if len(obs_p):
            obs_score_onsets[i, 0] = perf[obs_p[0], 0]

    return obs_score_onsets


def get_real_tempi(score, perf):
    N = score.shape[0]
    obs_score_onsets = get_real_onset_times(score, perf)
    obs_score_tempi = obs_score_onsets.copy()
    for i in range(1, N):
        last_played_note_idx = np.where(np.isfinite(obs_score_onsets[:i, 0]))[0][-1]
        obs_score_tempi[i, 0] = (
            60
            * (score[i, 0] - score[last_played_note_idx, 0])
            / (obs_score_onsets[i, 0] - obs_score_onsets[last_played_note_idx, 0])
        )
    obs_score_tempi[0, 0] = obs_score_tempi[1, 0]
    return obs_score_tempi


def get_all_estimation_info(score, perf, est_idx, est_tempi, last_idx=None):
    out = np.zeros((perf.shape[0], 6))
    out[:, 0] = perf[:, 0]
    out[:, 1] = est_idx
    out[:, 2] = est_tempi
    out[:, 3] = reconstruct_onsets_beats(score, est_idx)
    out[:, 4] = reconstruct_pitches(score, est_idx)
    if perf.shape[0] > 1:
        out[:, 5] = get_inserts_and_skips(est_idx)
    else:
        if (
            last_idx is None or est_idx == last_idx + 1
        ):  # the behaviour represents 0 for insertions,
            # 1 for normal and 2 for skips
            out[:, 5] = 1
        elif est_idx > last_idx + 1:
            out[:, 5] = 2

    return out  # [onset (s), position index, tempo, onset (beats), pitch, behaviour]


class InitialModel(object):
    """
    TODO: Move this class to hiddenmarkov
    """

    def generate(self, n_particles: int):
        raise NotImplementedError("must be implemented by subclass.")


class TransitionModel(object):
    """
    TODO: Move this class to hiddenmarkov
    """

    dim_state: int

    def generate(self, last_state):
        raise NotImplementedError("must be implemented by subclass.")


class ObservationModel(object):
    """
    TODO: Move this class to hiddenmarkov
    """

    def density(self, observation, hidden_state):
        raise NotImplementedError("must be implemented by subclass.")


class BootstrapFilter(object):
    """
    Class to perform particle filtering using the bootstrap filter algorithm
    Parameters
    ----------
    initial_model: `InitialModel`
        instance to generate initial particles

    transition_model: `TransitionModel`
        instance to generate next particles from current ones

    observation_model: `ObservationModel`
        instance to get the observation densities to compute the weights

    n_particles: int,
        number of particles

    resampling_rate: int,
        resampling will be done every `resampling_rate` step

    TODO: Move this class to hiddenmarkov

    """

    def __init__(
        self,
        initial_model: InitialModel,
        transition_model: TransitionModel,
        observation_model: ObservationModel,
        n_particles: int,
        resampling_rate: int = 1,
        rng: np.random.RandomState = RNG,
    ) -> None:
        self.initial_model = initial_model
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.n_particles = n_particles
        self.resampling_rate = resampling_rate
        self.rng = rng
        self.counter = 0

    def process_step(
        self,
        observation: Any,
        last_particles: np.ndarray,
        resample: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:

        particles = np.zeros((self.n_particles, self.transition_model.dim_state))
        weights = np.zeros(self.n_particles)

        # TODO: use list comprehension?
        for i in range(self.n_particles):
            particles[i, :] = self.transition_model.generate(last_particles[i, :])
            weights[i] = self.observation_model.density(observation, particles[i, :])

        # normalization

        weights = weights / sum(weights)

        if resample:
            neff = 1.0 / sum(weights**2)
            if (
                self.counter % self.resampling_rate == 0
                or neff < self.n_particles / 10.0
            ):
                particles, weights = self.resample(particles, weights)

        self.counter += 1

        return particles, weights

    def resample(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample particles
        """
        new_idx = self.rng.choice(
            np.arange(self.n_particles),
            self.n_particles,
            p=weights,
        )
        new_particles = particles[new_idx, :]
        new_weights = 1.0 / self.n_particles * np.ones(self.n_particles)
        return new_particles, new_weights

    def init_particles(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize particles and weights
        """
        particles = self.initial_model.generate(self.n_particles)
        weights = 1.0 / self.n_particles * np.ones(self.n_particles)
        return particles, weights

    def process(self, observations: np.ndarray):
        """
        Batch process observations

        Question. Do we need this method?
        """
        T = observations.shape[0]
        all_particles = np.zeros((T, self.n_particles, self.transition_model.dim_state))
        all_weights = np.zeros((T, self.n_particles))

        particles, weights = self.init_particles()
        for t in range(T):
            particles, weights = self.process_step(
                observations[t], particles, resample=True
            )
            # neff = 1.0 / sum(weights**2)
            # if t % self.resampling_rate == 0 or neff < self.n_particles / 10.0:
            #     particles, weights = self.resample(particles, weights)
            all_particles[t] = particles
            all_weights[t] = weights
        return all_particles, all_weights


class BasePF(BootstrapFilter):
    """
    Base Particle Filter for Score Follwing

    TODO:
    * Add interface for tempo model (for now, tempo_model instances are ignored)
    """

    observation_model: ObservationModel
    transition_model: TransitionModel
    initial_model: InitialModel
    tempo_model: Optional[TempoModel]
    has_insertions: bool
    # TODO: double check order of the warping path
    _warping_path: List[Tuple[float, int]]
    queue: Optional[RECVQueue]

    def __init__(
        self,
        observation_model: ObservationModel,
        transition_model: TransitionModel,
        initial_model: InitialModel,
        state_space: Union[NDArrayFloat, NDArrayInt],
        n_particles: int = 1000,
        resampling_rate: int = 1,
        tempo_model: Optional[TempoModel] = None,
        has_insertions: bool = False,
        queue: Optional[RECVQueue] = None,
        patience: int = 10,
        init_tempo: float = 0.5,
    ) -> None:

        BootstrapFilter.__init__(
            self,
            observation_model=observation_model,
            transition_model=transition_model,
            initial_model=initial_model,
            n_particles=n_particles,
            resampling_rate=resampling_rate,
        )

        self.state_space = state_space
        self.tempo_model = tempo_model
        self.has_insertions = has_insertions
        self.input_index = 0
        self._warping_path = []
        self.queue = queue
        self.patience = patience
        self.current_state = 0.0

        self.last_particles, self.last_weights = self.init_particles()

        self.est_tempo = init_tempo

        self.n_states = len(state_space)

    @property
    def warping_path(self) -> NDArrayInt:
        # TODO: how to adapt this correctly for non-integer positions
        return (np.array(self._warping_path).T).astype(np.int32)

    def __call__(self, input: NDArrayFloat) -> float:
        """
        TODO: Cythonize!!!
        """
        # Particles represent position in time, tempo
        particles, weights = self.process_step(
            observation=input,
            last_particles=self.last_particles,
        )

        # Potential check: Ensure that the generated particles are a 2D array
        # If they are a 3D array, they are being processed in batch form
        self.est_tempo = np.dot(weights, particles[:, 1])
        unique_states = np.unique(particles[:, 0])

        # sum all weights for given position
        w = np.array([np.sum(weights[particles[:, 0] == us]) for us in unique_states])
        current_state = unique_states[np.argmax(w)]

        self._warping_path.append((current_state, self.input_index))
        self.input_index += 1
        self.current_state = current_state

        return current_state

    def run(self) -> Generator[int, None, NDArrayInt]:
        """
        TODO: adapt this for the new run_offline interface.
        """
        if self.queue is not None:
            prev_state = self.current_state
            same_state_counter = 0
            while self.is_still_following():
                target_feature = self.queue.get()

                current_state = self(target_feature)

                # TODO: Update for graceful termination
                if current_state == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0

                yield current_state

            return self.warping_path

    def is_still_following(self) -> bool:
        """
        TODO: Adapt for the new run offline interface
        """
        if self.current_state is not None:
            return self.current_state <= self.n_states - 1

        return False


# TODO: Remove this function?
def generate_pitch_from_position(score, i):
    max_value = score.shape[0] - 1
    if i % 1:
        return -score[min([int(i + 1), max_value]), 2]
    else:
        return score[min([int(i), max_value]), 2]


# initial model to generate first particles
class MonophonicPitchInitialModel(InitialModel):
    def __init__(
        self,
        pitch: np.ndarray,
        unique_onsets: np.ndarray,
        min_bpm: float,
        max_bpm: float,
        initial_position: Optional[int] = None,
        rng: np.random.RandomState = RNG,
    ) -> None:
        self.pitch = pitch
        self.unique_onsets = unique_onsets
        self.n_states = len(pitch)
        self.min_tempo = 60.0 / max_bpm
        self.max_tempo = 60.0 / min_bpm
        self.initial_position = initial_position
        self.rng = rng

        # Question: why do we need the range to be between -1 and n_states?
        # Because -1 is setting the position to before the start
        self.initial_position_choices = np.arange(-1, self.n_states)

        self.state_dtype = np.dtype(
            [
                ("state", float),
                ("tempo", float),
                ("pitch", int),
                ("ref_time", float),
                ("perf_time", float),
            ]
        )

    def generate(self, n_particles: int) -> np.ndarray:

        out = np.zeros(n_particles, self.state_dtype)

        # position (index), tempo in beats per minute, d, onset, pitch,
        # i -> position (index of the state but continuous) (float)
        # v -> tempo in beats per minute (float)
        # d -> position in time (in beats) float
        # p -> pitch (int represeting MIDI pitch)
        # t -> position in time (in seconds) (float)
        for k in range(n_particles):
            # generate a position from the score positions
            if self.initial_position is not None:
                i = self.initial_position
            else:
                i = self.rng.choice(self.initial_position_choices)
            # generate a random tempo
            v = self.min_tempo + self.rng.rand() * (self.max_tempo - self.min_tempo)

            #
            o = self.unique_onsets[max([int(i), 0])]
            d = 0
            # p = generate_pitch_from_position(self.score, i)
            # Question: why p=0? Just for initialization?
            p = 0
            t = o * v

            out[k]["state"] = i
            out[k]["tempo"]= v
            out[k]["pitch"] = p
            out[k]["ref_time"] = d
            out[k]["perf_time"] = t

        return out


# initial model to generate first particles
class PolyphonicPitchInitialModel(InitialModel):
    def __init__(
        self,
        pitch: np.ndarray,
        unique_onsets: np.ndarray,
        min_bpm: float,
        max_bpm: float,
        initial_position: Optional[int] = None,
        rng: np.random.RandomState = RNG,
    ) -> None:
        self.pitch = pitch
        self.unique_onsets = unique_onsets
        self.n_states = len(pitch)
        self.min_tempo = 60.0 / max_bpm
        self.max_tempo = 60.0 / min_bpm
        self.initial_position = initial_position
        self.rng = rng

        # Question: why do we need the range to be between -1 and n_states?
        # Because -1 is setting the position to before the start
        self.initial_position_choices = np.arange(-1, self.n_states)

        self.state_dtype = np.dtype(
            [
                ("state", float),
                ("tempo", float),
                # Pitch is numpy array of different length (also potentially empty)
                ("pitch", object),
                ("ref_time", float),
                ("perf_time", float),
            ]
        )

    def generate(self, n_particles: int) -> np.ndarray:

        out = np.zeros(n_particles, self.state_dtype)

        # position (index), tempo in beats per minute, d, onset, pitch,
        # i -> position (index of the state but continuous) (float)
        # v -> tempo in beats per minute (float)
        # d -> position in time (in beats) float
        # p -> pitch (int represeting MIDI pitch)
        # t -> position in time (in seconds) (float)
        for k in range(n_particles):
            # generate a position from the score positions
            if self.initial_position is not None:
                i = self.initial_position
            else:
                i = self.rng.choice(self.initial_position_choices)
            # generate a random tempo
            v = self.min_tempo + self.rng.rand() * (self.max_tempo - self.min_tempo)

            #
            o = self.unique_onsets[max([int(i), 0])]
            d = 0
            # p = generate_pitch_from_position(self.score, i)
            # Question: why p=0? Just for initialization?
            p = np.array([0])
            t = o * v

            out[k]["state"] = i
            out[k]["tempo"]= v
            out[k]["pitch"] = p
            out[k]["ref_time"] = d
            out[k]["perf_time"] = t

        return out




def generate_next_perf_id(i, insert_prob, skip_prob, max_value):
    """
    Sample IDs of the performance


    """

    if i % 1:
        # Case for insertion
        return np.random.choice(
            [i, min([int(i + 1), max_value]), min([int(i + 2), max_value])],
            p=[insert_prob, 1 - insert_prob - skip_prob, skip_prob],
        )
    else:
        # Case of a normal position
        return np.random.choice(
            [
                min([i + 0.5, max_value]),
                min([i + 1, max_value]),
                min([i + 2, max_value]),
            ],
            p=[insert_prob, 1 - insert_prob - skip_prob, skip_prob],
        )


# transition model to generate particles from last ones
class NotesTransitionModel(TransitionModel):
    def __init__(
        self,
        score: np.ndarray,
        insert_prob: float,
        skip_prob: float,
        inserted_ioi_coeff: float,
        tempo_std: float,
    ) -> None:

        # A numpy array, dimension 0 is the onset time in beats
        self.score = score

        # probability of insertion
        self.insert_prob = insert_prob

        # probability of skips
        self.skip_prob = skip_prob

        # What is this? float?
        self.inserted_ioi_coeff = inserted_ioi_coeff

        # Tempo standard deviation
        self.tempo_std = tempo_std
        self.dim_state = 5

    def generate(self, last_state: np.ndarray):
        i, v, p, d, t = last_state
        # generate next position
        if i == -1:
            inew = 0
        else:
            inew = generate_next_perf_id(
                i=i,
                insert_prob=self.insert_prob,
                skip_prob=self.skip_prob,
                max_value=self.score.shape[0] - 1,
            )

        # generate next tempo
        vnew = np.random.normal(v, self.tempo_std)
        # vnew = 2**(np.random.normal(np.log2(v), self.tempo_std))

        # compute next pitch
        all_notes = np.arange(21, 109)
        if inew >= self.score.shape[0] - 1:
            pnew = np.random.choice(all_notes)
        elif inew % 1:
            possible_notes = all_notes[all_notes != self.score[int(inew + 1), 2]]
            # possible_notes = all_notes
            pnew = np.random.choice(possible_notes)
        else:
            pnew = self.score[int(inew), 2]

        # compute next ioi
        # Why the -d? (the starting d is 0 according to the NoteInitialModel)
        max_ioi = (
            self.score[int(np.ceil(inew)), 0] - self.score[max([int(i), 0]), 0] - d
        )
        if inew % 1:
            eps = self.inserted_ioi_coeff * max_ioi
            ioinew = np.random.uniform(eps, max_ioi - eps)
            dnew = ioinew + d
        else:
            ioinew = max_ioi
            dnew = 0
        tnew = t + v * ioinew

        return np.array([inew, vnew, pnew, dnew, tnew])


class NotesObservationModel(ObservationModel):
    def __init__(self, pitch_std, onset_std):
        self.pitch_std = pitch_std
        self.onset_std = onset_std

    def density(self, observation, hidden_state):

        onset_obs, pitch_obs = observation
        i, v, p, d, t = hidden_state

        # Pitch observation density
        pitches_density = discrete_normal_density(
            pitch_obs, p, self.pitch_std**2, grid=np.arange(21, 109)
        )

        # performed onset time observation density
        onsets_density = normal_density(onset_obs, t, self.onset_std**2)
        return onsets_density * pitches_density


# Particle filtering (i.e. sequential monte carlo) : Bootstrap filter algorithm
class BootstrapScoreFollowingProcessor(OnlineAlignment):
    """

    Parameters
    ----------
    score: array (Nscore, 2)
        Array representing the score notes as [onset (beats), pitch].
    min_bpm : float, optional
        Minimum tempo used for beat tracking [bpm].
    max_bpm : float, optional
        Maximum tempo used for beat tracking [bpm].
    initial_position: int
        position for initial particles
    insert_prob: float, optional
        Probability of inserting a note in the performance.
    skip_prob: float, optional
        Probability of skipping a note in the performance.
    inserted_ioi_coeff: float
        coefficient for the density of inserted iois
    tempo_std: float
        standard deviation for tempo
    pitch_std: float
        standard deviation for observed pitches
    onset_std: float
        standard deviation for observed onsets
    n_particles: int
        number of particles
    resampling_rate: int,
        resampling will be done every `resampling_rate` step
    """

    # initial model parameters
    INIT_TEMPO_RANGE = [50, 100]
    MIN_BPM = 5.0  # beats / min
    MAX_BPM = 500.0  # beats / min
    INIT_POS = None
    # transition model parameters
    INSERT_PROB = 0.25
    SKIP_PROB = 0.1
    INSERTED_IOI_COEFF = 0.05  # sec
    TEMPO_STD = 0.1  # sec / beats
    # observation model parameters
    PITCH_STD = 1.0 / 3  # midi id
    ONSET_STD = 0.3  # sec
    # SMC parameters
    N_PARTICLES = 1000
    RESAMPLING_RATE = 5

    def __init__(
        self,
        score,
        initial_tempo_range=INIT_TEMPO_RANGE,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
        initial_position=INIT_POS,
        insert_prob=INSERT_PROB,
        skip_prob=SKIP_PROB,
        inserted_ioi_coeff=INSERTED_IOI_COEFF,
        tempo_std=TEMPO_STD,
        pitch_std=PITCH_STD,
        onset_std=ONSET_STD,
        n_particles=N_PARTICLES,
        resampling_rate=RESAMPLING_RATE,
        online=False,
    ):

        self.im = NotesInitialModel(score, initial_tempo_range, initial_position)
        # transition model
        self.tm = NotesTransitionModel(
            score,
            insert_prob,
            skip_prob,
            inserted_ioi_coeff,
            tempo_std,
            min_bpm,
            max_bpm,
        )
        # observation model
        self.om = NotesObservationModel(pitch_std, onset_std)
        # particle filter (sequential monte carlo)
        self.smc = BootstrapFilter(
            self.im,
            self.tm,
            self.om,
            n_particles=n_particles,
            resampling_rate=resampling_rate,
        )
        # keep state in online mode
        self.online = online
        # save score
        self.score = score

        # initialise particles for online mode
        self.last_particles, self.last_weights = self.smc.init_particles()
        self.last_position = None

    def process_step(self, observation):
        obs = observation[1:]  # remove onsets
        particles, weights = self.smc.process_step(obs, self.last_particles)
        est_positions, est_tempi = self.get_position_and_tempo_from_particles(
            particles, weights
        )

        self.last_particles = particles
        self.last_weights = weights

        out = get_all_estimation_info(
            self.score,
            np.atleast_2d(observation),
            est_positions,
            est_tempi,
            self.last_position,
        )
        self.last_position = est_positions

        return out

    def process_sequence(self, observations):
        obs = observations[:, [0, 2]]  # remove iois
        particles, weights = self.smc.process(obs)
        est_positions, est_tempi = self.get_position_and_tempo_from_particles(
            particles, weights
        )

        return get_all_estimation_info(
            self.score, observations, est_positions, est_tempi
        )

    def process(self, observations, **kwargs):
        if self.online:
            return self.process_step(observations)
        else:
            return self.process_sequence(observations)

    def get_position_and_tempo_from_particles(self, particles, weights):
        """
        Since the positions take place in a discrete space (integers of half-integers), we can't just take
        the mean of the particles because this would lead to a non integer or half-integer
        position. Therefore we take the position which has the maximum sum of weights
        :return: the estimated positions and tempi
        """
        if particles.ndim == 2:  # this means only one time step
            # just add a dimension
            particles = particles[np.newaxis, :]
            weights = weights[np.newaxis, :]
        T, _, dim_state = particles.shape
        est_positions = np.zeros(T)
        est_tempi = np.zeros(T)
        for t in range(T):
            # for the tempi take the mean
            est_tempi[t] = np.dot(weights[t, :], particles[t, :, 1])
            # get all the possible positions given by the particles
            unique_pos = np.unique(particles[t, :, 0])
            w = np.zeros_like(unique_pos)
            for i in range(len(unique_pos)):
                # sum all weights for given position
                w[i] = np.sum(weights[t, particles[t, :, 0] == unique_pos[i]])
            # get the position which have the maximum weight
            est_positions[t] = unique_pos[np.argmax(w)]

        return est_positions, 60.0 / est_tempi
