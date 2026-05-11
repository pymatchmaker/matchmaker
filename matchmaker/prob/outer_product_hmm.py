from typing import List, Optional, Union

import progressbar
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END

try:
    # import the compiled function (name depends on your .pyx)
    from viterbi_step import viterbi_step_cy
except Exception:
    viterbi_step_cy = None

import numpy as np
from partitura.score import Part, Score, ScoreLike

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

DEFAULT_PITCH_ERROR_PROBS = {
    "correct_pitch_prob": 0.9497,
    "semi_tone_error_prob": 0.0145 / 2.0,
    "whole_tone_error_prob": 0.0224 / 2.0,
    "octave_error_prob": 0.0047 / 2.0,
    "within_one_octave_error_prob": 0.0086 / 9.0 / 2.0,
}

DEFAULT_TRANSITIONS = [
    (-3, 0.00509),
    (-2, 0.00516),
    (-1, 0.00886),
    (0, 0.01342),
    (1, 0.94531),
    (2, 0.00610),
    (3, 0.00073),
]

DEFAULT_D1 = 3
DEFAULT_D2 = 3

IOI_THRESHOLD = 0.035  # seconds


def compute_OuterProductHMM_pitch_probabilities(
    chords: List[set],
    pitch_error_probs: dict = None,
    other_prob: float = 1e-6,
) -> NDArrayFloat:
    """
    Precompute emission probabilities corresponding to neighbouring pitches for OuterProductHMM states.
    This function takes into consideration pitch errors such as semitone, whole tone, octave, and within one octave errors.

    Parameters
    ----------
    chords : list of sets
        chords[i] contains MIDI pitches (0–127) for score chord at state i.
        A chord is defined as all notes with the same onset time.
    pitch_error_probs : dict or None
        If None, uses DEFAULT_PITCH_ERROR_PROBS. These are the probabilities assigned to different pitch error categories.
    other_prob : float
        Probability assigned to any pitch not falling into error categories. Default is 1e-6.

    Returns
    -------
    b_table : ndarray (N x 128)
        b_table[i, p] = probability of observing pitch p at state i.
    """

    if pitch_error_probs is None:
        pitch_error_probs = DEFAULT_PITCH_ERROR_PROBS

    N = len(chords)
    max_pitch = 128
    b_table = np.full((N, max_pitch), other_prob, dtype=float)

    for i, chord in enumerate(chords):
        if not chord:
            continue
        correct = set(chord)
        semi = {p + 1 for p in chord if 0 <= p + 1 < max_pitch} | {
            p - 1 for p in chord if 0 <= p - 1 < max_pitch
        }
        semi -= correct
        whole = {p + 2 for p in chord if 0 <= p + 2 < max_pitch} | {
            p - 2 for p in chord if 0 <= p - 2 < max_pitch
        }
        whole -= correct | semi
        octv = {p + 12 for p in chord if 0 <= p + 12 < max_pitch} | {
            p - 12 for p in chord if 0 <= p - 12 < max_pitch
        }
        octv -= correct | semi | whole
        within_oct = {
            x
            for p in chord
            for x in range(p - 11, p + 12)
            if 0 <= x < max_pitch
            and x not in correct
            and x not in semi
            and x not in whole
            and x not in octv
        }

        probs = pitch_error_probs
        for p in correct:
            b_table[i, p] = probs["correct_pitch_prob"] / len(correct)
        for p in semi:
            b_table[i, p] = probs["semi_tone_error_prob"] / max(1, len(semi))
        for p in whole:
            b_table[i, p] = probs["whole_tone_error_prob"] / max(1, len(whole))
        for p in octv:
            b_table[i, p] = probs["octave_error_prob"] / max(1, len(octv))
        for p in within_oct:
            b_table[i, p] = probs["within_one_octave_error_prob"] / max(
                1, len(within_oct)
            )
    return b_table


def get_chords_from_score(
    score: ScoreLike,
    return_unique_onsets: bool = False,
) -> List[set]:
    """
    Extract chords from a score-like object.
    A chord is defined as all notes with the same onset time.

    Parameters
    ----------
    score : ScoreLike
        The score-like object to extract chords from.

    return_unique_onsets : bool
        If True, also return the unique onset times.

    Returns
    -------
    List[set]
        A list of sets, each containing the MIDI pitches for a chord.
    """

    if isinstance(score, (Score, Part)):
        note_array = score.note_array()

    if isinstance(score, np.ndarray):
        note_array = score

        if "onset_beat" not in note_array.dtype.names:
            raise ValueError("`score` is not a valid note array")

    # This code does not handle ornaments
    # We are using score-like objects, but we might want to have this to be more general

    unique_onsets = np.unique(note_array["onset_beat"])

    unique_onset_idxs = [
        np.where(note_array["onset_beat"] == uo)[0] for uo in unique_onsets
    ]

    chords = [set(note_array["pitch"][ui]) for ui in unique_onset_idxs]

    if return_unique_onsets:
        return chords, unique_onsets
    else:
        return chords


def compute_transition_matrix(
    N: int,
    transitions: list[tuple[int, float]] = None,
    D1: int = DEFAULT_D1,
    D2: int = DEFAULT_D2,
) -> tuple[NDArrayFloat, int, int]:
    """
    Construct banded transition matrix (α) from transition deltas and probabilities.

    Parameters
    ----------
    N : int
        Number of score states (chords)
    transitions : list of (delta, prob) or None
        If None, uses DEFAULT_TRANSITIONS.

    Returns
    -------
    alpha : ndarray (N x N)
        α[i,j] = probability of transitioning from state i -> j (banded structure)
    D1, D2 : int
        Fixed neighbourhood sizes (default 3)
    """
    if transitions is None:
        transitions = DEFAULT_TRANSITIONS

    # intialize transition matrix with epsilons
    alpha = np.full((N, N), 1e-6, dtype=float)
    for delta, prob in transitions:
        for i in range(N):
            j = i + delta
            if 0 <= j < N:
                alpha[i, j] = prob

    alpha += np.finfo(float).eps
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha, D1, D2


class OuterProductHMM(OnlineAlignment):
    def __init__(
        self,
        reference_features: Union[np.ndarray, ScoreLike],
        queue: Optional[RECVQueue] = None,
        transitions: Optional[List[tuple[int, float]]] = None,
        pitch_error_probs: Optional[dict[str, float]] = None,
        S: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        other_prob: float = 1e-6,
        patience: int = 10,
        **kwargs,
    ) -> None:
        """
        Outer-product Hidden Markov Model for score following.

        Parameters
        ----------
        reference_features : ndarray or ScoreLike
            Note array or score like object

        queue : RECVQueue or None
            Queue for receiving incoming observations

        pitch_error_probs : dict or None
            If None, uses DEFAULT_PITCH_ERROR_PROBS.

        transitions : list of (delta, prob), optional
            If None, uses DEFAULT_TRANSITIONS.

        S, r : 1D arrays or None (skip-from and skip-to)
            If None, uniform distributions are used.

        other_prob : float
            Small prob for unmodelled pitches

        """

        self.reference_features = reference_features
        OnlineAlignment.__init__(
            self,
            reference_features=reference_features,
        )

        self.queue = queue
        chords, unique_onsets = get_chords_from_score(
            self.reference_features, return_unique_onsets=True
        )
        self.n_states = len(chords)
        self.score_positions = unique_onsets
        self.transitions = (
            transitions if transitions is not None else DEFAULT_TRANSITIONS
        )
        self.pitch_error_probs = (
            pitch_error_probs
            if pitch_error_probs is not None
            else DEFAULT_PITCH_ERROR_PROBS
        )
        self.other_prob = other_prob

        # Transition setup
        self.alpha, self.D1, self.D2 = compute_transition_matrix(
            self.n_states, self.transitions
        )
        self.S = (
            np.ones(self.n_states) / self.n_states
            if S is None
            else np.array(S, dtype=float)
        )
        self.r = (
            np.ones(self.n_states) / self.n_states
            if r is None
            else np.array(r, dtype=float)
        )

        # Emission setup
        self.b_table = compute_OuterProductHMM_pitch_probabilities(
            chords, pitch_error_probs, other_prob
        )

        self.current_index = 0
        self.input_index = 0
        self._prev_perf_time = None
        self._alignment_path = []
        self._current_chord = np.zeros(88, dtype=int)
        self.patience = patience
        self.state_probabilities = np.zeros(self.n_states)
        self.state_probabilities[0] = 1.0
        self.is_first_observation = True

    def is_still_following(self) -> bool:
        # Viterbi can lock onto the final state mid-piece, but we still
        # want to consume remaining events. Termination is handled by the
        # patience counter in run() and by STREAM_END.
        return True

    def __call__(
        self, observation: np.ndarray, perf_time: float, *args, **kwargs
    ) -> Optional[float]:
        # Chord-continuation case: merge pitches into the pending chord and
        # signal `None` to run() so the patience counter stays put. We don't
        # call super() because no state advance / path append should happen.
        pitch_obs = observation
        ioi = (
            perf_time - self._prev_perf_time
            if self._prev_perf_time is not None
            else 0.0
        )
        self._prev_perf_time = perf_time
        if ioi < IOI_THRESHOLD:
            self._current_chord = np.maximum(self._current_chord, pitch_obs)
            return None
        return super().__call__(observation, perf_time)

    def step(self, features) -> None:
        self._current_chord = features
        self.state_probabilities = self.viterbi_step(
            self.state_probabilities, self._current_chord
        )
        self.current_index = int(np.argmax(self.state_probabilities))
        self.input_index += 1

    # Observation likelihood
    def compute_obs_likelihood(
        self,
        observation: np.ndarray,
    ) -> NDArrayFloat:
        """
        Given observed MIDI pitches, return likelihood vector b[i].

        Parameters
        ----------
        observation: iterable of MIDI note numbers

        Returns
        -------
        b : ndarray (N,)
            b[i] = likelihood of observing `observation` at state i.
        """

        log_b = np.log(np.maximum(self.b_table[:, 21:109], 1e-300))  # (N, 88)
        log_em = log_b @ observation  # (N,): log-product over active pitches
        log_em -= log_em.max()  # shift for numerical stability
        return np.exp(log_em)  # (N,)

    # Viterbi update
    def viterbi_step(
        self,
        prev_probs: NDArrayFloat,
        observation: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        This function performs a fast outer-product Viterbi update.
        Parameters
        ----------
        prev_probs : ndarray (N,)
            Previous state probabilities.
        observation : ndarray (88,)
            Current observed MIDI pitches (88 keys from A0 to C8).
        Returns
        -------
        new_probs : ndarray (N,)
            Updated state probabilities after the Viterbi step.
        """

        b = self.compute_obs_likelihood(observation)

        if viterbi_step_cy is not None:
            prev = np.ascontiguousarray(prev_probs, dtype=np.float64)
            alpha = np.ascontiguousarray(self.alpha, dtype=np.float64)
            S = np.ascontiguousarray(self.S, dtype=np.float64)
            r = np.ascontiguousarray(self.r, dtype=np.float64)
            b_cy = np.ascontiguousarray(b, dtype=np.float64)

            # D1, D2 must be ints
            D1 = int(self.D1)
            D2 = int(self.D2)

            # Call cython function and return its result
            # viterbi_step_cy(prev, alpha, S, r, b, D1, D2) -> numpy array
            new_probs = viterbi_step_cy(prev, alpha, S, r, b_cy, D1, D2)

            return new_probs

        skip_values = prev_probs * self.S
        global_skip_max = skip_values.max()
        new_probs = np.zeros(self.n_states, dtype=float)
        for i in range(self.n_states):
            j_start = max(0, i - self.D2)
            j_end = min(self.n_states, i + self.D1 + 1)
            local_max = 0.0
            for j in range(j_start, j_end):
                val = prev_probs[j] * self.alpha[j, i]
                if val > local_max:
                    local_max = val
            skip_contrib = self.r[i] * global_skip_max

            new_probs[i] = b[i] * (
                skip_contrib if skip_contrib >= local_max else local_max
            )
        if np.sum(new_probs) > 0:
            new_probs /= np.sum(new_probs)
        else:
            new_probs = np.ones(self.n_states) / self.n_states
        return new_probs

    def run(
        self,
        verbose: bool = True,
    ) -> NDArrayInt:
        same_state_counter = 0
        empty_counter = 0
        if verbose:
            pbar = progressbar.ProgressBar(
                maxval=self.n_states,  # redirect_stdout=True
            )
            pbar.start()

        while self.is_still_following():
            prev_state = self.current_index

            queue_input = self.queue.get()
            if queue_input is STREAM_END:
                break
            if queue_input is not None:
                beat = self(*queue_input)
                if beat is None:
                    # Chord continuation: no state advance, skip patience.
                    continue
                if self.current_index == prev_state:
                    if same_state_counter < self.patience:
                        same_state_counter += 1
                    else:
                        break
                else:
                    same_state_counter = 0

                if verbose:
                    pbar.update(int(self.current_index))
                yield beat

        if verbose:
            pbar.finish()
        return self.alignment_path
