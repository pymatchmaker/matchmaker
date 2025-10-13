from typing import List

import numpy as np

from partitura.score import Part, Score, ScoreLike

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


def compute_OuterProductHMM_pitch_probabilities(
    chords, pitch_error_probs=None, other_prob=1e-6
):
    """
    Precompute emission probabilities for OuterProductHMM states.

    Parameters
    ----------
    chords : list of sets
        chords[i] contains MIDI pitches (0–127) for score chord at state i.
    pitch_error_probs : dict or None
        If None, uses DEFAULT_PITCH_ERROR_PROBS.
    other_prob : float
        Probability assigned to any pitch not falling into error categories.

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


def get_chords_from_score(score: ScoreLike) -> List[set]:
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

    return chords


def compute_transition_matrix(N, transitions=None, D1=DEFAULT_D1, D2=DEFAULT_D2):
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

    alpha = np.zeros((N, N), dtype=float)
    for delta, prob in transitions:
        for i in range(N):
            j = i + delta
            if 0 <= j < N:
                alpha[i, j] = prob

    alpha += np.finfo(float).eps
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha, D1, D2


class OuterProductHMM:
    def __init__(
        self,
        chords,
        pitch_error_probs=None,
        transitions=None,
        S=None,
        r=None,
        other_prob=1e-6,
    ):
        """
        chords : list of sets of MIDI pitches (one per score state). Not to be confused with the general definition of a chord!
        pitch_error_probs : dict, optional
        transitions : list of (delta, prob), optional
        S, r : 1D arrays or None (skip-from and skip-to)
        other_prob : float, small prob for unmodelled pitches
        """
        self.N = len(chords)

        # Transition setup
        self.alpha, self.D1, self.D2 = compute_transition_matrix(self.N, transitions)
        self.S = np.ones(self.N) / self.N if S is None else np.array(S, dtype=float)
        self.r = np.ones(self.N) / self.N if r is None else np.array(r, dtype=float)

        # Emission setup
        self.b_table = compute_OuterProductHMM_pitch_probabilities(
            chords, pitch_error_probs, other_prob
        )

    # Observation likelihood
    def compute_obs_likelihood(self, observation):
        """
        Given observed MIDI pitches, return likelihood vector b[i].
        observation: iterable of MIDI note numbers
        """
        b = np.ones(self.N, dtype=float)
        for pitch in observation:
            b *= self.b_table[:, pitch]
        return b

    # Viterbi update
    def viterbi_step(self, prev_probs, observation):
        """Fast outer-product Viterbi update."""
        b = self.compute_obs_likelihood(observation)
        skip_values = prev_probs * self.S
        global_skip_max = skip_values.max()
        new_probs = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            j_start = max(0, i - self.D2)
            j_end = min(self.N, i + self.D1 + 1)
            local_max = 0.0
            for j in range(j_start, j_end):
                val = prev_probs[j] * self.alpha[j, i]
                if val > local_max:
                    local_max = val
            skip_contrib = self.r[i] * global_skip_max
            new_probs[i] = b[i] * (
                skip_contrib if skip_contrib >= local_max else local_max
            )
        return new_probs

    # --- Forward update ---
    def forward_step(self, prev_forward, observation):
        """Fast outer-product forward update."""
        b = self.compute_obs_likelihood(observation)
        skip_sum = np.dot(prev_forward, self.S)
        new_forward = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            j_start = max(0, i - self.D2)
            j_end = min(self.N, i + self.D1 + 1)
            local_sum = 0.0
            for j in range(j_start, j_end):
                local_sum += prev_forward[j] * self.alpha[j, i]
            new_forward[i] = b[i] * (local_sum + self.r[i] * skip_sum)
        return new_forward
