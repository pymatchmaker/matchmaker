import time
from queue import Empty
from typing import List, Optional

import numpy as np
import progressbar
from numpy.typing import NDArray
from partitura.score import Part, Score, ScoreLike

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import QUEUE_TIMEOUT
from matchmaker.utils.misc import RECVQueue, set_latency_stats

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

DEFAULT_PITCH_ERROR_PROBS = {
    "correct_pitch_prob": 0.9497,
    "semi_tone_error_prob": 0.0145 / 2.0,
    "whole_tone_error_prob": 0.0224 / 2.0,
    "octave_error_prob": 0.0047 / 2.0,
    "within_one_octave_error_prob": 0.0086 / 9.0 / 2.0,
}

# DEFAULT_TRANSITIONS = [
#     (1, 1.0),  # normal (i→i+1)
#     (2, 1e-50),  # deletion (i→i+2), HHMMState_simple.hpp: log10(-50)
# ]
DEFAULT_TRANSITIONS = [
    (-3, 0.001),
    (-2, 0.001),
    (-1, 0.002),
    (0, 0.01342),
    (1, 0.96),
    (2, 0.01),
    (3, 0.002),
]

DEFAULT_D1 = 3
DEFAULT_D2 = 3

IOI_THRESHOLD = 0.035  # seconds

_FLUX_EXIT_BOOST: float = 1.0
_OTHER_PROB: float = 1e-6
_PAUSE_ENTRY_PROB: float = 0.01  # probability of entering pause state from sound
_PAUSE_DURATION_SEC: float = 0.5
_PAUSE_EMISSION_MAX: float = 1e-3


def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten observation to 1D array, taking last frame if 2D."""
    obs = np.asarray(obs, dtype=float)
    if obs.ndim == 2:
        obs = obs[-1]
    return obs.reshape(-1)


def get_chords_from_score(
    score: ScoreLike,
    return_unique_onsets: bool = False,
) -> List[set]:
    if isinstance(score, (Score, Part)):
        note_array = score.note_array()
    if isinstance(score, np.ndarray):
        note_array = score
        if "onset_beat" not in note_array.dtype.names:
            raise ValueError("`score` is not a valid note array")

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
    Supports negative deltas for backward transitions (repeat).

    Parameters
    ----------
    N : int
        Number of score states (chords)
    transitions : list of (delta, prob) or None
        If None, uses DEFAULT_TRANSITIONS.
    D1, D2 : int
        Fixed neighbourhood sizes (default 3)

    Returns
    -------
    alpha : ndarray (N x N)
        α[i,j] = probability of transitioning from state i -> j (banded structure)
    D1, D2 : int
        Fixed neighbourhood sizes
    """
    if transitions is None:
        transitions = DEFAULT_TRANSITIONS

    # Initialize transition matrix with epsilons
    alpha = np.full((N, N), 1e-6, dtype=float)
    for delta, prob in transitions:
        for i in range(N):
            j = i + delta
            if 0 <= j < N:
                alpha[i, j] = prob

    alpha += np.finfo(float).eps
    alpha /= alpha.sum(axis=1, keepdims=True)
    return alpha, D1, D2


class AudioOuterProductHMM:
    def __init__(
        self,
        reference_features: np.ndarray,
        queue: Optional[RECVQueue] = None,
        transitions: Optional[List[tuple[int, float]]] = None,
        pitch_error_probs: Optional[dict[str, float]] = None,
        patience: int = 0,
        tone_model=None,
        sample_rate: int = 16000,
        hop_length: int = 320,
    ) -> None:
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
        self.state_space = unique_onsets
        # Harmonic mask includes fundamental + harmonics (II-C)
        self.chord_harmonic_mask = np.zeros((self.n_states, 88), dtype=float)
        for i, chord in enumerate(chords):
            # build harmonic mask for this chord (in 0..87 pitch-index domain)
            # offsets are approximate: octave=+12, 12th≈+19, 2oct=+24, etc.
            harm = [
                (0, 1.0),
                (12, 0.7),
                (19, 0.5),
                (24, 0.4),
                (28, 0.3),
                (31, 0.25),
                (34, 0.2),
            ]
            for p in chord:
                if not (21 <= p <= 108):
                    continue
                base = int(p - 21)
                for off, w in harm:
                    idx = base + int(off)
                    if 0 <= idx < 88:
                        self.chord_harmonic_mask[i, idx] += float(w)
            s = float(np.sum(self.chord_harmonic_mask[i]))
            if s > 0:
                self.chord_harmonic_mask[i] /= s
        self.transitions = (
            transitions if transitions is not None else DEFAULT_TRANSITIONS
        )
        self.pitch_error_probs = (
            pitch_error_probs
            if pitch_error_probs is not None
            else DEFAULT_PITCH_ERROR_PROBS
        )
        self.other_prob = _OTHER_PROB
        self.tone_model = tone_model
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.pause_entry_prob = _PAUSE_ENTRY_PROB
        self.pause_duration_sec = _PAUSE_DURATION_SEC
        self.pause_emission_max = _PAUSE_EMISSION_MAX

        # Transition setup with banded structure
        self.alpha, self.D1, self.D2 = compute_transition_matrix(
            self.n_states, self.transitions
        )

        # Remove top-level self-transitions (handled by bottom layer a00)
        if self.n_states > 1:
            np.fill_diagonal(self.alpha, 0.0)
            row_sums = self.alpha.sum(axis=1, keepdims=True)
            bad = row_sums.squeeze(-1) <= 0
            if np.any(bad):
                for j in np.where(bad)[0]:
                    self.alpha[j] = 1.0 / (self.n_states - 1)
                    self.alpha[j, j] = 0.0
                row_sums = self.alpha.sum(axis=1, keepdims=True)
            self.alpha = self.alpha / row_sums

        self.current_state = 0
        self._warping_path = []
        self._current_chord = np.zeros(88, dtype=int)
        self.patience = int(patience)
        self.state_probabilities = np.zeros(self.n_states * 2, dtype=float)
        self.state_probabilities[0] = 1.0
        self.is_first_observation = True
        self.input_index = 0
        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        # Bottom transitions a_{l',l}^{(i)} and exit probs e_l^{(i)} (Eq.(5))
        frame_rate = float(self.sample_rate) / float(self.hop_length)
        self.a00 = self._extract_chord_self_transition_probs(
            reference_features=self.reference_features,
            unique_onsets=unique_onsets,
            frame_rate=frame_rate,
        )
        self.a11 = float(
            np.clip(
                self._pause_self_transition_prob(self.pause_duration_sec, frame_rate),
                0.0,
                1.0,
            )
        )
        # Pause entry prob a01 (II-E)
        move_prob = 1.0 - self.a00
        p_pause = float(np.clip(self.pause_entry_prob, 0.0, 1.0))
        self.a01 = move_prob * p_pause
        # a10^(i)=0 per II-E
        self.a10 = np.zeros(self.n_states, dtype=float)

        # exit probabilities from top state:
        # e0^(i) = 1 - a00^(i) - a01^(i)
        # e1^(i) = 1 - a11
        self.e0 = np.clip(1.0 - self.a00 - self.a01, 1e-10, 1.0)
        self.e1 = float(np.clip(1.0 - self.a11, 1e-10, 1.0))

    @property
    def warping_path(self) -> NDArrayInt:
        return (np.array(self._warping_path).T).astype(np.int32)

    @staticmethod
    def _pause_self_transition_prob(
        pause_duration_sec: float, frame_rate: float
    ) -> float:
        if pause_duration_sec <= 0:
            return 0.0
        frame_time = 1.0 / max(frame_rate, 1e-6)
        exit_prob = frame_time / max(pause_duration_sec, frame_time)
        exit_prob = float(np.clip(exit_prob, 1e-6, 1.0))
        return float(1.0 - exit_prob)

    @staticmethod
    def _extract_chord_self_transition_probs(
        reference_features: np.ndarray,
        unique_onsets: np.ndarray,
        frame_rate: float,
        default_duration_sec: float = 0.2,  # 200ms default note duration
    ) -> np.ndarray:
        """
        Extract self-transition probabilities from note durations.

        Based on Nakamura et al. 2013 Eq.(2):
            d_i = 1 / (1 - a_i)  =>  a_i = 1 - 1/d_i

        where d_i is the expected duration in frames.

        Parameters
        ----------
        reference_features : np.ndarray
            Note array with onset_beat and duration_sec fields
        unique_onsets : np.ndarray
            Unique onset times (chord boundaries)
        frame_rate : float
            Audio frame rate (frames per second)
        default_duration_sec : float
            Default note duration when not available (default: 0.2s = 200ms)

        Returns
        -------
        np.ndarray
            Self-transition probability for each state (a_i values)
        """
        N = len(unique_onsets)
        frame_time = 1.0 / max(frame_rate, 1e-6)

        # Default: a_i = 1 - 1/d_i where d_i = default_duration_sec / frame_time
        default_d_i = max(1.0, default_duration_sec / frame_time)
        default_a_i = 1.0 - 1.0 / default_d_i  # e.g., 0.2s at 50fps → d=10 → a=0.90
        out = np.full(N, default_a_i, dtype=float)

        if not isinstance(reference_features, np.ndarray):
            return out
        names = getattr(reference_features.dtype, "names", None)
        if not names or "onset_beat" not in names:
            return out

        if "self_trans_prob" in names:
            for i, onset in enumerate(unique_onsets):
                idxs = np.where(reference_features["onset_beat"] == onset)[0]
                if idxs.size == 0:
                    continue
                v = float(reference_features["self_trans_prob"][idxs[0]])
                if np.isfinite(v):
                    out[i] = float(np.clip(v, 1e-6, 1.0 - 1e-6))
            return out

        if "duration_sec" in names:
            for i, onset in enumerate(unique_onsets):
                idxs = np.where(reference_features["onset_beat"] == onset)[0]
                if idxs.size == 0:
                    continue
                duration_sec = float(reference_features["duration_sec"][idxs[0]])
                if np.isfinite(duration_sec) and duration_sec > 0:
                    # Minimum duration floor to avoid too aggressive transitions
                    duration_sec = max(duration_sec, 0.05)  # at least 50ms
                    d_i = max(1.0, duration_sec / frame_time)
                    out[i] = float(np.clip(1.0 - 1.0 / d_i, 1e-6, 1.0 - 1e-6))

        return out

    def is_still_following(self) -> bool:
        if self.current_state is not None:
            return self.current_state <= self.n_states - 1
        return False

    def __call__(self, input, *args, **kwargs) -> Optional[int]:
        """
        Frame-based audio HMM update.

        Parameters
        ----------
        input : np.ndarray or tuple
            Current frame observation y_t (CQT magnitude 88-bin vector).
            If tuple, uses first element for backward compatibility.

        Returns
        -------
        current_state : int or None
            Current estimated score state index.
        """
        if isinstance(input, tuple):
            observation = np.asarray(input[0], dtype=float)
        else:
            observation = np.asarray(input, dtype=float)

        if observation.ndim == 2:
            observation = observation[-1]

        self.state_probabilities = self.forward_step(
            self.state_probabilities, observation
        )

        probs = self.state_probabilities
        top_scores = probs[0::2] + probs[1::2]
        new_top = int(np.argmax(top_scores))

        self.current_state = new_top
        self._warping_path.append((self.current_state, self.input_index))
        self.input_index += 1
        return self.current_state

    def compute_obs_likelihood(
        self,
        observation: np.ndarray,
    ) -> NDArrayFloat:
        """Compute per-top-state sound emission b_0^{(i)}(y_t) for current frame."""
        obs = _preprocess_obs(observation)

        # CQT-based emission: chord_harmonic_mask @ normalized_cqt
        cqt = np.maximum(obs[:88] if obs.size >= 88 else obs, 0.0)
        s = cqt.sum()
        if s <= 0:
            return np.full(self.n_states, 1e-300, dtype=float)
        cqt = cqt / s

        em = self.chord_harmonic_mask @ cqt
        return np.maximum(np.nan_to_num(em, nan=1e-12), 1e-12)

    def _compute_pause_emission(self, observation: np.ndarray) -> float:
        """Compute pause emission probability based on spectral flatness.

        Silence/noise has a flat spectrum (low variance), pitched sound has peaks (high variance).
        """
        obs = _preprocess_obs(observation)
        cqt = np.maximum(obs[:88] if obs.size >= 88 else obs, 0.0)
        s = cqt.sum()

        if s <= 0:
            emit = 1.0
        else:
            var = float(np.var(cqt / s))
            emit = 1.0 / (1.0 + 200.0 * var)

        emit = max(emit, 1e-300)
        return min(emit, self.pause_emission_max)

    def forward_step(
        self,
        prev_probs: NDArrayFloat,
        observation: NDArrayFloat,
    ) -> NDArrayFloat:
        """
        Forward(filtering) update for hierarchical HMM (L=2) with repeat/skip factorization.

        Mapping to paper:
        - II-F Eq.(9): α_{t,(i,l)} recursion on flattened standard HMM.
        - II-F Eq.(6): transition definition from (j,l') -> (i,l)
        - III-B Eq.(11): top transition factorization a_{j,i} = a^{(nbh)}_{j,i} + s_j r_i
        - Appendix B Eq.(19)(20): O(LN) update using global term

        Implementation summary (L=2, π_1^(i)=0):
        - prev_probs is length 2N: [sound0, pause0, sound1, pause1, ...]
        - exit_mass[j] = Σ_{l'} prev(j,l') * e_{l'}^{(j)}
        - neigh_sum_i = Σ_{j∈nbh(i)} exit_mass[j] * α_{j,i}
        - new_sound(i) = b_sound(i,y) * [ prev_sound(i)*a00(i) + neigh_sum_i ]
        - new_pause(i) = b_pause(y) * [ prev_sound(i)*a01(i) + prev_pause(i)*a11 ]
        """
        N = self.n_states
        if prev_probs.shape[0] != 2 * N:
            raise ValueError(
                f"Expected prev_probs shape {(2*N,)} but got {prev_probs.shape}"
            )
        prev_sound = np.asarray(prev_probs[0::2], dtype=float)
        prev_pause = np.asarray(prev_probs[1::2], dtype=float)

        # Emission
        emit_sound = self.compute_obs_likelihood(observation)
        emit_pause_scalar = self._compute_pause_emission(observation)
        emit_pause = np.full(N, emit_pause_scalar, dtype=float)

        # Spectral-flux-driven exit boost
        obs_flat = _preprocess_obs(observation)
        flux = float(obs_flat[88]) if obs_flat.size > 88 else 0.0
        f = flux / (flux + 1.0)  # [0,1)
        boost = 1.0 + _FLUX_EXIT_BOOST * f
        e0 = np.clip(self.e0 * boost, 1e-10, 1.0 - self.a01 - 1e-10)
        a00 = np.clip(1.0 - self.a01 - e0, 1e-10, 1.0 - 1e-10)

        # Exit masses from each top state j (Eq.(6))
        exit_mass = prev_sound * e0 + prev_pause * self.e1  # (N,)

        # Compute neigh_sum_i for each i (banded, Eq.(9))
        neigh_sum = np.zeros(N, dtype=float)
        for i in range(N):
            j_start = max(0, i - self.D2)
            j_end = min(N, i + self.D1 + 1)
            ssum = 0.0
            for j in range(j_start, j_end):
                a = float(self.alpha[j, i])
                if a <= 0:
                    continue
                ssum += exit_mass[j] * a
            neigh_sum[i] = ssum

        # Within-top bottom transitions
        within_sound = prev_sound * a00
        within_pause = prev_sound * self.a01 + prev_pause * self.a11

        # Entering a top state resets bottom to sound (π_0=1, π_1=0)
        new_sound = emit_sound * (within_sound + neigh_sum)
        new_pause = emit_pause * within_pause

        new_probs = np.empty(2 * N, dtype=float)
        new_probs[0::2] = new_sound
        new_probs[1::2] = new_pause

        new_probs = np.nan_to_num(new_probs, nan=1e-300, posinf=1.0, neginf=1e-300)
        new_probs = np.maximum(new_probs, 1e-300)
        z = float(new_probs.sum())
        if z > 0:
            new_probs /= z
        else:
            new_probs[:] = 0.0
            new_probs[0] = 1.0
        return new_probs

    def run(
        self,
        verbose: bool = True,
    ) -> NDArrayInt:
        same_state_counter = 0
        empty_counter = 0
        if verbose:
            pbar = progressbar.ProgressBar(maxval=self.n_states)
            pbar.start()

        while self.is_still_following():
            prev_state = self.current_state

            try:
                queue_input = self.queue.get(timeout=QUEUE_TIMEOUT)
            except Empty:
                break
            self.last_queue_update = time.time()
            if queue_input is not None:
                current_state = self(queue_input)
                empty_counter = 0
                if current_state == prev_state:
                    if self.patience > 0:
                        if same_state_counter < self.patience:
                            same_state_counter += 1
                        else:
                            break
                else:
                    same_state_counter = 0

                if verbose:
                    if current_state is not None:
                        pbar.update(int(current_state) + 1)  # states starts with 0
                latency = time.time() - self.last_queue_update
                self.latency_stats = set_latency_stats(
                    latency, self.latency_stats, self.input_index
                )
                yield current_state

        if verbose:
            pbar.finish()
        return self.warping_path
