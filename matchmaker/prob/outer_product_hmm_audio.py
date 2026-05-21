import time
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from partitura.score import Part, Score, ScoreLike

from matchmaker.base import OnlineAlignment
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats

NDArrayFloat = NDArray[np.float32]
NDArrayInt = NDArray[np.int32]

# Nakamura et al. 2016 Section IV-B experimental parameters.
# Neighbourhood transitions a^{(nbh)}_{j,i} from nakamura_data.py:
#   These are the "small transition probabilities" for the banded structure.
#   Paper: a_{i,i} = 0 (self-transition handled by bottom HMM a00),
#          a_{i,i+2} = 1e-50 (deletion, effectively 0).
# We use the empirical values from [13] (Nakamura JNMR 2014).
DEFAULT_TRANSITIONS = [
    (-3, 0.00509),
    (-2, 0.00516),
    (-1, 0.00886),
    (0, 0.01342),  # insertion (staying at same top state)
    (1, 0.94531),  # normal forward progression
    (2, 0.00610),  # deletion (skip one note)
    (3, 0.00073),
]

DEFAULT_D1 = 3
DEFAULT_D2 = 3

IOI_THRESHOLD = 0.035  # seconds
SUSTAINED_DECAY: float = 0.3  # exponential decay rate for sustained notes

_FLUX_EXIT_BOOST: float = 1.0
_OTHER_PROB: float = 1e-6
# Paper IV-B:
#   a_{0,1}^{(i)} = 1e-100  (pause entry: almost never enter pause)
#   a_{1,1}^{(i)} = 0.999   (pause self-transition: once in pause, stay)
_PAUSE_ENTRY_PROB: float = 1e-100
_PAUSE_SELF_TRANSITION: float = 0.999
_PAUSE_EMISSION_MAX: float = 1e-3


def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten observation to 1D array, taking last frame if 2D."""
    obs = np.asarray(obs, dtype=float)
    if obs.ndim == 2:
        obs = obs[-1]
    return obs.reshape(-1)


def get_weighted_chords_from_score(
    score: ScoreLike,
    decay: float = SUSTAINED_DECAY,
) -> tuple[list[dict[int, float]], np.ndarray]:
    """Get weighted chords at each unique onset.

    Onset notes get weight 1.0, sustained notes get exp(-decay * beats_elapsed).

    Returns (chords, unique_onsets) where each chord is a dict {pitch: weight}.
    """
    if isinstance(score, (Score, Part)):
        note_array = score.note_array()
    if isinstance(score, np.ndarray):
        note_array = score

    unique_onsets = np.unique(note_array["onset_beat"])
    onsets = note_array["onset_beat"]
    durations = note_array["duration_beat"]
    pitches = note_array["pitch"]

    chords = []
    for uo in unique_onsets:
        starting = onsets == uo
        sustained = (onsets < uo) & (onsets + durations > uo)
        chord = {}
        for p in pitches[starting]:
            chord[int(p)] = 1.0
        for idx in np.where(sustained)[0]:
            ip = int(pitches[idx])
            if ip not in chord:
                elapsed = float(uo - onsets[idx])
                chord[ip] = float(np.exp(-decay * elapsed))
        chords.append(chord)

    return chords, unique_onsets


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


class AudioOuterProductHMM(OnlineAlignment):
    def __init__(
        self,
        reference_features: np.ndarray,
        queue: Optional[RECVQueue] = None,
        transitions: Optional[List[tuple[int, float]]] = None,
        tempo: float = 120.0,
        sample_rate: int = 16000,
        hop_length: int = 320,
        s_j: float = 1e-5,
        r_i: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        super().__init__(reference_features=reference_features, queue=queue)

        weighted_chords, unique_onsets = get_weighted_chords_from_score(
            self.reference_features,
        )
        self.n_states = len(weighted_chords)
        self.score_positions = unique_onsets
        self.chord_harmonic_mask = np.zeros((self.n_states, 88), dtype=float)
        for i, chord in enumerate(weighted_chords):
            for p, note_weight in chord.items():
                if not (21 <= p <= 108):
                    continue
                base = int(p - 21)
                self.chord_harmonic_mask[i, base] += note_weight
            s = float(np.sum(self.chord_harmonic_mask[i]))
            if s > 0:
                self.chord_harmonic_mask[i] /= s
        self.transitions = (
            transitions if transitions is not None else DEFAULT_TRANSITIONS
        )
        self.other_prob = _OTHER_PROB
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.pause_entry_prob = _PAUSE_ENTRY_PROB
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

        # Repeat/skip factorization (Nakamura Eq.11):
        #   a_{j,i} = a^{(nbh)}_{j,i} + s_j * r_i
        # s_j: probability of stopping at event j before a repeat/skip
        # r_i: probability of resuming at event i after a repeat/skip
        self.S = np.full(self.n_states, float(s_j), dtype=float)
        if r_i is not None:
            self.r = np.asarray(r_i, dtype=float)
        else:
            self.r = np.ones(self.n_states, dtype=float) / self.n_states

        # Renormalize alpha to account for s_j mass:
        # Σ_i a_{j,i} = Σ_i a^{(nbh)}_{j,i} + s_j * Σ_i r_i = 1
        # => Σ_i a^{(nbh)}_{j,i} = 1 - s_j  (since Σ_i r_i = 1)
        if s_j > 0:
            self.alpha = self.alpha * (1.0 - self.S[:, None])

        self.current_index = 0
        self._current_chord = np.zeros(88, dtype=int)
        self.state_probabilities = np.zeros(self.n_states * 2, dtype=float)
        self.state_probabilities[0] = 1.0
        self.is_first_observation = True
        self.input_index = 0
        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }
        # Bottom transitions a_{l',l}^{(i)} and exit probs e_l^{(i)} (Eq.(5))
        frame_rate = float(self.sample_rate) / float(self.hop_length)
        self.a00 = self._compute_chord_self_transition_probs(
            unique_onsets=unique_onsets,
            tempo=tempo,
            frame_rate=frame_rate,
        )
        # Paper IV-B: a_{1,1}^{(i)} = 0.999 (pause self-transition)
        self.a11 = float(_PAUSE_SELF_TRANSITION)
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

        # Precompute alpha diagonals and sliding window indices for vectorized forward_step
        self._alpha_diags = []
        for d in range(-self.D2, self.D1 + 1):
            self._alpha_diags.append(np.diagonal(self.alpha, offset=-d).copy())
        self._j_starts = np.maximum(0, np.arange(self.n_states) - self.D2)
        self._j_ends = np.minimum(self.n_states, np.arange(self.n_states) + self.D1 + 1)

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
    def _compute_chord_self_transition_probs(
        unique_onsets: np.ndarray,
        tempo: float,
        frame_rate: float,
    ) -> np.ndarray:
        """
        Compute self-transition probabilities from chord durations (Eq.5).

        a_i = 1 - 1/d_i, where d_i = duration_sec / frame_time.
        """
        N = len(unique_onsets)
        frame_time = 1.0 / max(frame_rate, 1e-6)

        # Convert onset beats to seconds, then compute inter-onset durations
        onset_sec = unique_onsets * (60.0 / tempo)
        dur_sec = np.zeros(N, dtype=float)
        if N >= 2:
            dur_sec[:-1] = np.diff(onset_sec)
            dur_sec[-1] = dur_sec[-2]
        else:
            dur_sec[:] = 0.2

        dur_sec = np.maximum(dur_sec, 0.05)  # at least 50ms
        d_i = np.maximum(1.0, dur_sec / frame_time)
        return np.clip(1.0 - 1.0 / d_i, 1e-6, 1.0 - 1e-6)

    def is_still_following(self) -> bool:
        # Argmax over hierarchical states can lock onto the final state
        # mid-piece; rely on STREAM_END to terminate.
        return True

    def __call__(self, input, perf_time: float, *args, **kwargs) -> float:
        """Frame-based audio HMM update.

        Parameters
        ----------
        input : np.ndarray
            Current frame observation y_t (CQT magnitude 88-bin vector).
        perf_time : float
            Performance time (seconds) of this frame.

        Returns
        -------
        float
            Current beat position from score_positions[current_index].
        """
        t0 = time.time()
        observation = np.asarray(input, dtype=float)
        self.current_perf_time = float(perf_time)

        if observation.size == 0:
            return self.current_position

        if observation.ndim == 2:
            observation = observation[-1]

        self.state_probabilities = self.forward_step(
            self.state_probabilities, observation
        )

        probs = self.state_probabilities
        top_scores = probs[0::2] + probs[1::2]
        new_top = int(np.argmax(top_scores))

        self.current_index = new_top
        self.current_position = float(self.score_positions[self.current_index])
        self._alignment_path.append((self.current_position, self.current_perf_time))
        self.input_index += 1
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )
        return self.current_position

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

        # --- Single _preprocess_obs call + inline emission computation ---
        obs = _preprocess_obs(observation)
        cqt = np.maximum(obs[:88] if obs.size >= 88 else obs, 0.0)
        cqt_sum = cqt.sum()

        # Sound emission (from compute_obs_likelihood)
        if cqt_sum <= 0:
            emit_sound = np.full(N, 1e-300, dtype=float)
        else:
            cqt_norm = cqt / cqt_sum
            em = self.chord_harmonic_mask @ cqt_norm
            emit_sound = np.maximum(np.nan_to_num(em, nan=1e-12), 1e-12)

        # Pause emission (from _compute_pause_emission)
        if cqt_sum <= 0:
            emit_pause_scalar = min(1.0, self.pause_emission_max)
        else:
            var = float(np.var(cqt / cqt_sum))
            emit_pause_scalar = min(
                max(1.0 / (1.0 + 200.0 * var), 1e-300), self.pause_emission_max
            )
        emit_pause = np.full(N, emit_pause_scalar, dtype=float)

        # Spectral-flux-driven exit boost
        flux = float(obs[88]) if obs.size > 88 else 0.0
        f = flux / (flux + 1.0)  # [0,1)
        boost = 1.0 + _FLUX_EXIT_BOOST * f
        e0 = np.clip(self.e0 * boost, 1e-10, 1.0 - self.a01 - 1e-10)
        a00 = np.clip(1.0 - self.a01 - e0, 1e-10, 1.0 - 1e-10)

        # Exit masses from each top state j (Eq.(6))
        exit_mass = prev_sound * e0 + prev_pause * self.e1  # (N,)

        # --- Vectorized neighbourhood sum (replaces O(N*(D1+D2)) Python loop) ---
        # Global skip term: Σ_j exit_mass[j] * S[j]  (O(N), computed once)
        global_skip_sum = float(np.dot(exit_mass, self.S))

        # Local neighbourhood transition: sum over diagonals of alpha
        local_nbh = np.zeros(N, dtype=float)
        for k, d in enumerate(range(-self.D2, self.D1 + 1)):
            diag = self._alpha_diags[k]
            L = len(diag)
            src = max(0, d)  # source index offset in exit_mass
            dst = max(0, -d)  # destination index offset in local_nbh
            local_nbh[dst : dst + L] += exit_mass[src : src + L] * diag

        # Local skip via cumsum sliding window: O(N) instead of O(N*D)
        eS = exit_mass * self.S
        cumsum_eS = np.empty(N + 1, dtype=float)
        cumsum_eS[0] = 0.0
        np.cumsum(eS, out=cumsum_eS[1:])
        local_skip = cumsum_eS[self._j_ends] - cumsum_eS[self._j_starts]

        neigh_sum = local_nbh + self.r * (global_skip_sum - local_skip)

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
