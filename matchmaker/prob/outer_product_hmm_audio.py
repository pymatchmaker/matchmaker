import time
from queue import Empty
from typing import List, Optional, Tuple

import numpy as np
import progressbar
from numpy.typing import NDArray
from partitura.score import Part, Score, ScoreLike

from matchmaker.base import OnlineAlignment
from matchmaker.features.audio import QUEUE_TIMEOUT, GaussianToneModel
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
_EMISSION_SHARPENING: float = 1.06
_OTHER_PROB: float = 1e-6
_PAUSE_ENTRY_PROB: float = 0.01  # probability of entering pause state from sound
_PAUSE_DURATION_SEC: float = 0.5
_PAUSE_EMISSION_MAX: float = 1e-3
_DEFAULT_STOP_PROB: float = (
    0.0  # disabled by default; use S, r params to enable repeat/skip
)


def _pitch_to_template_index(pitch: int) -> Optional[int]:
    """MIDI pitch (21..108) -> tone template index (0..87). Out-of-range returns None."""
    if 21 <= pitch <= 108:
        return int(pitch - 21)
    return None


def _obs_to_chroma(obs: np.ndarray) -> np.ndarray:
    """Convert observation to normalized 12-bin chroma vector."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    obs = np.maximum(obs, 0.0)

    if obs.size == 12:
        chroma = obs.copy()
    elif obs.size >= 88:
        chroma = np.zeros(12, dtype=float)
        for i in range(88):
            chroma[(21 + i) % 12] += obs[i]
    else:
        chroma = np.zeros(12, dtype=float)
        chroma[: min(12, obs.size)] = obs[: min(12, obs.size)]

    s = chroma.sum()
    return chroma / s if s > 0 else chroma


def _preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """Flatten observation to 1D array, taking last frame if 2D."""
    obs = np.asarray(obs, dtype=float)
    if obs.ndim == 2:
        obs = obs[-1]
    return obs.reshape(-1)


def compute_chord_template_mixture_weights(
    chords: List[set],
    n_templates: int,
    pitch_error_probs: Optional[dict] = None,
    other_prob: float = 1e-10,
) -> np.ndarray:
    """
    Build per-top-state mixture weights w_k^{(i)} over tone templates.

    Extends Eq.(4) II-C mixture weights to polyphonic chords with pitch error probabilities.
    """
    if pitch_error_probs is None:
        pitch_error_probs = DEFAULT_PITCH_ERROR_PROBS

    K = int(n_templates)
    if K < 2:
        raise ValueError("n_templates must be >= 2 (including noise template).")
    noise_idx = K - 1

    w = np.full((len(chords), K), float(other_prob), dtype=float)

    for i, chord in enumerate(chords):
        # chord -> template index set
        correct = {
            ti for p in chord if (ti := _pitch_to_template_index(int(p))) is not None
        }
        if not correct:
            # no pitched notes -> keep floor + slightly prefer noise
            w[i, noise_idx] = max(w[i, noise_idx], 1.0)
            w[i] /= w[i].sum()
            continue

        # Build neighbor sets in template-index domain (0..87)
        def clamp_set(s: set[int]) -> set[int]:
            return {x for x in s if 0 <= x < noise_idx}

        semi = clamp_set({x + 1 for x in correct} | {x - 1 for x in correct}) - correct
        whole = (
            clamp_set({x + 2 for x in correct} | {x - 2 for x in correct})
            - correct
            - semi
        )
        octv = (
            clamp_set({x + 12 for x in correct} | {x - 12 for x in correct})
            - correct
            - semi
            - whole
        )
        within_oct = {
            x
            for p in correct
            for x in range(p - 11, p + 12)
            if 0 <= x < noise_idx
            and x not in correct
            and x not in semi
            and x not in whole
            and x not in octv
        }

        probs = pitch_error_probs
        # Distribute mass across sets; fall back to 1 element to avoid div-by-0.
        for t in correct:
            w[i, t] = probs["correct_pitch_prob"] / len(correct)
        for t in semi:
            w[i, t] = probs["semi_tone_error_prob"] / max(1, len(semi))
        for t in whole:
            w[i, t] = probs["whole_tone_error_prob"] / max(1, len(whole))
        for t in octv:
            w[i, t] = probs["octave_error_prob"] / max(1, len(octv))
        for t in within_oct:
            w[i, t] = probs["within_one_octave_error_prob"] / max(1, len(within_oct))

        # keep noise_idx at floor(other_prob) for sound-state
        w[i, noise_idx] = float(other_prob)
        w[i] /= w[i].sum()

    return w


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
        tone_model: Optional[GaussianToneModel] = None,
        sample_rate: int = 16000,
        hop_length: int = 320,
        emission_mode: str = "chord_mask",
        S: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
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
        self.chord_pitch_mask = np.zeros((self.n_states, 88), dtype=float)
        # Harmonic mask includes fundamental + harmonics (II-C)
        self.chord_harmonic_mask = np.zeros(
            (self.n_states, 88), dtype=float
        )  # (n_states, 88)
        self.chord_pc_mask = np.zeros(
            (self.n_states, 12), dtype=float
        )  # (n_states, 12)
        for i, chord in enumerate(chords):
            for p in chord:
                if 21 <= p <= 108:
                    self.chord_pitch_mask[i, int(p - 21)] = 1.0
                self.chord_pc_mask[i, int(p % 12)] = 1.0

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
        self.use_pause_state = True  # L=2 (sound/pause)
        self.pause_entry_prob = _PAUSE_ENTRY_PROB
        self.pause_duration_sec = _PAUSE_DURATION_SEC
        self.emission_mode = str(emission_mode)
        if self.emission_mode not in {"chord_mask", "tone_model_mixture", "chroma"}:
            raise ValueError(
                "Invalid emission_mode. Use 'chord_mask', 'tone_model_mixture', or 'chroma'."
            )
        self.pause_emission_max = _PAUSE_EMISSION_MAX

        # Transition setup with banded structure
        self.alpha, self.D1, self.D2 = compute_transition_matrix(
            self.n_states, self.transitions
        )

        # Repeat/skip parameters (Nakamura et al. factorization)
        # a_{j,i} = (1 - s_j) * a^{(nbh)}_{j,i} + s_j * r_i
        # - stop_probs: s_j in [0,1]
        # - resume_probs: r_i, sum=1
        if S is None:
            stop_probs = np.full(self.n_states, _DEFAULT_STOP_PROB, dtype=float)
        else:
            stop_probs = np.array(S, dtype=float)
        if r is None:
            resume_probs = np.ones(self.n_states, dtype=float) / self.n_states
        else:
            resume_probs = np.array(r, dtype=float)

        self.stop_probs, self.resume_probs = self._normalize_skip_params(
            stop_probs, resume_probs
        )
        # Backward-compat attribute names
        self.S = self.stop_probs
        self.r = self.resume_probs

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

        # Eq.(11) III-B: a_{j,i} = a^{(nbh)}_{j,i} + s_j * r_i
        self.a_nbh = (1.0 - self.stop_probs)[:, None] * self.alpha

        frame_rate = float(sample_rate) / float(hop_length)
        self.sound_self_prob = self._extract_chord_self_transition_probs(
            reference_features=self.reference_features,
            unique_onsets=unique_onsets,
            frame_rate=frame_rate,
        )
        self.pause_self_prob = self._pause_self_transition_prob(
            pause_duration_sec=self.pause_duration_sec, frame_rate=frame_rate
        )

        # Emission model (II-C Eq.(4) extended to polyphonic chords)
        if self.emission_mode == "chroma":
            # Chroma mode: no tone_model needed, uses chord_pc_mask only
            self.n_templates = 0
            self.noise_template_idx = -1
            self.w_sound = None
            self.w_pause = None
        elif self.tone_model is None:
            raise ValueError(
                "AudioOuterProductHMM requires tone_model (GaussianToneModel) "
                "unless emission_mode='chroma'. "
                "Pass tone_model from matchmaker.features.audio.GaussianToneModel.from_templates()."
            )
        else:
            self.n_templates = int(self.tone_model.n_models)
            self.noise_template_idx = self.n_templates - 1
            self.w_sound = compute_chord_template_mixture_weights(
                chords,
                n_templates=self.n_templates,
                pitch_error_probs=pitch_error_probs,
                other_prob=max(1e-12, self.other_prob),
            )
            # Pause state uses noise template only (II-E)
            self.w_pause = np.zeros(self.n_templates, dtype=float)
            self.w_pause[self.noise_template_idx] = 1.0

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
        # Hierarchical HMM parameters (L=2): l=0 sound, l=1 pause
        # Bottom init: π_0=1, π_1=0 (II-E)
        self.pi_bottom = np.zeros((self.n_states, 2), dtype=float)
        self.pi_bottom[:, 0] = 1.0
        self.pi_bottom[:, 1] = 0.0

        # bottom transitions a_{l',l}^{(i)} and exit probs e_l^{(i)}
        # sound self-transition a00^(i) is derived from duration (Eq.(5) d=1/(1-a00)).
        # pause self-transition a11^(i) derived from pause_duration_sec.
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

    @staticmethod
    def _normalize_skip_params(
        stop_probs: np.ndarray, resume_probs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize/validate repeat-skip parameters.

        stop_probs: elementwise probabilities in [0,1]
        resume_probs: distribution (sum=1)
        """
        stop_probs = np.asarray(stop_probs, dtype=float).copy()
        resume_probs = np.asarray(resume_probs, dtype=float).copy()

        if stop_probs.ndim != 1 or resume_probs.ndim != 1:
            raise ValueError("S(stop_probs) and r(resume_probs) must be 1D arrays.")
        if stop_probs.shape[0] != resume_probs.shape[0]:
            raise ValueError("S(stop_probs) and r(resume_probs) must have same length.")

        # stop_probs in [0,1]
        stop_probs = np.nan_to_num(stop_probs, nan=0.0, posinf=1.0, neginf=0.0)
        stop_probs = np.clip(stop_probs, 0.0, 1.0)

        # resume_probs >= 0 and sum to 1
        resume_probs = np.nan_to_num(resume_probs, nan=0.0, posinf=0.0, neginf=0.0)
        resume_probs = np.maximum(resume_probs, 0.0)
        s = resume_probs.sum()
        if s <= 0:
            resume_probs = np.ones_like(resume_probs) / resume_probs.size
        else:
            resume_probs = resume_probs / s

        return stop_probs, resume_probs

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

        # Chroma-based emission: chord_pc_mask (N x 12) @ chroma (12,)
        if self.emission_mode == "chroma":
            chroma = _obs_to_chroma(obs)
            if chroma.sum() <= 0:
                return np.full(self.n_states, 1e-12, dtype=float)
            em = self.chord_pc_mask @ chroma
            if _EMISSION_SHARPENING != 1.0:
                em = np.maximum(em, 1e-12)
                em = em**_EMISSION_SHARPENING
            return np.maximum(np.nan_to_num(em, nan=1e-12), 1e-12)

        # CQT-based emission
        cqt = np.maximum(obs[:88] if obs.size >= 88 else obs, 0.0)
        s = cqt.sum()
        if s <= 0:
            return np.full(self.n_states, 1e-300, dtype=float)
        cqt = cqt / s

        if self.emission_mode == "chord_mask":
            em = self.chord_harmonic_mask @ cqt
            # Apply emission sharpening to make emissions more discriminative
            # Higher power (>1) emphasizes differences between states
            if _EMISSION_SHARPENING != 1.0:
                em = np.maximum(em, 1e-12)
                em = em**_EMISSION_SHARPENING
            return np.maximum(np.nan_to_num(em, nan=1e-12), 1e-12)

        # tone_model_mixture mode
        if getattr(self.tone_model, "is_synthetic", False):
            em = self.chord_pitch_mask @ cqt
            return np.maximum(np.nan_to_num(em, nan=1e-12), 1e-12)

        # Crop/pad to template feature dimension
        template_fd = int(self.tone_model.n_features)
        if cqt.shape[0] > template_fd:
            x = cqt[:template_fd]
        elif cqt.shape[0] < template_fd:
            x = np.pad(cqt, (0, template_fd - cqt.shape[0]), constant_values=0.0)
        else:
            x = cqt

        logp = np.asarray(
            self.tone_model.compute_log_likelihood(x), dtype=float
        ).reshape(-1)
        if logp.shape[0] != self.n_templates:
            raise ValueError(
                f"tone_model returned {logp.shape[0]} templates, expected {self.n_templates}"
            )

        m = float(np.max(logp))
        if not np.isfinite(m):
            return np.full(self.n_states, 1e-300, dtype=float)

        p = np.exp(logp - m)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.sum(p))
        if s <= 0:
            return np.full(self.n_states, 1e-300, dtype=float)
        p = p / s

        em = self.w_sound @ p
        return np.maximum(np.nan_to_num(em, nan=1e-300), 1e-300)

    def _compute_pause_emission(self, observation: np.ndarray) -> float:
        """Compute pause emission probability based on spectral flatness or noise template."""
        obs = _preprocess_obs(observation)
        cqt = np.maximum(obs[:88] if obs.size >= 88 else obs, 0.0)
        s = cqt.sum()

        if s <= 0:
            emit = 1.0
        elif self.emission_mode == "chroma":
            # Flatness-based: flat distribution -> higher pause probability
            chroma = _obs_to_chroma(obs)
            var = float(np.var(chroma))
            emit = 1.0 / (1.0 + 200.0 * var)
        elif getattr(self.tone_model, "is_synthetic", False):
            var = float(np.var(cqt / s))
            emit = 1.0 / (1.0 + 200.0 * var)
        else:
            # Use noise template from tone_model
            cqt = cqt / s
            template_fd = int(self.tone_model.n_features)
            if cqt.shape[0] > template_fd:
                x = cqt[:template_fd]
            elif cqt.shape[0] < template_fd:
                x = np.pad(cqt, (0, template_fd - cqt.shape[0]), constant_values=0.0)
            else:
                x = cqt

            logp = np.asarray(
                self.tone_model.compute_log_likelihood(x), dtype=float
            ).reshape(-1)
            m = float(np.max(logp)) if logp.size else -np.inf
            if not np.isfinite(m):
                emit = 1e-300
            else:
                p = np.exp(logp - m)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                ps = p.sum()
                emit = float(p[self.noise_template_idx] / ps) if ps > 0 else 1e-300

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
        - global_stop_exit = Σ_j exit_mass[j] * s_j
        - neigh_sum_i = Σ_{j∈nbh(i)} exit_mass[j] * a^{(nbh)}_{j,i}
        - new_sound(i) = b_sound(i,y) * [ prev_sound(i)*a00(i) + π_0^(i)*(neigh_sum_i + r_i*global_stop_exit) ]
        - new_pause(i) = b_pause(y) * [ prev_sound(i)*a01(i) + prev_pause(i)*a11 ]
          where b_pause(y) = N(y|μ_noise,Σ_noise) (II-E)
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

        # Spectral-flux-driven exit boost (onset heuristic)
        e0 = self.e0
        a00 = self.a00
        obs_flat = _preprocess_obs(observation)
        flux = float(obs_flat[88]) if obs_flat.size > 88 else 0.0
        f = flux / (flux + 1.0)  # [0,1)
        boost = 1.0 + _FLUX_EXIT_BOOST * f
        e0_eff = np.clip(e0 * boost, 1e-10, 1.0 - self.a01 - 1e-10)
        a00_eff = np.clip(1.0 - self.a01 - e0_eff, 1e-10, 1.0 - 1e-10)
        e0 = e0_eff
        a00 = a00_eff

        # Exit masses from each top state j (Eq.(6) uses e_{l'}^{(j)})
        exit_mass = prev_sound * e0 + prev_pause * self.e1  # (N,)

        # Global stop-exit term: Σ_j exit_mass[j] * s_j
        global_stop_exit = float(np.dot(exit_mass, self.stop_probs))

        # Compute neigh_sum_i for each i (banded)
        neigh_sum = np.zeros(N, dtype=float)
        for i in range(N):
            j_start = max(0, i - self.D2)
            j_end = min(N, i + self.D1 + 1)
            ssum = 0.0
            for j in range(j_start, j_end):
                a = float(self.a_nbh[j, i])
                if a <= 0:
                    continue
                ssum += exit_mass[j] * a
            neigh_sum[i] = ssum

        # Combine with repeat/skip global term (Appendix B)
        trans_enter = neigh_sum + self.resume_probs * global_stop_exit  # (N,)

        # Within-top bottom transitions
        # sound:
        within_sound = prev_sound * a00  # + prev_pause * a10(=0)
        # pause:
        within_pause = prev_sound * self.a01 + prev_pause * self.a11

        # Entering a top state resets bottom according to π_l^(i).
        # Since π_0=1, π_1=0, only sound receives the enter term.
        new_sound = emit_sound * (within_sound + trans_enter)
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
