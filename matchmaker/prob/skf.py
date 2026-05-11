"""
Switching Kalman Filter (SKF) for Audio Score Following with Hidden Tempo.

Implements the method from:
    Jiang, Y. and Raphael, C. "Score Following with Hidden Tempo Using a
    Switching State-Space Model", ISMIR 2020.

The model tracks tempo as a continuous latent variable using a Kalman filter,
and infers the current chord/note position via a switching state-space model
at the audio frame level.  Hypotheses are managed as a beam of (chord_index,
age) pairs, each carrying a Gaussian tempo belief (mean, variance).  Nodes
with identical (chord_index, age) are merged via moment matching.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import partitura
from partitura.score import Note as PtNote
from partitura.score import Part as PtPart
from partitura.score import TimeSignature as PtTimeSignature
from scipy.signal import get_window
from scipy.special import ndtr  # faster than scipy.stats.norm.cdf

from matchmaker.base import OnlineAlignment
from matchmaker.io.audio import QUEUE_TIMEOUT
from matchmaker.io.queue import RECVQueue
from matchmaker.utils.misc import set_latency_stats

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_N_FFT = 512  # Paper: "512-sample frame size (64 ms)" at 8 kHz
DEFAULT_HOP_LENGTH = 128  # paper: "128-sample hop size" -> 16 ms at 8 kHz

# Kalman noise scales (paper Section 3.1, tuned experimentally)
# Paper: "σε,k was modified to be proportional to tk, and ση,k
# proportional to lk*tk, with manually set scales."
SIGMA_EPS_SCALE = 0.05  # σ_ε = scale * t_k (onset jitter)
SIGMA_ETA_SCALE = 0.01  # σ_η = scale * l_k * t_k (tempo drift)

# Beam search (paper Section 5.3: "limit hypotheses at each frame to ≤200")
MAX_HYPOTHESES = 200

# Minimum probability floor
PROB_FLOOR = 1e-300


def build_chord_sequence(note_array: np.ndarray):
    """Build homophonic chord sequence from a partitura note_array.

    Returns
    -------
    chords : list of list of int
        chords[k] = list of MIDI pitches in chord k
    lengths : np.ndarray (K,)
        Nominal length of each chord in whole-note units.
    onset_beats : np.ndarray (K,)
        Score onset in beat units for each chord.
    """
    onsets = note_array["onset_beat"]
    pitches = note_array["pitch"]
    durations = note_array["duration_beat"]

    unique_onsets = np.unique(onsets)
    K = len(unique_onsets)

    chords: List[List[int]] = []
    onset_beats = np.zeros(K)

    for k, ub in enumerate(unique_onsets):
        mask = onsets == ub
        chords.append(pitches[mask].tolist())
        onset_beats[k] = ub

    # Nominal length l_k = onset_{k+1} - onset_k (in beat units, then /4
    # to convert to whole-note units since 1 whole note = 4 beats)
    lengths = np.zeros(K)
    for k in range(K - 1):
        lengths[k] = (onset_beats[k + 1] - onset_beats[k]) / 4.0
    # Last chord: use same length as previous, or 1 beat
    lengths[K - 1] = lengths[K - 2] if K > 1 else 0.25

    # Ensure all lengths are positive
    lengths = np.maximum(lengths, 1e-6)

    return chords, lengths, onset_beats


def build_spectral_templates(
    chords: List[List[int]],
    n_fft: int = DEFAULT_N_FFT,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_harmonics: int = 5,
) -> np.ndarray:
    """Build normalized spectral templates q_k for each chord.

    Following Raphael 2006 [13]: templates are formed from the average
    magnitude spectra of synthesized audio for each chord. Falls back to
    Gaussian-peak templates if synthesis fails.

    Parameters
    ----------
    chords : list of list of int (MIDI pitches)
    n_fft : int
    sample_rate : int
    n_harmonics : int (used only in fallback)

    Returns
    -------
    templates : np.ndarray (K, n_bins)
        Each row sums to 1.
    """
    # Try synthesized templates first (Raphael 2006 approach)
    try:
        templates = _build_synthesized_templates(chords, n_fft, sample_rate)
        if templates is not None:
            print("[SKF] Using synthesized spectral templates")
            return templates
    except Exception as e:
        print(f"[SKF] Synthesis failed ({e}), falling back to Gaussian peaks")

    print("[SKF] Using Gaussian-peak spectral templates (fallback)")
    return _build_gaussian_templates(chords, n_fft, sample_rate, n_harmonics)


def _build_synthesized_templates(
    chords: List[List[int]],
    n_fft: int,
    sample_rate: int,
    note_duration: float = 0.3,
) -> Optional[np.ndarray]:
    """Build templates by synthesizing all chords via partitura + FluidSynth.

    Creates a partitura Part with each chord at a different time offset,
    synthesizes once with FluidSynth, then extracts average magnitude
    spectra at each chord's location.
    """
    # MIDI pitch -> (step, alter)
    _PITCH_MAP = [
        ("C", 0),
        ("C", 1),
        ("D", 0),
        ("D", 1),
        ("E", 0),
        ("F", 0),
        ("F", 1),
        ("G", 0),
        ("G", 1),
        ("A", 0),
        ("A", 1),
        ("B", 0),
    ]

    n_bins = n_fft // 2 + 1
    hop_length = DEFAULT_HOP_LENGTH
    K = len(chords)
    bpm = 120.0
    quarter_dur = 480  # divs per quarter note
    secs_per_div = 60.0 / (bpm * quarter_dur)
    gap = 0.1  # silence between chords (seconds)
    chord_interval = note_duration + gap

    note_dur_divs = int(note_duration / secs_per_div)
    chord_interval_divs = int(chord_interval / secs_per_div)

    # Build a partitura Part with all chords at sequential offsets
    part = PtPart(id="P0", part_name="Piano", quarter_duration=quarter_dur)
    part.add(PtTimeSignature(4, 4), start=0)

    note_id = 0
    for k, chord_pitches in enumerate(chords):
        onset = k * chord_interval_divs
        offset = onset + note_dur_divs
        for pitch in chord_pitches:
            step, alter = _PITCH_MAP[int(pitch) % 12]
            octave = (int(pitch) // 12) - 1
            note = PtNote(
                id=f"n{note_id}",
                step=step,
                octave=octave,
                alter=alter,
                voice=1,
            )
            part.add(note, start=onset, end=offset)
            note_id += 1

    # Convert to PerformedPart so we can set velocity=100
    # (partitura default is 64, matching pretty_midi's 100 matters for template quality)
    ppart = partitura.utils.music.performance_from_part(part, bpm=bpm)
    for n in ppart.notes:
        n["velocity"] = 100

    # Single synthesis call (uses partitura's bundled MuseScore soundfont)
    audio = partitura.save_wav_fluidsynth(
        ppart,
        out=None,
        samplerate=sample_rate,
    )

    # Extract spectra for each chord
    window = get_window("hann", n_fft)
    templates = np.zeros((K, n_bins))

    for k in range(K):
        # Start sample for this chord (skip first few ms of attack)
        chord_start_sample = int((k * chord_interval + 0.02) * sample_rate)
        chord_end_sample = int((k * chord_interval + note_duration) * sample_rate)

        spec_sum = np.zeros(n_bins)
        count = 0
        pos = chord_start_sample
        while pos + n_fft <= chord_end_sample and pos + n_fft <= len(audio):
            frame = audio[pos : pos + n_fft]
            mag = np.abs(np.fft.rfft(frame * window, n=n_fft))
            spec_sum += mag
            count += 1
            pos += hop_length

        if count > 0:
            templates[k] = spec_sum / count
        else:
            templates[k] = 1.0

    # Add small floor and normalize
    templates = np.maximum(templates, 1e-10)
    row_sums = templates.sum(axis=1, keepdims=True)
    templates /= np.maximum(row_sums, 1e-20)

    return templates


def _build_gaussian_templates(
    chords: List[List[int]],
    n_fft: int = DEFAULT_N_FFT,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_harmonics: int = 5,
) -> np.ndarray:
    """Fallback: Gaussian-peak templates."""
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, n_bins)
    K = len(chords)
    templates = np.full((K, n_bins), 1e-10)

    freq_resolution = sample_rate / n_fft
    sigma_hz = 1.7 * freq_resolution

    for k, chord_pitches in enumerate(chords):
        for midi_pitch in chord_pitches:
            f0 = 440.0 * (2.0 ** ((midi_pitch - 69) / 12.0))
            for h in range(1, n_harmonics + 1):
                fh = f0 * h
                if fh >= sample_rate / 2:
                    break
                amp = 1.0 / h
                templates[k] += amp * np.exp(-0.5 * ((freqs - fh) / sigma_hz) ** 2)

    row_sums = templates.sum(axis=1, keepdims=True)
    templates /= np.maximum(row_sums, 1e-20)

    return templates


class SwitchingKalmanFilterFollower(OnlineAlignment):
    """
    Audio score follower using a Switching State-Space Model with hidden tempo.

    Each hypothesis is a tuple (k, a) where k is the chord index and a is
    the age (number of frames the current chord has lasted).  Each hypothesis
    carries:
      - prob : float, filtered probability p(k_n, a_n | y^n_1)
      - mu_t : float, Kalman-filtered tempo mean E[t_k | ...]
      - var_t: float, Kalman-filtered tempo variance Var[t_k | ...]

    Parameters
    ----------
    reference_features : np.ndarray
        Score note_array from partitura.
    queue : RECVQueue
        Audio frame queue from AudioStream.
    tempo : float
        Initial tempo in BPM.
    sample_rate : int
        Audio sample rate (default 8000).
    n_fft : int
        FFT size (default 512).
    hop_length : int
        Hop size in samples (default 128).
    max_hypotheses : int
        Beam width (default 200).
    sigma_eps_scale : float
        Scale for onset noise σ_ε (default 0.15).
    sigma_eta_scale : float
        Scale for tempo noise σ_η (default 0.05).
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        queue: Optional[RECVQueue] = None,
        tempo: float = 120.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        max_hypotheses: int = MAX_HYPOTHESES,
        sigma_eps_scale: float = SIGMA_EPS_SCALE,
        sigma_eta_scale: float = SIGMA_ETA_SCALE,
        **kwargs,
    ):
        super().__init__(reference_features=reference_features)
        self.queue = queue
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_hypotheses = max_hypotheses
        self.sigma_eps_scale = sigma_eps_scale
        self.sigma_eta_scale = sigma_eta_scale

        # Frame duration in seconds
        self.delta = hop_length / sample_rate

        # Build chord sequence from score
        self.chords, self.lengths, self.onset_beats = build_chord_sequence(
            reference_features
        )
        self.K = len(self.chords)
        self.score_positions = self.onset_beats

        # Build spectral templates (Raphael 2006: synthesized audio)
        self.templates = build_spectral_templates(
            self.chords, n_fft=n_fft, sample_rate=sample_rate
        )

        # Log templates for efficient likelihood (Eq. 1 in log domain)
        self.log_templates = np.log(np.maximum(self.templates, 1e-300))

        # Initial tempo: seconds per whole note
        # BPM = quarter notes per minute → seconds per whole note = 240 / BPM
        self.init_tempo = 240.0 / tempo  # seconds per whole note
        self.init_tempo_var = (self.init_tempo * 0.1) ** 2  # +-10% initial uncertainty

        total_dur_est = float(np.sum(self.lengths)) * self.init_tempo
        print(
            f"[SKF] K={self.K} chords, tempo={tempo:.1f} BPM "
            f"(init_t={self.init_tempo:.3f} s/wn), "
            f"spectral({n_fft}), est_dur={total_dur_est:.1f}s"
        )

        # Hypothesis state: dict of (k, a) -> (prob, mu_t, var_t)
        # k: chord index (0..K-1), a: age in frames (1, 2, ...)
        # Initialize: chord 0, age 1
        self.hypotheses = {(0, 1): (1.0, self.init_tempo, self.init_tempo_var)}

        # Tracking state
        self.input_index = 0
        self.current_index = 0
        self.queue_timeout = QUEUE_TIMEOUT
        self.latency_stats = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

        # For compatibility with matchmaker eval
        self.patience = 0

    def _sigma_eps(self, t_k: float) -> float:
        """Onset noise std: proportional to tempo."""
        return self.sigma_eps_scale * abs(t_k)

    def _sigma_eta(self, l_k: float, t_k: float) -> float:
        """Tempo noise std: proportional to l_k * t_k."""
        return self.sigma_eta_scale * abs(l_k * t_k)

    def _transition_prob(
        self, age: int, l_k: float, mu_t: float, var_t: float, k: int
    ) -> float:
        """Compute stay probability p1 (Eq. 12).

        p1 = P(L_k >= (a+1)*delta | L_k >= a*delta)

        where L_k ~ N(l_k * mu_t, sigma_eps^2 + l_k^2 * sigma_t^2).

        Returns the probability of staying in the current chord.
        """
        # Mean and variance of note duration L_k
        mu_L = l_k * mu_t
        sigma_eps = self._sigma_eps(mu_t)
        var_L = sigma_eps**2 + l_k**2 * var_t

        if var_L <= 0:
            var_L = 1e-12
        sigma_L = np.sqrt(var_L)

        a_cur = age * self.delta
        a_next = (age + 1) * self.delta

        # p1 = (1 - Phi((a_next - mu) / sigma)) / (1 - Phi((a_cur - mu) / sigma))
        # Using ndtr (normal CDF) which is faster than scipy.stats.norm.cdf
        z_cur = (a_cur - mu_L) / sigma_L
        z_next = (a_next - mu_L) / sigma_L
        denom = 1.0 - ndtr(z_cur)
        if denom < 1e-300:
            return 0.0  # note has exceeded expected duration, must advance

        numer = 1.0 - ndtr(z_next)
        p1 = numer / denom

        return float(np.clip(p1, 0.0, 1.0))

    def _kalman_update(
        self,
        mu_t_prev: float,
        var_t_prev: float,
        k_prev: int,
        age_prev: int,
    ) -> Tuple[float, float]:
        """Kalman filter update when a new note onset is observed.

        When transitioning from chord k_prev to k_prev+1, we observe
        the duration of chord k_prev (= age_prev * delta).

        The observation model is:
            L_k = l_k * t_k + eps_k  (Eq. 2)
            t_{k+1} = t_k + eta_k    (Eq. 3)

        Returns updated (mu_t, var_t) for the new chord's tempo.
        """
        l_k = self.lengths[k_prev]
        observed_duration = age_prev * self.delta  # actual note duration in seconds

        sigma_eps = self._sigma_eps(mu_t_prev)

        # Kalman update: observation L_k = l_k * t_k + eps_k
        # Prior: t_k ~ N(mu_t_prev, var_t_prev)
        # Likelihood: L_k | t_k ~ N(l_k * t_k, sigma_eps^2)
        # Posterior: t_k | L_k ~ N(mu_post, var_post)
        obs_var = sigma_eps**2
        prior_var = var_t_prev

        if l_k < 1e-10:
            mu_post = mu_t_prev
            var_post = prior_var
        else:
            innov = observed_duration - l_k * mu_t_prev
            innov_var = l_k**2 * prior_var + obs_var

            if innov_var < 1e-20:
                innov_var = 1e-20

            K_gain = (l_k * prior_var) / innov_var
            mu_post = mu_t_prev + K_gain * innov
            var_post = prior_var - K_gain * l_k * prior_var

        # Predict: t_{k+1} = t_k + eta_k
        # sigma_eta uses posterior mean (paper: proportional to l_k * t_k)
        sigma_eta = self._sigma_eta(l_k, mu_post)
        mu_new = mu_post
        var_new = max(var_post, 0.0) + sigma_eta**2

        return mu_new, var_new

    def _compute_all_likelihoods(self, y: np.ndarray) -> np.ndarray:
        """Compute spectral likelihoods for all chords at once (Eq. 1).

        p(y|k) = prod_w (q^k_w)^{y'_w}
        log p(y|k) = sum_w y'_w * log(q^k_w) = log_templates[k] @ y_norm

        We subtract max(log_liks) so the best-matching chord gets
        likelihood 1.0. This is equivalent to dividing all likelihoods
        by a common constant (the evidence p(y)), which cancels in
        the posterior normalization (Eq. 14-17).
        """
        s = y.sum()
        if s <= 0:
            return np.full(self.K, 1e-300)
        y_norm = y / s

        # (K, n_bins) @ (n_bins,) -> (K,)
        log_liks = self.log_templates @ y_norm
        log_liks -= log_liks.max()
        liks = np.exp(np.clip(log_liks, -500, 0))
        return np.maximum(liks, 1e-300)

    def _extract_features(self, audio_frame: np.ndarray) -> np.ndarray:
        """Extract magnitude spectrum from audio frame (paper Eq. 1)."""
        frame = np.asarray(audio_frame, dtype=float)
        if frame.ndim == 2:
            frame = frame[-1]
        frame = frame.reshape(-1)

        # Already a spectrum from RawSpectrumProcessor
        if len(frame) == self.n_fft // 2 + 1:
            return np.maximum(frame, 0.0)

        # Raw audio: compute magnitude spectrum
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))
        elif len(frame) > self.n_fft:
            frame = frame[: self.n_fft]

        spectrum = np.abs(np.fft.rfft(frame, n=self.n_fft))
        return spectrum

    def __call__(self, observation, perf_time: float) -> float:
        """Process one audio frame.

        Parameters
        ----------
        observation : np.ndarray
            Feature vector from AudioStream.
        perf_time : float
            Performance time (seconds) of this frame.

        Returns
        -------
        float
            Current beat position (chord onset + within-chord interpolation).
        """
        t0 = time.time()
        self.current_perf_time = float(perf_time)
        y = self._extract_features(observation)

        # Precompute data likelihoods for all chords (Eq. 1)
        likelihoods = self._compute_all_likelihoods(y)

        new_hypotheses = {}

        def _merge_into(hyps, key, prob, mu, var):
            """Merge a hypothesis into the dict via moment matching."""
            if key in hyps:
                old_prob, old_mu, old_var = hyps[key]
                total = old_prob + prob
                w_old = old_prob / total
                w_new = prob / total
                m_mu = w_old * old_mu + w_new * mu
                m_var = w_old * (old_var + old_mu**2) + w_new * (var + mu**2) - m_mu**2
                hyps[key] = (total, m_mu, max(m_var, 1e-20))
            else:
                hyps[key] = (prob, mu, var)

        for (k, a), (prob, mu_t, var_t) in self.hypotheses.items():
            # --- Case 1: Stay in same chord (Eq. 14-15)
            p_stay = self._transition_prob(a, self.lengths[k], mu_t, var_t, k)

            stay_prob = prob * p_stay * likelihoods[k]
            if stay_prob > PROB_FLOOR:
                _merge_into(new_hypotheses, (k, a + 1), stay_prob, mu_t, var_t)

            # --- Case 2: Advance to next chord (Eq. 16-17)
            next_k = k + 1
            if next_k < self.K:
                p_advance = 1.0 - p_stay
                adv_mu, adv_var = self._kalman_update(mu_t, var_t, k, a)
                adv_prob = prob * p_advance * likelihoods[next_k]
                if adv_prob > PROB_FLOOR:
                    _merge_into(new_hypotheses, (next_k, 1), adv_prob, adv_mu, adv_var)

        # --- Beam pruning: keep top max_hypotheses by probability
        if len(new_hypotheses) > self.max_hypotheses:
            sorted_items = sorted(
                new_hypotheses.items(), key=lambda x: x[1][0], reverse=True
            )
            new_hypotheses = dict(sorted_items[: self.max_hypotheses])

        # --- Normalize probabilities
        total_prob = sum(p for p, _, _ in new_hypotheses.values())
        if total_prob > 0:
            self.hypotheses = {
                key: (p / total_prob, mu, var)
                for key, (p, mu, var) in new_hypotheses.items()
            }
        else:
            # Fallback: reset to current best state
            self.hypotheses = {
                (self.current_index, 1): (
                    1.0,
                    self.init_tempo,
                    self.init_tempo_var,
                )
            }

        # --- Determine current state: highest total probability per chord k
        chord_probs = np.zeros(self.K)
        chord_weighted_age = np.zeros(self.K)
        chord_weighted_mu = np.zeros(self.K)
        for (k, a), (prob, mu_t, _) in self.hypotheses.items():
            chord_probs[k] += prob
            chord_weighted_age[k] += prob * a
            chord_weighted_mu[k] += prob * mu_t

        best_k = int(np.argmax(chord_probs))
        self.current_index = best_k

        # Interpolate position within chord using expected age and tempo
        base_beat = float(self.onset_beats[best_k])
        if chord_probs[best_k] > 0:
            avg_age = chord_weighted_age[best_k] / chord_probs[best_k]
            avg_mu = chord_weighted_mu[best_k] / chord_probs[best_k]
            elapsed_in_chord = (avg_age - 1) * self.delta
            if best_k + 1 < self.K:
                chord_dur_beats = (
                    self.onset_beats[best_k + 1] - self.onset_beats[best_k]
                )
                expected_dur_sec = self.lengths[best_k] * avg_mu
                if expected_dur_sec > 0:
                    frac = min(elapsed_in_chord / expected_dur_sec, 1.0)
                    position = base_beat + frac * chord_dur_beats
                else:
                    position = base_beat
            else:
                position = base_beat
        else:
            position = base_beat

        self.current_position = position
        self._alignment_path.append((position, self.current_perf_time))
        self.input_index += 1
        self.latency_stats = set_latency_stats(
            time.time() - t0, self.latency_stats, self.input_index
        )

        return self.current_position
