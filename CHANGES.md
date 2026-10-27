# Changes

## Version 0.4.0

New score followers, a method registry, and a standardized evaluation output. Not backwards compatible with 0.3.

### New Features

- Korzeniowski particle filter for audio and MIDI (`method="pfkorz"`), and the OLTW variants `arzt_tempo` and `arzt_multi_fold`.
- `register_method()` and a YAML method registry for custom score followers.
- SPARC smoothness in `evaluate_alignment()`, using configurable onset-wise windows and last-position hold resampling. Compare smoothness only within the same input type (audio or MIDI).
- Python 3.13 and NumPy 2 support.

### Breaking Changes

- `method="pf"` (`ParticleFilter`) is removed; use `method="pfkorz"`.
- `alignment_path` rows are now `(performance time in seconds, score position in beats)`, the reverse of 0.3.
- Custom followers now receive observations through `__call__(observation, perf_time)` instead of a single `(observation, perf_time)` tuple.
- `evaluate_alignment()` returns flat fields (`mean_b`, `median_b`, `std_b`, `accuracy_<tolerance>b`, `skewness`, `kurtosis`, `smoothness`).
- Removed millisecond-based alignment metrics (`in_seconds`, `TOLERANCES_IN_MILLISECONDS`, `ms_tolerances`) and the `return_details` option on `evaluate_alignment()`.
- Latency keys are renamed from `f_avg_latency` and `i_avg_latency` to `feature_latency_ms` and `inference_latency_ms`.

### Bug Fixes

- Corrected CQT/STFT magnitude features, non-4/4 chord durations, MIDI onset timestamps, and grace-note tempo initialization.
- Included bundled example assets in distribution wheels.

## Version 0.3.0

This is a major feature release introducing new symbolic and audio score-following trackers, a unified interface refactor, particle filters, bug fixes, and documentation improvements. This version is not backwards compatible with 0.2 versions.

### New Features

#### Outer Product HMM — MIDI Symbolic Tracker (PR #41)

- New `OuterProductHMM` tracker for MIDI score following
- Cython-accelerated Viterbi step for improved performance
- State probability normalization inside the Cython Viterbi step
- Set as the default tracker for MIDI in the `Matchmaker` class

#### Symbolic OLTWArzt Tracker (PR #49)

- New symbolic OLTWArzt tracker for event-based score following
- Major refactor to a fully event-based interface

#### Symbolic  OLTWDixon Tracker (PR #50)

- New symbolic `OLTWDixon` tracker (event-based)
- Updated implementation and integration into the unified wrapper

#### Audio Trackers (PR #51)

- New audio score-following trackers added and integrated
- Wrapper extended for four Parangonar-based trackers
- Updated benchmark code for audio trackers

#### MIDI Trackers — Unified Interface (PR #55)

- Major MIDI tracker refactor for a unified interface
- Fixed MIDI score-following pipeline for HMM methods:
  - Skip `None` frames in `MidiStream` queue (empty polling frames)
  - Removed patience-based early termination in `BaseHMM` / `PitchIOIHMM`
  - Warping path now stored as `(beat, time)` floats instead of `int32`
  - `return_pitch_list=True` enabled for `PitchIOIHMM`
  - Handle `(pitch, ioi)` tuple input in `PitchHMM` for time tracking
  - Fixed NaN handling in `plot_alignment` beat ticks
- Default MIDI tracker changed to `PitchIOIHMM` (pthmm)

#### Particle Filters (PR #56)

- Single `ParticleFilter` class implemented for both MIDI and Audio
- Removed `auto_adjust_tempo` dependency

#### Tempo Parsing from Score (PR #42)

- Parse tempo directly from MusicXML scores
- Score marking and text tempo parsing
- Tempo adjustment now configurable as an option

#### BytesMidiStream & BytesAudioStream

- Added `BytesAudioStream` for web audio input
- Added `polling_period` argument in `BytesMidiStream` for web MIDI input
- Applied 1ms `polling_period` to arzt/dixon/outerhmm trackers

### Bug Fixes

- Fixed bug in calculating adjusted BPM of the score (PR #41)
- Fixed symbolic ARZT and Dixon trackers to be fully event-based
- Fixed bug in external matcher `__call__` method
- Fixed bug when empty observation is received
- Fixed typo in `feature_type` parameter
- Fixed lazy import for `parangonar` on Python 3.10
- Fixed Cython file naming issues
- Fixed NaN handling in `plot_alignment` beat ticks
- Fixed incorrect path in README installation instructions (PR #52)

### Improvements & Refactoring

- Unified interface refactor across MIDI and audio trackers
- Separated arguments in `__call__` method for clarity
- Removed `PitchChordProcessor` (deprecated)
- Fully deprecated `pkg_resources` — replaced with modern alternatives (PR #60)
- Updated assets and package management
- Normalized state probabilities in Viterbi step (Python and Cython)
- Score performance used directly for time mapping
- Added note IDs to the test score file
- Source length retrieved directly from score MIDI
- Added `pandas` as a dependency in `pyproject.toml`

### Documentation & Project

- Updated README with current usage, API, and installation instructions
- Added `CONTRIBUTING.md` and how-to-contribute guide
- Updated version strings to `0.3.0` (PR #60)

## Version 0.2.1

- Add evaluation function with performance annotation file
- Add `run_examples.py` for simple testing of the package
- Add more feature processors
- Bug fixes

## Version 0.2.0

- Add beat-level evaluation and assessment capabilities
- Adjustable parameters for distance functions and frame rates
- New HMM features for audio analysis
- Particle filter implementations

## Version 0.1.3

- Fix version scope of the `scipy` dependency

## Version 0.1.2

- Fix pinned versions of `numpy` and `scipy`
- Add automatic linting and formatting via `ruff`
- Update recommended Python version to 3.12

## Version 0.1.1

- Add documentation via ReadTheDocs
- Publish package on PyPI via GitHub Actions
- Rename package to `pymatchmaker`

## Version 0.1.0

First release.
