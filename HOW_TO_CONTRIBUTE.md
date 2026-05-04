# How to Contribute to Matchmaker

This guide walks through adding a new score follower (online alignment method)
to Matchmaker. Every score follower subclasses `matchmaker.base.OnlineAlignment`.

For the package architecture and pipeline overview (Stream → Processor →
OnlineAlignment), see the [Architecture section in the README](README.md#architecture).

## Implementing a new score follower

A new score follower class must inherit from `matchmaker.base.OnlineAlignment` and implement the `step(features)` method.
The base class provides defaults for `__call__`, `run`, `alignment_path`.

**Key attributes inside `step()`**:

| Attribute | Type | Meaning |
|---|---|---|
| `self.reference_features` | ndarray (shape depends on score follower — e.g., `(N_ref, n_features)` for frame DP, structured `note_array` for HMM) | Score-side features (set at construction). |
| `self.score_positions` | ndarray of shape `(N_states,)`, dtype float | Per-state beat values; one entry per unique onset (or per state of your model). |
| `self.current_index` | int | Index into `score_positions`. **Update this in `step()`.** |
| `self.current_position` | float | `score_positions[current_index]` by default. Override `get_current_position()` for finer precision. |
| `self.current_perf_time` | float | Performance time (seconds) of the latest observation. |
| `self.alignment_path` | ndarray of shape `(2, T)`, dtype float | Read-only; row 0 = score beats, row 1 = perf times. |

## Boilerplate

```python
from matchmaker.base import OnlineAlignment


class MyScoreFollower(OnlineAlignment):
    def __init__(self, reference_features, score_positions, queue=None):
        super().__init__(
            reference_features=reference_features,
            score_positions=score_positions,
            queue=queue,
        )
        # initialize any algorithm-specific state here

    def step(self, features) -> None:
        # update self.current_index from the observation
        ...
```

## What `OnlineAlignment` provides

The base class implements the rest of the loop.

| Method | Returns | Default behavior |
|---|---|---|
| `step(features: Any)` | (none) — updates `self.current_index` | **abstract** — must be implemented |
| `__call__(observation: Tuple[Any, float])` | `current_position: float` | unpacks `(features, perf_time)`, calls `step()`, sets `current_position`, appends to `alignment_path` |
| `run(verbose: bool = True)` | <ul><li>yields: <code>current_position: float</code> per step</li><li>returns: <code>alignment_path: np.ndarray</code> of shape <code>(2, T)</code></li></ul> | generator pulling items from `self.queue` until `STREAM_END` |
| `get_current_position()` | `current_position: float` | `score_positions[current_index]` (snaps to nearest onset) |
| `is_still_following()` | `bool` | `current_index < len(score_positions) - 1` |
| `alignment_path` (property) | `np.ndarray` of shape `(2, T)` | accumulated from each `__call__`; row [0] = score beats, row [1] = perf times (seconds) |

### The `alignment_path`

`score_follower.alignment_path` is a `(2, T)` numpy array:

- Row 0: score beat positions (`current_position` values, float).
- Row 1: performance times in seconds (`perf_time` values).

Each column is one `(score, perf)` pairing — `wp[1][i]` is when the performer
reached beat `wp[0][i]`.

### Choosing `reference_features` for a new score follower

The base class accepts any object as `reference_features`; pick whatever
shape your algorithm reads from. Two common cases:

**1. Your follower can reuse an existing representation.**
Existing followers fall into three groups:

| Representation | Used by | Provided by |
|---|---|---|
| partitura `note_array()` (default) | `PitchHMM`, `PitchIOIHMM`, `OuterProductHMM`, `AudioOuterProductHMM`, `SwitchingKalmanFilterFollower` | `Matchmaker.preprocess_score()` returns it directly |
| `(N_ref_frames, n_features)` audio feature matrix | `OnlineTimeWarpingArztFrame`, `OnlineTimeWarpingDixonFrame` | `Matchmaker.preprocess_score()` synthesizes score audio and runs the processor |
| `(N_states, 88)` onset pianoroll | `OnlineTimeWarpingArztEvent`, `OnlineTimeWarpingDixonEvent` | `Matchmaker._build_symbolic_follower()` calls `onset_pianoroll(...)` |

If your follower wants one of these, add a branch in
`_build_audio_follower` / `_build_symbolic_follower` that takes
`self.reference_features` and constructs your class.

**2. Your follower needs a new representation.**
Add a new branch in `Matchmaker.preprocess_score()` (or transform inside
`_build_*_follower`) that produces what you need from `self.score_part`,
then pass it to your constructor.

Examples in the codebase:
- `OnlineParangonarAlignment` calls `score_part.note_array(include_grace_notes=True)` inside its build branch (parangonar matchers need the `is_grace` column).
- MIDI OLTW transforms `note_array` → onset pianoroll inside the build function.

## Two ways to run the score follower

The score follower can be fed observations directly or via a queue. Both produce the
same `current_position` and `alignment_path`.

| | `__call__(observation)` | `run()` |
|---|---|---|
| Source of observations | caller passes them | `self.queue.get()` |
| Termination | caller stops iterating | `STREAM_END`(sentinel `object()`) arrives, or `is_still_following()` returns `False` (score follower reached the last `score_positions` entry) |

```python
# 1. Per-observation
score_follower = MyScoreFollower(reference_features=..., score_positions=...)
for features, perf_time in observations:
    beat = score_follower((features, perf_time))

# 2. Stream-driven
score_follower = MyScoreFollower(reference_features=..., score_positions=..., queue=q)
for beat in score_follower.run(verbose=True):
    ...
wp = score_follower.alignment_path  # (2, T)
```

`run()` calls `__call__` internally, so all bookkeeping (path append, position
update) goes through one place.

## Plugging into `Matchmaker`

Register the tracker in `matchmaker/matchmaker.py`:

1. Add the method name to `AVAILABLE_METHODS["audio"]` or
   `AVAILABLE_METHODS["midi"]`.
2. Add an entry in `DEFAULT_KWARGS` if your method needs specific kwargs.
3. Add a branch in `_build_audio_follower` or `_build_symbolic_follower`.

Then it works through the public API:

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="matchmaker/assets/simple_mozart_k265_var1.musicxml",
    performance_file="matchmaker/assets/simple_mozart_k265_var1.mp3",
    input_type="audio",
    method="my_score_follower",
)
for beat in mm.run():
    print(beat)
```
