# How to implement your own score follower

This guide walks you through adding a new score follower (online alignment method) along with its feature processing to the Matchmaker framework.

You will need:
1. a score follower class
2. an input feature processor

In the following we look the requirements for each part.
For the package architecture and pipeline overview (Stream → Processor →
OnlineAlignment), see the [Architecture section in the README](README.md#architecture).

## 1. Custom Score Followers

Every score follower subclasses `matchmaker.base.OnlineAlignment`.
A new score follower class **must** inherit from `matchmaker.base.OnlineAlignment` and implement the `step(features)` and `get_current_position` methods.

| Method                        | Input                                                                                                                            | Output                         | Internal                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| `self.step(features)`         | features: whatever input observation your score follower requires. choose a fitting or custom `Processor` to provide this object | None                           | main update step, must update reference position estimation. |
| `self.__init__(**kwargs)`     | add arguments as needed                                                                                                          | None                           | all setup logic                                              |
| `self.get_current_position()` | None                                                                                                                             | current position in beat units | defaults to indexing                                         |
### Examples

Here is an extremely simple score follower with random steps:

```python
from matchmaker.base import OnlineAlignment
import numpy as np

class SimplestFollower(OnlineAlignment):

    def step(self, features) -> None:
        pass

    def get_current_position(self):
	    return self.current_position + np.random.rand()
```

The default `OnlineAlignment` already contains some useful optional logic. Very often, a score follower moves between pre-defined positions which can be indexed. To this end, the default  `OnlineAlignment` defines an index `self.current_idx`. We can use this with an externally computed score_positions (a numpy vector, beat positions corresponding to each position):
```python
from matchmaker.base import OnlineAlignment

class IndexFollower(OnlineAlignment):
     def __init__(self,
	     score_positions,
	     **kwargs):
        super().__init__(
            score_positions=score_positions,
            **kwargs,
        )

    def step(self, features) -> None:
	    # this tracker just marches forward for every input
	    self.current_idx += 1


```

Note that we didn't have to define `self.get_current_position()` as its default behavior is given by `return float(self.score_positions[self.current_index])`, and we have all this in place in the base class.
### Fixed internals

The `matchmaker.base.OnlineAlignment` base class provides defaults for `__call__`, `run`, `alignment_path`. This enables full use of the matchmaker ecosystem including real-time tracking and offline evaluation. To this end the base class uses the following attributes and methods. They can be accessed by your custom tracking logic, but **must not be overwritten** to keep matchmaker functionality.

#### OnlineAlignment base class attributes

| Attribute                | Type                                   | Meaning                                                                                                                      |
| ------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `self.current_position`  | float                                  | `score_positions[current_index]` by default. Override `get_current_position()` for finer precision.                          |
| `self.current_perf_time` | float                                  | Performance time (seconds) of the latest observation.                                                                        |
| `self.alignment_path`    | ndarray of shape `(2, T)`, dtype float | Read-only; row 0 = perf times (seconds), row 1 = score beats. Accessible for a running or finished tracker, None for a yet unused one. |
#### OnlineAlignment base class methods

| Method                             | Returns                                                                                                                                                       | Behavior                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `__call__(observation, perf_time)` | `current_position: float`                                                                                                                                     | unpacks `(features, perf_time)`, calls `step()`, sets `current_position`, appends to `alignment_path` |
| `run(verbose: bool = True)`        | <ul><li>yields: <code>current_position: float</code> per step</li><li>returns: <code>alignment_path: np.ndarray</code> of shape <code>(2, T)</code></li></ul> | generator pulling items from `self.queue` until `STREAM_END`                                          |
| `get_current_position()`           | `current_position: float`                                                                                                                                     | `score_positions[current_index]` (snaps to nearest onset)                                             |
| `is_still_following()`             | `bool`                                                                                                                                                        | `current_index < len(score_positions) - 1`                                                            |
| `alignment_path` (property)        | `np.ndarray` of shape `(2, T)`                                                                                                                                | accumulated from each `__call__`; row [0] = perf times (seconds), row [1] = score beats               |

#### OnlineAlignment reference_features

The default `OnlineAlignment` class has an optional `reference_features` argument. When using the `MatchMaker` top-level object, it passes a score note array as reference_feature to the score follower, i.e. a structured numpy array with field such as "onset_beat" and "duration_beat". This feature is computed with the help of the `partitura` library. Any `OnlineAlignment` subclass is responsible for converting this note array into features it can use internally.

The `reference_features = score note array` is the default interface for passing reference information to `OnlineAlignment` subclasses and it is the preferred way whenever possible. Custom processing for other types of references is also possible, albeit not within the `MatchMaker` object.

---

## 2. Custom Input Processors

TODO: add minimal outline of a Processor (streamed and real-time)
- must implement call method
- input depends on MidiStream or Audiostream
- must return features, p_time -> features must fit the step method of the score follower
- is stateless / called by stream
- must not change: __
- can optionally change/add: __


---

## Use a Custom Score Follower

The score follower can be fed observations directly or via a Stream. Both produce the
same `current_position` and `alignment_path`.

|                        | `__call__(observation)` | `run()`                                                                                                                                        |
| ---------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Source of observations | caller passes them      | `self.queue.get()`                                                                                                                             |
| Termination            | caller stops iterating  | `STREAM_END`(sentinel `object()`) arrives, or `is_still_following()` returns `False` (score follower reached the last `score_positions` entry) |

This means you can process the inputs as they are expected by your score follower and loop over them. This is useful for debugging and offline tests:

```
score_follower = MyScoreFollower(**kwargs)
for features, perf_time in my_processed_features:
    beat = score_follower((features, perf_time))
```

For real-time usage, you can plug your custom Processor into a stream object (which requires further arguments to set up ports etc.) and use it in your Score Follower like so:

```
# 2. Stream-driven

processor = MyInputProcessor(**kwargs)
with MidiStream(processor = processor) as stream: # or AudioStream
    score_follower = MyScoreFollower(queue = stream.queue, **kwargs)
	for current_position in self.score_follower.run():
		print("current position [beats]", current_position)
```

## Make it accessible in `Matchmaker`

Call `register_method()`. Your follower does not have to live in this package —
registration works from any module, so a tracker you keep in your own project
(or submit to the benchmark) plugs into the same pipeline as a built-in one.

```python
import numpy as np
from matchmaker import register_method

register_method(
    "my_score_follower",
    input_type="audio",
    build_follower=lambda mm: MyScoreFollower(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        frame_rate=mm.frame_rate,
        **mm.config,
    ),
    default_kwargs={"processor": "chroma", "frame_rate": 30},
)
```

`build_follower(mm)` is called once per `Matchmaker`, after the score is loaded
and the stream exists. Read what you need off `mm`:

| | |
| --- | --- |
| `mm.score_part` | the unfolded, merged partitura `Part` |
| `mm.score_positions` | ascending score beat of every note onset |
| `mm.reference_features` | the score-side features (see below) |
| `mm.tempo` | notated tempo in BPM, or 120 if the score has no marking |
| `mm.frame_rate`, `mm.sample_rate`, `mm.hop_length` | audio stream settings |
| `mm.config` | whatever `default_kwargs` / `kwargs=` supplied |
| `mm.stream.queue` | pass this as your follower's `queue` |

Two optional hooks handle the cases where the defaults do not fit:

- **`build_processor(mm)`** — omit it and Matchmaker builds the standard
  processor named by `default_kwargs["processor"]`, which is usually what you
  want. Pass one only for a processor of your own.
- **`build_reference(mm)`** — omit it for the score note array. Audio followers
  that align against a synthesised score rendering override it:

  ```python
  def build_reference(mm):
      from matchmaker.utils.misc import generate_score_audio
      audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
      features, _ = mm.processor((audio.astype(np.float32), 0.0))
      mm.processor.reset()   # the same processor then handles the live input
      return features
  ```

  Frame-based audio followers usually also want `mm.ref_frame_to_beat()`, the
  score beat of each reference frame.

Registering under the name of a built-in method, or twice under the same name,
raises — pass `overwrite=True` if replacing is what you meant.
`unregister_method(name, input_type)` undoes it, which is mostly useful in
tests.

Either way, it then works through the public API:

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
