# Matchmaker

Matchmaker is a Python library for real-time music alignment.

Music alignment is a fundamental MIR task, and real-time music alignment is a necessary component of many interactive applications (e.g., automatic accompaniment systems, automatic page turning).

Unlike offline alignment methods, for which state-of-the-art implementations are publicly available, real-time (online) methods have no standard implementation, forcing researchers and developers to build them from scratch for their projects.

We aim to provide efficient reference implementations of score followers for use in real-time applications which can be easily integrated into existing projects.

The full documentation for matchmaker is available online at [readthedocs.org](https://pymatchmaker.readthedocs.io/).

## Setup

### Prerequisites

- Available Python version: 3.10, 3.11, 3.12, 3.13

### Install

```bash
pip install pymatchmaker
```

The base installation supports **simulation mode** (testing with performance files). You can also implement your own `Stream` subclass to integrate external input sources (e.g., WebSocket) without any extra dependencies.

To use **local audio/MIDI devices** (microphone, MIDI keyboard), install with the `devices` extra, which adds [PyAudio](https://pypi.org/project/PyAudio/), [python-rtmidi](https://pypi.org/project/python-rtmidi/), and [pyfluidsynth](https://pypi.org/project/pyfluidsynth/):

```bash
pip install pymatchmaker[devices]
```

> **Note:** `pyfluidsynth` requires [Fluidsynth](https://www.fluidsynth.org/) to be installed on your system, and `pyaudio` requires [PortAudio](http://www.portaudio.com/). We recommend installing them via conda: `conda install -c conda-forge fluidsynth portaudio`.

### Install from source using conda

Setting up the code as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

```bash
# Clone matchmaker
git clone https://github.com/pymatchmaker/matchmaker.git

# Create the conda environment
conda create -n matchmaker python=3.12

conda activate matchmaker

# Go to matchmaker directory
cd matchmaker

# Install matchmaker in editable mode (dev includes devices)
pip install -e ".[dev]"

# Install GCC
conda install -c conda-forge gcc=12.1.0

# Install glib, fluidsynth, and portaudio
conda install -c conda-forge glib fluidsynth portaudio
```

Because of the dependency of `partitura`, which uses `MuseScore_General.sf3` (free soundfont provided by MuseScore) as the default soundfont, the soundfont will be installed automatically inside the `partitura` package. This might take a while for the first time.

### Known Setup Issues

#### Missing Visual C++ build tools (on Windows)

The solution seems to be to download vs_BuildTools.exe from <https://visualstudio.microsoft.com/visual-cpp-build-tools/> and then execute

```bash
vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools
```

#### Issues with Fluidsynth and pyfluidsynth on Windows

On Windows, pyfluidsynth expects fluidsynth.exe to be located in `C:\tools\bin` (other users have reported that it is expected in `C:\tools\fluidsynth\bin`). You can fix the issue by

1. Get the ZIP file for your Windows version from <https://github.com/FluidSynth/fluidsynth/releases/latest>
2. Extract the contents to `C:\tools` (or wherever pyfluidsynth expects the executable to be).

#### Using Fluidsynth installed from Homebrew on MacOS

We recommend to install Fluidsynth from conda in a dedicated environemnt. If however, you want to use the system-wide Fluidsynth installed with homebrew, you might run into an `ImportError("Couldn't find the FluidSynth library.")` with `pyfluidsynth`.  Please refer to the following [link](https://stackoverflow.com/a/75339618).

## Usage Examples

### Quickstart (simulation mode)

You can test the alignment with a score and a performance file. No extra dependencies needed beyond the base install.

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file=”path/to/score.musicxml”,
    performance_file=”path/to/performance.wav”,
    input_type=”audio”,
)
for current_position in mm.run():
    print(current_position)  # beat position in the score
```

The returned value is the current position in the score, represented in beats defined by `partitura` library's note array system.
Specifically, each position is calculated for every frame input and interpolated within the score's `onset_beat` array.
Please refer to [here](https://partitura.readthedocs.io/en/latest/Tutorial/notebook.html) for more information about the `onset_beat` concept.

### Live streaming (requires `[devices]`)

To run with a live audio or MIDI input, install with `pip install pymatchmaker[devices]`.

```python
from matchmaker import Matchmaker

# Audio input (microphone)
mm = Matchmaker(
    score_file=”path/to/score.musicxml”,
    input_type=”audio”,
)
for current_position in mm.run():
    print(current_position)
```

You can also specify a device by name or index:

```python
mm = Matchmaker(
    score_file=”path/to/score.musicxml”,
    input_type=”audio”,
    device_name_or_index=”MacBookPro Microphone”,
)
```

### Using a Custom Stream

You can inject your own stream (e.g., from a WebSocket or any external source) without needing `pyaudio` or `python-rtmidi`:

```python
import queue
from matchmaker import Matchmaker
from matchmaker.io.stream import Stream
from matchmaker.features.audio import ChromagramProcessor

class MyStream(Stream):
    """Custom stream that receives audio data from an external source."""

    def __init__(self, processor, data_source):
        super().__init__(processor=processor, mock=False)
        self.data_source = data_source
        self.queue = queue.Queue()

    def run(self):
        """Read data from your source and push features to the queue."""
        for chunk in self.data_source:
            features = self.processor(chunk)
            self.queue.put((features, time.time()))

my_stream = MyStream(
    processor=ChromagramProcessor(),
    data_source=my_websocket_source,
)

mm = Matchmaker(
    score_file="path/to/score",
    input_type="audio",
    stream=my_stream,
)
for current_position in mm.run():
    print(current_position)
```

### Running Examples

The repository includes a ready-to-use example script that demonstrates the complete workflow:

```bash
# Run with audio input (uses arzt method as default)
python run_examples.py --audio

# Run with MIDI input and specific method
python run_examples.py --midi --method hmm
```

This script runs a complete example with score following and evaluation, saving results to the `results/` directory.

### Testing with Different Methods or Features

You can specify the alignment method and feature processor as follows:

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="path/to/score",
    input_type="audio",
    method="arzt",       # see Alignment Methods section
    processor="chroma",  # see Features section
)
for current_position in mm.run():
    print(current_position)
```

For options regarding the `method`, please refer to the [Alignment Methods](#alignment-methods) section.
For options regarding the `processor`, please refer to the [Features](#features) section.


## Alignment Methods

Matchmaker currently supports the following alignment methods:

- `"arzt"`: On-line time warping algorithm adapted from Brazier and Widmer (2020) (based on the work by Arzt et al. (2010)). Supports audio and MIDI input.
- `"dixon"`: On-line time warping algorithm by S. Dixon (2005). Supports audio and MIDI input.
- `"outerhmm"`: Outer-product HMM score follower by E. Nakamura (2014). Supports audio and MIDI input.
- `"hmm"`: Hidden Markov Model-based score follower by Cancino-Chacón et al. (2023), based on the state-space score followers by Duan et al. (2011) and Jiang and Raphael (2020). Supports MIDI input.
- `"pthmm"`: Pitch-based HMM score follower. Supports MIDI input.

## Features

Matchmaker currently supports the following feature types:

- For audio:
  - `"chroma"`: Chroma features. Default for audio input.
  - `"mfcc"`: Mel-frequency cepstral coefficients.
  - `"cqt"`: Constant-Q transform.
  - `"mel"`: Mel-spectrogram.
  - `"lse"`: Log-spectral energy features used in Dixon (2005).
  - `"cqt_spectral_flux"`: CQT-based spectral flux used in Nakamura et al. (2014).
- For MIDI:
  - `"pitch_ioi"`: Pitch and inter-onset interval features. Default for MIDI input.
  - `"pianoroll"`: Piano-roll features.
  - `"pitchclass"`: Pitch class features.

## Configurations

Initialization parameters for the `Matchmaker` class:

- `score_file` (str): Path to the score file.
- `input_type` (str): Type of input data. Options: `"audio"`, `"midi"`.
- `method` (str): Alignment method to use. See [Alignment Methods](#alignment-methods) for available options.
- `processor` (str): Type of feature processor to use. See [Features](#features) for available options.
- `stream` (Stream or None): A custom `Stream` instance to use instead of the built-in `AudioStream`/`MidiStream`. Useful for integrating external input sources (e.g., WebSocket). If `None`, the stream is built automatically based on `input_type`.
- `device_name_or_index` (str or int): The audio/MIDI device name or index you want to use. If `None`, the default device will be used. Requires `pymatchmaker[devices]`.

## Citing Matchmaker

If you find Matchmaker useful, we would appreciate if you could cite us!

```bibtex
@inproceedings{park_matchmaker_2025,
	title = {Matchmaker: {An} {Open}-{Source} {Library} for {Real}-{Time} {Piano} {Score} {Following} and {Systematic} {Evaluation}},
	booktitle = {Proceedings of the 26th {International} {Society} for {Music} {Information} {Retrieval} {Conference} ({ISMIR} 2025)},
	author = {Park, Jiyun and Cancino-Chacón, Carlos and Chiruthapudi, Suhit and Nam, Juhan},
    address = {Daejeon, South Korea}
	year = {2025}
}
```

```bibtex
@inproceedings{matchmaker_lbd,
  title={{Matchmaker: A Python library for Real-time Music Alignment}},
  author={Park, Jiyun and Cancino-Chac\'{o}n, Carlos and Kwon, Taegyun and Nam, Juhan},
  booktitle={{Proceedings of the Late Breaking/Demo Session at the 25th International Society for Music Information Retrieval Conference}},
  address={San Francisco, USA.},
  year={2024}
}
```



## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("*Rach3: A Computational Approach to Study Piano Rehearsals*"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
