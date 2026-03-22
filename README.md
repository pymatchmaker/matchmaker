# Matchmaker

Matchmaker is a Python library for real-time music alignment.

Music alignment is a fundamental MIR task, and real-time music alignment is a necessary component of many interactive applications (e.g., automatic accompaniment systems, automatic page turning).

Unlike offline alignment methods, for which state-of-the-art implementations are publicly available, real-time (online) methods have no standard implementation, forcing researchers and developers to build them from scratch for their projects.

We aim to provide efficient reference implementations of score followers for use in real-time applications which can be easily integrated into existing projects.

The full documentation for matchmaker is available online at [readthedocs.org](https://pymatchmaker.readthedocs.io/).

## Setup

### Prerequisites

- Available Python version: 3.10, 3.11, 3.12, 3.13
- [Fluidsynth](https://www.fluidsynth.org/)
- [PortAudio](http://www.portaudio.com/)

First, install Fluidsynth, and then install the `pyfluidsynth` Python library. We recommend to install Fluidsynth using conda as well (see instructions below).

Note that `pyfluidsynth` only provides Python bindings for Fluidsynth; it does not install Fluidsynth itself. Be aware that there is also a `fluidsynth` Python library (without the `py-` prefix), but it is not compatible with `matchmaker`. We recommend installing Fluidsynth using conda

### Install from source using conda

Setting up the code as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
# Clone matchmaker
git clone https://github.com/pymatchmaker/matchmaker.git

# Create the conda environment
conda create -n matchmaker python=3.12

conda activate matchmaker

# Go to matchmaker directory
cd matchmaker

# Install matchmaker in editable mode
pip install -e ."[dev]"

# Install GCC
conda install -c conda-forge gcc=12.1.0

# Install glib and fluidsynth
conda install -c conda-forge glib fluidsynth
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

### Quickstart for live streaming

To get started quickly, you can use the `Matchmaker` class, which provides a simple interface for running the alignment process. You can use a `musicxml` or `midi` file as the score file. Specify `"audio"` or `"midi"` as the `input_type` argument, and the default device for that input type will be automatically set up.

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="path/to/score",
    input_type="audio",
)
for current_position in mm.run():
    print(current_position)  # beat position in the score
```

The returned value is the current position in the score, represented in beats defined by `partitura` library's note array system.
Specifically, each position is calculated for every frame input and interpolated within the score's `onset_beat` array.
Please refer to [here](https://partitura.readthedocs.io/en/latest/Tutorial/notebook.html) for more information about the `onset_beat` concept.

### Testing with the performance file

You can simulate the real-time alignment by putting a specific performance file as input, rather than running it as a live stream.
The type of performance file can be either audio file or midi file, depending on the `input_type`.

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="path/to/score",
    performance_file="path/to/performance.mid",
    input_type="midi",
)
for current_position in mm.run():
    print(current_position)
```

### Testing with Specific Input Device

To use a specific audio or MIDI device that is not the default device, you can pass the device name or index.
By default, `input_type` is set to `“audio”`. If you are using a MIDI device, you can change the input type to `“midi”`.

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="path/to/score",
    input_type="audio",
    device_name_or_index="MacBookPro Microphone",
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
- `device_name_or_index` (str or int): The audio/MIDI device name or index you want to use. If `None`, the default device will be used.

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
