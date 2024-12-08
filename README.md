# Matchmaker

Matchmaker is a Python library for real-time music alignment.

Music alignment is a fundamental MIR task, and real-time music alignment is a necessary component of many interactive applications (e.g., automatic accompaniment systems, automatic page turning).

Unlike offline alignment methods, for which state-of-the-art implementations are publicly available, real-time (online) methods have no standard implementation, forcing researchers and developers to build them from scratch for their projects.
  
We aim to provide efficient reference implementations of score followers for use in real-time applications which can be easily integrated into existing projects.

The full documentation for matchmaker is available online at [readthedocs.org](https://matchmaker.readthedocs.io).


## Setup

### Prerequisites

- Available Python version: 3.9, 3.10, 3.11, 3.12 (3.12 recommended)
- [Fluidsynth](https://www.fluidsynth.org/)
- [PortAudio](http://www.portaudio.com/)

Please ensure that you've installed the above packages before proceeding.
You should not install `fluidsynth` using `pip install fluidsynth` as it is not compatible with `matchmaker`.

### Install from PyPI

```bash
pip install pymatchmaker
```

### Install from source using conda

Please refer to the [requirements.txt](requirements.txt) file for the minimum required versions of the packages.
Setting up the code as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
# Clone matchmaker
git clone https://github.com/pymatchmaker/matchmaker.git

cd matchmaker

# Create the conda environment
conda create -n matchmaker python=3.12
conda activate matchmaker

# Install matchmaker
pip install -e .

# Install matchmaker with dev tools
pip install -e .[dev]
```

If you have a ImportError with 'Fluidsynth' by `pyfluidsynth` library on MacOS, please refer to the following [link](https://stackoverflow.com/a/75339618).

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

### Testing with Different Methods or Features

For testing with Audio input, you can specify the alignment method as follows:

```python
from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="path/to/score",
    input_type="audio",
    method="dixon",  # or "arzt" (default)
)
for current_position in mm.run():
    print(current_position)
```

For options regarding the `method`, please refer to the [Alignment Methods](#alignment-methods) section.
For options regarding the `feature_type`, please refer to the [Features](#features) section.

### Custom Example

If you want to use a different alignment method or custom method, you can do so by importing the specific class and passing the necessary parameters.
In order to define a custom alignment class, you need to inherit from the Base `OnlineAlignment` class and implement the `run` method. Note that the returned value from the `OnlineAlignment` class should be the current frame number in the reference features, not in beats.

```python
from matchmaker.dp import OnlineTimeWarpingDixon
from matchmaker.io.audio import AudioStream
from matchmaker.features import ChromagramProcessor

feature_processor = ChromagramProcessor()
reference_features = feature_processor('path/to/score/audio.wav')

with AudioStream(processor=feature_processor) as stream:
    score_follower = OnlineTimeWarpingDixon(reference_features, stream.queue)
    for current_frame in score_follower.run():
        print(current_frame)  # frame number in the reference features
```

## Alignment Methods

Matchmaker currently supports the following alignment methods:

- `"dixon"`: On-line time warping algorithm by S. Dixon (2005). Supports audio input only.
- `"arzt"`: On-line time warping algorithm adapted from Brazier and Widmer (2020) (based on the work by Arzt et al. (2010)). Supports audio input only.
- `"hmm"`: Hidden Markov Model-based score follower by Cancino-Chacón et al. (2023), based on the state-space score followers by Duan et al. (2011) and Jiang and Raphael (2020). Supports MIDI input only.

## Features

Matchmaker currently supports the following feature types:

- For audio:
  - `"chroma"`: Chroma features. Default feature type for audio input.
  - `"mfcc"`: Mel-frequency cepstral coefficients.
  - `"mel"`: Mel-Spectrogram.
  - `"logspectral"`: Log-spectral features used in Dixon (2005).
- For MIDI:
  - `pianoroll`: Piano-roll features. Default feature type for MIDI input.
  - `"pitch"`: Pitch features for MIDI input.
  - `"pitchclass"`: Pitch class features for MIDI input.

## Configurations

Initialization parameters for the `Matchmaker` class:

- `score_file` (str): Path to the score file.
- `input_type` (str): Type of input data. Options: `"audio"`, `"midi"`.
- `feature_type` (str): Type of feature to use. Options: `"chroma"`, `"mfcc"`, `"cqt"`, `"spectrogram"`, `"onset"`.
- `method` (str): Alignment method to use. Options: `"dixon"`, `"arzt"`, `"hmm"`.
- `sample_rate` (int): Sample rate of the input audio data.
- `frame_rate` (int): Frame rate of the input audio/MIDI data.
- `device_name_or_index` (str or int): The audio/MIDI device name or index you want to use. If `None`, the default device will be used.

## Citing Matchmaker

If you find Matchmaker useful, we would appreciate if you could cite us!

```
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
