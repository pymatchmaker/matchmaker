# Matchmaker

Matchmaker is a Python library for real-time music alignment.

Music alignment is a fundamental MIR task, and real-time music alignment is a necessary component of many interactive applications (e.g., automatic accompaniment systems, automatic page turning).

Unlike offline alignment methods, for which state-of-the-art implementations are publicly available, real-time (online) methods have no standard implementation, forcing researchers and developers to build them from scratch for their projects.
  
We aim to provide efficient reference implementations of score followers for use in real-time applications which can be easily integrated into existing projects.


## Setup

### Install from PyPI

TBD!!


### Install from source using conda

Setting up the code as described here requires [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Follow the instructions for your OS.

To setup the experiments, use the following script.

```bash
# Clone matchmaker
git clone https://github.com/CarlosCancino-Chacon/matchmaker.git

cd matchmaker

# Create the conda environment
conda env create -f environment.yml

# Install matchmaker
pip install -e .
```

## Example Usage

TBD

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("*Rach3: A Computational Approach to Study Piano Rehearsals*"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).