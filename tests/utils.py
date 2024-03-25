#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utilities for tests
"""
import numbers
import threading
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import mido
import numpy as np
import partitura as pt
from partitura.performance import PerformanceLike

from matchmaker.io.midi import MockFramedMidiStream, MockMidiStream, POLLING_PERIOD
from matchmaker.io.audio import MockAudioStream, SAMPLE_RATE, HOP_LENGTH, CHUNK_SIZE
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.symbolic import save_wav_fluidsynth
import tempfile

# Random number generator
RNG = np.random.RandomState(1984)


def make_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    centers: Optional[int] = None,
    cluster_std=1.0,
    center_box=(-10.0, 10.0),
    random_state: np.random.RandomState = RNG,
    dtype=np.float32,
):
    """
    Generate isotropic Gaussian blobs for clustering.

    This code and its documentation were slightly adapted from

    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_samples_generator.py

    Following Scikit-learn's license, this function is distributed under the
    BSD 3-Clause License (see OPEN_SOURCE_LICENSES.md).

    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of points equally divided among
        clusters.

    n_features : int, default=2
        The number of features for each sample.

    centers : int or ndarray of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.

    cluster_std : float or array-like of float, default=1.0
        The standard deviation of the clusters.

    center_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_centers : bool, default=False
        If True, then return the centers of each cluster.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.

    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster. Only returned if
        ``return_centers=True``.

    """

    if isinstance(n_samples, numbers.Integral):
        # Set n_centers by looking at centers arg
        if centers is None:
            centers = 3

        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = random_state.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )

        else:
            n_features = centers.shape[1]
            n_centers = centers.shape[0]

    else:
        # Set n_centers by looking at [n_samples] arg
        n_centers = len(n_samples)
        if centers is None:
            centers = random_state.uniform(
                center_box[0], center_box[1], size=(n_centers, n_features)
            )
        try:
            assert len(centers) == n_centers
        except TypeError as e:
            raise ValueError(
                "Parameter `centers` must be array-like. Got {!r} instead".format(
                    centers
                )
            ) from e
        except AssertionError as e:
            raise ValueError(
                "Length of `n_samples` not consistent with number of "
                f"centers. Got n_samples = {n_samples} and centers = {centers}"
            ) from e
        else:
            n_features = centers.shape[1]

    # stds: if cluster_std is given as list, it must be consistent
    # with the n_centers
    if hasattr(cluster_std, "__len__") and len(cluster_std) != n_centers:
        raise ValueError(
            "Length of `clusters_std` not consistent with "
            "number of centers. Got centers = {} "
            "and cluster_std = {}".format(centers, cluster_std)
        )

    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.full(len(centers), cluster_std)

    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers

        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1

    cum_sum_n_samples = np.cumsum(n_samples_per_center)
    X = np.empty(
        shape=(sum(n_samples_per_center), n_features),
        dtype=dtype,
    )
    y = np.empty(shape=(sum(n_samples_per_center),), dtype=int)

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        start_idx = cum_sum_n_samples[i - 1] if i > 0 else 0
        end_idx = cum_sum_n_samples[i]
        X[start_idx:end_idx] = random_state.normal(
            loc=centers[i], scale=std, size=(n, n_features)
        )
        y[start_idx:end_idx] = i

    # shuffle arrays
    # This is a simplified version of sklearn's utils_shuffle
    # to avoid duplicating more code.
    shuffle_idxs = np.arange(len(X))
    random_state.shuffle(shuffle_idxs)
    X = X[shuffle_idxs]
    y = y[shuffle_idxs]

    return X, y


def generate_example_sequences(
    lenX: int = 100,
    centers: int = 3,
    n_features: int = 5,
    maxreps: int = 4,
    minreps: int = 1,
    noise_scale: float = 0.01,
    random_state: np.random.RandomState = RNG,
    dtype=np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates example pairs of related sequences. Sequence X are samples of
    an K-dimensional space around a specified number of centroids.
    Sequence Y is a non-constant "time-streched" version of X with some
    noise added.

    Parameters
    ----------
    lenX : int
        Number of elements in the X sequence

    centers: int
        Number of different centers ("classes") that the elements
        of the sequences represent

    n_features: int
        Dimensionality of the features ($K$) in the notation of the
        Notebook

    noise_scale: float
        Scale of the noise

    Returns
    -------
    X : np.ndarray
        Sequence X (a matrix where each row represents
        an element of the sequence)

    Y: np.ndarray
        Sequence Y

    ground_truth_path: np.ndarray
        Alignment between X and Y where the first column represents the indices
        in X and the second column represents the corresponding index in Y.
    """

    X, _ = make_blobs(
        n_samples=lenX,
        centers=centers,
        n_features=n_features,
        dtype=dtype,
    )
    # Time stretching X! each element in sequence X is
    # repeated a random number of times
    # and then we add some noise to spice things up :)

    if minreps == maxreps:
        n_reps = np.ones(len(X), dtype=int) * minreps
    else:
        n_reps = random_state.randint(minreps, maxreps, len(X))
    y_idxs = [rp * [i] for i, rp in enumerate(n_reps)]
    y_idxs = np.array([el for reps in y_idxs for el in reps], dtype=int)
    # Add a bias, so that Y has a different "scaling" than X
    Y = X[y_idxs]
    # add some noise
    Y += noise_scale * random_state.randn(*Y.shape)
    ground_truth_path = np.column_stack((y_idxs, np.arange(len(Y))))

    X = X.astype(dtype)
    Y = Y.astype(dtype)
    return X, Y, ground_truth_path


def process_midi_offline(
    perf_info: Union[PerformanceLike, str],
    features: List[Callable],
    polling_period: Optional[float] = POLLING_PERIOD,
) -> List[Any]:
    """
    Helper method to process all MIDI
    """

    queue = RECVQueue()

    if polling_period is not None:
        input_stream = MockFramedMidiStream(
            file_path=perf_info,
            queue=queue,
            polling_period=polling_period,
            features=features,
            return_midi_messages=False,
            mediator=None,
        )
    else:
        input_stream = MockMidiStream(
            file_path=perf_info,
            queue=queue,
            features=features,
            return_midi_messages=False,
            mediator=None,
        )

    input_stream.start()
    input_stream.join()

    outputs = list(queue.queue)

    return outputs


def process_audio_offline(
        perf_info: Union[PerformanceLike, str],
        features: List[Callable],
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        chunk_size: int = CHUNK_SIZE,
) -> List[Any]:
    
    queue = RECVQueue()

    if isinstance(perf_info, str):
        file_path = perf_info
        temp_file = None
    elif isinstance(perf_info, (pt.score.Score, pt.performance.Performance, pt.score.Part, pt.performance.PerformedPart,)):
        # TODO modify mock audio stream to handle
        # an input numpy array

        # Create a temporary file and get its path
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav") 
        file_path = temp_file.name

        save_wav_fluidsynth(
            input_data=perf_info,
            out=file_path,
        )
    
    input_stream = MockAudioStream(
        file_path=file_path,
        queue=queue,
        features=features,
        sample_rate=sample_rate,
        hop_length=hop_length,
        chunk_size=chunk_size,
    )

    input_stream.start()
    input_stream.join()

    outputs = list(queue.queue)

    if temp_file is not None:
        temp_file.close()

    return outputs
# Create a temporary file and get its path
temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file_path = temp_file.name

if __name__ == "__main__":

    lenX = 5
    centers = 3
    n_features = 5
    maxreps = 4
    minreps = 1
    noise_scale = 0.01

    X, Y, ground_truth_path = generate_example_sequences(
        lenX=lenX,
        centers=centers,
        n_features=n_features,
        maxreps=maxreps,
        minreps=minreps,
        noise_scale=noise_scale,
        random_state=RNG,
    )


class DummyMidiPlayer(threading.Thread):
    def __init__(self, port: mido.ports.BaseOutput, filename: str) -> None:
        threading.Thread.__init__(self)
        self.port = port
        self.mf = mido.MidiFile(filename)
        self.is_playing = False

    def run(self) -> None:
        self.is_playing = True
        for msg in self.mf.play():
            self.port.send(msg)

        # close port
        self.is_playing = False
        self.port.close()
