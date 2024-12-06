Installation
============

Setup
-----

Install from source using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting up the code as described here requires `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Follow the instructions for your OS.

To setup the experiments, use the following script:

.. code-block:: bash

    # Clone matchmaker
    git clone https://github.com/pymatchmaker/matchmaker.git

    cd matchmaker

    # Create the conda environment
    conda env create -f environment.yml

    # Install matchmaker
    pip install -e .

If you have a ImportError with 'Fluidsynth' by ``pyfluidsynth`` library on MacOS, please refer to the following `link <https://stackoverflow.com/a/75339618>`_.

Usage Examples
------------

Quickstart for live streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~

To get started quickly, you can use the ``Matchmaker`` class, which provides a simple interface for running the alignment process. You can use a ``musicxml`` or ``midi`` file as the score file. Specify ``"audio"`` or ``"midi"`` as the ``input_type`` argument, and the default device for that input type will be automatically set up. For options regarding the ``method``, please refer to the :ref:`alignment-methods` section.

.. code-block:: python

    from matchmaker import Matchmaker

    mm = Matchmaker(
        score_file="path/to/score",
        input_type="audio",
        method="dixon",
    )
    for current_position in mm.run():
        print(current_position)  # beat position in the score

The returned value is the current position in the score, represented in beats defined by ``partitura`` library's note array system.
Specifically, each position is calculated for every frame input and interpolated within the score's ``onset_beat`` array.
Please refer to `here <https://partitura.readthedocs.io/en/latest/Tutorial/notebook.html>`_ for more information about the ``onset_beat`` concept.

Testing with the performance file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can simulate the real-time alignment by putting a specific performance file as input, rather than running it as a live stream.
The type of performance file can be either audio file or midi file, depending on the ``input_type``.

.. code-block:: python

    from matchmaker import Matchmaker

    mm = Matchmaker(
        score_file="path/to/score",
        performance_file="path/to/performance.mid",
        input_type="midi",
        feature_type="mel",
        method="hmm",
    )
    for current_position in mm.run():
        print(current_position)  # beat position in the score

Testing with Specific Input Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a specific audio or MIDI device that is not the default device, you can pass the device name or index.

.. code-block:: python

    from matchmaker import Matchmaker

    mm = Matchmaker(
        score_file="path/to/score",
        input_type="audio",
        feature_type="chroma",
        method="arzt",
        device_name_or_index="MacBookPro Microphone",
    )
    for current_position in mm.run():
        print(current_position)  # beat position in the score