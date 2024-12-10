#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module contains a mediator for filtering out
MIDI messages "performed" by an internal process/thread
from the input MIDI (from a true/non-internal source).
Using the mediator is only necessary for old instruments
that do not filter the messages automatically. So far we have only
encountered this issue with the Bösendorfer CEUS, and thus
the name of the class.
"""

import collections
import threading


class ThreadMediator(object):
    """
    Mediator class for communications between a main system
    (e.g., an accompaniment system) modules running in
    concurrent threads or processes. The class ensures thread safety.

    Parameters
    ----------
    _comms_buffer : deque
        A buffer to receive the output from the one process and send it to
        another when prompted. Follows LIFO (Last In, First Out) logic.
        For the buffer a deque object is used as this ensures thread safety.
    """

    _comms_buffer: collections.deque
    _mediator_type: str

    def __init__(self, **kwargs) -> None:
        """
        The initialization method.
        """
        # Define the comms buffer:
        self._comms_buffer = collections.deque(maxlen=200)
        # A name variable to store the type of the mediator:
        self._mediator_type = "default"
        # Call the superconstructor:
        super().__init__(**kwargs)

    def is_empty(self) -> bool:
        """
        Returns True if the comms buffer is empty.
        False if it has at least one element.

        Returns
        -------
        empty : Boolean
            True if the comms buffer is empty.
            False if it has at least one element.
        """
        # Check if the buffer is empty:
        if len(self._comms_buffer) == 0:
            return True
        # Otherwise return false:
        return False

    def get_message(self) -> collections.namedtuple:
        """
        Get the first from the previously sent messages MIDI messages.
        Returns IndexError if there is no element in the buffer.
        This should only be called by the main thread producing MIDI messages.

        Returns
        -------
        message : collections.namedtuple
            The message to be returned.
        """
        # Return the first element:
        return self._comms_buffer.popleft()

    def put_message(self, message: collections.namedtuple) -> None:
        """
        Put a message into the comms buffer.
        This should only be called by matchmaker.

        Parameters
        ----------
        message : collections.namedtuple
            The message to be put into the buffer.
        """
        self._comms_buffer.append(message)

    @property
    def mediator_type(self) -> str:
        """
        Property method to return the value of the mediator_type variable.
        """
        return self._mediator_type


class CeusMediator(ThreadMediator):
    """
    Encapsulates the trans-module communication in the context of a
    Bösendorfer CEUS System. It also filters notes (MIDI pitches)
    that are played by the Ceus (i.e., sent from an internal MIDI port).

    Parameters
    ----------
    _Ceus_filter : deque
        The filter buffer for notes (MIDI pitches) played back by Ceus.

    _comms_buffer : deque
        A buffer to receive the output from the one process and send it to
        another when promped. Follows LIFO (Last In, First Out) logic. For the
        buffer a deque object is used as this ensures thread safety.
    """

    _ceus_lock: threading.RLock

    def __init__(self, **kwargs):
        """
        The initialization method.
        """
        # A lock to ensure thread safety of the Ceus filter:
        self._ceus_lock = threading.RLock()

        # Define the Ceus filter:
        self._ceus_filter = collections.deque(maxlen=10)
        # Call the superconstructor:
        super().__init__(**kwargs)

        # A name variable to store the type of the mediator:
        self._mediator_type = "ceus"

    def filter_check(
        self,
        midi_pitch: int,
        delete_entry: bool = True,
    ) -> bool:
        """
        Check if the midi pitch is in the Ceus filter. Return True if yes,
        False if it is not present. Delete the filter entry if specified by
        delete_entry.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be checked against the filter.

        delete_entry : bool
            Specifies whether to delete the filter entry if such is found.
            Default: True

        Returns
        -------
        indicate : bool
            True if pitch is in the filter, False if it is not.
        """

        with self._ceus_lock:
            # Check if the entry is in the filter:
            if midi_pitch in self._ceus_filter:
                # Whether to delete the entry:
                if delete_entry:
                    self.filter_remove_pitch(midi_pitch)

                # Return true:
                return True

            # If it is not, return False:
            return False

    def filter_append_pitch(self, midi_pitch: int) -> None:
        """
        Append a MIDI pitch to the Ceus filter. This should only be called by
        the main thread producing/sending MIDI messages.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be appended to the filter.
        """
        with self._ceus_lock:
            # Append the midi pitch to be filtered:
            self._ceus_filter.append(midi_pitch)

    def filter_remove_pitch(self, midi_pitch: int) -> None:
        """
        Remove a MIDI pitch from the Ceus filter. This should only be
        called by a score follower.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be removed from the filter.
        """
        with self._ceus_lock:
            # Remove the pitch from filter:
            self._ceus_filter.remove(midi_pitch)
