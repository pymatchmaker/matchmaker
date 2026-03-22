#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Queue utilities for matchmaker streaming.
"""

from queue import Empty, Queue
from typing import Any


class RECVQueue(Queue):
    """Queue with a recv method (like Pipe).

    Uses Queue.get with a timeout to remain interruptable via
    KeyboardInterrupt. Periodically queries with a 1s timeout as a
    middle ground between busy-waiting and uninterruptable blocked waiting.
    """

    def __init__(self) -> None:
        Queue.__init__(self)

    def recv(self) -> Any:
        """Return and remove an item from the queue."""
        while True:
            try:
                return self.get(timeout=1)
            except Empty:  # pragma: no cover
                pass

    def poll(self) -> bool:
        return self.empty()
