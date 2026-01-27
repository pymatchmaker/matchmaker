#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Top module for alignment methods imported from the parangonar library:
https://github.com/sildater/parangonar
"""
import parangonar as pa
from matchmaker.base import OnlineAlignment
from typing import Callable, Dict, Generator
from numpy.typing import NDArray

QUEUE_SENTINEL = object()

class OnlineParangonarAlignment(OnlineAlignment):
    def __init__(self, 
                 parangonar_tracker):
        # an instance of 
        self.parangonar_tracker = parangonar_tracker

    def __call__(self, performance_note):
        # process
        score_position = self.parangonar_tracker.step(performance_note)
        return score_position

    def run(self) -> Generator[int, None, float]:
        while self.parangonar_tracker.is_still_following():
            input_feature = self.queue.get(block=True)
            if input_feature == QUEUE_SENTINEL:
                print("empty queue")
                return None
            else:
                current_state = self(input_feature)
                yield current_state

        return None
    

