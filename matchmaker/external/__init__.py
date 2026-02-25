#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Top module for alignment methods imported from the parangonar library:
https://github.com/sildater/parangonar
"""
import parangonar as pa
# from matchmaker.base import OnlineAlignment
from typing import Callable, Dict, Generator
from numpy.typing import NDArray

QUEUE_SENTINEL = object()

class OnlineParangonarAlignment():#(OnlineAlignment):
    def __init__(self, 
                 queue,
                 score_note_array,
                 parangonar_tracker_type: str = "SLT_OLTW"):
        # an instance of 
        self.queue = queue
        if parangonar_tracker_type == "SLT_OLTW":
            self.parangonar_tracker = pa.TOLTWMatcher(score_note_array, 
                                                      tracker_type=parangonar_tracker_type)
        elif parangonar_tracker_type == "SL_OLTW":
            self.parangonar_tracker = pa.OLTWMatcher(score_note_array, 
                                                    tracker_type=parangonar_tracker_type)
        elif parangonar_tracker_type == "OPTM":
            self.parangonar_tracker = pa.OnlinePureTransformerMatcher(score_note_array)
        elif parangonar_tracker_type == "OTM":
            self.parangonar_tracker = pa.OnlineTransformerMatcher(score_note_array)

    def __call__(self, performance_note):
        # process
        score_position = self.parangonar_tracker(performance_note)
        return score_position

    def run(self) -> Generator[int, None, float]:
        while self.parangonar_tracker.is_still_following():
            input_feature = self.queue.get(block=True)
            if input_feature is QUEUE_SENTINEL:
                print("empty queue")
                return None
            else:
                current_state = self(input_feature)
                yield current_state

        return None
    
    def run_offline(self):
        self.queue.put(QUEUE_SENTINEL)
        for position in self.run():
            print(position, self.parangonar_tracker.unique_onsets[position])
        
        return self.parangonar_tracker.warping_path
    
if __name__ == "__main__":
    import partitura as pt
    from queue import Queue
    # load the example match file included in the library
    perf_match, groundtruth_alignment, score_match = pt.load_match(
        filename= pa.EXAMPLE, # 
        create_score=True
    )

    # compute note arrays from the loaded score and performance
    pna_match = perf_match[0].note_array()
    sna_match = score_match[0].note_array(include_grace_notes=True)

    # create queue
    input_queue = Queue()
    for note_row in pna_match:
        input_queue.put(note_row)

    # create matchmaker follower
    score_follower = OnlineParangonarAlignment(
        input_queue, sna_match, "OTM")

    # run the follower offline
    warping_path = score_follower.run_offline()



    

