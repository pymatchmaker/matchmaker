import pickle

import matplotlib.pyplot as plt
import numpy as np

from matchmaker import TOLERANCES_IN_MILLISECONDS, Matchmaker
from matchmaker.utils.eval import (
    get_evaluation_results,
    transfer_from_perf_to_predicted_score,
    transfer_from_score_to_predicted_perf,
)

input_type = "audio"

score_file = "./matchmaker/assets/mozart_k265_var1.musicxml"
score_file2 = "/home/suhit/Documents/Datasets/datasets/vienna4x22/musicxml_adjusted/Mozart_K331_1st-mov.musicxml"

perf_file_midi = "./matchmaker/assets/mozart_k265_var1.mid"
perf_file_midi2 = "/home/suhit/Documents/Datasets/datasets/vienna4x22/midi/Mozart_K331_1st-mov_p01.mid"

perf_file_audio = "./matchmaker/assets/mozart_k265_var1.mp3"
perf_file_audio2 = "/home/suhit/Documents/Datasets/datasets/vienna4x22/audio/Mozart/Mozart_K331_1st-mov_p02.wav"

perf_annotations = "./matchmaker/assets/mozart_k265_var1_annotations.txt"

if input_type == "midi":
    feature_type = "pitch_ioi"
    performance_file = perf_file_midi2
else:
    feature_type = "chroma"
    performance_file = perf_file_audio

mm = Matchmaker(
    score_file=score_file,
    performance_file=performance_file,
    input_type=input_type,
    feature_type=feature_type,
    method="pf",
)

for current_position in mm.run():
    print(current_position)


score_annots = mm.build_score_annotations(level="beat")
perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)

perf_annots_predicted = transfer_from_score_to_predicted_perf(
    mm.score_follower.warping_path,
    score_annots,
    frame_rate=mm.frame_rate,
)

score_annots_predicted = transfer_from_perf_to_predicted_score(
    mm.score_follower.warping_path,
    perf_annots,
    frame_rate=mm.frame_rate,
)
score_annots = score_annots[: len(score_annots_predicted)]

get_evaluation_results(
    perf_annots,
    perf_annots_predicted,
    tolerances=TOLERANCES_IN_MILLISECONDS,
    total_length=len(score_annots_predicted),
)
