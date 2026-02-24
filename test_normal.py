from matchmaker import Matchmaker
import matplotlib.pyplot as plt
import pickle

input_type = "audio"

score_file = "/home/suhit/JKU/Repos/matchmaker/matchmaker/assets/mozart_k265_var1.musicxml"
score_file2 = '/home/suhit/Documents/Datasets/datasets/vienna4x22/musicxml_adjusted/Mozart_K331_1st-mov.musicxml'

perf_file_midi = "/home/suhit/JKU/Repos/matchmaker/matchmaker/assets/mozart_k265_var1.mid"
perf_file_midi2 = "/home/suhit/Documents/Datasets/datasets/vienna4x22/midi/Mozart_K331_1st-mov_p01.mid"

perf_file_audio = "/home/suhit/JKU/Repos/matchmaker/matchmaker/assets/mozart_k265_var1.mp3"
perf_file_audio2 = "/home/suhit/Documents/Datasets/datasets/vienna4x22/audio/Mozart/Mozart_K331_1st-mov_p02.wav"

if input_type == "midi":
    feature_type = "pitch_ioi"
    performance_file = perf_file_midi2
else:
    feature_type = "chroma"
    performance_file = perf_file_audio2

mm = Matchmaker(
    score_file=score_file2,
    performance_file=performance_file,
    input_type=input_type,
    feature_type=feature_type,
    method="pf",
)

for current_position in mm.run():
    print(current_position)