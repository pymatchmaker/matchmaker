from matchmaker import Matchmaker

mm = Matchmaker(
    score_file="matchmaker/assets/mozart_k265_var1.musicxml",
    performance_file="matchmaker/assets/mozart_k265_var1.mid",
    input_type="midi",
    feature_type="pitchclass",
    method="outerhmm",
)

for current_position in mm.run():
    print(current_position)